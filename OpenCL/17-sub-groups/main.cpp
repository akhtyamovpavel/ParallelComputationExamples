#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <string>

#define CL_TARGET_OPENCL_VERSION 210
#include <CL/cl.h>

#define CL_CHECK(call) do {                                                   \
    cl_int _err = (call);                                                     \
    if (_err != CL_SUCCESS) {                                                 \
        std::cerr << "OpenCL error " << _err << " at "                        \
                  << __FILE__ << ":" << __LINE__ << std::endl;                \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// ============================================================================
// Sub-groups (подгруппы) — аналог варпов (warps) в CUDA.
//
// На аппаратном уровне work-items внутри одной подгруппы выполняются
// синхронно (в lockstep). Для NVIDIA GPU размер подгруппы = 32 (варп),
// для AMD — 32 или 64 (волновой фронт, wavefront).
//
// Ключевое преимущество: операции внутри подгруппы НЕ требуют барьеров
// и работы с __local памятью, что даёт прирост производительности.
// Это аналог CUDA-примера 04.7-warp-primitives, где используются
// __shfl_down_sync и warp-level редукция.
//
// В OpenCL 2.0 подгруппы доступны через расширение cl_khr_subgroups.
// В OpenCL 2.1+ подгруппы стали частью стандарта (core feature).
//
// Мы реализуем два ядра редукции:
//   1) reduce_local — классическая редукция через __local память и барьеры
//      (как в примере 03-reduction)
//   2) reduce_subgroup — двухуровневая редукция: сначала внутри подгруппы
//      через sub_group_reduce_add() (без барьеров!), затем комбинирование
//      результатов подгрупп в __local памяти
// ============================================================================

// Ядро 1: классическая редукция через __local память (базовая линия).
static const char* kernel_local_source = R"CLC(
__kernel void reduce_local(__global const int* in,
                           __global int* out,
                           __local int* scratch,
                           const int n) {
    unsigned int tid = get_local_id(0);
    unsigned int gid = get_global_id(0);

    // Загрузка в local memory; за границей массива — 0 (нейтральный элемент).
    scratch[tid] = (gid < n) ? in[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Дерево редукции — log2(local_size) шагов, каждый с барьером.
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        out[get_group_id(0)] = scratch[0];
    }
}
)CLC";

// Ядро 2: двухуровневая редукция с использованием sub-groups.
// Первый уровень — sub_group_reduce_add() внутри подгруппы (без барьера!).
// Второй уровень — один work-item от каждой подгруппы записывает результат
// в __local память, а затем первая подгруппа финализирует.
static const char* kernel_subgroup_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

__kernel void reduce_subgroup(__global const int* in,
                              __global int* out,
                              __local int* scratch,
                              const int n) {
    unsigned int tid = get_local_id(0);
    unsigned int gid = get_global_id(0);

    // Каждый work-item загружает свой элемент.
    int val = (gid < n) ? in[gid] : 0;

    // --- Уровень 1: редукция внутри подгруппы ---
    // sub_group_reduce_add() суммирует val по всем work-items подгруппы.
    // Это аппаратная операция — аналог __shfl_down_sync + цикла в CUDA.
    // Барьер НЕ нужен: work-items в подгруппе идут синхронно.
    int sg_sum = sub_group_reduce_add(val);

    // get_sub_group_local_id() — индекс внутри подгруппы (0..sub_group_size-1)
    // get_sub_group_id()       — номер подгруппы внутри work-group
    // get_sub_group_size()     — размер подгруппы (аналог warpSize в CUDA)
    unsigned int sg_lid = get_sub_group_local_id();
    unsigned int sg_id  = get_sub_group_id();

    // Первый work-item каждой подгруппы записывает частичную сумму в scratch.
    if (sg_lid == 0) {
        scratch[sg_id] = sg_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Уровень 2: финальная редукция частичных сумм подгрупп ---
    // Количество подгрупп в work-group = get_num_sub_groups().
    // Первая подгруппа делает финальную редукцию.
    unsigned int num_sg = get_num_sub_groups();
    if (sg_id == 0) {
        int partial = (sg_lid < num_sg) ? scratch[sg_lid] : 0;
        int total = sub_group_reduce_add(partial);
        if (sg_lid == 0) {
            out[get_group_id(0)] = total;
        }
    }
}
)CLC";

static cl_device_id get_gpu_device() {
    cl_uint num_platforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    for (cl_uint p = 0; p < num_platforms; ++p) {
        cl_uint nd = 0;
        cl_int err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU,
                                    0, nullptr, &nd);
        if (err != CL_SUCCESS || nd == 0) continue;
        std::vector<cl_device_id> devs(nd);
        CL_CHECK(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU,
                                nd, devs.data(), nullptr));
        return devs[0];
    }
    std::cerr << "No OpenCL GPU device found." << std::endl;
    std::exit(1);
}

// Проверка: поддерживает ли устройство OpenCL >= 2.0 и расширение cl_khr_subgroups.
static bool check_subgroup_support(cl_device_id dev) {
    // Проверяем версию OpenCL C, поддерживаемую устройством.
    char version_str[256] = {};
    CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_VERSION,
                             sizeof(version_str), version_str, nullptr));
    std::cout << "Device OpenCL C version: " << version_str << std::endl;

    // Формат строки: "OpenCL C X.Y ..."
    int major = 0, minor = 0;
    if (std::sscanf(version_str, "OpenCL C %d.%d", &major, &minor) != 2) {
        std::cerr << "Не удалось разобрать версию OpenCL C." << std::endl;
        return false;
    }

    if (major < 2) {
        std::cerr << "Требуется OpenCL C 2.0+, доступна " << major << "."
                  << minor << ". Sub-groups недоступны." << std::endl;
        return false;
    }

    // В OpenCL 2.1+ подгруппы — core feature.
    // В OpenCL 2.0 нужно расширение cl_khr_subgroups.
    if (major == 2 && minor == 0) {
        size_t ext_size = 0;
        CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size));
        std::string extensions(ext_size, '\0');
        CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, ext_size,
                                 &extensions[0], nullptr));

        if (extensions.find("cl_khr_subgroups") == std::string::npos) {
            std::cerr << "OpenCL 2.0, но расширение cl_khr_subgroups отсутствует.\n"
                      << "Sub-groups недоступны на данном устройстве." << std::endl;
            return false;
        }
        std::cout << "OpenCL 2.0: расширение cl_khr_subgroups найдено." << std::endl;
    } else {
        std::cout << "OpenCL " << major << "." << minor
                  << ": подгруппы — core feature." << std::endl;
    }
    return true;
}

static cl_program build_program(cl_context ctx, cl_device_id dev,
                                const char* source, const char* options) {
    cl_int err = 0;
    size_t src_len = std::strlen(source);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &source, &src_len, &err);
    CL_CHECK(err);
    err = clBuildProgram(prog, 1, &dev, options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                              log_size, &log[0], nullptr);
        std::cerr << "Build failed:\n" << log << std::endl;
        clReleaseProgram(prog);
        std::exit(1);
    }
    return prog;
}

int main() {
    const int N = 1 << 22;  // ~4 миллиона элементов
    const size_t local_size = 256;
    const size_t num_groups = (N + local_size - 1) / local_size;
    const size_t global_size = num_groups * local_size;

    // Входной массив — все единицы, ожидаемая сумма = N.
    std::vector<int> h_in(N, 1);

    cl_device_id dev = get_gpu_device();

    // Проверяем поддержку sub-groups.
    if (!check_subgroup_support(dev)) {
        std::cout << "Sub-groups не поддерживаются — завершение." << std::endl;
        return 0;
    }

    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev,
                                                  CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    // --- Сборка двух ядер ---
    // Ядро с __local памятью не требует OpenCL 2.0 — собираем без -cl-std.
    cl_program prog_local = build_program(ctx, dev, kernel_local_source, nullptr);
    // Ядро с sub-groups требует -cl-std=CL2.0.
    cl_program prog_sg = build_program(ctx, dev, kernel_subgroup_source,
                                       "-cl-std=CL2.0");

    cl_kernel k_local = clCreateKernel(prog_local, "reduce_local", &err);
    CL_CHECK(err);
    cl_kernel k_sg = clCreateKernel(prog_sg, "reduce_subgroup", &err);
    CL_CHECK(err);

    // --- Выводим размер подгруппы ---
    // Узнаём preferred sub-group size у ядра. Это аналог warpSize в CUDA.
    size_t sg_size = 0;
    CL_CHECK(clGetKernelSubGroupInfo(k_sg, dev,
                                     CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
                                     sizeof(local_size), &local_size,
                                     sizeof(sg_size), &sg_size, nullptr));
    std::cout << "\nРазмер подгруппы (sub-group size): " << sg_size
              << "  (аналог warpSize в CUDA)" << std::endl;
    std::cout << "Подгрупп в work-group: " << local_size / sg_size << std::endl;
    std::cout << std::endl;

    // --- Буферы ---
    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 N * sizeof(int), h_in.data(), &err);
    CL_CHECK(err);
    cl_mem d_out_local = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                        num_groups * sizeof(int), nullptr, &err);
    CL_CHECK(err);
    cl_mem d_out_sg = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                     num_groups * sizeof(int), nullptr, &err);
    CL_CHECK(err);

    // ======================================================================
    // Запуск 1: классическая редукция через __local память
    // ======================================================================
    int n_int = N;
    CL_CHECK(clSetKernelArg(k_local, 0, sizeof(cl_mem), &d_in));
    CL_CHECK(clSetKernelArg(k_local, 1, sizeof(cl_mem), &d_out_local));
    CL_CHECK(clSetKernelArg(k_local, 2, local_size * sizeof(int), nullptr));
    CL_CHECK(clSetKernelArg(k_local, 3, sizeof(int), &n_int));

    cl_event ev_local;
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_local, 1, nullptr,
                                    &global_size, &local_size,
                                    0, nullptr, &ev_local));
    CL_CHECK(clWaitForEvents(1, &ev_local));

    cl_ulong t0 = 0, t1 = 0;
    CL_CHECK(clGetEventProfilingInfo(ev_local, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &t0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(ev_local, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &t1, nullptr));
    double ms_local = (t1 - t0) * 1e-6;

    std::vector<int> h_out_local(num_groups);
    CL_CHECK(clEnqueueReadBuffer(queue, d_out_local, CL_TRUE, 0,
                                 num_groups * sizeof(int),
                                 h_out_local.data(), 0, nullptr, nullptr));
    long long sum_local = 0;
    for (size_t i = 0; i < num_groups; ++i) sum_local += h_out_local[i];

    // ======================================================================
    // Запуск 2: редукция через sub-groups
    // ======================================================================
    // Для scratch нужно: (local_size / sg_size) элементов — по одному на подгруппу.
    size_t num_subgroups_per_wg = local_size / sg_size;
    CL_CHECK(clSetKernelArg(k_sg, 0, sizeof(cl_mem), &d_in));
    CL_CHECK(clSetKernelArg(k_sg, 1, sizeof(cl_mem), &d_out_sg));
    CL_CHECK(clSetKernelArg(k_sg, 2, num_subgroups_per_wg * sizeof(int), nullptr));
    CL_CHECK(clSetKernelArg(k_sg, 3, sizeof(int), &n_int));

    cl_event ev_sg;
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_sg, 1, nullptr,
                                    &global_size, &local_size,
                                    0, nullptr, &ev_sg));
    CL_CHECK(clWaitForEvents(1, &ev_sg));

    CL_CHECK(clGetEventProfilingInfo(ev_sg, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &t0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(ev_sg, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &t1, nullptr));
    double ms_sg = (t1 - t0) * 1e-6;

    std::vector<int> h_out_sg(num_groups);
    CL_CHECK(clEnqueueReadBuffer(queue, d_out_sg, CL_TRUE, 0,
                                 num_groups * sizeof(int),
                                 h_out_sg.data(), 0, nullptr, nullptr));
    long long sum_sg = 0;
    for (size_t i = 0; i < num_groups; ++i) sum_sg += h_out_sg[i];

    // ======================================================================
    // Результаты
    // ======================================================================
    std::cout << "=== Редукция через __local память ===" << std::endl;
    std::cout << "N=" << N
              << "  sum=" << sum_local
              << "  expected=" << N
              << "  check=" << (sum_local == N ? "OK" : "FAIL")
              << "  kernel_time=" << std::fixed << std::setprecision(3)
              << ms_local << " ms" << std::endl;

    std::cout << "\n=== Редукция через sub-groups ===" << std::endl;
    std::cout << "N=" << N
              << "  sum=" << sum_sg
              << "  expected=" << N
              << "  check=" << (sum_sg == N ? "OK" : "FAIL")
              << "  kernel_time=" << std::fixed << std::setprecision(3)
              << ms_sg << " ms" << std::endl;

    // Сравнение производительности.
    std::cout << "\n=== Сравнение ===" << std::endl;
    std::cout << "__local память:  " << std::setprecision(3) << ms_local << " ms"
              << std::endl;
    std::cout << "sub-groups:      " << std::setprecision(3) << ms_sg << " ms"
              << std::endl;
    if (ms_local > 0) {
        double speedup = ms_local / ms_sg;
        std::cout << "Ускорение (sub-groups vs __local): "
                  << std::setprecision(2) << speedup << "x" << std::endl;
    }
    std::cout << "\n// Sub-groups быстрее, т.к. первый уровень редукции обходится\n"
              << "// без барьеров и без записи/чтения __local памяти.\n"
              << "// На NVIDIA: подгруппа = warp (32 потока).\n"
              << "// На AMD:    подгруппа = wavefront (32 или 64 потока)." << std::endl;

    // --- Очистка ---
    clReleaseEvent(ev_local);
    clReleaseEvent(ev_sg);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out_local);
    clReleaseMemObject(d_out_sg);
    clReleaseKernel(k_local);
    clReleaseKernel(k_sg);
    clReleaseProgram(prog_local);
    clReleaseProgram(prog_sg);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
