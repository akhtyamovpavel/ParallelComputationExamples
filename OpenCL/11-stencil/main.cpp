#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#define CL_CHECK(call) do {                                                   \
    cl_int _err = (call);                                                     \
    if (_err != CL_SUCCESS) {                                                 \
        std::cerr << "OpenCL error " << _err << " at "                        \
                  << __FILE__ << ":" << __LINE__ << std::endl;                \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// 1D stencil (трёхточечный) с использованием __local памяти и halo-ячеек.
//
// Формула теплопроводности: out[i] = 0.25*(in[i-1] + 2*in[i] + in[i+1]).
// Граничные элементы (i==0, i==N-1) не обновляются.
//
// Два ядра:
//   stencil_global — каждый work-item читает соседей напрямую из __global.
//                    Элементы на границах тайлов читаются дважды: один раз
//                    правым соседом одного work-group, один раз левым соседом
//                    следующего. Это аналог наивного CUDA-ядра.
//
//   stencil_local  — work-group загружает тайл + halo (по одному элементу
//                    слева и справа) в __local массив, делает barrier,
//                    затем считает из __local. Каждый глобальный элемент
//                    читается из DRAM ровно один раз, а затем переиспользуется
//                    через __local (аналог __shared__ в CUDA).
//
// Сравните с CUDA/03-memory-model/08-stencil: логика та же, но вместо
// __shared__ используется __local, вместо __syncthreads() —
// barrier(CLK_LOCAL_MEM_FENCE).
static const char* kernel_source = R"CLC(
// Наивный вариант: все чтения из глобальной памяти.
// Элементы на стыке тайлов читаются дважды разными work-group'ами.
__kernel void stencil_global(__global const float* in,
                             __global float* out,
                             const int n) {
    int i = get_global_id(0);
    // Граничные элементы не обновляем.
    if (i <= 0 || i >= n - 1) return;
    out[i] = 0.25f * (in[i - 1] + 2.0f * in[i] + in[i + 1]);
}

// Оптимизированный вариант с __local памятью.
//
// Паттерн halo: массив в __local имеет размер (BLOCK_SIZE + 2).
// Индекс 0 — halo слева, индексы 1..BLOCK_SIZE — основной тайл,
// индекс BLOCK_SIZE+1 — halo справа.
//
// Почему это выгодно: на границе тайла элемент in[i-1] или in[i+1]
// принадлежит соседнему work-group. Без __local он читался бы из DRAM
// дважды (своим и соседним work-group). С __local каждый элемент
// загружается из глобальной памяти ровно один раз, а потом все три
// обращения идут из быстрой on-chip памяти.
__kernel void stencil_local(__global const float* in,
                            __global float* out,
                            __local float* tile,
                            const int n) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int block_size = get_local_size(0);

    // Основной элемент тайла: tile[lid + 1] = in[gid].
    // +1 потому что tile[0] зарезервирован под halo слева.
    if (gid < n) {
        tile[lid + 1] = in[gid];
    }

    // Загрузка halo-ячеек.
    // Первый work-item группы загружает левый halo.
    if (lid == 0) {
        tile[0] = (gid > 0) ? in[gid - 1] : 0.0f;
    }
    // Последний work-item группы загружает правый halo.
    if (lid == block_size - 1) {
        tile[block_size + 1] = (gid + 1 < n) ? in[gid + 1] : 0.0f;
    }

    // Барьер: все элементы tile[] должны быть загружены
    // перед тем, как кто-то начнёт читать.
    barrier(CLK_LOCAL_MEM_FENCE);

    // Граничные элементы массива не обновляем.
    if (gid <= 0 || gid >= n - 1) return;

    // Обращение только к __local: tile[lid], tile[lid+1], tile[lid+2]
    // соответствуют in[gid-1], in[gid], in[gid+1].
    out[gid] = 0.25f * (tile[lid] + 2.0f * tile[lid + 1] + tile[lid + 2]);
}
)CLC";

static cl_device_id get_gpu_device() {
    cl_uint np = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &np));
    std::vector<cl_platform_id> platforms(np);
    CL_CHECK(clGetPlatformIDs(np, platforms.data(), nullptr));

    for (cl_uint p = 0; p < np; ++p) {
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

int main() {
    const int N = 1 << 22;          // ~4M элементов
    const size_t local_size = 256;
    const size_t num_groups = (N + local_size - 1) / local_size;
    const size_t global_size = num_groups * local_size;

    // Инициализация: in[i] = sin(i * 0.01), чтобы результат был нетривиальным.
    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = std::sin(i * 0.01f);

    // Эталонный результат на CPU.
    std::vector<float> h_ref(N, 0.0f);
    for (int i = 1; i < N - 1; ++i) {
        h_ref[i] = 0.25f * (h_in[i - 1] + 2.0f * h_in[i] + h_in[i + 1]);
    }

    cl_device_id dev = get_gpu_device();
    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev,
                                                  CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    size_t src_len = std::strlen(kernel_source);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_source,
                                                &src_len, &err);
    CL_CHECK(err);
    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                              log_size, &log[0], nullptr);
        std::cerr << "Build failed:\n" << log << std::endl;
        return 1;
    }

    cl_kernel k_global = clCreateKernel(prog, "stencil_global", &err);
    CL_CHECK(err);
    cl_kernel k_local = clCreateKernel(prog, "stencil_local", &err);
    CL_CHECK(err);

    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 N * sizeof(float), h_in.data(), &err);
    CL_CHECK(err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                  N * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    // ---------------------------------------------------------------
    // 1. stencil_global
    // ---------------------------------------------------------------
    int n_int = N;
    CL_CHECK(clSetKernelArg(k_global, 0, sizeof(cl_mem), &d_in));
    CL_CHECK(clSetKernelArg(k_global, 1, sizeof(cl_mem), &d_out));
    CL_CHECK(clSetKernelArg(k_global, 2, sizeof(int),    &n_int));

    // Прогрев
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_global, 1, nullptr,
                                    &global_size, &local_size,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));

    cl_event ev_global;
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_global, 1, nullptr,
                                    &global_size, &local_size,
                                    0, nullptr, &ev_global));
    CL_CHECK(clWaitForEvents(1, &ev_global));

    cl_ulong t0 = 0, t1 = 0;
    CL_CHECK(clGetEventProfilingInfo(ev_global, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &t0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(ev_global, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &t1, nullptr));
    double ms_global = (t1 - t0) * 1e-6;

    // Считываем результат и проверяем.
    std::vector<float> h_out(N, 0.0f);
    CL_CHECK(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                                 N * sizeof(float),
                                 h_out.data(), 0, nullptr, nullptr));

    bool ok_global = true;
    for (int i = 1; i < N - 1; ++i) {
        if (std::fabs(h_out[i] - h_ref[i]) > 1e-4f) {
            ok_global = false;
            break;
        }
    }

    std::cout << "stencil_global  N=" << N
              << "  time=" << std::fixed << std::setprecision(3) << ms_global << " ms"
              << "  check=" << (ok_global ? "OK" : "FAIL") << std::endl;

    // ---------------------------------------------------------------
    // 2. stencil_local (с __local памятью и halo)
    // ---------------------------------------------------------------
    CL_CHECK(clSetKernelArg(k_local, 0, sizeof(cl_mem), &d_in));
    CL_CHECK(clSetKernelArg(k_local, 1, sizeof(cl_mem), &d_out));
    // __local буфер: local_size + 2 элемента (halo слева и справа).
    CL_CHECK(clSetKernelArg(k_local, 2, (local_size + 2) * sizeof(float), nullptr));
    CL_CHECK(clSetKernelArg(k_local, 3, sizeof(int), &n_int));

    // Прогрев
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_local, 1, nullptr,
                                    &global_size, &local_size,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));

    cl_event ev_local;
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_local, 1, nullptr,
                                    &global_size, &local_size,
                                    0, nullptr, &ev_local));
    CL_CHECK(clWaitForEvents(1, &ev_local));

    CL_CHECK(clGetEventProfilingInfo(ev_local, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &t0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(ev_local, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &t1, nullptr));
    double ms_local = (t1 - t0) * 1e-6;

    CL_CHECK(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                                 N * sizeof(float),
                                 h_out.data(), 0, nullptr, nullptr));

    bool ok_local = true;
    for (int i = 1; i < N - 1; ++i) {
        if (std::fabs(h_out[i] - h_ref[i]) > 1e-4f) {
            ok_local = false;
            break;
        }
    }

    std::cout << "stencil_local   N=" << N
              << "  time=" << std::fixed << std::setprecision(3) << ms_local << " ms"
              << "  check=" << (ok_local ? "OK" : "FAIL") << std::endl;

    // Ускорение: ожидаем, что __local-версия будет быстрее за счёт
    // устранения дублирующих чтений из глобальной памяти на стыках тайлов.
    if (ms_local > 0.0) {
        std::cout << "speedup (global/local) = "
                  << std::setprecision(2) << ms_global / ms_local << "x"
                  << std::endl;
    }

    clReleaseEvent(ev_global);
    clReleaseEvent(ev_local);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseKernel(k_global);
    clReleaseKernel(k_local);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
