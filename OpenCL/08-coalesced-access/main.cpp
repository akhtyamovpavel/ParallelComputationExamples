#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cmath>

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

// Сравнение coalesced и strided (неколесцированного) доступа к глобальной памяти.
// Аналог CUDA примера 03-memory-model/01-uncoalesced-access.
//
// Coalesced доступ: соседние work-item'ы в wavefront/warp обращаются к
// СОСЕДНИМ адресам в памяти (stride=1). Контроллер памяти GPU объединяет эти
// обращения в ОДНУ транзакцию на 128 байт (размер кэш-линии L2).
//
// Strided доступ: соседние work-item'ы читают/пишут адреса с шагом STRIDE.
// Каждый work-item попадает в отдельную кэш-линию => контроллер памяти
// выполняет до 32 отдельных транзакций вместо одной. Пропускная способность
// падает пропорционально количеству лишних транзакций.
//
// На GPU с кэш-линией 128 байт и warp=32, при stride=16 (64 байта) каждая
// пара work-item'ов попадает в разные кэш-линии. При stride >= 32 — каждый
// work-item в отдельной кэш-линии, и bandwidth деградирует в ~32 раз.
static const char* kernel_source = R"CLC(
__kernel void copy_coalesced(__global const float* in,
                             __global float* out,
                             const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        out[gid] = in[gid];
    }
}

__kernel void copy_strided(__global const float* in,
                           __global float* out,
                           const int n,
                           const int stride) {
    int gid = get_global_id(0);
    int idx = gid * stride;
    if (idx < n) {
        out[idx] = in[idx];
    }
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

static cl_program build_program(cl_context ctx, cl_device_id dev, const char* src) {
    cl_int err = 0;
    size_t len = std::strlen(src);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &len, &err);
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
        std::exit(1);
    }
    return prog;
}

int main() {
    // N — количество work-item'ов (элементов для coalesced-копирования).
    // Для strided-версии массив должен вмещать N * STRIDE элементов.
    const int N = 1 << 24;              // 16M work-item'ов
    const int STRIDES[] = {1, 2, 4, 8, 16, 32};
    const int NUM_STRIDES = sizeof(STRIDES) / sizeof(STRIDES[0]);
    const int MAX_STRIDE = STRIDES[NUM_STRIDES - 1];

    // Выделяем массив, достаточный для максимального stride.
    const size_t total_elems = static_cast<size_t>(N) * MAX_STRIDE;
    const size_t total_bytes = total_elems * sizeof(float);

    std::vector<float> h_in(total_elems);
    for (size_t i = 0; i < total_elems; ++i) h_in[i] = static_cast<float>(i % 1000) * 0.001f;
    std::vector<float> h_out(total_elems, 0.0f);

    cl_device_id dev = get_gpu_device();
    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev,
                                                  CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    cl_program prog = build_program(ctx, dev, kernel_source);
    cl_kernel k_coalesced = clCreateKernel(prog, "copy_coalesced", &err);
    CL_CHECK(err);
    cl_kernel k_strided = clCreateKernel(prog, "copy_strided", &err);
    CL_CHECK(err);

    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 total_bytes, h_in.data(), &err);
    CL_CHECK(err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, total_bytes, nullptr, &err);
    CL_CHECK(err);

    size_t local_size = 256;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;
    int n_int = N;

    std::cout << "=== Coalesced vs Strided memory access ===" << std::endl;
    std::cout << "N = " << N << " work-items (" << (N * sizeof(float)) / (1024 * 1024)
              << " MB per coalesced copy)" << std::endl << std::endl;

    // --- Coalesced (stride=1) ---
    {
        CL_CHECK(clSetKernelArg(k_coalesced, 0, sizeof(cl_mem), &d_in));
        CL_CHECK(clSetKernelArg(k_coalesced, 1, sizeof(cl_mem), &d_out));
        CL_CHECK(clSetKernelArg(k_coalesced, 2, sizeof(int), &n_int));

        // Прогрев
        CL_CHECK(clEnqueueNDRangeKernel(queue, k_coalesced, 1, nullptr,
                                        &global_size, &local_size,
                                        0, nullptr, nullptr));
        CL_CHECK(clFinish(queue));

        cl_event event;
        CL_CHECK(clEnqueueNDRangeKernel(queue, k_coalesced, 1, nullptr,
                                        &global_size, &local_size,
                                        0, nullptr, &event));
        CL_CHECK(clWaitForEvents(1, &event));

        cl_ulong t_start = 0, t_end = 0;
        CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), &t_start, nullptr));
        CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), &t_end, nullptr));
        double ms = (t_end - t_start) * 1e-6;
        // Effective bandwidth: читаем N float + пишем N float = 2 * N * sizeof(float)
        double gbps = (2.0 * N * sizeof(float)) / ((t_end - t_start) * 1e-9)
                      / (1024.0 * 1024.0 * 1024.0);

        // Проверка корректности
        CL_CHECK(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                                     N * sizeof(float), h_out.data(),
                                     0, nullptr, nullptr));
        bool ok = true;
        for (int i = 0; i < 1024 && ok; ++i) {
            if (std::fabs(h_out[i] - h_in[i]) > 1e-6f) ok = false;
        }

        std::cout << std::left << std::setw(22) << "coalesced (stride=1)"
                  << " time=" << std::right << std::setw(8)
                  << std::fixed << std::setprecision(3) << ms << " ms"
                  << "  bw=" << std::setw(7)
                  << std::setprecision(2) << gbps << " GB/s"
                  << "  check=" << (ok ? "OK" : "FAIL") << std::endl;

        clReleaseEvent(event);
    }

    // --- Strided (stride > 1) ---
    // Для каждого stride: work-item i обращается к in[i * stride] и out[i * stride].
    // Чем больше stride, тем больше кэш-линий задействует один warp/wavefront,
    // и тем ниже эффективная пропускная способность.
    for (int s = 1; s < NUM_STRIDES; ++s) {
        int stride = STRIDES[s];
        int total_n = N * stride;  // полный диапазон адресов

        // Обнуляем выходной буфер
        std::vector<float> zeros(total_elems, 0.0f);
        CL_CHECK(clEnqueueWriteBuffer(queue, d_out, CL_TRUE, 0,
                                      total_bytes, zeros.data(),
                                      0, nullptr, nullptr));

        CL_CHECK(clSetKernelArg(k_strided, 0, sizeof(cl_mem), &d_in));
        CL_CHECK(clSetKernelArg(k_strided, 1, sizeof(cl_mem), &d_out));
        CL_CHECK(clSetKernelArg(k_strided, 2, sizeof(int), &total_n));
        CL_CHECK(clSetKernelArg(k_strided, 3, sizeof(int), &stride));

        // Прогрев
        CL_CHECK(clEnqueueNDRangeKernel(queue, k_strided, 1, nullptr,
                                        &global_size, &local_size,
                                        0, nullptr, nullptr));
        CL_CHECK(clFinish(queue));

        cl_event event;
        CL_CHECK(clEnqueueNDRangeKernel(queue, k_strided, 1, nullptr,
                                        &global_size, &local_size,
                                        0, nullptr, &event));
        CL_CHECK(clWaitForEvents(1, &event));

        cl_ulong t_start = 0, t_end = 0;
        CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), &t_start, nullptr));
        CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), &t_end, nullptr));
        double ms = (t_end - t_start) * 1e-6;
        // Effective bandwidth считаем по ПОЛЕЗНЫМ данным: 2 * N * sizeof(float),
        // хотя реально GPU пересылает гораздо больше кэш-линий.
        double gbps = (2.0 * N * sizeof(float)) / ((t_end - t_start) * 1e-9)
                      / (1024.0 * 1024.0 * 1024.0);

        // Проверка: out[i * stride] == in[i * stride]
        CL_CHECK(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                                     total_bytes, h_out.data(),
                                     0, nullptr, nullptr));
        bool ok = true;
        for (int i = 0; i < 1024 && ok; ++i) {
            int idx = i * stride;
            if (std::fabs(h_out[idx] - h_in[idx]) > 1e-6f) ok = false;
        }

        char label[64];
        std::snprintf(label, sizeof(label), "strided (stride=%d)", stride);
        std::cout << std::left << std::setw(22) << label
                  << " time=" << std::right << std::setw(8)
                  << std::fixed << std::setprecision(3) << ms << " ms"
                  << "  bw=" << std::setw(7)
                  << std::setprecision(2) << gbps << " GB/s"
                  << "  check=" << (ok ? "OK" : "FAIL") << std::endl;

        clReleaseEvent(event);
    }

    // Вывод: при coalesced-доступе пропускная способность близка к пиковой
    // пропускной способности памяти GPU. При stride=32 каждый work-item в warp'е
    // попадает в свою кэш-линию, и bandwidth падает в десятки раз.
    std::cout << "\n// Вывод: coalesced доступ (stride=1) использует пропускную"
              << std::endl
              << "// способность памяти эффективно, т.к. все work-item'ы в warp/wavefront"
              << std::endl
              << "// обращаются к одной кэш-линии. С ростом stride каждый work-item"
              << std::endl
              << "// попадает в отдельную кэш-линию => транзакций в разы больше,"
              << std::endl
              << "// а полезных данных в каждой — в разы меньше." << std::endl;

    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseKernel(k_coalesced);
    clReleaseKernel(k_strided);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
