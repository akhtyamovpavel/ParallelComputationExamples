#include <iostream>
#include <iomanip>
#include <vector>
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

// Два алгоритма scan для одного work-group: Хиллис-Стил и Блеллох.
//
// ВАЖНО: здесь реализован scan только для одного work-group.
// Полный scan большого массива (multi-block) — тема домашнего задания.
//
// Аналог CUDA-примера: 05-scan (single-block version).
// В OpenCL вместо __shared__ используем __local, вместо __syncthreads() —
// barrier(CLK_LOCAL_MEM_FENCE).
//
// Ссылки:
//   Mark Harris, "Parallel Prefix Sum (Scan) with CUDA"
//   GPU Gems 3, Chapter 39
//   Ben-Gurion University, лекция по Scan
static const char* kernel_source = R"CLC(
// ===================================================================
// Алгоритм 1: Хиллис--Стил (Hillis--Steele) — inclusive scan.
//
// Идея: на каждом шаге d = 0, 1, 2, ... каждый поток i прибавляет
// к своему элементу элемент на расстоянии 2^d слева.
// Сложность: O(N log N) работы, log N шагов.
// Проще, но делает больше работы, чем Блеллох.
// ===================================================================
__kernel void hillis_steele_scan(__global const int* in,
                                __global int* out,
                                __local int* buf0,
                                __local int* buf1) {
    unsigned int tid = get_local_id(0);
    unsigned int n   = get_local_size(0);

    buf0[tid] = in[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    __local int* src = buf0;
    __local int* dst = buf1;

    for (unsigned int offset = 1; offset < n; offset <<= 1) {
        if (tid >= offset) {
            dst[tid] = src[tid] + src[tid - offset];
        } else {
            dst[tid] = src[tid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        __local int* tmp = src;
        src = dst;
        dst = tmp;
    }

    out[tid] = src[tid];
}

// ===================================================================
// Алгоритм 2: Блеллох (Blelloch) — exclusive scan.
//
// Две фазы:
//   1) Up-sweep (reduce): строим дерево частичных сумм снизу вверх.
//   2) Down-sweep: распространяем prefix-суммы сверху вниз.
//
// Сложность: O(N) работы (work-efficient), 2*log(N) шагов.
// Делает меньше работы, чем Хиллис-Стил, но имеет больше
// зависимостей между шагами и потенциальные bank conflicts.
//
// Результат: exclusive scan (out[i] = sum of in[0..i-1], out[0] = 0).
// ===================================================================
__kernel void blelloch_scan(__global const int* in,
                            __global int* out,
                            __local int* temp) {
    unsigned int tid = get_local_id(0);
    unsigned int n   = get_local_size(0);

    temp[tid] = in[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Up-sweep (reduce) ---
    for (unsigned int stride = 1; stride < n; stride <<= 1) {
        unsigned int idx = (tid + 1) * stride * 2 - 1;
        if (idx < n) {
            temp[idx] += temp[idx - stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Обнуляем последний элемент (корень дерева)
    if (tid == 0) {
        temp[n - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Down-sweep ---
    for (unsigned int stride = n >> 1; stride > 0; stride >>= 1) {
        unsigned int idx = (tid + 1) * stride * 2 - 1;
        if (idx < n) {
            int t = temp[idx - stride];
            temp[idx - stride] = temp[idx];
            temp[idx] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    out[tid] = temp[tid];
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
    // Размер массива = размер work-group (один work-group).
    // Обычно 256 или 1024 — зависит от устройства.
    const int N = 256;

    // Входной массив: все единицы.
    // Ожидаемый результат inclusive scan: 1, 2, 3, ..., N.
    std::vector<int> h_in(N, 1);
    std::vector<int> h_out(N, 0);

    cl_device_id dev = get_gpu_device();
    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev,
                                                  CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    // --- Компиляция ядра ---
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
    cl_kernel hs_kernel = clCreateKernel(prog, "hillis_steele_scan", &err);
    CL_CHECK(err);
    cl_kernel bl_kernel = clCreateKernel(prog, "blelloch_scan", &err);
    CL_CHECK(err);

    // --- Буферы ---
    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 N * sizeof(int), h_in.data(), &err);
    CL_CHECK(err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                  N * sizeof(int), nullptr, &err);
    CL_CHECK(err);

    size_t local_size  = static_cast<size_t>(N);
    size_t global_size = static_cast<size_t>(N);

    // ===================================================================
    // Тест 1: Хиллис-Стил (inclusive scan)
    // ===================================================================
    {
        CL_CHECK(clSetKernelArg(hs_kernel, 0, sizeof(cl_mem), &d_in));
        CL_CHECK(clSetKernelArg(hs_kernel, 1, sizeof(cl_mem), &d_out));
        CL_CHECK(clSetKernelArg(hs_kernel, 2, N * sizeof(int), nullptr));
        CL_CHECK(clSetKernelArg(hs_kernel, 3, N * sizeof(int), nullptr));

        cl_event event;
        CL_CHECK(clEnqueueNDRangeKernel(queue, hs_kernel, 1, nullptr,
                                        &global_size, &local_size,
                                        0, nullptr, &event));
        CL_CHECK(clWaitForEvents(1, &event));

        cl_ulong t_start = 0, t_end = 0;
        CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), &t_start, nullptr));
        CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), &t_end, nullptr));
        double ms = (t_end - t_start) * 1e-6;

        CL_CHECK(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                                     N * sizeof(int),
                                     h_out.data(), 0, nullptr, nullptr));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (h_out[i] != i + 1) {
                std::cerr << "HS mismatch at i=" << i
                          << ": got " << h_out[i]
                          << ", expected " << (i + 1) << std::endl;
                ok = false;
                break;
            }
        }

        std::cout << "=== Hillis-Steele (inclusive scan, O(N log N) work) ==="
                  << std::endl;
        std::cout << "N=" << N
                  << "  check=" << (ok ? "OK" : "FAIL")
                  << "  kernel_time=" << std::fixed << std::setprecision(3)
                  << ms << " ms" << std::endl;
        std::cout << "out[0..7]  = ";
        for (int i = 0; i < 8 && i < N; ++i) std::cout << h_out[i] << " ";
        std::cout << std::endl;

        clReleaseEvent(event);
    }

    // ===================================================================
    // Тест 2: Блеллох (exclusive scan)
    // ===================================================================
    {
        CL_CHECK(clSetKernelArg(bl_kernel, 0, sizeof(cl_mem), &d_in));
        CL_CHECK(clSetKernelArg(bl_kernel, 1, sizeof(cl_mem), &d_out));
        CL_CHECK(clSetKernelArg(bl_kernel, 2, N * sizeof(int), nullptr));

        cl_event event;
        CL_CHECK(clEnqueueNDRangeKernel(queue, bl_kernel, 1, nullptr,
                                        &global_size, &local_size,
                                        0, nullptr, &event));
        CL_CHECK(clWaitForEvents(1, &event));

        cl_ulong t_start = 0, t_end = 0;
        CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), &t_start, nullptr));
        CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), &t_end, nullptr));
        double ms = (t_end - t_start) * 1e-6;

        CL_CHECK(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                                     N * sizeof(int),
                                     h_out.data(), 0, nullptr, nullptr));

        // Exclusive scan единиц: 0, 1, 2, ..., N-1
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (h_out[i] != i) {
                std::cerr << "Blelloch mismatch at i=" << i
                          << ": got " << h_out[i]
                          << ", expected " << i << std::endl;
                ok = false;
                break;
            }
        }

        std::cout << "\n=== Blelloch (exclusive scan, O(N) work) ==="
                  << std::endl;
        std::cout << "N=" << N
                  << "  check=" << (ok ? "OK" : "FAIL")
                  << "  kernel_time=" << std::fixed << std::setprecision(3)
                  << ms << " ms" << std::endl;
        std::cout << "out[0..7]  = ";
        for (int i = 0; i < 8 && i < N; ++i) std::cout << h_out[i] << " ";
        std::cout << std::endl;

        clReleaseEvent(event);
    }

    // --- Cleanup ---
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseKernel(hs_kernel);
    clReleaseKernel(bl_kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
