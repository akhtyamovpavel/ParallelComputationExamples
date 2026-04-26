#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cstdlib>

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

// Атомарные операции в OpenCL. Два подпримера:
//
// Часть A — Атомарный счётчик:
//   Каждый work-item выполняет atomic_add(&counter, 1).
//   Результат должен равняться общему числу work-item'ов.
//   Аналог atomicAdd в CUDA.
//
// Часть B — Гистограмма:
//   Дан массив случайных чисел 0..255. Строим 256-бинную гистограмму.
//
//   histogram_naive — каждый work-item делает atomic_add на глобальной
//   гистограмме. При большом числе work-item'ов возникает высокая
//   конкуренция (contention) за одни и те же ячейки.
//
//   histogram_privatized — каждый work-group строит локальную гистограмму
//   в __local памяти, затем после barrier один work-item на бин сливает
//   локальную гистограмму в глобальную через atomic_add.
//   Приватизация снижает конкуренцию: вместо того чтобы все work-item'ы
//   толкались в 256 глобальных ячеек, каждый work-group работает со
//   своей копией в быстрой __local памяти, а слияние происходит лишь
//   один раз на work-group.
//
// Сравните с CUDA/04.5-atomics: atomicAdd → atomic_add,
// __shared__ → __local, __syncthreads() → barrier(CLK_LOCAL_MEM_FENCE).
static const char* kernel_source = R"CLC(
// ---- Часть A: атомарный счётчик ----
__kernel void atomic_counter(__global int* counter) {
    atomic_add(counter, 1);
}

// ---- Часть B: наивная гистограмма ----
// Каждый work-item читает один элемент и атомарно инкрементирует
// соответствующий бин глобальной гистограммы. Высокая конкуренция.
__kernel void histogram_naive(__global const uchar* data,
                              __global int* hist,
                              const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        atomic_add(&hist[data[gid]], 1);
    }
}

// ---- Часть B: приватизированная гистограмма ----
// 1) Обнуляем __local гистограмму (256 бинов).
// 2) Каждый work-item атомарно инкрементирует __local бин.
// 3) barrier — ждём, пока все закончат.
// 4) Первые 256 work-item'ов группы сливают __local → __global
//    через atomic_add. Это существенно дешевле, чем N атомарных
//    операций по глобальной памяти.
__kernel void histogram_privatized(__global const uchar* data,
                                   __global int* hist,
                                   __local int* local_hist,
                                   const int n) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int block_size = get_local_size(0);

    // Обнуляем __local гистограмму. Каждый work-item обнуляет
    // несколько бинов, чтобы покрыть все 256.
    for (int i = lid; i < 256; i += block_size) {
        local_hist[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Накопление в __local — конкуренция только внутри work-group.
    if (gid < n) {
        atomic_add(&local_hist[data[gid]], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Слияние: каждый work-item сливает часть бинов в глобальную гистограмму.
    for (int i = lid; i < 256; i += block_size) {
        if (local_hist[i] > 0) {
            atomic_add(&hist[i], local_hist[i]);
        }
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

int main() {
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

    cl_kernel k_counter   = clCreateKernel(prog, "atomic_counter",       &err);
    CL_CHECK(err);
    cl_kernel k_hist_naive = clCreateKernel(prog, "histogram_naive",     &err);
    CL_CHECK(err);
    cl_kernel k_hist_priv  = clCreateKernel(prog, "histogram_privatized", &err);
    CL_CHECK(err);

    // ===============================================================
    // Часть A: Атомарный счётчик
    // ===============================================================
    {
        const int TOTAL = 1 << 20;          // 1M work-item'ов
        const size_t local_size = 256;
        const size_t global_size = TOTAL;   // уже кратно 256

        // Счётчик на устройстве, инициализируем нулём.
        int zero = 0;
        cl_mem d_counter = clCreateBuffer(ctx,
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(int), &zero, &err);
        CL_CHECK(err);

        CL_CHECK(clSetKernelArg(k_counter, 0, sizeof(cl_mem), &d_counter));

        cl_event ev;
        CL_CHECK(clEnqueueNDRangeKernel(queue, k_counter, 1, nullptr,
                                        &global_size, &local_size,
                                        0, nullptr, &ev));
        CL_CHECK(clWaitForEvents(1, &ev));

        cl_ulong t0 = 0, t1 = 0;
        CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), &t0, nullptr));
        CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), &t1, nullptr));
        double ms = (t1 - t0) * 1e-6;

        int result = 0;
        CL_CHECK(clEnqueueReadBuffer(queue, d_counter, CL_TRUE, 0,
                                     sizeof(int), &result, 0, nullptr, nullptr));

        std::cout << "=== Part A: atomic counter ===" << std::endl;
        std::cout << "work-items=" << TOTAL
                  << "  counter=" << result
                  << "  expected=" << TOTAL
                  << "  check=" << (result == TOTAL ? "OK" : "FAIL")
                  << "  time=" << std::fixed << std::setprecision(3) << ms << " ms"
                  << std::endl << std::endl;

        clReleaseEvent(ev);
        clReleaseMemObject(d_counter);
    }

    // ===============================================================
    // Часть B: Гистограмма
    // ===============================================================
    {
        const int N = 1 << 22;              // ~4M элементов
        const size_t local_size = 256;
        const size_t num_groups = (N + local_size - 1) / local_size;
        const size_t global_size = num_groups * local_size;
        const int NUM_BINS = 256;

        // Генерация случайных данных 0..255.
        std::vector<unsigned char> h_data(N);
        std::srand(42);
        for (int i = 0; i < N; ++i) h_data[i] = static_cast<unsigned char>(std::rand() % 256);

        // Эталонная гистограмма на CPU.
        std::vector<int> h_ref(NUM_BINS, 0);
        for (int i = 0; i < N; ++i) ++h_ref[h_data[i]];

        cl_mem d_data = clCreateBuffer(ctx,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       N * sizeof(unsigned char), h_data.data(), &err);
        CL_CHECK(err);

        // -----------------------------------------------------------
        // histogram_naive
        // -----------------------------------------------------------
        {
            // Гистограмма на устройстве, обнулённая.
            std::vector<int> zeros(NUM_BINS, 0);
            cl_mem d_hist = clCreateBuffer(ctx,
                                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                           NUM_BINS * sizeof(int), zeros.data(), &err);
            CL_CHECK(err);

            int n_int = N;
            CL_CHECK(clSetKernelArg(k_hist_naive, 0, sizeof(cl_mem), &d_data));
            CL_CHECK(clSetKernelArg(k_hist_naive, 1, sizeof(cl_mem), &d_hist));
            CL_CHECK(clSetKernelArg(k_hist_naive, 2, sizeof(int),    &n_int));

            // Прогрев
            CL_CHECK(clEnqueueNDRangeKernel(queue, k_hist_naive, 1, nullptr,
                                            &global_size, &local_size,
                                            0, nullptr, nullptr));
            CL_CHECK(clFinish(queue));

            // Обнуляем гистограмму перед замером.
            CL_CHECK(clEnqueueWriteBuffer(queue, d_hist, CL_TRUE, 0,
                                          NUM_BINS * sizeof(int),
                                          zeros.data(), 0, nullptr, nullptr));

            cl_event ev;
            CL_CHECK(clEnqueueNDRangeKernel(queue, k_hist_naive, 1, nullptr,
                                            &global_size, &local_size,
                                            0, nullptr, &ev));
            CL_CHECK(clWaitForEvents(1, &ev));

            cl_ulong t0 = 0, t1 = 0;
            CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                             sizeof(cl_ulong), &t0, nullptr));
            CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                             sizeof(cl_ulong), &t1, nullptr));
            double ms_naive = (t1 - t0) * 1e-6;

            std::vector<int> h_hist(NUM_BINS);
            CL_CHECK(clEnqueueReadBuffer(queue, d_hist, CL_TRUE, 0,
                                         NUM_BINS * sizeof(int),
                                         h_hist.data(), 0, nullptr, nullptr));

            bool ok = (h_hist == h_ref);

            std::cout << "=== Part B: histogram ===" << std::endl;
            std::cout << "histogram_naive       N=" << N
                      << "  time=" << std::fixed << std::setprecision(3)
                      << ms_naive << " ms"
                      << "  check=" << (ok ? "OK" : "FAIL") << std::endl;

            clReleaseEvent(ev);
            clReleaseMemObject(d_hist);
        }

        // -----------------------------------------------------------
        // histogram_privatized (с __local гистограммой)
        // -----------------------------------------------------------
        {
            std::vector<int> zeros(NUM_BINS, 0);
            cl_mem d_hist = clCreateBuffer(ctx,
                                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                           NUM_BINS * sizeof(int), zeros.data(), &err);
            CL_CHECK(err);

            int n_int = N;
            CL_CHECK(clSetKernelArg(k_hist_priv, 0, sizeof(cl_mem), &d_data));
            CL_CHECK(clSetKernelArg(k_hist_priv, 1, sizeof(cl_mem), &d_hist));
            CL_CHECK(clSetKernelArg(k_hist_priv, 2, NUM_BINS * sizeof(int), nullptr));
            CL_CHECK(clSetKernelArg(k_hist_priv, 3, sizeof(int), &n_int));

            // Прогрев
            CL_CHECK(clEnqueueNDRangeKernel(queue, k_hist_priv, 1, nullptr,
                                            &global_size, &local_size,
                                            0, nullptr, nullptr));
            CL_CHECK(clFinish(queue));

            // Обнуляем гистограмму перед замером.
            CL_CHECK(clEnqueueWriteBuffer(queue, d_hist, CL_TRUE, 0,
                                          NUM_BINS * sizeof(int),
                                          zeros.data(), 0, nullptr, nullptr));

            cl_event ev;
            CL_CHECK(clEnqueueNDRangeKernel(queue, k_hist_priv, 1, nullptr,
                                            &global_size, &local_size,
                                            0, nullptr, &ev));
            CL_CHECK(clWaitForEvents(1, &ev));

            cl_ulong t0 = 0, t1 = 0;
            CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                             sizeof(cl_ulong), &t0, nullptr));
            CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                             sizeof(cl_ulong), &t1, nullptr));
            double ms_priv = (t1 - t0) * 1e-6;

            std::vector<int> h_hist(NUM_BINS);
            CL_CHECK(clEnqueueReadBuffer(queue, d_hist, CL_TRUE, 0,
                                         NUM_BINS * sizeof(int),
                                         h_hist.data(), 0, nullptr, nullptr));

            bool ok = (h_hist == h_ref);

            std::cout << "histogram_privatized  N=" << N
                      << "  time=" << std::fixed << std::setprecision(3)
                      << ms_priv << " ms"
                      << "  check=" << (ok ? "OK" : "FAIL") << std::endl;

            clReleaseEvent(ev);
            clReleaseMemObject(d_hist);
        }

        clReleaseMemObject(d_data);
    }

    clReleaseKernel(k_counter);
    clReleaseKernel(k_hist_naive);
    clReleaseKernel(k_hist_priv);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
