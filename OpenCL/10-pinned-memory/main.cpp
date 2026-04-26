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

// Сравнение скорости передачи Host <-> Device для обычных и pinned-буферов.
// Аналог CUDA-примера 03-memory-model/03-host-alloc-benchmarks.
//
// Обычный путь (pageable memory):
//   host malloc -> clEnqueueWriteBuffer -> device buffer
//   Драйвер внутри делает ДВА копирования:
//     1) memcpy из pageable-памяти в pinned staging buffer (CPU)
//     2) DMA из staging buffer на GPU
//   Первый шаг нужен, потому что DMA-контроллер GPU не может гарантированно
//   работать с pageable-памятью (ОС может переместить страницу в swap).
//
// Pinned путь (CL_MEM_ALLOC_HOST_PTR):
//   clCreateBuffer(CL_MEM_ALLOC_HOST_PTR) — драйвер выделяет page-locked
//   (pinned) память, которая никогда не уходит в swap.
//   clEnqueueMapBuffer → получаем указатель на эту память → заполняем данными.
//   clEnqueueUnmapMemObject → сообщаем драйверу, что host закончил работу.
//   Теперь данные можно передать на GPU одним DMA, без промежуточного memcpy.
//
// В CUDA аналог — cudaHostAlloc / cudaMallocHost.
static const char* kernel_source = R"CLC(
__kernel void scale(__global const float* in,
                    __global float* out,
                    const float factor,
                    const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        out[gid] = in[gid] * factor;
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

static double event_ms(cl_event ev) {
    cl_ulong t0 = 0, t1 = 0;
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &t0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &t1, nullptr));
    return (t1 - t0) * 1e-6;
}

int main() {
    const int N = 1 << 24;   // 16M float = 64 MB
    const size_t bytes = N * sizeof(float);
    const int NUM_ITERS = 5;

    cl_device_id dev = get_gpu_device();
    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev,
                                                  CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    cl_program prog = build_program(ctx, dev, kernel_source);
    cl_kernel kernel = clCreateKernel(prog, "scale", &err);
    CL_CHECK(err);

    std::cout << "=== Pinned vs Pageable memory transfer ===" << std::endl;
    std::cout << "Buffer size: " << bytes / (1024 * 1024) << " MB"
              << "  (" << NUM_ITERS << " iterations)" << std::endl << std::endl;

    // ================================================================
    // Вариант 1: обычная (pageable) память
    // ================================================================
    {
        std::vector<float> h_in(N);
        for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i) * 0.001f;
        std::vector<float> h_out(N, 0.0f);

        cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        CL_CHECK(err);
        cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
        CL_CHECK(err);

        double total_h2d = 0.0, total_d2h = 0.0;

        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            // Host -> Device
            cl_event ev_write;
            CL_CHECK(clEnqueueWriteBuffer(queue, d_in, CL_FALSE, 0, bytes,
                                          h_in.data(), 0, nullptr, &ev_write));
            CL_CHECK(clWaitForEvents(1, &ev_write));
            total_h2d += event_ms(ev_write);
            clReleaseEvent(ev_write);

            // Запускаем ядро, чтобы на GPU были валидные данные для чтения
            float factor = 2.0f;
            int n_int = N;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(float), &factor));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &n_int));
            size_t local_size = 256;
            size_t global_size = ((N + local_size - 1) / local_size) * local_size;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                            &global_size, &local_size,
                                            0, nullptr, nullptr));
            CL_CHECK(clFinish(queue));

            // Device -> Host
            cl_event ev_read;
            CL_CHECK(clEnqueueReadBuffer(queue, d_out, CL_FALSE, 0, bytes,
                                         h_out.data(), 0, nullptr, &ev_read));
            CL_CHECK(clWaitForEvents(1, &ev_read));
            total_d2h += event_ms(ev_read);
            clReleaseEvent(ev_read);
        }

        double avg_h2d = total_h2d / NUM_ITERS;
        double avg_d2h = total_d2h / NUM_ITERS;
        double bw_h2d = (bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_h2d * 1e-3);
        double bw_d2h = (bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_d2h * 1e-3);

        // Проверка корректности (последняя итерация)
        bool ok = true;
        for (int i = 0; i < 1024 && ok; ++i) {
            float expected = h_in[i] * 2.0f;
            if (std::fabs(h_out[i] - expected) > 1e-3f) ok = false;
        }

        std::cout << "--- Pageable (clEnqueueWrite/ReadBuffer) ---" << std::endl;
        std::cout << "  H->D:  " << std::fixed << std::setprecision(3)
                  << avg_h2d << " ms  (" << std::setprecision(2) << bw_h2d
                  << " GB/s)" << std::endl;
        std::cout << "  D->H:  " << std::fixed << std::setprecision(3)
                  << avg_d2h << " ms  (" << std::setprecision(2) << bw_d2h
                  << " GB/s)" << std::endl;
        std::cout << "  check=" << (ok ? "OK" : "FAIL") << std::endl << std::endl;

        clReleaseMemObject(d_in);
        clReleaseMemObject(d_out);
    }

    // ================================================================
    // Вариант 2: pinned-память (CL_MEM_ALLOC_HOST_PTR)
    // ================================================================
    // CL_MEM_ALLOC_HOST_PTR указывает драйверу выделить page-locked буфер
    // на хосте. clEnqueueMapBuffer позволяет получить указатель на эту
    // память без промежуточного копирования.
    //
    // Схема использования:
    //   1. clCreateBuffer(CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, ...)
    //   2. ptr = clEnqueueMapBuffer(CL_MAP_WRITE, ...) — получаем указатель
    //   3. memcpy(ptr, source, bytes) — заполняем данные прямо в pinned буфер
    //   4. clEnqueueUnmapMemObject(buf, ptr) — говорим драйверу: данные готовы
    //   5. Ядро может использовать буфер — DMA без лишнего копирования
    {
        // Pinned input buffer (host -> device)
        cl_mem pinned_in = clCreateBuffer(ctx,
                                          CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                          bytes, nullptr, &err);
        CL_CHECK(err);
        // Pinned output buffer (device -> host)
        cl_mem pinned_out = clCreateBuffer(ctx,
                                           CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                           bytes, nullptr, &err);
        CL_CHECK(err);
        // Device-side буферы для вычислений
        cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        CL_CHECK(err);
        cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
        CL_CHECK(err);

        double total_h2d = 0.0, total_d2h = 0.0;

        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            // --- Host -> Device (pinned) ---
            // 1. Map pinned buffer для записи (host получает указатель на pinned-память)
            float* mapped_in = static_cast<float*>(
                clEnqueueMapBuffer(queue, pinned_in, CL_TRUE, CL_MAP_WRITE,
                                   0, bytes, 0, nullptr, nullptr, &err));
            CL_CHECK(err);

            // 2. Заполняем данные напрямую в pinned-буфер (без промежуточного memcpy)
            for (int i = 0; i < N; ++i) mapped_in[i] = static_cast<float>(i) * 0.001f;

            // 3. Unmap — данные теперь доступны драйверу
            CL_CHECK(clEnqueueUnmapMemObject(queue, pinned_in, mapped_in,
                                             0, nullptr, nullptr));
            CL_CHECK(clFinish(queue));

            // 4. Копируем из pinned-буфера в device-буфер (одно DMA, без staging)
            cl_event ev_write;
            CL_CHECK(clEnqueueCopyBuffer(queue, pinned_in, d_in, 0, 0, bytes,
                                         0, nullptr, &ev_write));
            CL_CHECK(clWaitForEvents(1, &ev_write));
            total_h2d += event_ms(ev_write);
            clReleaseEvent(ev_write);

            // Запускаем ядро
            float factor = 2.0f;
            int n_int = N;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(float), &factor));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &n_int));
            size_t local_size = 256;
            size_t global_size = ((N + local_size - 1) / local_size) * local_size;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                            &global_size, &local_size,
                                            0, nullptr, nullptr));
            CL_CHECK(clFinish(queue));

            // --- Device -> Host (pinned) ---
            // Копируем результат из device-буфера в pinned-буфер
            cl_event ev_read;
            CL_CHECK(clEnqueueCopyBuffer(queue, d_out, pinned_out, 0, 0, bytes,
                                         0, nullptr, &ev_read));
            CL_CHECK(clWaitForEvents(1, &ev_read));
            total_d2h += event_ms(ev_read);
            clReleaseEvent(ev_read);
        }

        double avg_h2d = total_h2d / NUM_ITERS;
        double avg_d2h = total_d2h / NUM_ITERS;
        double bw_h2d = (bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_h2d * 1e-3);
        double bw_d2h = (bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_d2h * 1e-3);

        // Проверка: читаем результат из pinned output
        float* mapped_out = static_cast<float*>(
            clEnqueueMapBuffer(queue, pinned_out, CL_TRUE, CL_MAP_READ,
                               0, bytes, 0, nullptr, nullptr, &err));
        CL_CHECK(err);

        bool ok = true;
        for (int i = 0; i < 1024 && ok; ++i) {
            float expected = static_cast<float>(i) * 0.001f * 2.0f;
            if (std::fabs(mapped_out[i] - expected) > 1e-3f) ok = false;
        }

        CL_CHECK(clEnqueueUnmapMemObject(queue, pinned_out, mapped_out,
                                         0, nullptr, nullptr));
        CL_CHECK(clFinish(queue));

        std::cout << "--- Pinned (CL_MEM_ALLOC_HOST_PTR + Map/Unmap) ---" << std::endl;
        std::cout << "  H->D:  " << std::fixed << std::setprecision(3)
                  << avg_h2d << " ms  (" << std::setprecision(2) << bw_h2d
                  << " GB/s)" << std::endl;
        std::cout << "  D->H:  " << std::fixed << std::setprecision(3)
                  << avg_d2h << " ms  (" << std::setprecision(2) << bw_d2h
                  << " GB/s)" << std::endl;
        std::cout << "  check=" << (ok ? "OK" : "FAIL") << std::endl << std::endl;

        clReleaseMemObject(pinned_in);
        clReleaseMemObject(pinned_out);
        clReleaseMemObject(d_in);
        clReleaseMemObject(d_out);
    }

    // Пояснение результатов
    std::cout << "// Pinned-память позволяет избежать промежуточного копирования"
              << std::endl
              << "// из pageable-памяти в staging buffer, что даёт выигрыш в bandwidth."
              << std::endl
              << "// Особенно заметно на больших буферах и при частых пересылках."
              << std::endl
              << "// Недостаток: page-locked память не вытесняется в swap, поэтому"
              << std::endl
              << "// чрезмерное использование может исчерпать физическую RAM."
              << std::endl;

    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
