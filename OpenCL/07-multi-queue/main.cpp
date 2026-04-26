#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cmath>
#include <climits>

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

// Перекрытие копирования и вычислений через несколько очередей команд.
// Аналог CUDA streams (08-streams-events/02-overlap-compute-copy):
//
//   queue 0: write chunk_0 → kernel chunk_0 → read chunk_0
//   queue 1: write chunk_1 → kernel chunk_1 → read chunk_1
//
// Если драйвер поддерживает параллельное исполнение, write chunk_1 может
// идти одновременно с kernel chunk_0 на разных движках (copy engine + compute).
//
// Зависимости внутри каждой очереди гарантируются in-order очередью.
// Между очередями синхронизация — через cl_event (аналог cudaStreamWaitEvent).
static const char* kernel_source = R"CLC(
__kernel void square(__global const float* in,
                     __global float* out,
                     const int n) {
    int i = get_global_id(0);
    if (i < n) {
        float v = in[i];
        out[i] = v * v;
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
    const int N = 1 << 24;
    const int NUM_CHUNKS = 4;
    const int CHUNK = N / NUM_CHUNKS;
    const size_t chunk_bytes = CHUNK * sizeof(float);

    std::vector<float> h_in(N);
    std::vector<float> h_out(N, 0.0f);
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 100) * 0.01f;

    cl_device_id dev = get_gpu_device();
    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);

    // Создаём NUM_CHUNKS in-order очередей с профилированием.
    // Каждая очередь — аналог отдельного CUDA stream'а.
    std::vector<cl_command_queue> queues(NUM_CHUNKS);
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        queues[i] = clCreateCommandQueue(ctx, dev,
                                         CL_QUEUE_PROFILING_ENABLE, &err);
        CL_CHECK(err);
    }

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
    cl_kernel kernel = clCreateKernel(prog, "square", &err);
    CL_CHECK(err);

    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                 N * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                  N * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    // --- Вариант 1: одна очередь, последовательно ---
    cl_event ev_seq_start, ev_seq_end;
    CL_CHECK(clEnqueueWriteBuffer(queues[0], d_in, CL_FALSE, 0,
                                  N * sizeof(float), h_in.data(),
                                  0, nullptr, &ev_seq_start));

    int n_int = N;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &n_int));

    size_t local_size = 256;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;
    CL_CHECK(clEnqueueNDRangeKernel(queues[0], kernel, 1, nullptr,
                                    &global_size, &local_size,
                                    0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(queues[0], d_out, CL_FALSE, 0,
                                 N * sizeof(float), h_out.data(),
                                 0, nullptr, &ev_seq_end));
    CL_CHECK(clFinish(queues[0]));

    cl_ulong seq_start = 0, seq_end = 0;
    CL_CHECK(clGetEventProfilingInfo(ev_seq_start, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &seq_start, nullptr));
    CL_CHECK(clGetEventProfilingInfo(ev_seq_end, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &seq_end, nullptr));
    double ms_seq = (seq_end - seq_start) * 1e-6;

    clReleaseEvent(ev_seq_start);
    clReleaseEvent(ev_seq_end);

    // --- Вариант 2: NUM_CHUNKS очередей, перекрытие copy/compute ---
    std::vector<cl_event> ev_first(NUM_CHUNKS);
    std::vector<cl_event> ev_last(NUM_CHUNKS);

    for (int i = 0; i < NUM_CHUNKS; ++i) {
        size_t offset = static_cast<size_t>(i) * CHUNK * sizeof(float);

        CL_CHECK(clEnqueueWriteBuffer(queues[i], d_in, CL_FALSE, offset,
                                      chunk_bytes,
                                      h_in.data() + i * CHUNK,
                                      0, nullptr, &ev_first[i]));

        int chunk_int = CHUNK;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &chunk_int));

        size_t g_off = static_cast<size_t>(i) * CHUNK;
        size_t g_size = ((CHUNK + local_size - 1) / local_size) * local_size;
        CL_CHECK(clEnqueueNDRangeKernel(queues[i], kernel, 1, &g_off,
                                        &g_size, &local_size,
                                        0, nullptr, nullptr));

        CL_CHECK(clEnqueueReadBuffer(queues[i], d_out, CL_FALSE, offset,
                                     chunk_bytes,
                                     h_out.data() + i * CHUNK,
                                     0, nullptr, &ev_last[i]));
    }

    for (int i = 0; i < NUM_CHUNKS; ++i) CL_CHECK(clFinish(queues[i]));

    cl_ulong par_start = ULONG_MAX, par_end = 0;
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        cl_ulong t0 = 0, t1 = 0;
        CL_CHECK(clGetEventProfilingInfo(ev_first[i], CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), &t0, nullptr));
        CL_CHECK(clGetEventProfilingInfo(ev_last[i], CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), &t1, nullptr));
        if (t0 < par_start) par_start = t0;
        if (t1 > par_end)   par_end = t1;
    }
    double ms_par = (par_end - par_start) * 1e-6;

    for (int i = 0; i < NUM_CHUNKS; ++i) {
        clReleaseEvent(ev_first[i]);
        clReleaseEvent(ev_last[i]);
    }

    // Проверка корректности
    bool ok = true;
    for (int i = 0; i < 1024 && ok; ++i) {
        float expected = h_in[i] * h_in[i];
        if (std::fabs(h_out[i] - expected) > 1e-5f) ok = false;
    }

    std::cout << "N=" << N << "  chunks=" << NUM_CHUNKS << std::endl
              << "  sequential (1 queue):    "
              << std::fixed << std::setprecision(3) << ms_seq << " ms" << std::endl
              << "  overlapped (" << NUM_CHUNKS << " queues):  "
              << ms_par << " ms" << std::endl
              << "  speedup: " << std::setprecision(2) << ms_seq / ms_par << "x"
              << "  check=" << (ok ? "OK" : "FAIL") << std::endl;

    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    for (int i = 0; i < NUM_CHUNKS; ++i) clReleaseCommandQueue(queues[i]);
    clReleaseContext(ctx);
    return 0;
}
