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

// Наивное матричное умножение: каждый work-item вычисляет один элемент C.
// C[row][col] = sum_k A[row][k] * B[k][col].
//
// Это прямой аналог CUDA/03.5-matrix-multiplication-example — вся работа
// идёт через глобальную память, без __local. Обратите внимание на 2D NDRange:
// global_size = (cols_C, rows_C), local_size = (BLOCK, BLOCK).
static const char* kernel_source = R"CLC(
__kernel void matmul_naive(__global const float* A,
                           __global const float* B,
                           __global float* C,
                           const int M,
                           const int K,
                           const int N) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
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
    const int M = 1024;
    const int K = 512;
    const int N = 1024;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(i % 7);
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(i % 5);

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
    cl_kernel kernel = clCreateKernel(prog, "matmul_naive", &err);
    CL_CHECK(err);

    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                M * K * sizeof(float), h_A.data(), &err);
    CL_CHECK(err);
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                K * N * sizeof(float), h_B.data(), &err);
    CL_CHECK(err);
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                M * N * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    int m_int = M, k_int = K, n_int = N;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &m_int));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &k_int));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &n_int));

    const size_t block = 16;
    size_t global[2] = {((N + block - 1) / block) * block,
                        ((M + block - 1) / block) * block};
    size_t local[2] = {block, block};

    // Прогрев
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                                    global, local, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));

    cl_event event;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                                    global, local, 0, nullptr, &event));
    CL_CHECK(clWaitForEvents(1, &event));

    cl_ulong t_start = 0, t_end = 0;
    CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &t_start, nullptr));
    CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &t_end, nullptr));
    double ms = (t_end - t_start) * 1e-6;
    double gflops = (2.0 * M * K * N) / ((t_end - t_start) * 1e-9) / 1e9;

    CL_CHECK(clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                                 M * N * sizeof(float),
                                 h_C.data(), 0, nullptr, nullptr));

    // Проверка на CPU: первые 3 строки, первые 3 столбца.
    bool ok = true;
    for (int r = 0; r < 3 && ok; ++r) {
        for (int c = 0; c < 3 && ok; ++c) {
            float ref = 0.0f;
            for (int kk = 0; kk < K; ++kk)
                ref += h_A[r * K + kk] * h_B[kk * N + c];
            if (std::fabs(h_C[r * N + c] - ref) > 1e-2f) ok = false;
        }
    }

    std::cout << "matmul_naive  M=" << M << " K=" << K << " N=" << N
              << "  time=" << std::fixed << std::setprecision(3) << ms << " ms"
              << "  gflops=" << std::setprecision(2) << gflops
              << "  check=" << (ok ? "OK" : "FAIL") << std::endl;

    clReleaseEvent(event);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
