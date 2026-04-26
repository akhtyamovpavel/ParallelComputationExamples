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

// Дерево редукции в __local памяти. Каждый work-group суммирует свой кусок
// массива и записывает частичную сумму в out[get_group_id(0)].
// Финальное суммирование частичных сумм — на хосте (можно и вторым запуском
// ядра, но для учебного примера достаточно).
//
// Сравните с CUDA/04-reduction/01-default-sum: логика та же,
// но вместо __shared__ используется __local, вместо __syncthreads() —
// barrier(CLK_LOCAL_MEM_FENCE).
static const char* kernel_source = R"CLC(
__kernel void reduce(__global const int* in,
                     __global int* out,
                     __local int* scratch,
                     const int n) {
    unsigned int tid = get_local_id(0);
    unsigned int gid = get_global_id(0);

    // Загрузка в local memory; за границей массива — 0 (нейтральный элемент).
    scratch[tid] = (gid < n) ? in[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Классическое дерево: на каждом шаге складываем пары с расстоянием s.
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

int main() {
    const int N = 1 << 22;
    const size_t local_size = 256;
    const size_t num_groups = (N + local_size - 1) / local_size;
    const size_t global_size = num_groups * local_size;

    std::vector<int> h_in(N, 1);

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

    cl_kernel kernel = clCreateKernel(prog, "reduce", &err);
    CL_CHECK(err);

    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 N * sizeof(int), h_in.data(), &err);
    CL_CHECK(err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                  num_groups * sizeof(int), nullptr, &err);
    CL_CHECK(err);

    int n_int = N;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
    CL_CHECK(clSetKernelArg(kernel, 2, local_size * sizeof(int), nullptr));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &n_int));

    cl_event event;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                    &global_size, &local_size,
                                    0, nullptr, &event));
    CL_CHECK(clWaitForEvents(1, &event));

    cl_ulong t_start = 0, t_end = 0;
    CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &t_start, nullptr));
    CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &t_end, nullptr));
    double ms = (t_end - t_start) * 1e-6;

    std::vector<int> h_out(num_groups);
    CL_CHECK(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                                 num_groups * sizeof(int),
                                 h_out.data(), 0, nullptr, nullptr));

    long long sum = 0;
    for (size_t i = 0; i < num_groups; ++i) sum += h_out[i];

    std::cout << "N=" << N
              << "  sum=" << sum
              << "  expected=" << N
              << "  check=" << (sum == N ? "OK" : "FAIL")
              << "  kernel_time=" << std::fixed << std::setprecision(3)
              << ms << " ms" << std::endl;

    clReleaseEvent(event);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
