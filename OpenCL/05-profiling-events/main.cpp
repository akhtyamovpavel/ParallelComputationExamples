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

// В OpenCL замер времени устроен иначе, чем в CUDA: у каждой операции
// (enqueue*) можно попросить cl_event, а у event'а — прочитать временные метки:
//   CL_PROFILING_COMMAND_QUEUED   — когда команда попала в очередь (host)
//   CL_PROFILING_COMMAND_SUBMIT   — когда команда отправлена на устройство
//   CL_PROFILING_COMMAND_START    — когда устройство начало выполнение
//   CL_PROFILING_COMMAND_END      — когда устройство закончило
//
// Для этого очередь должна быть создана с флагом CL_QUEUE_PROFILING_ENABLE.
// В CUDA аналог — cudaEvent + cudaEventElapsedTime.
static const char* kernel_source = R"CLC(
__kernel void saxpy(__global const float* x,
                    __global float* y,
                    const float a,
                    const int n) {
    int i = get_global_id(0);
    if (i < n) {
        y[i] = a * x[i] + y[i];
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

static double event_ms(cl_event ev, cl_profiling_info from, cl_profiling_info to) {
    cl_ulong t0 = 0, t1 = 0;
    CL_CHECK(clGetEventProfilingInfo(ev, from, sizeof(cl_ulong), &t0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(ev, to,   sizeof(cl_ulong), &t1, nullptr));
    return (t1 - t0) * 1e-6;
}

int main() {
    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_x(N, 1.0f);
    std::vector<float> h_y(N, 2.0f);

    cl_device_id dev = get_gpu_device();
    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    // CL_QUEUE_PROFILING_ENABLE — ключевой флаг; без него
    // clGetEventProfilingInfo вернёт CL_PROFILING_INFO_NOT_AVAILABLE.
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
    cl_kernel kernel = clCreateKernel(prog, "saxpy", &err);
    CL_CHECK(err);

    // Буферы без CL_MEM_COPY_HOST_PTR — мы хотим замерить копирование отдельно.
    cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  bytes, nullptr, &err);
    CL_CHECK(err);
    cl_mem d_y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    CL_CHECK(err);

    // --- Замер: Host → Device ---
    cl_event ev_write_x, ev_write_y;
    CL_CHECK(clEnqueueWriteBuffer(queue, d_x, CL_FALSE, 0, bytes,
                                  h_x.data(), 0, nullptr, &ev_write_x));
    CL_CHECK(clEnqueueWriteBuffer(queue, d_y, CL_FALSE, 0, bytes,
                                  h_y.data(), 0, nullptr, &ev_write_y));

    // --- Замер: Kernel ---
    float a = 2.0f;
    int n_int = N;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_y));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(float),  &a));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &n_int));

    size_t local_size = 256;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;

    cl_event ev_kernel;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                    &global_size, &local_size,
                                    0, nullptr, &ev_kernel));

    // --- Замер: Device → Host ---
    cl_event ev_read;
    CL_CHECK(clEnqueueReadBuffer(queue, d_y, CL_FALSE, 0, bytes,
                                 h_y.data(), 0, nullptr, &ev_read));

    CL_CHECK(clFinish(queue));

    // Выводим все четыре стадии event'а для каждой операции.
    auto print_event = [](const char* label, cl_event ev) {
        double queued_to_submit = event_ms(ev, CL_PROFILING_COMMAND_QUEUED,
                                               CL_PROFILING_COMMAND_SUBMIT);
        double submit_to_start  = event_ms(ev, CL_PROFILING_COMMAND_SUBMIT,
                                               CL_PROFILING_COMMAND_START);
        double exec_time        = event_ms(ev, CL_PROFILING_COMMAND_START,
                                               CL_PROFILING_COMMAND_END);
        std::cout << std::left << std::setw(14) << label
                  << "  queued→submit=" << std::right << std::setw(8)
                  << std::fixed << std::setprecision(3) << queued_to_submit << " ms"
                  << "  submit→start=" << std::setw(8) << submit_to_start << " ms"
                  << "  exec=" << std::setw(8) << exec_time << " ms"
                  << std::endl;
    };

    std::cout << "SAXPY  N=" << N << "  a=" << a << std::endl;
    print_event("write x", ev_write_x);
    print_event("write y", ev_write_y);
    print_event("kernel",  ev_kernel);
    print_event("read y",  ev_read);

    double bw_write = (2.0 * bytes) /
        (event_ms(ev_write_x, CL_PROFILING_COMMAND_START,
                  CL_PROFILING_COMMAND_END) * 1e-3) / (1024.0*1024.0*1024.0);
    double bw_read = bytes /
        (event_ms(ev_read, CL_PROFILING_COMMAND_START,
                  CL_PROFILING_COMMAND_END) * 1e-3) / (1024.0*1024.0*1024.0);
    std::cout << "\nEffective bandwidth:  H→D ~ "
              << std::setprecision(2) << bw_write << " GB/s"
              << "   D→H ~ " << bw_read << " GB/s" << std::endl;

    clReleaseEvent(ev_write_x);
    clReleaseEvent(ev_write_y);
    clReleaseEvent(ev_kernel);
    clReleaseEvent(ev_read);
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_y);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
