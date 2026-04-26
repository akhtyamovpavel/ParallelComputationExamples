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

// Наивный транспоз: каждый work-item читает один элемент из (row, col)
// и пишет в (col, row). Чтение coalesced (соседние WI читают соседние float),
// но запись — со страйдом rows, то есть НЕ coalesced.
//
// Тайловый транспоз: блок work-item'ов загружает тайл в __local память,
// делает barrier, затем записывает из __local в глобальную память с coalesced-
// доступом. Аналог shared memory в CUDA.
static const char* kernel_source = R"CLC(
#define TILE 16

__kernel void transpose_naive(__global const float* in,
                              __global float* out,
                              const int rows,
                              const int cols) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col];
    }
}

__kernel void transpose_local(__global const float* in,
                              __global float* out,
                              const int rows,
                              const int cols) {
    __local float tile[TILE][TILE];

    int col = get_group_id(0) * TILE + get_local_id(0);
    int row = get_group_id(1) * TILE + get_local_id(1);

    if (row < rows && col < cols) {
        tile[get_local_id(1)][get_local_id(0)] = in[row * cols + col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Выходной блок транспонирован: (group_x, group_y) -> (group_y, group_x).
    int out_col = get_group_id(1) * TILE + get_local_id(0);
    int out_row = get_group_id(0) * TILE + get_local_id(1);
    if (out_row < cols && out_col < rows) {
        out[out_row * rows + out_col] = tile[get_local_id(0)][get_local_id(1)];
    }
}
)CLC";

static cl_device_id get_gpu_device() {
    cl_uint num_platforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    for (cl_uint p = 0; p < num_platforms; ++p) {
        cl_uint num_devices = 0;
        cl_int err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0,
                                    nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) continue;
        std::vector<cl_device_id> devices(num_devices);
        CL_CHECK(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU,
                                num_devices, devices.data(), nullptr));
        return devices[0];
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

static void run_transpose(cl_command_queue queue, cl_kernel kernel,
                          cl_mem d_in, cl_mem d_out,
                          int rows, int cols, const char* label) {
    int n_int = rows;
    int c_int = cols;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &n_int));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &c_int));

    const size_t tile = 16;
    size_t global[2] = {((cols + tile - 1) / tile) * tile,
                        ((rows + tile - 1) / tile) * tile};
    size_t local[2] = {tile, tile};

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
    double gbps = (2.0 * rows * cols * sizeof(float)) / ((t_end - t_start) * 1e-9)
                  / (1024.0 * 1024.0 * 1024.0);

    // Проверка: читаем результат и сверяем угол 32x32.
    size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    std::vector<float> h_out(rows * cols);
    CL_CHECK(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, bytes,
                                 h_out.data(), 0, nullptr, nullptr));

    std::vector<float> h_in(rows * cols);
    CL_CHECK(clEnqueueReadBuffer(queue, d_in, CL_TRUE, 0, bytes,
                                 h_in.data(), 0, nullptr, nullptr));
    bool ok = true;
    for (int r = 0; r < 32 && ok; ++r)
        for (int c = 0; c < 32 && ok; ++c)
            if (h_out[c * rows + r] != h_in[r * cols + c]) ok = false;

    std::cout << std::left << std::setw(20) << label
              << " time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << "  bw=" << std::right << std::setw(7)
              << std::fixed << std::setprecision(2) << gbps << " GB/s"
              << "  check=" << (ok ? "OK" : "FAIL") << std::endl;

    clReleaseEvent(event);
}

int main() {
    const int ROWS = 4096;
    const int COLS = 4096;
    const size_t bytes = static_cast<size_t>(ROWS) * COLS * sizeof(float);

    std::vector<float> h_in(ROWS * COLS);
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS; ++c)
            h_in[r * COLS + c] = static_cast<float>(r * COLS + c);

    cl_device_id dev = get_gpu_device();
    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev,
                                                  CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    cl_program prog = build_program(ctx, dev, kernel_source);
    cl_kernel k_naive = clCreateKernel(prog, "transpose_naive", &err);
    CL_CHECK(err);
    cl_kernel k_local = clCreateKernel(prog, "transpose_local", &err);
    CL_CHECK(err);

    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 bytes, h_in.data(), &err);
    CL_CHECK(err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    CL_CHECK(err);

    run_transpose(queue, k_naive, d_in, d_out, ROWS, COLS, "naive");
    run_transpose(queue, k_local, d_in, d_out, ROWS, COLS, "local_memory");

    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseKernel(k_naive);
    clReleaseKernel(k_local);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
