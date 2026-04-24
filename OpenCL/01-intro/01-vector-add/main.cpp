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

// Исходник ядра встроен в строку — это главный визуальный контраст с CUDA:
// в OpenCL kernel компилируется на лету силами drivers'а (clBuildProgram),
// а не nvcc'ом при сборке хост-программы. Ту же строку можно было бы
// держать в отдельном .cl файле и грузить с диска — ничего больше не меняется.
static const char* kernel_source = R"CLC(
__kernel void vector_add(__global const float* x,
                         __global const float* y,
                         __global float* z,
                         const int n) {
    int idx = get_global_id(0);
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}
)CLC";

int main() {
    const int N = 1 << 22;

    // --- 1. Выбираем платформу и GPU-устройство на ней. ---
    // В CUDA весь этот слой скрыт: есть просто cudaSetDevice(i). В OpenCL
    // платформа-vendor живёт параллельно устройствам, и контекст строится
    // явно из указанных device id.
    cl_uint num_platforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    cl_platform_id chosen_platform = nullptr;
    cl_device_id chosen_device = nullptr;
    for (cl_uint p = 0; p < num_platforms && chosen_device == nullptr; ++p) {
        cl_uint num_devices = 0;
        cl_int err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0,
                                    nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(num_devices);
        CL_CHECK(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU,
                                num_devices, devices.data(), nullptr));
        chosen_platform = platforms[p];
        chosen_device = devices[0];
    }
    if (chosen_device == nullptr) {
        std::cerr << "No OpenCL GPU device found." << std::endl;
        return 1;
    }

    // --- 2. Контекст + очередь команд. ---
    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &chosen_device,
                                     nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, chosen_device, 0, &err);
    CL_CHECK(err);

    // --- 3. Компиляция kernel'а во время работы. ---
    size_t src_len = std::strlen(kernel_source);
    cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_source,
                                                   &src_len, &err);
    CL_CHECK(err);
    err = clBuildProgram(program, 1, &chosen_device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // На ошибке билда забираем log и печатаем — иначе отлаживать почти нечем.
        size_t log_size = 0;
        clGetProgramBuildInfo(program, chosen_device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, chosen_device, CL_PROGRAM_BUILD_LOG,
                              log_size, &log[0], nullptr);
        std::cerr << "clBuildProgram failed (" << err << "):" << std::endl
                  << log << std::endl;
        return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    CL_CHECK(err);

    // --- 4. Буферы на устройстве + копия входа с host. ---
    std::vector<float> h_x(N, 1.0f);
    std::vector<float> h_y(N, 2.0f);
    std::vector<float> h_z(N, 0.0f);
    const size_t bytes = N * sizeof(float);

    cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                bytes, h_x.data(), &err);
    CL_CHECK(err);
    cl_mem d_y = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                bytes, h_y.data(), &err);
    CL_CHECK(err);
    cl_mem d_z = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    CL_CHECK(err);

    // --- 5. Аргументы ядра + enqueue. ---
    int n_int = N;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_y));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_z));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &n_int));

    size_t local_size = 256;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                    &global_size, &local_size, 0, nullptr, nullptr));

    CL_CHECK(clEnqueueReadBuffer(queue, d_z, CL_TRUE, 0, bytes,
                                 h_z.data(), 0, nullptr, nullptr));

    // --- 6. Проверка. ---
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float e = std::fabs(h_z[i] - 3.0f);
        if (e > max_err) {
            max_err = e;
        }
    }
    std::cout << "N=" << N << "  max_err=" << max_err << std::endl;

    // --- 7. Явный cleanup. В OpenCL все объекты — refcounted handles. ---
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_y);
    clReleaseMemObject(d_z);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}
