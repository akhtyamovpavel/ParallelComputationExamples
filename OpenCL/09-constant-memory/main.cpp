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

// Демонстрация __constant-памяти в OpenCL.
// Аналог CUDA-примера 03-memory-model/06-constant.
//
// В OpenCL __constant — это отдельное адресное пространство с особыми свойствами:
//  1. Данные read-only со стороны ядра.
//  2. Все work-item'ы в wavefront/warp, читающие ОДИН И ТОТ ЖЕ адрес,
//     получают данные за ОДНУ транзакцию (broadcast). В CUDA это __constant__
//     память, обслуживаемая constant cache.
//  3. Размер ограничен (обычно 64 KB на большинстве GPU — см.
//     CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE).
//
// Когда все work-item'ы читают одни и те же коэффициенты (как в полиноме),
// __constant идеален: один broadcast вместо N обращений через L1/L2.
//
// Если же work-item'ы читают РАЗНЫЕ адреса из __constant, broadcast не работает
// и производительность может быть даже ХУЖЕ, чем __global + кэш L1.
//
// Сравниваем два ядра:
//   poly_global   — коэффициенты через __global const float*
//   poly_constant — коэффициенты через __constant float*
static const char* kernel_source = R"CLC(
// Вычисляем полином a*x^2 + b*x + c для массива значений x.
// coeffs[0] = a, coeffs[1] = b, coeffs[2] = c.

__kernel void poly_global(__global const float* x,
                          __global float* y,
                          __global const float* coeffs,
                          const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float xi = x[gid];
        // Все work-item'ы читают coeffs[0..2] — одни и те же адреса.
        // Через __global они идут через L1/L2 кэш. Первое обращение
        // загружает кэш-линию, дальше — cache hit, но всё равно проходит
        // через кэш-иерархию.
        float a = coeffs[0];
        float b = coeffs[1];
        float c = coeffs[2];
        y[gid] = a * xi * xi + b * xi + c;
    }
}

__kernel void poly_constant(__global const float* x,
                            __global float* y,
                            __constant float* coeffs,
                            const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float xi = x[gid];
        // Через __constant: hardware broadcast — все work-item'ы в wavefront
        // читают один адрес, constant cache отдаёт его за одну транзакцию.
        // Это аналог CUDA __constant__ памяти.
        float a = coeffs[0];
        float b = coeffs[1];
        float c = coeffs[2];
        y[gid] = a * xi * xi + b * xi + c;
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
    const int N = 1 << 24;   // 16M элементов
    const size_t bytes = N * sizeof(float);
    const int NUM_ITERS = 10;  // число запусков для усреднения

    // Коэффициенты полинома: 2*x^2 + 3*x + 1
    float coeffs[3] = {2.0f, 3.0f, 1.0f};
    const size_t coeffs_bytes = 3 * sizeof(float);

    std::vector<float> h_x(N);
    for (int i = 0; i < N; ++i) h_x[i] = static_cast<float>(i % 10000) * 0.0001f;
    std::vector<float> h_y(N, 0.0f);

    cl_device_id dev = get_gpu_device();

    // Выводим максимальный размер constant-буфера на устройстве.
    cl_ulong max_const_size = 0;
    CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                             sizeof(cl_ulong), &max_const_size, nullptr));
    cl_uint max_const_args = 0;
    CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_MAX_CONSTANT_ARGS,
                             sizeof(cl_uint), &max_const_args, nullptr));

    std::cout << "=== __constant memory demo ===" << std::endl;
    std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = "
              << max_const_size / 1024 << " KB" << std::endl;
    std::cout << "CL_DEVICE_MAX_CONSTANT_ARGS        = "
              << max_const_args << std::endl;
    std::cout << "N = " << N << " elements, polynomial: 2*x^2 + 3*x + 1"
              << std::endl << std::endl;

    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev,
                                                  CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    cl_program prog = build_program(ctx, dev, kernel_source);
    cl_kernel k_global   = clCreateKernel(prog, "poly_global", &err);
    CL_CHECK(err);
    cl_kernel k_constant = clCreateKernel(prog, "poly_constant", &err);
    CL_CHECK(err);

    cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                bytes, h_x.data(), &err);
    CL_CHECK(err);
    cl_mem d_y = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    CL_CHECK(err);

    // Буфер коэффициентов — обычный __global.
    cl_mem d_coeffs_global = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            coeffs_bytes, coeffs, &err);
    CL_CHECK(err);

    // Буфер коэффициентов для __constant — тот же clCreateBuffer, но ядро
    // принимает его как __constant float*. Драйвер размещает его в constant cache.
    cl_mem d_coeffs_const = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           coeffs_bytes, coeffs, &err);
    CL_CHECK(err);

    size_t local_size = 256;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;
    int n_int = N;

    // --- Ядро с __global const коэффициентами ---
    {
        CL_CHECK(clSetKernelArg(k_global, 0, sizeof(cl_mem), &d_x));
        CL_CHECK(clSetKernelArg(k_global, 1, sizeof(cl_mem), &d_y));
        CL_CHECK(clSetKernelArg(k_global, 2, sizeof(cl_mem), &d_coeffs_global));
        CL_CHECK(clSetKernelArg(k_global, 3, sizeof(int), &n_int));

        // Прогрев
        CL_CHECK(clEnqueueNDRangeKernel(queue, k_global, 1, nullptr,
                                        &global_size, &local_size,
                                        0, nullptr, nullptr));
        CL_CHECK(clFinish(queue));

        double total_ms = 0.0;
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            cl_event event;
            CL_CHECK(clEnqueueNDRangeKernel(queue, k_global, 1, nullptr,
                                            &global_size, &local_size,
                                            0, nullptr, &event));
            CL_CHECK(clWaitForEvents(1, &event));

            cl_ulong t_start = 0, t_end = 0;
            CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                             sizeof(cl_ulong), &t_start, nullptr));
            CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                             sizeof(cl_ulong), &t_end, nullptr));
            total_ms += (t_end - t_start) * 1e-6;
            clReleaseEvent(event);
        }
        double avg_ms = total_ms / NUM_ITERS;

        // Проверка
        CL_CHECK(clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, bytes,
                                     h_y.data(), 0, nullptr, nullptr));
        bool ok = true;
        for (int i = 0; i < 1024 && ok; ++i) {
            float xi = h_x[i];
            float expected = coeffs[0] * xi * xi + coeffs[1] * xi + coeffs[2];
            if (std::fabs(h_y[i] - expected) > 1e-4f) ok = false;
        }

        std::cout << std::left << std::setw(22) << "__global const"
                  << " avg_time=" << std::right << std::setw(8)
                  << std::fixed << std::setprecision(3) << avg_ms << " ms"
                  << "  (" << NUM_ITERS << " iters)"
                  << "  check=" << (ok ? "OK" : "FAIL") << std::endl;
    }

    // --- Ядро с __constant коэффициентами ---
    {
        CL_CHECK(clSetKernelArg(k_constant, 0, sizeof(cl_mem), &d_x));
        CL_CHECK(clSetKernelArg(k_constant, 1, sizeof(cl_mem), &d_y));
        CL_CHECK(clSetKernelArg(k_constant, 2, sizeof(cl_mem), &d_coeffs_const));
        CL_CHECK(clSetKernelArg(k_constant, 3, sizeof(int), &n_int));

        // Прогрев
        CL_CHECK(clEnqueueNDRangeKernel(queue, k_constant, 1, nullptr,
                                        &global_size, &local_size,
                                        0, nullptr, nullptr));
        CL_CHECK(clFinish(queue));

        double total_ms = 0.0;
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            cl_event event;
            CL_CHECK(clEnqueueNDRangeKernel(queue, k_constant, 1, nullptr,
                                            &global_size, &local_size,
                                            0, nullptr, &event));
            CL_CHECK(clWaitForEvents(1, &event));

            cl_ulong t_start = 0, t_end = 0;
            CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                             sizeof(cl_ulong), &t_start, nullptr));
            CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                             sizeof(cl_ulong), &t_end, nullptr));
            total_ms += (t_end - t_start) * 1e-6;
            clReleaseEvent(event);
        }
        double avg_ms = total_ms / NUM_ITERS;

        // Проверка
        CL_CHECK(clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, bytes,
                                     h_y.data(), 0, nullptr, nullptr));
        bool ok = true;
        for (int i = 0; i < 1024 && ok; ++i) {
            float xi = h_x[i];
            float expected = coeffs[0] * xi * xi + coeffs[1] * xi + coeffs[2];
            if (std::fabs(h_y[i] - expected) > 1e-4f) ok = false;
        }

        std::cout << std::left << std::setw(22) << "__constant"
                  << " avg_time=" << std::right << std::setw(8)
                  << std::fixed << std::setprecision(3) << avg_ms << " ms"
                  << "  (" << NUM_ITERS << " iters)"
                  << "  check=" << (ok ? "OK" : "FAIL") << std::endl;
    }

    // Пояснение результатов
    std::cout << "\n// На данном ядре разница может быть невелика, т.к. всего 3 float"
              << std::endl
              << "// коэффициента легко помещаются в L1 кэш даже через __global."
              << std::endl
              << "// Преимущество __constant проявляется сильнее когда:" << std::endl
              << "//  - коэффициентов больше (десятки-сотни, но < 64KB)" << std::endl
              << "//  - ядро compute-bound и constant cache разгружает L1 для данных"
              << std::endl
              << "//  - ВСЕ work-item'ы читают ОДНИ И ТЕ ЖЕ адреса (broadcast)"
              << std::endl;

    clReleaseMemObject(d_x);
    clReleaseMemObject(d_y);
    clReleaseMemObject(d_coeffs_global);
    clReleaseMemObject(d_coeffs_const);
    clReleaseKernel(k_global);
    clReleaseKernel(k_constant);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
