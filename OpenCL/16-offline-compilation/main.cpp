#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>

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

// Кэширование скомпилированного OpenCL-бинарника.
//
// Компиляция из исходного кода (clBuildProgram) может занимать десятки
// миллисекунд — это заметно при запуске приложения, особенно если ядер
// много. OpenCL позволяет сохранить скомпилированный бинарник на диск
// и при следующем запуске загрузить его через clCreateProgramWithBinary,
// полностью пропустив этап компиляции.
//
// Алгоритм:
//   1. Если файл kernel.bin существует — загружаем бинарник оттуда.
//   2. Иначе — компилируем из исходника и сохраняем бинарник в kernel.bin.
//
// В обоих случаях замеряем время build/load и выводим его, чтобы
// наглядно показать выигрыш от кэширования.
//
// В продакшене бинарник привязан к конкретному устройству и версии
// драйвера: при смене GPU или обновлении драйвера кэш нужно сбросить.
static const char* kernel_source = R"CLC(
__kernel void vector_add(__global const float* x,
                         __global const float* y,
                         __global float* z,
                         const int n) {
    int i = get_global_id(0);
    if (i < n) {
        z[i] = x[i] + y[i];
    }
}
)CLC";

static const char* BINARY_FILE = "kernel.bin";

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

// Проверяет, существует ли файл.
static bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}

// Сохраняет скомпилированный бинарник программы на диск.
static void save_program_binary(cl_program prog, const char* path) {
    // Узнаём размер бинарника.
    size_t binary_size = 0;
    CL_CHECK(clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES,
                              sizeof(size_t), &binary_size, nullptr));

    // Забираем сам бинарник.
    std::vector<unsigned char> binary(binary_size);
    unsigned char* ptrs[1] = { binary.data() };
    CL_CHECK(clGetProgramInfo(prog, CL_PROGRAM_BINARIES,
                              sizeof(unsigned char*), ptrs, nullptr));

    // Пишем на диск.
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Cannot write to " << path << std::endl;
        std::exit(1);
    }
    f.write(reinterpret_cast<const char*>(binary.data()),
            static_cast<std::streamsize>(binary_size));
    std::cout << "Saved binary to " << path
              << " (" << binary_size << " bytes)" << std::endl;
}

// Читает бинарный файл целиком.
static std::vector<unsigned char> read_binary_file(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << path << std::endl;
        std::exit(1);
    }
    size_t size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<unsigned char> data(size);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(size));
    return data;
}

int main() {
    const int N = 1 << 22;
    const size_t bytes = N * sizeof(float);

    cl_device_id dev = get_gpu_device();
    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev,
                                                  CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    // --- Создание программы: из кэша или из исходника ---
    cl_program prog = nullptr;
    bool from_cache = false;

    auto t_build_start = std::chrono::high_resolution_clock::now();

    if (file_exists(BINARY_FILE)) {
        // Путь 2: загружаем из кэша (clCreateProgramWithBinary).
        std::vector<unsigned char> binary = read_binary_file(BINARY_FILE);
        size_t bin_size = binary.size();
        const unsigned char* bin_ptr = binary.data();
        cl_int binary_status = 0;

        prog = clCreateProgramWithBinary(ctx, 1, &dev,
                                         &bin_size, &bin_ptr,
                                         &binary_status, &err);
        CL_CHECK(err);
        if (binary_status != CL_SUCCESS) {
            std::cerr << "Binary status error: " << binary_status << std::endl;
            return 1;
        }
        err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                                  0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                                  log_size, &log[0], nullptr);
            std::cerr << "Build from binary failed:\n" << log << std::endl;
            return 1;
        }
        from_cache = true;
        std::cout << "Loaded program from cached binary: " << BINARY_FILE
                  << std::endl;
    } else {
        // Путь 1: компилируем из исходника и сохраняем бинарник.
        size_t src_len = std::strlen(kernel_source);
        prog = clCreateProgramWithSource(ctx, 1, &kernel_source,
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
            std::cerr << "Build from source failed:\n" << log << std::endl;
            return 1;
        }
        from_cache = false;
        std::cout << "Compiled program from source." << std::endl;

        // Сохраняем бинарник для следующих запусков.
        save_program_binary(prog, BINARY_FILE);
    }

    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(
                          t_build_end - t_build_start).count();

    std::cout << "Build mode: " << (from_cache ? "cached binary" : "source")
              << "  build_time=" << std::fixed << std::setprecision(3)
              << build_ms << " ms" << std::endl;

    // --- Создание ядра ---
    cl_kernel kernel = clCreateKernel(prog, "vector_add", &err);
    CL_CHECK(err);

    // --- Данные ---
    std::vector<float> h_x(N, 1.0f);
    std::vector<float> h_y(N, 2.0f);
    std::vector<float> h_z(N, 0.0f);

    cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                bytes, h_x.data(), &err);
    CL_CHECK(err);
    cl_mem d_y = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                bytes, h_y.data(), &err);
    CL_CHECK(err);
    cl_mem d_z = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    CL_CHECK(err);

    // --- Аргументы + запуск ---
    int n_int = N;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_y));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_z));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &n_int));

    size_t local_size = 256;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;

    cl_event event;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                    &global_size, &local_size,
                                    0, nullptr, &event));
    CL_CHECK(clWaitForEvents(1, &event));

    // --- Замер времени ядра ---
    cl_ulong t0 = 0, t1 = 0;
    CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &t0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &t1, nullptr));
    double kernel_ms = (t1 - t0) * 1e-6;

    // --- Чтение и проверка ---
    CL_CHECK(clEnqueueReadBuffer(queue, d_z, CL_TRUE, 0, bytes,
                                 h_z.data(), 0, nullptr, nullptr));

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float e = std::fabs(h_z[i] - 3.0f);
        if (e > max_err) max_err = e;
    }

    std::cout << "N=" << N
              << "  max_err=" << max_err
              << "  check=" << (max_err < 1e-6f ? "OK" : "FAIL")
              << "  kernel_time=" << std::fixed << std::setprecision(3)
              << kernel_ms << " ms" << std::endl;

    if (!from_cache) {
        std::cout << "\nПодсказка: запустите программу ещё раз — при повторном "
                     "запуске\nбудет использован кэшированный бинарник (kernel.bin), "
                     "и время\nсборки должно уменьшиться." << std::endl;
    }

    // --- Cleanup ---
    clReleaseEvent(event);
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_y);
    clReleaseMemObject(d_z);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
