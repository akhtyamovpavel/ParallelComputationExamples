#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
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

// Загрузка kernel из внешнего .cl файла.
//
// Во всех предыдущих примерах исходник ядра хранился как inline-строка
// (static const char* kernel_source = R"CLC(...)CLC"). Это удобно для
// коротких учебных примеров, но в реальных проектах ядра живут в
// отдельных .cl файлах, и вот почему:
//   1. Подсветка синтаксиса — большинство редакторов и IDE поддерживают
//      .cl файлы (OpenCL C), но не подсвечивают код внутри C++ строки.
//   2. Удобство редактирования — kernel может быть длинным, с include'ами,
//      макросами; в отдельном файле всё это читается естественнее.
//   3. Переиспользование — один .cl файл можно подгрузить из нескольких
//      хост-программ или даже языков (Python через PyOpenCL и т.д.).
//
// Файл square.cl лежит рядом с main.cpp и содержит простое ядро
// возведения в квадрат: out[i] = in[i] * in[i].

static std::string read_file(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Cannot open file: " << path << std::endl;
        std::exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

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
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    // --- 1. Читаем исходник ядра с диска. ---
    // Путь square.cl предполагает запуск из той же директории.
    std::string source = read_file("square.cl");
    const char* src_ptr = source.c_str();
    size_t src_len = source.size();

    std::cout << "Loaded kernel source from square.cl ("
              << src_len << " bytes)" << std::endl;

    // --- 2. Инициализация OpenCL: устройство, контекст, очередь. ---
    cl_device_id dev = get_gpu_device();
    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev,
                                                  CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    // --- 3. Компиляция — точно так же, как с inline-строкой. ---
    // clCreateProgramWithSource принимает массив строк; ему всё равно,
    // откуда строка взялась — из R"CLC()" литерала или с диска.
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr,
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

    // --- 4. Данные: in[i] = i, ожидаемый out[i] = i*i. ---
    std::vector<float> h_in(N);
    std::vector<float> h_out(N, 0.0f);
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 bytes, h_in.data(), &err);
    CL_CHECK(err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    CL_CHECK(err);

    // --- 5. Аргументы + запуск. ---
    int n_int = N;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &n_int));

    size_t local_size = 256;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;

    cl_event event;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                    &global_size, &local_size,
                                    0, nullptr, &event));
    CL_CHECK(clWaitForEvents(1, &event));

    // --- 6. Замер времени. ---
    cl_ulong t_start = 0, t_end = 0;
    CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &t_start, nullptr));
    CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &t_end, nullptr));
    double ms = (t_end - t_start) * 1e-6;

    // --- 7. Чтение и проверка. ---
    CL_CHECK(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, bytes,
                                 h_out.data(), 0, nullptr, nullptr));

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float expected = static_cast<float>(i) * static_cast<float>(i);
        float e = std::fabs(h_out[i] - expected);
        if (e > max_err) max_err = e;
    }

    std::cout << "N=" << N
              << "  max_err=" << max_err
              << "  check=" << (max_err < 1e-3f ? "OK" : "FAIL")
              << "  kernel_time=" << std::fixed << std::setprecision(3)
              << ms << " ms" << std::endl;

    // --- 8. Cleanup ---
    clReleaseEvent(event);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
