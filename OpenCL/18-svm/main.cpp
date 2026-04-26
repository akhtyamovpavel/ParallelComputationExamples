#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cmath>

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#define CL_CHECK(call) do {                                                   \
    cl_int _err = (call);                                                     \
    if (_err != CL_SUCCESS) {                                                 \
        std::cerr << "OpenCL error " << _err << " at "                        \
                  << __FILE__ << ":" << __LINE__ << std::endl;                \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// ============================================================================
// SVM (Shared Virtual Memory) — аналог Unified Memory (cudaMallocManaged)
// в CUDA.
//
// В классическом OpenCL (1.x) данные передаются между хостом и устройством
// через буферы: clCreateBuffer + clEnqueueWriteBuffer / clEnqueueReadBuffer.
// Это аналог cudaMalloc + cudaMemcpy — требует явного копирования.
//
// SVM (OpenCL 2.0+) позволяет хосту и устройству работать с одним и тем же
// указателем. Это упрощает код и устраняет необходимость явных копирований.
//
// Виды SVM:
//   1) Coarse-grained buffer SVM — хост и устройство разделяют память,
//      но синхронизация через map/unmap (как в этом примере).
//      Аналог: cudaMallocManaged с ручными cudaDeviceSynchronize.
//
//   2) Fine-grained buffer SVM — синхронизация на уровне отдельных
//      обращений; не нужен map/unmap.
//
//   3) Fine-grained system SVM — любой malloc() на хосте доступен
//      на устройстве. Максимальная прозрачность.
//
//   4) Fine-grained SVM с атомарными операциями — хост и устройство
//      могут одновременно атомарно обращаться к одним данным.
//      Требует аппаратной поддержки, доступно не на всех GPU.
//
// В этом примере демонстрируется coarse-grained SVM на задаче SAXPY:
//   y[i] = a * x[i] + y[i]
//
// Сравните с примером 05-profiling-events, где используются буферы:
//   - Нет clCreateBuffer
//   - Нет clEnqueueWriteBuffer / clEnqueueReadBuffer
//   - Вместо clSetKernelArg(sizeof(cl_mem)) → clSetKernelArgSVMPointer
//   - Синхронизация: clEnqueueSVMMap / clEnqueueSVMUnmap
// ============================================================================

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

// Проверяем поддержку OpenCL 2.0+ и SVM.
static bool check_svm_support(cl_device_id dev) {
    // Проверяем версию OpenCL C.
    char version_str[256] = {};
    CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_VERSION,
                             sizeof(version_str), version_str, nullptr));
    std::cout << "Device OpenCL C version: " << version_str << std::endl;

    int major = 0, minor = 0;
    if (std::sscanf(version_str, "OpenCL C %d.%d", &major, &minor) != 2) {
        std::cerr << "Не удалось разобрать версию OpenCL C." << std::endl;
        return false;
    }
    if (major < 2) {
        std::cerr << "Требуется OpenCL C 2.0+, доступна " << major << "."
                  << minor << ". SVM недоступна." << std::endl;
        return false;
    }

    // Проверяем SVM capabilities.
    cl_device_svm_capabilities svm_caps = 0;
    CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_SVM_CAPABILITIES,
                             sizeof(svm_caps), &svm_caps, nullptr));

    if (svm_caps == 0) {
        std::cerr << "Устройство не поддерживает SVM." << std::endl;
        return false;
    }

    std::cout << "SVM capabilities:" << std::endl;
    if (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
        std::cout << "  - Coarse-grained buffer SVM" << std::endl;
    if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
        std::cout << "  - Fine-grained buffer SVM" << std::endl;
    if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)
        std::cout << "  - Fine-grained system SVM" << std::endl;
    if (svm_caps & CL_DEVICE_SVM_ATOMICS)
        std::cout << "  - SVM atomics (хост и устройство могут атомарно "
                  << "обращаться к общей памяти)" << std::endl;

    if (!(svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
        std::cerr << "Coarse-grained buffer SVM не поддерживается." << std::endl;
        return false;
    }

    return true;
}

int main() {
    const int N = 1 << 22;  // ~4 миллиона элементов
    const size_t bytes = N * sizeof(float);
    const float a = 2.0f;

    cl_device_id dev = get_gpu_device();

    // Проверяем поддержку SVM.
    if (!check_svm_support(dev)) {
        std::cout << "SVM не поддерживается — завершение." << std::endl;
        return 0;
    }

    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev,
                                                  CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    // --- Сборка ядра ---
    size_t src_len = std::strlen(kernel_source);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_source,
                                                &src_len, &err);
    CL_CHECK(err);
    err = clBuildProgram(prog, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr);
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

    // ======================================================================
    // SVM-аллокация: вместо clCreateBuffer используем clSVMAlloc.
    // Возвращается обычный указатель, который можно использовать и на хосте,
    // и передать в ядро. Аналог cudaMallocManaged().
    // ======================================================================
    float* svm_x = static_cast<float*>(
        clSVMAlloc(ctx, CL_MEM_READ_ONLY, bytes, 0));
    float* svm_y = static_cast<float*>(
        clSVMAlloc(ctx, CL_MEM_READ_WRITE, bytes, 0));

    if (!svm_x || !svm_y) {
        std::cerr << "clSVMAlloc вернул nullptr — недостаточно памяти." << std::endl;
        if (svm_x) clSVMFree(ctx, svm_x);
        if (svm_y) clSVMFree(ctx, svm_y);
        clReleaseKernel(kernel);
        clReleaseProgram(prog);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return 1;
    }

    // ======================================================================
    // Заполнение на хосте: для coarse-grained SVM перед доступом хоста
    // нужно clEnqueueSVMMap (аналог "захвата" буфера хостом).
    // ======================================================================
    CL_CHECK(clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE,
                             svm_x, bytes, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE,
                             svm_y, bytes, 0, nullptr, nullptr));

    // Теперь хост может записать данные напрямую в SVM-указатели.
    // Никакого clEnqueueWriteBuffer не нужно!
    for (int i = 0; i < N; ++i) {
        svm_x[i] = 1.0f;
        svm_y[i] = 2.0f;
    }

    // Отдаём буферы обратно устройству.
    CL_CHECK(clEnqueueSVMUnmap(queue, svm_x, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueSVMUnmap(queue, svm_y, 0, nullptr, nullptr));

    // ======================================================================
    // Передача SVM-указателей в ядро.
    // Вместо clSetKernelArg(sizeof(cl_mem), &d_x) — clSetKernelArgSVMPointer.
    // ======================================================================
    CL_CHECK(clSetKernelArgSVMPointer(kernel, 0, svm_x));
    CL_CHECK(clSetKernelArgSVMPointer(kernel, 1, svm_y));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(float), &a));
    int n_int = N;
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &n_int));

    size_t local_size = 256;
    size_t global_size = ((N + local_size - 1) / local_size) * local_size;

    cl_event ev_kernel;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                    &global_size, &local_size,
                                    0, nullptr, &ev_kernel));
    CL_CHECK(clWaitForEvents(1, &ev_kernel));

    // --- Профилирование ядра ---
    cl_ulong t_start = 0, t_end = 0;
    CL_CHECK(clGetEventProfilingInfo(ev_kernel, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &t_start, nullptr));
    CL_CHECK(clGetEventProfilingInfo(ev_kernel, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &t_end, nullptr));
    double ms = (t_end - t_start) * 1e-6;

    // ======================================================================
    // Чтение результата на хосте: снова map, затем обращение к svm_y.
    // Никакого clEnqueueReadBuffer!
    // ======================================================================
    CL_CHECK(clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ,
                             svm_y, bytes, 0, nullptr, nullptr));

    // Проверка: y[i] = a * x[i] + y_old[i] = 2.0 * 1.0 + 2.0 = 4.0
    const float expected = a * 1.0f + 2.0f;
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(svm_y[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at i=" << i << ": got " << svm_y[i]
                      << ", expected " << expected << std::endl;
            ok = false;
            break;
        }
    }

    CL_CHECK(clEnqueueSVMUnmap(queue, svm_y, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));

    // --- Результаты ---
    std::cout << "\nSAXPY (SVM): N=" << N << "  a=" << a
              << "  y = a*x + y = " << a << "*1.0 + 2.0 = " << expected
              << std::endl;
    std::cout << "check=" << (ok ? "OK" : "FAIL")
              << "  kernel_time=" << std::fixed << std::setprecision(3)
              << ms << " ms" << std::endl;

    // Сравнение подходов (комментарий).
    std::cout << "\n// === Сравнение SVM и классических буферов ===" << std::endl;
    std::cout << "// Классический подход (OpenCL 1.x):" << std::endl;
    std::cout << "//   clCreateBuffer + clEnqueueWriteBuffer → ядро → "
              << "clEnqueueReadBuffer" << std::endl;
    std::cout << "//   Аналог в CUDA: cudaMalloc + cudaMemcpy" << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "// SVM подход (OpenCL 2.0+):" << std::endl;
    std::cout << "//   clSVMAlloc → SVMMap → заполнение указателя → SVMUnmap → "
              << "ядро → SVMMap → чтение" << std::endl;
    std::cout << "//   Аналог в CUDA: cudaMallocManaged" << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "// Преимущество: один указатель на хосте и устройстве,"
              << std::endl;
    std::cout << "// не нужны явные команды копирования." << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "// Fine-grained SVM (с атомиками между хостом и устройством)"
              << std::endl;
    std::cout << "// требует аппаратной поддержки и доступен не на всех GPU."
              << std::endl;

    // --- Очистка ---
    // clSVMFree вместо clReleaseMemObject — SVM-память освобождается отдельно.
    clSVMFree(ctx, svm_x);
    clSVMFree(ctx, svm_y);
    clReleaseEvent(ev_kernel);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
