#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
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

// Демонстрация разделения работы между несколькими устройствами OpenCL.
//
// Это уникальная возможность OpenCL, которой нет в CUDA: платформенная модель
// OpenCL позволяет одновременно использовать устройства разных типов и вендоров
// (например, CPU + GPU, или GPU NVIDIA + GPU AMD). В CUDA каждое ядро привязано
// к одному GPU NVIDIA, и для использования CPU потребуется отдельный фреймворк.
//
// Стратегия:
// 1. Перечисляем ВСЕ устройства на всех платформах (GPU + CPU).
// 2. Для каждого устройства создаём отдельный контекст, очередь, программу.
//    (Устройства из разных платформ не могут разделять контекст.)
// 3. Делим большой вектор на части — каждое устройство обрабатывает свой кусок.
// 4. Замеряем время на каждом устройстве через события профилирования.
// 5. Собираем результаты и проверяем корректность.
//
// Замечание: на практике для CPU и GPU нужно подбирать соотношение разбиения,
// потому что GPU обычно намного быстрее на параллельных задачах. Здесь мы
// делим поровну для простоты демонстрации.

static const char* kernel_source = R"CLC(
__kernel void vector_add(__global const float* a,
                         __global const float* b,
                         __global float* c,
                         const int n) {
    int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
)CLC";

// Информация об одном устройстве и его ресурсах OpenCL
struct DeviceInfo {
    cl_device_id   device;
    cl_context     context;
    cl_command_queue queue;
    cl_program     program;
    cl_kernel      kernel;
    cl_mem         d_a;
    cl_mem         d_b;
    cl_mem         d_c;
    std::string    name;
    cl_device_type type;
};

static std::string get_device_name(cl_device_id dev) {
    size_t size = 0;
    CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &size));
    std::string s(size, '\0');
    CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_NAME, size, &s[0], nullptr));
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

static const char* device_type_str(cl_device_type t) {
    if (t & CL_DEVICE_TYPE_GPU)         return "GPU";
    if (t & CL_DEVICE_TYPE_CPU)         return "CPU";
    if (t & CL_DEVICE_TYPE_ACCELERATOR) return "ACCEL";
    return "OTHER";
}

// Собираем все устройства со всех платформ.
// В CUDA аналог — cudaGetDeviceCount + cudaGetDeviceProperties, но только
// для GPU NVIDIA. Здесь мы получаем устройства любого типа и вендора.
static std::vector<std::pair<cl_platform_id, cl_device_id>> enumerate_all_devices() {
    std::vector<std::pair<cl_platform_id, cl_device_id>> result;

    cl_uint np = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &np));
    std::vector<cl_platform_id> platforms(np);
    CL_CHECK(clGetPlatformIDs(np, platforms.data(), nullptr));

    for (cl_uint p = 0; p < np; ++p) {
        cl_uint nd = 0;
        cl_int err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                                    0, nullptr, &nd);
        if (err != CL_SUCCESS || nd == 0) continue;

        std::vector<cl_device_id> devs(nd);
        CL_CHECK(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                                nd, devs.data(), nullptr));
        for (cl_uint d = 0; d < nd; ++d) {
            result.push_back({platforms[p], devs[d]});
        }
    }
    return result;
}

// Создаём контекст, очередь, компилируем программу для одного устройства.
// Каждое устройство получает свой контекст, потому что устройства из разных
// платформ (например, Intel CPU + NVIDIA GPU) не могут разделять один контекст.
static DeviceInfo setup_device(cl_platform_id platform, cl_device_id device) {
    DeviceInfo info;
    info.device = device;

    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_TYPE,
                             sizeof(cl_device_type), &info.type, nullptr));
    info.name = get_device_name(device);

    // Свойства контекста привязывают его к конкретной платформе
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0
    };

    cl_int err = 0;
    info.context = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);

    info.queue = clCreateCommandQueue(info.context, device,
                                      CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    size_t src_len = std::strlen(kernel_source);
    info.program = clCreateProgramWithSource(info.context, 1, &kernel_source,
                                             &src_len, &err);
    CL_CHECK(err);

    // Компиляция ядра под конкретное устройство. В CUDA ядро компилируется
    // nvcc заранее (PTX/CUBIN). В OpenCL — JIT-компиляция в runtime, что
    // позволяет запускать один и тот же исходный код на CPU, GPU, FPGA и т.д.
    err = clBuildProgram(info.program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(info.program, device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(info.program, device, CL_PROGRAM_BUILD_LOG,
                              log_size, &log[0], nullptr);
        std::cerr << "Build failed for " << info.name << ":\n"
                  << log << std::endl;
        std::exit(1);
    }

    info.kernel = clCreateKernel(info.program, "vector_add", &err);
    CL_CHECK(err);

    info.d_a = nullptr;
    info.d_b = nullptr;
    info.d_c = nullptr;

    return info;
}

static void cleanup_device(DeviceInfo& info) {
    if (info.d_a) clReleaseMemObject(info.d_a);
    if (info.d_b) clReleaseMemObject(info.d_b);
    if (info.d_c) clReleaseMemObject(info.d_c);
    clReleaseKernel(info.kernel);
    clReleaseProgram(info.program);
    clReleaseCommandQueue(info.queue);
    clReleaseContext(info.context);
}

int main() {
    const int N = 1 << 24;  // 16M элементов
    const size_t bytes = N * sizeof(float);

    // Инициализация входных данных
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i % 1000) * 0.001f;
        h_b[i] = static_cast<float>((i + 500) % 1000) * 0.002f;
    }

    // --- Шаг 1: перечисляем все устройства ---
    auto all_devices = enumerate_all_devices();
    if (all_devices.empty()) {
        std::cerr << "Не найдено ни одного OpenCL устройства." << std::endl;
        return 1;
    }

    std::cout << "Найдено OpenCL устройств: " << all_devices.size() << std::endl;
    for (size_t i = 0; i < all_devices.size(); ++i) {
        cl_device_type type;
        CL_CHECK(clGetDeviceInfo(all_devices[i].second, CL_DEVICE_TYPE,
                                 sizeof(cl_device_type), &type, nullptr));
        std::cout << "  [" << i << "] "
                  << device_type_str(type) << ": "
                  << get_device_name(all_devices[i].second) << std::endl;
    }
    std::cout << std::endl;

    // --- Шаг 2: настраиваем устройства ---
    size_t num_devices = all_devices.size();
    std::vector<DeviceInfo> devices(num_devices);
    for (size_t i = 0; i < num_devices; ++i) {
        devices[i] = setup_device(all_devices[i].first, all_devices[i].second);
    }

    if (num_devices == 1) {
        // Если доступно только одно устройство, всё равно демонстрируем паттерн.
        // На практике для полноценной демонстрации нужно 2+ устройства
        // (например, GPU + CPU через POCL или Intel OpenCL Runtime).
        std::cout << "ВНИМАНИЕ: найдено только 1 устройство. Для полноценной "
                  << "демонстрации\nмульти-девайсного исполнения нужно 2+ "
                  << "устройства (GPU + CPU)." << std::endl
                  << "Запускаем весь объём на единственном устройстве.\n"
                  << std::endl;
    }

    // --- Шаг 3: разбиваем данные и запускаем на каждом устройстве ---
    // Делим вектор на num_devices частей. Каждое устройство обрабатывает свой
    // непрерывный кусок. В реальном коде стоит учитывать относительную
    // производительность устройств и делить непропорционально (GPU получает
    // больше работы, чем CPU).
    int base_chunk = N / static_cast<int>(num_devices);
    int remainder  = N % static_cast<int>(num_devices);

    std::vector<int> chunk_sizes(num_devices);
    std::vector<int> offsets(num_devices);
    int offset = 0;
    for (size_t i = 0; i < num_devices; ++i) {
        chunk_sizes[i] = base_chunk + (static_cast<int>(i) < remainder ? 1 : 0);
        offsets[i] = offset;
        offset += chunk_sizes[i];
    }

    // Выделяем буферы и отправляем данные на каждое устройство
    std::vector<cl_event> ev_write(num_devices);
    std::vector<cl_event> ev_kernel(num_devices);
    std::vector<cl_event> ev_read(num_devices);

    for (size_t i = 0; i < num_devices; ++i) {
        size_t chunk_bytes = chunk_sizes[i] * sizeof(float);
        cl_int err = 0;

        // Каждое устройство имеет свой контекст → свои буферы.
        // В CUDA для multi-GPU нужен cudaSetDevice() + cudaMalloc на каждом GPU.
        devices[i].d_a = clCreateBuffer(devices[i].context, CL_MEM_READ_ONLY,
                                        chunk_bytes, nullptr, &err);
        CL_CHECK(err);
        devices[i].d_b = clCreateBuffer(devices[i].context, CL_MEM_READ_ONLY,
                                        chunk_bytes, nullptr, &err);
        CL_CHECK(err);
        devices[i].d_c = clCreateBuffer(devices[i].context, CL_MEM_WRITE_ONLY,
                                        chunk_bytes, nullptr, &err);
        CL_CHECK(err);

        // Асинхронная запись данных: каждое устройство получает свой кусок
        CL_CHECK(clEnqueueWriteBuffer(devices[i].queue, devices[i].d_a,
                                      CL_FALSE, 0, chunk_bytes,
                                      h_a.data() + offsets[i],
                                      0, nullptr, &ev_write[i]));
        CL_CHECK(clEnqueueWriteBuffer(devices[i].queue, devices[i].d_b,
                                      CL_FALSE, 0, chunk_bytes,
                                      h_b.data() + offsets[i],
                                      0, nullptr, nullptr));

        // Устанавливаем аргументы ядра
        int chunk_n = chunk_sizes[i];
        CL_CHECK(clSetKernelArg(devices[i].kernel, 0, sizeof(cl_mem),
                                &devices[i].d_a));
        CL_CHECK(clSetKernelArg(devices[i].kernel, 1, sizeof(cl_mem),
                                &devices[i].d_b));
        CL_CHECK(clSetKernelArg(devices[i].kernel, 2, sizeof(cl_mem),
                                &devices[i].d_c));
        CL_CHECK(clSetKernelArg(devices[i].kernel, 3, sizeof(int), &chunk_n));

        // Запускаем ядро на устройстве
        size_t local_size  = 256;
        size_t global_size = ((chunk_sizes[i] + local_size - 1) / local_size)
                             * local_size;
        CL_CHECK(clEnqueueNDRangeKernel(devices[i].queue, devices[i].kernel,
                                        1, nullptr, &global_size, &local_size,
                                        0, nullptr, &ev_kernel[i]));

        // Асинхронное чтение результата обратно в host-массив
        CL_CHECK(clEnqueueReadBuffer(devices[i].queue, devices[i].d_c,
                                     CL_FALSE, 0, chunk_bytes,
                                     h_c.data() + offsets[i],
                                     0, nullptr, &ev_read[i]));
    }

    // --- Шаг 4: ждём завершения на всех устройствах ---
    // В CUDA аналог — cudaDeviceSynchronize() на каждом GPU.
    for (size_t i = 0; i < num_devices; ++i) {
        CL_CHECK(clFinish(devices[i].queue));
    }

    // --- Шаг 5: собираем и выводим тайминги ---
    std::cout << "N=" << N << "  устройств=" << num_devices << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    cl_ulong total_start = ULONG_MAX, total_end = 0;

    for (size_t i = 0; i < num_devices; ++i) {
        cl_ulong t_write_start = 0, t_kernel_start = 0, t_kernel_end = 0,
                 t_read_end = 0;
        CL_CHECK(clGetEventProfilingInfo(ev_write[i],
                     CL_PROFILING_COMMAND_START,
                     sizeof(cl_ulong), &t_write_start, nullptr));
        CL_CHECK(clGetEventProfilingInfo(ev_kernel[i],
                     CL_PROFILING_COMMAND_START,
                     sizeof(cl_ulong), &t_kernel_start, nullptr));
        CL_CHECK(clGetEventProfilingInfo(ev_kernel[i],
                     CL_PROFILING_COMMAND_END,
                     sizeof(cl_ulong), &t_kernel_end, nullptr));
        CL_CHECK(clGetEventProfilingInfo(ev_read[i],
                     CL_PROFILING_COMMAND_END,
                     sizeof(cl_ulong), &t_read_end, nullptr));

        double ms_total  = (t_read_end - t_write_start) * 1e-6;
        double ms_kernel = (t_kernel_end - t_kernel_start) * 1e-6;

        std::cout << "  Устройство " << i << " [" << device_type_str(devices[i].type)
                  << "] " << devices[i].name << std::endl
                  << "    элементов:   " << chunk_sizes[i] << std::endl
                  << "    ядро:        " << std::fixed << std::setprecision(3)
                  << ms_kernel << " ms" << std::endl
                  << "    всего (H2D+kernel+D2H): " << ms_total << " ms"
                  << std::endl;

        if (t_write_start < total_start) total_start = t_write_start;
        if (t_read_end    > total_end)   total_end   = t_read_end;
    }

    // Общее время — от первого события до последнего. Если устройства работают
    // параллельно (разные очереди на разных устройствах), это время будет
    // меньше суммы времён каждого устройства.
    // Замечание: если устройства на разных платформах, их таймеры могут быть
    // несогласованы — в этом случае wall-clock время точнее.
    double ms_wall = (total_end - total_start) * 1e-6;
    std::cout << std::string(60, '-') << std::endl
              << "  Общее время (от первого write до последнего read): "
              << std::fixed << std::setprecision(3) << ms_wall << " ms"
              << std::endl;

    // --- Шаг 6: проверка корректности ---
    bool ok = true;
    for (int i = 0; i < N && ok; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            std::cerr << "Ошибка: c[" << i << "]=" << h_c[i]
                      << ", ожидалось " << expected << std::endl;
            ok = false;
        }
    }
    std::cout << "  check=" << (ok ? "OK" : "FAIL") << std::endl;

    // --- Освобождаем ресурсы ---
    for (size_t i = 0; i < num_devices; ++i) {
        clReleaseEvent(ev_write[i]);
        clReleaseEvent(ev_kernel[i]);
        clReleaseEvent(ev_read[i]);
        cleanup_device(devices[i]);
    }

    return 0;
}
