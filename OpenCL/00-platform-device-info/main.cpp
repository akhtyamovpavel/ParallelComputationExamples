#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

// OpenCL 1.2 — базовый целевой уровень, гарантированно есть везде.
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

// clGet*Info возвращает C-строку (с \0 на конце). Запрашиваем в два приёма:
// сначала — размер, потом — саму строку.
static std::string platform_str(cl_platform_id p, cl_platform_info what) {
    size_t size = 0;
    CL_CHECK(clGetPlatformInfo(p, what, 0, nullptr, &size));
    std::string s(size, '\0');
    CL_CHECK(clGetPlatformInfo(p, what, size, &s[0], nullptr));
    if (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    return s;
}

static std::string device_str(cl_device_id d, cl_device_info what) {
    size_t size = 0;
    CL_CHECK(clGetDeviceInfo(d, what, 0, nullptr, &size));
    std::string s(size, '\0');
    CL_CHECK(clGetDeviceInfo(d, what, size, &s[0], nullptr));
    if (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    return s;
}

template <typename T>
static T device_scalar(cl_device_id d, cl_device_info what) {
    T v{};
    CL_CHECK(clGetDeviceInfo(d, what, sizeof(T), &v, nullptr));
    return v;
}

static const char* device_type_name(cl_device_type t) {
    if (t & CL_DEVICE_TYPE_GPU) {
        return "GPU";
    }
    if (t & CL_DEVICE_TYPE_CPU) {
        return "CPU";
    }
    if (t & CL_DEVICE_TYPE_ACCELERATOR) {
        return "ACCEL";
    }
    return "OTHER";
}

int main() {
    // В OpenCL между программой и устройствами стоит "платформа" — реализация
    // OpenCL от конкретного вендора (NVIDIA, AMD, Intel, POCL, ...). На одной
    // машине часто несколько платформ (например, Intel CPU + NVIDIA GPU).
    cl_uint num_platforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    if (num_platforms == 0) {
        std::cout << "No OpenCL platforms found." << std::endl;
        return 0;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    std::cout << "OpenCL platforms: " << num_platforms << std::endl;

    for (cl_uint p = 0; p < num_platforms; ++p) {
        std::cout << std::endl
                  << "[platform " << p << "] "
                  << platform_str(platforms[p], CL_PLATFORM_NAME) << std::endl
                  << "  vendor:  " << platform_str(platforms[p], CL_PLATFORM_VENDOR) << std::endl
                  << "  version: " << platform_str(platforms[p], CL_PLATFORM_VERSION) << std::endl;

        cl_uint num_devices = 0;
        cl_int err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0,
                                    nullptr, &num_devices);
        if (err == CL_DEVICE_NOT_FOUND || num_devices == 0) {
            std::cout << "  (no devices)" << std::endl;
            continue;
        }
        CL_CHECK(err);

        std::vector<cl_device_id> devices(num_devices);
        CL_CHECK(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                                num_devices, devices.data(), nullptr));

        for (cl_uint d = 0; d < num_devices; ++d) {
            cl_device_id dev = devices[d];
            cl_device_type type = device_scalar<cl_device_type>(dev, CL_DEVICE_TYPE);
            cl_uint compute_units = device_scalar<cl_uint>(dev, CL_DEVICE_MAX_COMPUTE_UNITS);
            cl_ulong global_mem = device_scalar<cl_ulong>(dev, CL_DEVICE_GLOBAL_MEM_SIZE);
            cl_ulong local_mem = device_scalar<cl_ulong>(dev, CL_DEVICE_LOCAL_MEM_SIZE);
            cl_ulong const_mem = device_scalar<cl_ulong>(dev, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
            size_t max_wg = device_scalar<size_t>(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE);
            cl_uint max_freq = device_scalar<cl_uint>(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY);

            std::cout << std::endl
                      << "  [device " << d << ", " << device_type_name(type) << "] "
                      << device_str(dev, CL_DEVICE_NAME) << std::endl
                      << "    vendor:          " << device_str(dev, CL_DEVICE_VENDOR) << std::endl
                      << "    driver:          " << device_str(dev, CL_DRIVER_VERSION) << std::endl
                      << "    OpenCL C:        " << device_str(dev, CL_DEVICE_OPENCL_C_VERSION) << std::endl
                      << "    compute units:   " << compute_units << std::endl
                      << "    max work-group:  " << max_wg << std::endl
                      << "    max clock:       " << max_freq << " MHz" << std::endl
                      << "    global mem:      "
                      << std::fixed << std::setprecision(2)
                      << (global_mem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl
                      << "    local mem:       " << (local_mem / 1024) << " KB" << std::endl
                      << "    constant mem:    " << (const_mem / 1024) << " KB" << std::endl;
        }
    }

    return 0;
}
