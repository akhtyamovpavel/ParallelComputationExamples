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

// Image objects — особенность OpenCL, не имеющая прямого аналога в CUDA
// (в CUDA есть текстуры, но API совершенно другое).
//
// image2d_t + sampler_t дают:
//  - аппаратную интерполяцию (CLK_FILTER_LINEAR / NEAREST)
//  - автоматическую обработку краёв (CLK_ADDRESS_CLAMP_TO_EDGE / REPEAT / ...)
//  - доступ через нормализованные [0,1] или целочисленные координаты
//
// В этом примере реализуем Box Blur 3x3: каждый пиксель заменяется средним
// по окрестности 3x3. read_imagef + CLK_ADDRESS_CLAMP_TO_EDGE автоматически
// обрабатывает пиксели на границе, не требуя ручных if-ов.
static const char* kernel_source = R"CLC(
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE   |
                          CLK_FILTER_NEAREST;

__kernel void box_blur_3x3(__read_only  image2d_t src,
                           __write_only image2d_t dst) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    float4 sum = (float4)(0.0f);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            sum += read_imagef(src, sampler, (int2)(x + dx, y + dy));
        }
    }
    write_imagef(dst, (int2)(x, y), sum / 9.0f);
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

int main() {
    const int W = 1024;
    const int H = 1024;

    // Генерируем «шахматный» паттерн в RGBA float — легко проверить визуально.
    std::vector<float> h_src(W * H * 4);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float v = ((x / 32 + y / 32) % 2 == 0) ? 1.0f : 0.0f;
            int idx = (y * W + x) * 4;
            h_src[idx + 0] = v;
            h_src[idx + 1] = v;
            h_src[idx + 2] = v;
            h_src[idx + 3] = 1.0f;
        }
    }

    cl_device_id dev = get_gpu_device();
    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev,
                                                  CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    // Проверяем поддержку image. Некоторые OpenCL-устройства (особенно CPU-реализации)
    // могут не поддерживать image2d.
    cl_bool img_support = CL_FALSE;
    CL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_IMAGE_SUPPORT,
                             sizeof(cl_bool), &img_support, nullptr));
    if (!img_support) {
        std::cerr << "Device does not support images." << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return 1;
    }

    cl_image_format fmt;
    fmt.image_channel_order     = CL_RGBA;
    fmt.image_channel_data_type = CL_FLOAT;

    cl_image_desc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width  = W;
    desc.image_height = H;

    cl_mem img_src = clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   &fmt, &desc, h_src.data(), &err);
    CL_CHECK(err);
    cl_mem img_dst = clCreateImage(ctx, CL_MEM_WRITE_ONLY,
                                   &fmt, &desc, nullptr, &err);
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
    cl_kernel kernel = clCreateKernel(prog, "box_blur_3x3", &err);
    CL_CHECK(err);

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_src));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &img_dst));

    size_t global[2] = {static_cast<size_t>(W), static_cast<size_t>(H)};
    size_t local[2]  = {16, 16};

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

    // Считываем результат. Для image используется clEnqueueReadImage,
    // а не clEnqueueReadBuffer.
    std::vector<float> h_dst(W * H * 4);
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {static_cast<size_t>(W), static_cast<size_t>(H), 1};
    CL_CHECK(clEnqueueReadImage(queue, img_dst, CL_TRUE,
                                origin, region, 0, 0,
                                h_dst.data(), 0, nullptr, nullptr));

    // Проверка: центральный пиксель внутри белого квадрата должен остаться ~1.0,
    // а пиксель на границе шахматки — ~0.5..0.7 (усреднение с соседями).
    float center = h_dst[(16 * W + 16) * 4];
    float edge   = h_dst[(32 * W + 32) * 4];
    bool ok = (std::fabs(center - 1.0f) < 0.01f) && (edge > 0.2f && edge < 0.9f);

    std::cout << "box_blur_3x3  " << W << "x" << H
              << "  time=" << std::fixed << std::setprecision(3) << ms << " ms"
              << "  center_pixel=" << std::setprecision(4) << center
              << "  edge_pixel=" << edge
              << "  check=" << (ok ? "OK" : "FAIL") << std::endl;

    clReleaseEvent(event);
    clReleaseMemObject(img_src);
    clReleaseMemObject(img_dst);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
