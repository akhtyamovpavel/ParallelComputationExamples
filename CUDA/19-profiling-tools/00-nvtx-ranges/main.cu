#include <iostream>
#include <iomanip>
#include <nvToolsExt.h>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// Демонстрация NVTX (NVIDIA Tools Extension) — пользовательских аннотаций
// для визуального профилировщика Nsight Systems.
//
// NVTX позволяет «подписать» участки кода: nvtxRangePush / nvtxRangePop
// создают именованные диапазоны, которые видны на timeline в Nsight Systems.
// Это гораздо информативнее, чем искать ядра по имени в профиле.
//
// Как профилировать:
//   nsys profile --trace=cuda,nvtx -o report ./main
//   nsys-ui report.nsys-rep        # или открыть в Nsight Systems GUI
//
// Или через Nsight Compute (метрики ядер):
//   ncu --set full ./main
//
// Старый nvprof (для CUDA <= 11):
//   nvprof --print-gpu-trace ./main
//
// Ссылки:
//   https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof
//   https://developer.nvidia.com/nsight-systems
//   https://developer.nvidia.com/nsight-compute

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void saxpy(float* y, const float* x, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

__global__ void busy_kernel(float* data, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float x = data[i];
        for (int k = 0; k < iters; ++k) {
            x = x * 1.0001f + 0.5f;
        }
        data[i] = x;
    }
}

static void print_row(const char* name, float ms) {
    std::cout << std::left << std::setw(30) << name
              << " time=" << std::right << std::setw(9)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << std::endl;
}

int main() {
    const int N = 1 << 22;
    const size_t bytes = N * sizeof(float);
    const int block = 256;
    const int grid = (N + block - 1) / block;

    // --- Выделение памяти (помечаем NVTX-диапазоном) ---
    nvtxRangePush("Allocate memory");

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    float* h_a = new float[N];
    float* h_b = new float[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    nvtxRangePop();

    // --- H2D копирование ---
    nvtxRangePush("Host-to-Device copy");
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    nvtxRangePop();

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // --- Ядро 1: vector_add ---
    nvtxRangePush("vector_add kernel");
    CUDA_CHECK(cudaEventRecord(t0));
    vector_add<<<grid, block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    print_row("vector_add", ms);
    nvtxRangePop();

    // --- Ядро 2: saxpy ---
    nvtxRangePush("saxpy kernel");
    CUDA_CHECK(cudaEventRecord(t0));
    saxpy<<<grid, block>>>(d_c, d_a, 2.0f, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    print_row("saxpy", ms);
    nvtxRangePop();

    // --- Ядро 3: busy_kernel (compute-bound) ---
    nvtxRangePush("busy_kernel (compute-bound)");
    CUDA_CHECK(cudaEventRecord(t0));
    busy_kernel<<<grid, block>>>(d_c, N, 500);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    print_row("busy_kernel (500 iters)", ms);
    nvtxRangePop();

    // --- D2H копирование ---
    nvtxRangePush("Device-to-Host copy");
    float* h_c = new float[N];
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    std::cout << "\nResult sample: c[0]=" << h_c[0]
              << "  c[N-1]=" << h_c[N - 1] << std::endl;
    std::cout << "\nNVTX ranges recorded. Profile with:\n"
              << "  nsys profile --trace=cuda,nvtx -o report ./main\n"
              << "  ncu --set full ./main\n"
              << "  nvprof --print-gpu-trace ./main  (CUDA <= 11)\n"
              << std::endl;

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return 0;
}
