#include <iostream>
#include <iomanip>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

__global__ void vec_add(const float* x, const float* y, float* z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        z[i] = x[i] + y[i];
    }
}

static float time_kernel(const float* x, const float* y, float* z, int n,
                         int grid, int block) {
    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    vec_add<<<grid, block>>>(x, y, z, n);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms;
}

static void print_row(const char* name, float ms) {
    std::cout << std::left << std::setw(34) << name
              << " time=" << std::right << std::setw(9)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << std::endl;
}

int main() {
    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);
    const int block = 256;
    const int grid = (N + block - 1) / block;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    // Ветка A — managed БЕЗ prefetch.
    // После инициализации на host все страницы принадлежат host. Первое
    // обращение ядра приводит к page fault'ам и ленивой миграции на device.
    {
        float* x = nullptr;
        float* y = nullptr;
        float* z = nullptr;
        CUDA_CHECK(cudaMallocManaged(&x, bytes));
        CUDA_CHECK(cudaMallocManaged(&y, bytes));
        CUDA_CHECK(cudaMallocManaged(&z, bytes));

        for (int i = 0; i < N; ++i) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }

        float ms = time_kernel(x, y, z, N, grid, block);
        print_row("managed, no prefetch (faulting)", ms);

        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
        CUDA_CHECK(cudaFree(z));
    }

    // Ветка B — managed С prefetch.
    // cudaMemPrefetchAsync заранее перегоняет страницы на device, поэтому
    // ядро не платит за page fault'ы на первом обращении.
    {
        float* x = nullptr;
        float* y = nullptr;
        float* z = nullptr;
        CUDA_CHECK(cudaMallocManaged(&x, bytes));
        CUDA_CHECK(cudaMallocManaged(&y, bytes));
        CUDA_CHECK(cudaMallocManaged(&z, bytes));

        for (int i = 0; i < N; ++i) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }

        CUDA_CHECK(cudaMemPrefetchAsync(x, bytes, device));
        CUDA_CHECK(cudaMemPrefetchAsync(y, bytes, device));
        CUDA_CHECK(cudaMemPrefetchAsync(z, bytes, device));
        CUDA_CHECK(cudaDeviceSynchronize());

        float ms = time_kernel(x, y, z, N, grid, block);
        print_row("managed, with prefetch", ms);

        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
        CUDA_CHECK(cudaFree(z));
    }

    return 0;
}
