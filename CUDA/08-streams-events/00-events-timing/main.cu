#include <iostream>
#include <iomanip>
#include <chrono>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

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
    std::cout << std::left << std::setw(40) << name
              << " time=" << std::right << std::setw(9)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << std::endl;
}

int main() {
    const int N = 1 << 22;
    const int ITERS = 200;
    const int block = 256;
    const int grid = (N + block - 1) / block;

    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    // Warm-up — загрузить JIT и кеши, чтобы первые замеры не врали.
    busy_kernel<<<grid, block>>>(d_data, N, ITERS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 1) std::chrono БЕЗ cudaDeviceSynchronize — запуск ядра асинхронный,
    //    поэтому чаще всего мы измерим только overhead lauch API.
    {
        auto t0 = std::chrono::steady_clock::now();
        busy_kernel<<<grid, block>>>(d_data, N, ITERS);
        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        print_row("std::chrono (no sync, WRONG)", ms);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 2) std::chrono С cudaDeviceSynchronize — корректно, но включает overhead.
    {
        auto t0 = std::chrono::steady_clock::now();
        busy_kernel<<<grid, block>>>(d_data, N, ITERS);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        print_row("std::chrono + cudaDeviceSynchronize", ms);
    }

    // 3) cudaEvent — точное GPU-время, без host-overhead.
    {
        cudaEvent_t t0;
        cudaEvent_t t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));

        CUDA_CHECK(cudaEventRecord(t0));
        busy_kernel<<<grid, block>>>(d_data, N, ITERS);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        print_row("cudaEvent (GPU time)", ms);

        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));
    }

    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
