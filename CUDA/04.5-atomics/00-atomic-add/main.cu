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

__global__ void increment_racy(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int v = *counter;
        *counter = v + 1;
    }
}

__global__ void increment_atomic(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(counter, 1);
    }
}

static float run_and_time(void (*kernel)(int*, int), int* d_counter, int n,
                          int grid, int block) {
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    kernel<<<grid, block>>>(d_counter, n);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms;
}

int main() {
    const int N = 1 << 20;
    const int block = 256;
    const int grid = (N + block - 1) / block;

    int* d_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));

    std::cout << std::left << std::setw(24) << "threads:"
              << std::right << std::setw(10) << N << std::endl;

    {
        float ms = run_and_time(increment_racy, d_counter, N, grid, block);
        int result = 0;
        CUDA_CHECK(cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
        std::cout << std::left << std::setw(24) << "racy (no atomic):"
                  << std::right << std::setw(10) << result
                  << "  time=" << std::fixed << std::setprecision(3)
                  << std::setw(8) << ms << " ms"
                  << "  (expected " << N << ")" << std::endl;
    }

    {
        float ms = run_and_time(increment_atomic, d_counter, N, grid, block);
        int result = 0;
        CUDA_CHECK(cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
        std::cout << std::left << std::setw(24) << "atomicAdd:"
                  << std::right << std::setw(10) << result
                  << "  time=" << std::fixed << std::setprecision(3)
                  << std::setw(8) << ms << " ms"
                  << "  (expected " << N << ")" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_counter));
    return 0;
}
