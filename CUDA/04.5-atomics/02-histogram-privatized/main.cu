#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

__global__ void histogram_privatized(const int* data, int n, int* bins, int num_bins) {
    extern __shared__ int s_bins[];

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        s_bins[i] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        int bin = data[i] % num_bins;
        atomicAdd(&s_bins[bin], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&bins[i], s_bins[i]);
    }
}

int main() {
    const int N = 1 << 24;
    const int NUM_BINS = 64;
    const int block = 256;
    const int grid = 1024;

    std::vector<int> h_data(N);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, NUM_BINS - 1);
    for (int i = 0; i < N; ++i) {
        h_data[i] = dist(rng);
    }

    int* d_data = nullptr;
    int* d_bins = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bins, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(int)));

    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    size_t shmem = NUM_BINS * sizeof(int);

    CUDA_CHECK(cudaEventRecord(t0));
    histogram_privatized<<<grid, block, shmem>>>(d_data, N, d_bins, NUM_BINS);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    std::vector<int> h_bins(NUM_BINS);
    CUDA_CHECK(cudaMemcpy(h_bins.data(), d_bins, NUM_BINS * sizeof(int),
                          cudaMemcpyDeviceToHost));

    long long total = 0;
    for (int b : h_bins) {
        total += b;
    }

    double bytes = static_cast<double>(N) * sizeof(int);
    std::cout << std::left << std::setw(24) << "histogram_privatized"
              << " time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << "  bw=" << std::right << std::setw(7)
              << std::fixed << std::setprecision(2)
              << bytes / (ms * 1e-3) / (1024.0 * 1024.0 * 1024.0) << " GB/s"
              << "  total=" << total << " (expected " << N << ")" << std::endl;

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_bins));
    return 0;
}
