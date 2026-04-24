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

// Для каждого warp'а:
//  - ballot_mask — 32-битная маска, i-й бит = 1, если лейн i удовлетворил предикат
//  - __popc(mask) — быстрое количество единичных битов (по сути popcount)
//  - __any_sync / __all_sync — "есть хоть один" / "все лейны"
// Лейн 0 пишет результат для своего warp'а в per-warp слот.
__global__ void warp_vote(const int* in, int* counts, int* any_flags,
                          int* all_flags, int n, int threshold) {
    const unsigned FULL = 0xffffffffu;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int global_warp_id = blockIdx.x * warps_per_block + (threadIdx.x >> 5);

    int val = (idx < n) ? in[idx] : 0;
    int predicate = val > threshold;

    unsigned ballot = __ballot_sync(FULL, predicate);
    int any_pred = __any_sync(FULL, predicate);
    int all_pred = __all_sync(FULL, predicate);

    if (lane == 0) {
        counts[global_warp_id] = __popc(ballot);
        any_flags[global_warp_id] = any_pred;
        all_flags[global_warp_id] = all_pred;
    }
}

int main() {
    const int N = 1 << 20;
    const int THRESHOLD = 500;
    const int block = 256;
    const int grid = (N + block - 1) / block;
    const int warps_per_block = block / 32;
    const int total_warps = grid * warps_per_block;

    int* h_in = new int[N];
    int cpu_count = 0;
    for (int i = 0; i < N; ++i) {
        h_in[i] = i % 1000;
        if (h_in[i] > THRESHOLD) {
            ++cpu_count;
        }
    }

    int* d_in = nullptr;
    int* d_counts = nullptr;
    int* d_any = nullptr;
    int* d_all = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, total_warps * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_any, total_warps * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all, total_warps * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    warp_vote<<<grid, block>>>(d_in, d_counts, d_any, d_all, N, THRESHOLD);
    CUDA_CHECK(cudaDeviceSynchronize());

    int* h_counts = new int[total_warps];
    int* h_any = new int[total_warps];
    int* h_all = new int[total_warps];
    CUDA_CHECK(cudaMemcpy(h_counts, d_counts, total_warps * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_any, d_any, total_warps * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_all, d_all, total_warps * sizeof(int),
                          cudaMemcpyDeviceToHost));

    long long gpu_count = 0;
    int any_warps_nonzero = 0;
    int all_warps_nonzero = 0;
    for (int i = 0; i < total_warps; ++i) {
        gpu_count += h_counts[i];
        if (h_any[i]) {
            ++any_warps_nonzero;
        }
        if (h_all[i]) {
            ++all_warps_nonzero;
        }
    }

    std::cout << "threshold:            " << THRESHOLD << std::endl;
    std::cout << "GPU count (ballot):   " << gpu_count
              << "  (CPU expected: " << cpu_count << ")" << std::endl;
    std::cout << "warps with any>thr:   " << any_warps_nonzero
              << " / " << total_warps << std::endl;
    std::cout << "warps with all>thr:   " << all_warps_nonzero
              << " / " << total_warps << std::endl;

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_any));
    CUDA_CHECK(cudaFree(d_all));
    delete[] h_in;
    delete[] h_counts;
    delete[] h_any;
    delete[] h_all;
    return 0;
}
