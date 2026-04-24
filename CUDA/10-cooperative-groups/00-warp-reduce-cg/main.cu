#include <iostream>
#include <iomanip>

#include <cooperative_groups.h>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

namespace cg = cooperative_groups;

// Та же butterfly-редукция, что в 04.7-warp-primitives/01-warp-reduce,
// но через API cooperative groups. Плюсы против ручного __shfl_down_sync:
//  - тип warp'а задан явно через thread_block_tile<32> — значит компилятор
//    ловит ошибки "неправильный width";
//  - не нужно указывать маску активных лейнов: её держит сама группа;
//  - shfl_down — метод группы, а не свободная функция, что даёт
//    композицию (warp внутри блока, под-варпы через tiled_partition<N>).
__device__ int warp_sum_cg(int val) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += warp.shfl_down(val, offset);
    }
    return val;
}

__global__ void reduce_cg(const int* in, int* out_warp_sums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int global_warp_id = blockIdx.x * warps_per_block + (threadIdx.x >> 5);

    int val = 0;
    if (idx < n) {
        val = in[idx];
    }

    int sum = warp_sum_cg(val);

    if (lane == 0) {
        out_warp_sums[global_warp_id] = sum;
    }
}

int main() {
    const int N = 1 << 20;
    const int block = 256;
    const int grid = (N + block - 1) / block;
    const int warps_per_block = block / 32;
    const int total_warps = grid * warps_per_block;

    int* h_in = new int[N];
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1;
    }

    int* d_in = nullptr;
    int* d_warp_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_warp_sums, total_warps * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    reduce_cg<<<grid, block>>>(d_in, d_warp_sums, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    int* h_warp_sums = new int[total_warps];
    CUDA_CHECK(cudaMemcpy(h_warp_sums, d_warp_sums,
                          total_warps * sizeof(int), cudaMemcpyDeviceToHost));

    long long total = 0;
    for (int i = 0; i < total_warps; ++i) {
        total += h_warp_sums[i];
    }

    std::cout << std::left << std::setw(24) << "warp_reduce_cg"
              << " time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << "  total=" << total << "  (expected " << N << ")"
              << std::endl;

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_warp_sums));
    delete[] h_in;
    delete[] h_warp_sums;
    return 0;
}
