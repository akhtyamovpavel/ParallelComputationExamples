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

// Один блок из 32 тредов = ровно один warp. Каждый лейн забирает значения
// у соседей четырьмя способами и пишет их в строку общей таблицы.
__global__ void shuffle_demo(int* out) {
    const unsigned FULL = 0xffffffffu;

    int lane = threadIdx.x;
    int val = lane * 10;

    // Broadcast: все лейны получают значение лейна 0.
    int bcast = __shfl_sync(FULL, val, 0);

    // Down-shift: лейн L читает значение лейна L+1 (лейн 31 сохраняет своё).
    int down = __shfl_down_sync(FULL, val, 1);

    // XOR-butterfly: лейн L меняется парами с лейном L^1 (0<->1, 2<->3, ...).
    int xorpair = __shfl_xor_sync(FULL, val, 1);

    out[lane * 4 + 0] = val;
    out[lane * 4 + 1] = bcast;
    out[lane * 4 + 2] = down;
    out[lane * 4 + 3] = xorpair;
}

int main() {
    const int WARP = 32;
    const int COLS = 4;

    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, WARP * COLS * sizeof(int)));

    shuffle_demo<<<1, WARP>>>(d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_out[WARP * COLS];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, WARP * COLS * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::cout << std::left << std::setw(6) << "lane"
              << std::right << std::setw(8) << "val"
              << std::setw(12) << "bcast(0)"
              << std::setw(12) << "shfl_down"
              << std::setw(12) << "shfl_xor1"
              << std::endl;
    for (int lane = 0; lane < WARP; ++lane) {
        std::cout << std::left << std::setw(6) << lane
                  << std::right << std::setw(8) << h_out[lane * COLS + 0]
                  << std::setw(12) << h_out[lane * COLS + 1]
                  << std::setw(12) << h_out[lane * COLS + 2]
                  << std::setw(12) << h_out[lane * COLS + 3]
                  << std::endl;
    }

    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
