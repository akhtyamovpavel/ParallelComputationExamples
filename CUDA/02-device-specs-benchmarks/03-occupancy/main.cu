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

// Простое compute-bound ядро: внутри цикла чередуются FMA, так что время
// зависит не только от пропускной способности памяти, а и от того, сколько
// warp'ов удаётся держать резидентными на SM (занятость = occupancy).
__global__ void busy_kernel(float* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float x = 1.0f;
        for (int k = 0; k < iters; ++k) {
            x = x * 1.0001f + 0.5f;
        }
        out[i] = x;
    }
}

static float time_kernel(float* d_out, int n, int iters, int block, int grid) {
    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // Warm-up.
    busy_kernel<<<grid, block>>>(d_out, n, iters);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(t0));
    busy_kernel<<<grid, block>>>(d_out, n, iters);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms;
}

int main() {
    const int N = 1 << 22;
    const int ITERS = 200;

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // 1) Рекомендация CUDA runtime: оптимальный размер блока и минимальный
    //    размер grid'а, при которых SM заполняются с максимальной occupancy.
    int min_grid = 0;
    int best_block = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &best_block,
                                                  busy_kernel, 0, 0));

    // 2) Для того же best_block считаем активные блоки на SM и переводим
    //    в теоретический occupancy ratio.
    int max_active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, busy_kernel, best_block, 0));

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    double theoretical_occ = static_cast<double>(max_active_blocks * best_block)
                             / props.maxThreadsPerMultiProcessor;

    std::cout << "device: " << props.name
              << "  (SMs=" << props.multiProcessorCount
              << ", max threads/SM=" << props.maxThreadsPerMultiProcessor << ")"
              << std::endl;
    std::cout << "cudaOccupancyMaxPotentialBlockSize → best_block="
              << best_block << "  (min_grid=" << min_grid << ")" << std::endl;
    std::cout << "theoretical occupancy at best_block: "
              << std::fixed << std::setprecision(2)
              << (theoretical_occ * 100.0) << "%" << std::endl
              << std::endl;

    // 3) Ручной свип по размерам блока — смотрим, как реальное время
    //    коррелирует с рекомендацией API.
    const int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    std::cout << std::left << std::setw(12) << "block"
              << std::right << std::setw(10) << "grid"
              << std::setw(12) << "time (ms)" << std::endl;
    for (int bs : block_sizes) {
        int grid = (N + bs - 1) / bs;
        float ms = time_kernel(d_out, N, ITERS, bs, grid);
        std::cout << std::left << std::setw(12) << bs
                  << std::right << std::setw(10) << grid
                  << std::setw(12) << std::fixed << std::setprecision(3) << ms
                  << std::endl;
    }

    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
