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

#define TILE 32

// Tiled транспоз с padding +1 по последней размерности shared-tile.
//
// В 01-tiled строка занимала ровно 32 float = 32 банка shared-памяти.
// При чтении tile[threadIdx.x][threadIdx.y] все 32 лейна warp'а попадали
// в один банк — 32-way conflict.
//
// Здесь строка занимает 33 float. Тогда позиция элемента
// tile[i][j] = i * 33 + j в shared, и банк = (i * 33 + j) % 32.
// На фиксированном j и i = 0..31 банки все разные — конфликт уходит.
__global__ void transpose_padded(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = in[row * cols + col];
    }
    __syncthreads();

    int out_col = blockIdx.y * TILE + threadIdx.x;
    int out_row = blockIdx.x * TILE + threadIdx.y;
    if (out_row < cols && out_col < rows) {
        out[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    const int ROWS = 4096;
    const int COLS = 4096;
    const size_t bytes = static_cast<size_t>(ROWS) * COLS * sizeof(float);

    float* h_in = new float[ROWS * COLS];
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            h_in[r * COLS + c] = static_cast<float>(r * COLS + c);
        }
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((COLS + TILE - 1) / TILE, (ROWS + TILE - 1) / TILE);

    transpose_padded<<<grid, block>>>(d_in, d_out, ROWS, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    transpose_padded<<<grid, block>>>(d_in, d_out, ROWS, COLS);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    float* h_out = new float[ROWS * COLS];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int r = 0; r < 32 && ok; ++r) {
        for (int c = 0; c < 32 && ok; ++c) {
            if (h_out[c * ROWS + r] != h_in[r * COLS + c]) {
                ok = false;
            }
        }
    }

    double gbps = (2.0 * bytes) / (ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);
    std::cout << std::left << std::setw(20) << "transpose_padded"
              << " time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << "  bw=" << std::right << std::setw(7)
              << std::fixed << std::setprecision(2) << gbps << " GB/s"
              << "  check=" << (ok ? "OK" : "FAIL") << std::endl;

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;
    return 0;
}
