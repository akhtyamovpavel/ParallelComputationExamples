#include <iostream>
#include <iomanip>
#include <cmath>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// 5-point стенсил с shared-memory tile'ом и halo.
//
// Block размера (TILE+2) x (TILE+2) = 16 x 16. Каждый поток грузит ровно
// один элемент в shared[ty][tx], включая halo (внешнее «кольцо» шириной 1).
// Потом внутренние потоки (tx, ty ∈ [1..TILE]) собирают стенсил по
// 5 ячейкам shared-памяти — быстро и с нулевой redundancy.
#define TILE 14

__global__ void stencil_shared(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[TILE + 2][TILE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Глобальная координата текущего потока; -1 сдвиг = левая/верхняя кайма halo.
    int gx = blockIdx.x * TILE + tx - 1;
    int gy = blockIdx.y * TILE + ty - 1;

    // Clamp к границам (replicate boundary). Это позволяет halo-потокам безопасно
    // грузить что-то осмысленное, даже когда их глобальная координата вне массива.
    int cx = max(0, min(cols - 1, gx));
    int cy = max(0, min(rows - 1, gy));
    tile[ty][tx] = in[cy * cols + cx];
    __syncthreads();

    // Только "внутренние" потоки блока считают и пишут выход, и только для
    // внутренних точек глобального массива (кайма out остаётся неинициализированной).
    if (tx >= 1 && tx <= TILE && ty >= 1 && ty <= TILE) {
        if (gx >= 1 && gx < cols - 1 && gy >= 1 && gy < rows - 1) {
            float sum = tile[ty][tx]
                      + tile[ty - 1][tx] + tile[ty + 1][tx]
                      + tile[ty][tx - 1] + tile[ty][tx + 1];
            out[gy * cols + gx] = 0.2f * sum;
        }
    }
}

int main() {
    const int ROWS = 4096;
    const int COLS = 4096;
    const size_t bytes = static_cast<size_t>(ROWS) * COLS * sizeof(float);

    float* h_in = new float[ROWS * COLS];
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            h_in[r * COLS + c] = 1.0f;
        }
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE + 2, TILE + 2);
    dim3 grid((COLS + TILE - 1) / TILE, (ROWS + TILE - 1) / TILE);

    stencil_shared<<<grid, block>>>(d_in, d_out, ROWS, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    stencil_shared<<<grid, block>>>(d_in, d_out, ROWS, COLS);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    float* h_out = new float[ROWS * COLS];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int r = 1; r < ROWS - 1; ++r) {
        for (int c = 1; c < COLS - 1; ++c) {
            float e = std::fabs(h_out[r * COLS + c] - 1.0f);
            if (e > max_err) {
                max_err = e;
            }
        }
    }

    double gbps = (2.0 * bytes) / (ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);
    std::cout << std::left << std::setw(24) << "stencil_shared"
              << " time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << "  bw=" << std::right << std::setw(7)
              << std::fixed << std::setprecision(2) << gbps << " GB/s"
              << "  max_err=" << std::scientific << std::setprecision(2)
              << max_err << std::endl;

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;
    return 0;
}
