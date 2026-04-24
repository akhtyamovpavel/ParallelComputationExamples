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

// 5-point 2D-стенсил без shared memory.
// out[r,c] = 0.2 * (in[r,c] + in[r-1,c] + in[r+1,c] + in[r,c-1] + in[r,c+1])
// Каждое внутреннее значение массива читается из global memory 5 раз
// (как центр собственного шаблона и как сосед 4-х окружающих) — это и есть
// та избыточность, которую в 01-shared-halo закроет shared tile с halo.
__global__ void stencil_global(const float* in, float* out, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r <= 0 || r >= rows - 1 || c <= 0 || c >= cols - 1) {
        return;
    }
    float center = in[r * cols + c];
    float north  = in[(r - 1) * cols + c];
    float south  = in[(r + 1) * cols + c];
    float west   = in[r * cols + (c - 1)];
    float east   = in[r * cols + (c + 1)];
    out[r * cols + c] = 0.2f * (center + north + south + west + east);
}

int main() {
    const int ROWS = 4096;
    const int COLS = 4096;
    const size_t bytes = static_cast<size_t>(ROWS) * COLS * sizeof(float);
    const int block_x = 32;
    const int block_y = 8;

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

    dim3 block(block_x, block_y);
    dim3 grid((COLS + block_x - 1) / block_x, (ROWS + block_y - 1) / block_y);

    stencil_global<<<grid, block>>>(d_in, d_out, ROWS, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    stencil_global<<<grid, block>>>(d_in, d_out, ROWS, COLS);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    float* h_out = new float[ROWS * COLS];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // На входе все единицы, значит во внутренних точках выход должен быть
    // 0.2 * (1+1+1+1+1) = 1.0.
    float max_err = 0.0f;
    for (int r = 1; r < ROWS - 1; ++r) {
        for (int c = 1; c < COLS - 1; ++c) {
            float e = std::fabs(h_out[r * COLS + c] - 1.0f);
            if (e > max_err) {
                max_err = e;
            }
        }
    }

    // Учитываем обе стороны — чтение + запись (byte counting).
    double gbps = (2.0 * bytes) / (ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);
    std::cout << std::left << std::setw(24) << "stencil_global"
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
