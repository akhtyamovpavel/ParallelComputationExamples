#include <iostream>
#include <iomanip>
#include <cmath>

#include <cuda_fp16.h>
#include <mma.h>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// Требует sm_70+. CMake под этот target переопределяет архитектуры.
using namespace nvcuda;

// Полный matmul N x N через 16x16 WMMA-плитки.
// Структура: один блок = один warp (32 треда), один warp = одна 16x16 тайла C.
// Grid размером (N/16, N/16) покрывает всю матрицу C.
//
// Внутренний цикл проходит по K-оси шагами по 16 элементов: на каждой
// итерации грузится новая A-плитка и новая B-плитка, и mma_sync прибавляет
// их произведение к аккумулятору frag_c.
__global__ void wmma_matmul(const __half* A, const __half* B, float* C, int N) {
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int num_tiles_k = N / 16;
    for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
        const __half* a_ptr = A + (tile_row * 16) * N + (k_tile * 16);
        const __half* b_ptr = B + (k_tile * 16) * N + (tile_col * 16);
        wmma::load_matrix_sync(a, a_ptr, N);
        wmma::load_matrix_sync(b, b_ptr, N);
        wmma::mma_sync(acc, a, b, acc);
    }

    float* c_ptr = C + (tile_row * 16) * N + (tile_col * 16);
    wmma::store_matrix_sync(c_ptr, acc, N, wmma::mem_row_major);
}

int main() {
    const int N = 1024;
    const size_t bytes_half = static_cast<size_t>(N) * N * sizeof(__half);
    const size_t bytes_float = static_cast<size_t>(N) * N * sizeof(float);

    // A - все 1.0, B - все 2.0 => C[i][j] = 2 * N.
    __half* h_A = new __half[N * N];
    __half* h_B = new __half[N * N];
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = __float2half(1.0f);
        h_B[i] = __float2half(2.0f);
    }

    __half* d_A = nullptr;
    __half* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_half));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_half));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_float));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_half, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_half, cudaMemcpyHostToDevice));

    dim3 grid(N / 16, N / 16);
    dim3 block(32);

    // Warm-up.
    wmma_matmul<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    wmma_matmul<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    float* h_C = new float[N * N];
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_float, cudaMemcpyDeviceToHost));

    float expected = 2.0f * N;
    float max_err = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float e = std::fabs(h_C[i] - expected);
        if (e > max_err) {
            max_err = e;
        }
    }

    // 2 * N^3 FMA-операций, каждая = 2 FLOP (mul + add).
    double tflops = (2.0 * N * N * N) / (ms * 1e-3) / 1e12;
    std::cout << std::left << std::setw(24) << "WMMA matmul 1024"
              << " time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << "  " << std::right << std::setw(7)
              << std::fixed << std::setprecision(2) << tflops << " TFLOPS"
              << "  max_err=" << std::scientific << std::setprecision(2)
              << max_err << std::endl;

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    return 0;
}
