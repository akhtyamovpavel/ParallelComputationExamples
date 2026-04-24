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

// Требует Volta+ (sm_70 и выше). На Pascal (sm_60/61) не скомпилируется
// или даст "invalid device function" при запуске. В CMake под этот target
// архитектуры переопределены на 70;75;80;86;89.
using namespace nvcuda;

// Минимальная демонстрация WMMA: один warp (32 треда) выполняет
//   C[16x16] = A[16x16] * B[16x16]
// где A, B хранятся в FP16, а аккумулятор C — в FP32 (классическая
// "mixed precision"-связка, которую используют тензорные ядра).
//
// Все четыре wmma-вызова — warp-коллективные: данные внутри fragment
// распределены по регистрам 32 лейнов warp'а, но пользовательский
// код не видит этого распределения.
__global__ void wmma_16x16(const __half* A, const __half* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;

    wmma::fill_fragment(frag_c, 0.0f);
    wmma::load_matrix_sync(frag_a, A, 16);
    wmma::load_matrix_sync(frag_b, B, 16);
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    wmma::store_matrix_sync(C, frag_c, 16, wmma::mem_row_major);
}

int main() {
    const int M = 16;
    const int N = 16;
    const int K = 16;

    // Простейшие значения для проверки: A — все 1.0, B — все 2.0.
    // Тогда C[i][j] = sum_k 1 * 2 = 2 * K = 32.
    __half h_A[M * K];
    __half h_B[K * N];
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = __float2half(1.0f);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = __float2half(2.0f);
    }

    __half* d_A = nullptr;
    __half* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(h_A)));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(h_B)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice));

    wmma_16x16<<<1, 32>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_C[M * N];
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost));

    float expected = 2.0f * K;
    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float e = std::fabs(h_C[i] - expected);
        if (e > max_err) {
            max_err = e;
        }
    }

    std::cout << "WMMA 16x16 x 16x16 -> 16x16 (FP16 x FP16 -> FP32)"
              << std::endl
              << "  C[0,0]=" << std::fixed << std::setprecision(2) << h_C[0]
              << "  expected=" << expected
              << "  max_err=" << std::scientific << std::setprecision(2)
              << max_err << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
