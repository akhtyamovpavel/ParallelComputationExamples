#include <iostream>
#include <iomanip>
#include <cmath>

#include <curand.h>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

#define CURAND_CHECK(call) do {                                               \
    curandStatus_t st = (call);                                               \
    if (st != CURAND_STATUS_SUCCESS) {                                        \
        std::cerr << "cuRAND error " << st << " at "                          \
                  << __FILE__ << ":" << __LINE__ << std::endl;                \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// Считает, сколько (x, y) попали в единичную четверть круга (x^2 + y^2 <= 1).
// Каждый поток суммирует попадания в регистре и делает ровно одну atomicAdd
// на всю свою работу — глобальный counter не превращается в bottleneck.
__global__ void count_hits(const float* x, const float* y, int n,
                           unsigned long long* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned int hits = 0;
    for (int i = idx; i < n; i += stride) {
        float a = x[i];
        float b = y[i];
        if (a * a + b * b <= 1.0f) {
            ++hits;
        }
    }
    atomicAdd(out, static_cast<unsigned long long>(hits));
}

int main() {
    const int N = 1 << 24;
    const double PI_REF = 3.141592653589793;

    float* d_x = nullptr;
    float* d_y = nullptr;
    unsigned long long* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));

    // Host API — генератор создаётся и управляется с host'а. curandGenerateUniform
    // одним вызовом заполняет весь device-буфер равномерным U[0,1).
    // Подходит, когда все sample'ы нужны разом.
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND_CHECK(curandGenerateUniform(gen, d_x, N));
    CURAND_CHECK(curandGenerateUniform(gen, d_y, N));

    const int block = 256;
    const int grid = 1024;
    count_hits<<<grid, block>>>(d_x, d_y, N, d_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    double pi_est = 4.0 * static_cast<double>(h_count) / N;
    double err = std::fabs(pi_est - PI_REF);

    std::cout << std::left << std::setw(28) << "Monte Carlo (host API)"
              << " N=" << N
              << "  hits=" << h_count
              << "  pi=" << std::fixed << std::setprecision(6) << pi_est
              << "  err=" << std::scientific << std::setprecision(2) << err
              << std::endl;

    CURAND_CHECK(curandDestroyGenerator(gen));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_count));
    return 0;
}
