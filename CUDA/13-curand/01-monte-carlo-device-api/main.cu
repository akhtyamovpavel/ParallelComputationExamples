#include <iostream>
#include <iomanip>
#include <cmath>

#include <curand_kernel.h>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// Device API — каждый поток держит собственный curandState и генерирует
// sample'ы прямо внутри ядра. Отдельного буфера под случайные числа не нужно:
// числа появляются и сразу потребляются. Это удобно, когда sample'ов много,
// а требования к памяти хочется минимизировать.

// Первое ядро — инициализация состояний. curand_init принимает seed +
// subsequence: чтобы у потоков не было корреляции, делаем subsequence = idx.
__global__ void init_states(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Второе ядро — собственно Monte Carlo. Каждый поток делает samples_per_thread
// бросаний, считает локальные попадания в регистрах и один раз atomicAdd'ит
// суммарно в глобальный счётчик.
__global__ void monte_carlo_device(curandState* states, int samples_per_thread,
                                   unsigned long long* out, int n_threads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_threads) {
        return;
    }
    curandState local = states[idx];
    unsigned int hits = 0;
    for (int i = 0; i < samples_per_thread; ++i) {
        float x = curand_uniform(&local);
        float y = curand_uniform(&local);
        if (x * x + y * y <= 1.0f) {
            ++hits;
        }
    }
    states[idx] = local;
    atomicAdd(out, static_cast<unsigned long long>(hits));
}

int main() {
    const int NUM_THREADS = 1 << 16;
    const int SAMPLES_PER_THREAD = 256;
    const long long TOTAL = static_cast<long long>(NUM_THREADS) * SAMPLES_PER_THREAD;
    const double PI_REF = 3.141592653589793;

    curandState* d_states = nullptr;
    unsigned long long* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, NUM_THREADS * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));

    const int block = 256;
    const int grid = (NUM_THREADS + block - 1) / block;

    init_states<<<grid, block>>>(d_states, 1234ULL, NUM_THREADS);
    monte_carlo_device<<<grid, block>>>(d_states, SAMPLES_PER_THREAD,
                                        d_count, NUM_THREADS);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    double pi_est = 4.0 * static_cast<double>(h_count) / TOTAL;
    double err = std::fabs(pi_est - PI_REF);

    std::cout << std::left << std::setw(28) << "Monte Carlo (device API)"
              << " samples=" << TOTAL
              << "  hits=" << h_count
              << "  pi=" << std::fixed << std::setprecision(6) << pi_est
              << "  err=" << std::scientific << std::setprecision(2) << err
              << std::endl;

    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_count));
    return 0;
}
