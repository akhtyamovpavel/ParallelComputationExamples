#include <iostream>
#include <iomanip>
#include <chrono>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

__global__ void saxpy(float* z, const float* x, const float* y, int n,
                      float a, float b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        z[i] = a * x[i] + b * y[i];
    }
}

int main() {
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices < 2) {
        std::cout << "Need at least 2 GPUs; found " << num_devices
                  << ". Skipping." << std::endl;
        return 0;
    }

    const int N = 1 << 24;
    const int half = N / 2;
    const size_t bytes_full = N * sizeof(float);
    const size_t bytes_half = half * sizeof(float);
    const int block = 256;
    const int grid_full = (N + block - 1) / block;
    const int grid_half = (half + block - 1) / block;

    float* h_x = new float[N];
    float* h_y = new float[N];
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    // --- 1. Baseline: всё на GPU 0, один большой launch. ---
    float* d0_x = nullptr;
    float* d0_y = nullptr;
    float* d0_z = nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&d0_x, bytes_full));
    CUDA_CHECK(cudaMalloc(&d0_y, bytes_full));
    CUDA_CHECK(cudaMalloc(&d0_z, bytes_full));
    CUDA_CHECK(cudaMemcpy(d0_x, h_x, bytes_full, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d0_y, h_y, bytes_full, cudaMemcpyHostToDevice));

    // Warm-up.
    saxpy<<<grid_full, block>>>(d0_z, d0_x, d0_y, N, 2.0f, 3.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::steady_clock::now();
    saxpy<<<grid_full, block>>>(d0_z, d0_x, d0_y, N, 2.0f, 3.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    float ms_single = std::chrono::duration<float, std::milli>(t1 - t0).count();

    CUDA_CHECK(cudaFree(d0_x));
    CUDA_CHECK(cudaFree(d0_y));
    CUDA_CHECK(cudaFree(d0_z));

    // --- 2. Split: первая половина на GPU 0, вторая на GPU 1. ---
    // Два device'а работают параллельно: пока cudaSetDevice(1) запускает свой
    // kernel, GPU 0 уже считает. Суммарное wall-clock время должно быть
    // ~ms_single / 2 (минус накладные расходы).
    float* dA_x = nullptr;
    float* dA_y = nullptr;
    float* dA_z = nullptr;
    float* dB_x = nullptr;
    float* dB_y = nullptr;
    float* dB_z = nullptr;

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&dA_x, bytes_half));
    CUDA_CHECK(cudaMalloc(&dA_y, bytes_half));
    CUDA_CHECK(cudaMalloc(&dA_z, bytes_half));
    CUDA_CHECK(cudaMemcpy(dA_x, h_x, bytes_half, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_y, h_y, bytes_half, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&dB_x, bytes_half));
    CUDA_CHECK(cudaMalloc(&dB_y, bytes_half));
    CUDA_CHECK(cudaMalloc(&dB_z, bytes_half));
    CUDA_CHECK(cudaMemcpy(dB_x, h_x + half, bytes_half, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_y, h_y + half, bytes_half, cudaMemcpyHostToDevice));

    // Warm-up обоих.
    CUDA_CHECK(cudaSetDevice(0));
    saxpy<<<grid_half, block>>>(dA_z, dA_x, dA_y, half, 2.0f, 3.0f);
    CUDA_CHECK(cudaSetDevice(1));
    saxpy<<<grid_half, block>>>(dB_z, dB_x, dB_y, half, 2.0f, 3.0f);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t2 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaSetDevice(0));
    saxpy<<<grid_half, block>>>(dA_z, dA_x, dA_y, half, 2.0f, 3.0f);
    CUDA_CHECK(cudaSetDevice(1));
    saxpy<<<grid_half, block>>>(dB_z, dB_x, dB_y, half, 2.0f, 3.0f);
    // Обязательно синхронизировать оба device'а — иначе замер будет неверным.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t3 = std::chrono::steady_clock::now();
    float ms_split = std::chrono::duration<float, std::milli>(t3 - t2).count();

    std::cout << std::left << std::setw(30) << "single GPU  (all on dev 0)"
              << " time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms_single << " ms"
              << std::endl;
    std::cout << std::left << std::setw(30) << "split 2 GPUs (dev 0 + dev 1)"
              << " time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms_split << " ms"
              << "  speedup=" << std::fixed << std::setprecision(2)
              << (ms_single / ms_split) << "x" << std::endl;

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(dA_x));
    CUDA_CHECK(cudaFree(dA_y));
    CUDA_CHECK(cudaFree(dA_z));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(dB_x));
    CUDA_CHECK(cudaFree(dB_y));
    CUDA_CHECK(cudaFree(dB_z));
    delete[] h_x;
    delete[] h_y;
    return 0;
}
