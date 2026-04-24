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

// Ядро считает y = 2*x. Запускается на GPU 1, но указатель x указывает
// в память GPU 0 — доступ идёт прямиком через NVLink/PCIe без staging'а
// в host memory. Это работает, только если обе карты разрешили peer access
// друг другу через cudaDeviceEnablePeerAccess.
__global__ void scale_from_peer(const float* x_on_dev0, float* y_on_dev1, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        y_on_dev1[i] = 2.0f * x_on_dev0[i];
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

    // Проверяем, что GPU 1 может читать из GPU 0 (и наоборот).
    // На некоторых конфигурациях (разные PCIe root complex'ы, отключенный
    // NVLink) peer access может быть недоступен — тогда просто выходим.
    int can_0_to_1 = 0;
    int can_1_to_0 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_1_to_0, 1, 0));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_0_to_1, 0, 1));
    if (!can_0_to_1 || !can_1_to_0) {
        std::cout << "Peer access not available between dev 0 and dev 1"
                  << " (can_1->0=" << can_1_to_0
                  << ", can_0->1=" << can_0_to_1 << "). Skipping." << std::endl;
        return 0;
    }

    // Включаем peer access с обеих сторон. Флаг (2-й аргумент) зарезервирован,
    // передаём 0.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));

    const int N = 1 << 22;
    const size_t bytes = N * sizeof(float);
    const int block = 256;
    const int grid = (N + block - 1) / block;

    // Буфер x — на GPU 0.
    CUDA_CHECK(cudaSetDevice(0));
    float* d0_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d0_x, bytes));

    float* h_x = new float[N];
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.5f;
    }
    CUDA_CHECK(cudaMemcpy(d0_x, h_x, bytes, cudaMemcpyHostToDevice));

    // Буфер y — на GPU 1.
    CUDA_CHECK(cudaSetDevice(1));
    float* d1_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d1_y, bytes));

    // Ядро запускается с текущим device'ом = GPU 1, читает указатель
    // d0_x (память GPU 0) напрямую. Никакого cudaMemcpyPeer не потребовалось.
    scale_from_peer<<<grid, block>>>(d0_x, d1_y, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_y = new float[N];
    CUDA_CHECK(cudaMemcpy(h_y, d1_y, bytes, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float e = std::fabs(h_y[i] - 3.0f);
        if (e > max_err) {
            max_err = e;
        }
    }

    std::cout << "Peer-access kernel (dev 1 reads dev 0 buffer)  N=" << N
              << "  max_err=" << std::scientific << std::setprecision(2)
              << max_err << std::endl;

    CUDA_CHECK(cudaFree(d1_y));
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(d0_x));

    // Симметрично выключаем peer access (не строго обязательно —
    // cudaDeviceReset при выходе и так чистит).
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceDisablePeerAccess(1));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaDeviceDisablePeerAccess(0));

    delete[] h_x;
    delete[] h_y;
    return 0;
}
