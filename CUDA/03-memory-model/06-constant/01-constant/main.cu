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

#define MASK_W 31
#define MASK_R (MASK_W / 2)

// __constant__ — переменная в отдельном сегменте на устройстве.
//  - Размер всей constant memory ограничен 64 KB.
//  - Host заливает туда значение через cudaMemcpyToSymbol.
//  - Device-код читает как обычную глобальную переменную, но чтение идёт
//    через отдельный constant cache, оптимизированный под broadcast:
//    если все треды warp'а запрашивают один и тот же адрес,
//    выдача такая же быстрая, как из регистра.
__constant__ float c_mask[MASK_W];

__global__ void conv1d_constant(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < MASK_W; ++k) {
        int src = idx + k - MASK_R;
        if (src >= 0 && src < n) {
            sum += in[src] * c_mask[k];
        }
    }
    out[idx] = sum;
}

int main() {
    const int N = 1 << 22;
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);
    const int block = 256;
    const int grid = (N + block - 1) / block;

    float* h_in = new float[N];
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;
    }

    float h_mask[MASK_W];
    float mask_sum = 0.0f;
    for (int k = 0; k < MASK_W; ++k) {
        h_mask[k] = 1.0f / MASK_W;
        mask_sum += h_mask[k];
    }

    // Копирование в constant memory — не cudaMemcpy, а cudaMemcpyToSymbol
    // по имени __constant__-символа.
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask, h_mask, MASK_W * sizeof(float)));

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    conv1d_constant<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    conv1d_constant<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    float* h_out = new float[N];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = MASK_R; i < N - MASK_R; ++i) {
        float e = std::fabs(h_out[i] - mask_sum);
        if (e > max_err) {
            max_err = e;
        }
    }

    std::cout << std::left << std::setw(28) << "conv1d (mask in constant)"
              << " time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms << " ms"
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
