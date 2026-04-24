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

__global__ void vec_add(const float* x, const float* y, float* z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        z[i] = x[i] + y[i];
    }
}

static void print_row(const char* name, float ms) {
    std::cout << std::left << std::setw(34) << name
              << " time=" << std::right << std::setw(9)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << std::endl;
}

int main() {
    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);
    const int block = 256;
    const int grid = (N + block - 1) / block;

    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // Вариант 1 — классика: раздельные host/device буферы и явные cudaMemcpy.
    {
        float* h_x = new float[N];
        float* h_y = new float[N];
        float* h_z = new float[N];
        for (int i = 0; i < N; ++i) {
            h_x[i] = 1.0f;
            h_y[i] = 2.0f;
        }

        float* d_x = nullptr;
        float* d_y = nullptr;
        float* d_z = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, bytes));
        CUDA_CHECK(cudaMalloc(&d_y, bytes));
        CUDA_CHECK(cudaMalloc(&d_z, bytes));

        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
        vec_add<<<grid, block>>>(d_x, d_y, d_z, N);
        CUDA_CHECK(cudaMemcpy(h_z, d_z, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        print_row("explicit (cudaMalloc+cudaMemcpy)", ms);

        // Проверка корректности.
        float max_err = 0.0f;
        for (int i = 0; i < N; ++i) {
            float e = std::fabs(h_z[i] - 3.0f);
            if (e > max_err) {
                max_err = e;
            }
        }
        std::cout << "  max error: " << max_err << std::endl;

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_z));
        delete[] h_x;
        delete[] h_y;
        delete[] h_z;
    }

    // Вариант 2 — unified memory: один указатель, видимый и host, и device.
    // Миграция страниц между CPU и GPU делается драйвером по требованию.
    {
        float* x = nullptr;
        float* y = nullptr;
        float* z = nullptr;
        CUDA_CHECK(cudaMallocManaged(&x, bytes));
        CUDA_CHECK(cudaMallocManaged(&y, bytes));
        CUDA_CHECK(cudaMallocManaged(&z, bytes));

        for (int i = 0; i < N; ++i) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }

        CUDA_CHECK(cudaEventRecord(t0));
        vec_add<<<grid, block>>>(x, y, z, N);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        print_row("managed (cudaMallocManaged)", ms);

        // Доступ с host сразу после ядра — драйвер мигрирует страницы обратно.
        float max_err = 0.0f;
        for (int i = 0; i < N; ++i) {
            float e = std::fabs(z[i] - 3.0f);
            if (e > max_err) {
                max_err = e;
            }
        }
        std::cout << "  max error: " << max_err << std::endl;

        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
        CUDA_CHECK(cudaFree(z));
    }

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return 0;
}
