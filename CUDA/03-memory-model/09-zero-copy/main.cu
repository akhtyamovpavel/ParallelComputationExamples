#include <iostream>
#include <iomanip>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// Zero-copy (mapped) memory — память, выделенная на хосте и отображённая
// в адресное пространство GPU через PCIe/NVLink.
//
// cudaHostAlloc + cudaHostAllocMapped: выделяет pinned-память на хосте
// и делает её доступной для GPU без явного cudaMemcpy.
//
// cudaHostGetDevicePointer: получаем device-указатель на ту же физическую
// память (zero-copy — нет копирования, GPU читает напрямую через PCIe).
//
// Когда использовать:
//   + Однократный доступ к данным (streaming) — экономим на cudaMemcpy
//   + Данные не помещаются в GPU memory
//   + CPU и GPU работают с данными поочередно
//
// Когда НЕ использовать:
//   - Многократный доступ к одним данным (каждое чтение через PCIe)
//   - Latency-critical код (PCIe latency >> DRAM latency)
//
// Сравните с cudaMallocHost (03-host-alloc-benchmarks) — там pinned-память
// ускоряет КОПИРОВАНИЕ, здесь мы копирование убираем совсем.

__global__ void saxpy(float* y, const float* x, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

int main() {
    const int N = 1 << 22;
    const size_t bytes = N * sizeof(float);
    const int block = 256;
    const int grid = (N + block - 1) / block;

    // Проверяем, поддерживает ли устройство mapped memory
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (!prop.canMapHostMemory) {
        std::cerr << "Device does not support mapped host memory" << std::endl;
        return 1;
    }
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "N=" << N << "  bytes=" << (bytes >> 20) << " MB" << std::endl;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // =====================================================
    // Способ 1: классический — cudaMalloc + cudaMemcpy
    // =====================================================
    {
        float* h_x = new float[N];
        float* h_y = new float[N];
        for (int i = 0; i < N; ++i) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

        float *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, bytes));
        CUDA_CHECK(cudaMalloc(&d_y, bytes));

        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
        saxpy<<<grid, block>>>(d_y, d_x, 2.0f, N);
        CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        std::cout << std::left << std::setw(40) << "cudaMalloc + cudaMemcpy (total)"
                  << " time=" << std::right << std::setw(9)
                  << std::fixed << std::setprecision(3) << ms << " ms"
                  << "  y[0]=" << h_y[0] << std::endl;

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        delete[] h_x;
        delete[] h_y;
    }

    // =====================================================
    // Способ 2: zero-copy — cudaHostAlloc + cudaHostAllocMapped
    // =====================================================
    {
        float *h_x, *h_y;
        CUDA_CHECK(cudaHostAlloc(&h_x, bytes,
                                 cudaHostAllocMapped | cudaHostAllocWriteCombined));
        CUDA_CHECK(cudaHostAlloc(&h_y, bytes, cudaHostAllocMapped));
        for (int i = 0; i < N; ++i) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

        float *d_x, *d_y;
        CUDA_CHECK(cudaHostGetDevicePointer(&d_x, h_x, 0));
        CUDA_CHECK(cudaHostGetDevicePointer(&d_y, h_y, 0));

        CUDA_CHECK(cudaEventRecord(t0));
        saxpy<<<grid, block>>>(d_y, d_x, 2.0f, N);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        std::cout << std::left << std::setw(40) << "zero-copy (mapped, kernel only)"
                  << " time=" << std::right << std::setw(9)
                  << std::fixed << std::setprecision(3) << ms << " ms"
                  << "  y[0]=" << h_y[0] << std::endl;

        CUDA_CHECK(cudaFreeHost(h_x));
        CUDA_CHECK(cudaFreeHost(h_y));
    }

    // =====================================================
    // Способ 3: unified memory — cudaMallocManaged
    // =====================================================
    {
        float *x, *y;
        CUDA_CHECK(cudaMallocManaged(&x, bytes));
        CUDA_CHECK(cudaMallocManaged(&y, bytes));
        for (int i = 0; i < N; ++i) { x[i] = 1.0f; y[i] = 2.0f; }

        CUDA_CHECK(cudaEventRecord(t0));
        saxpy<<<grid, block>>>(y, x, 2.0f, N);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        std::cout << std::left << std::setw(40) << "unified memory (kernel only)"
                  << " time=" << std::right << std::setw(9)
                  << std::fixed << std::setprecision(3) << ms << " ms"
                  << "  y[0]=" << y[0] << std::endl;

        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(y));
    }

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return 0;
}
