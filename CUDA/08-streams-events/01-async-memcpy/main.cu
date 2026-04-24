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

static float time_memcpy(void* dst, const void* src, size_t bytes,
                         cudaMemcpyKind kind) {
    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms;
}

static float time_memcpy_async(void* dst, const void* src, size_t bytes,
                               cudaMemcpyKind kind, cudaStream_t stream) {
    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0, stream));
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, stream));
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms;
}

static void print_row(const char* name, float ms, size_t bytes) {
    double gbps = static_cast<double>(bytes) / (ms * 1e-3)
                  / (1024.0 * 1024.0 * 1024.0);
    std::cout << std::left << std::setw(36) << name
              << " time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << "  bw=" << std::right << std::setw(7)
              << std::fixed << std::setprecision(2) << gbps << " GB/s"
              << std::endl;
}

int main() {
    const size_t N = 1 << 26;
    const size_t bytes = N * sizeof(float);

    float* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, bytes));

    // Pageable host memory (обычный new[] / malloc): драйвер вынужден делать
    // staging через внутренний pinned-буфер, поэтому скорость ниже.
    {
        float* h_pageable = new float[N];
        for (size_t i = 0; i < N; ++i) {
            h_pageable[i] = 1.0f;
        }
        float ms = time_memcpy(d_buf, h_pageable, bytes, cudaMemcpyHostToDevice);
        print_row("cudaMemcpy (pageable host)", ms, bytes);
        delete[] h_pageable;
    }

    // Pinned host memory (cudaMallocHost / cudaHostAlloc) — DMA-движок копирует
    // напрямую, без staging; плюс только такая память даёт настоящий async.
    {
        float* h_pinned = nullptr;
        CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
        for (size_t i = 0; i < N; ++i) {
            h_pinned[i] = 1.0f;
        }
        float ms = time_memcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice);
        print_row("cudaMemcpy (pinned host)", ms, bytes);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        float ms_async = time_memcpy_async(d_buf, h_pinned, bytes,
                                           cudaMemcpyHostToDevice, stream);
        print_row("cudaMemcpyAsync (pinned + stream)", ms_async, bytes);
        CUDA_CHECK(cudaStreamDestroy(stream));

        CUDA_CHECK(cudaFreeHost(h_pinned));
    }

    CUDA_CHECK(cudaFree(d_buf));
    return 0;
}
