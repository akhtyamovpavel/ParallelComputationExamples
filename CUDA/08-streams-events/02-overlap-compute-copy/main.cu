#include <iostream>
#include <iomanip>
#include <vector>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

__global__ void busy_kernel(const float* in, float* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float x = in[i];
        for (int k = 0; k < iters; ++k) {
            x = x * 1.0001f + 0.5f;
        }
        out[i] = x;
    }
}

static void print_row(const char* name, float ms, size_t bytes) {
    double gbps = static_cast<double>(bytes) / (ms * 1e-3)
                  / (1024.0 * 1024.0 * 1024.0);
    std::cout << std::left << std::setw(32) << name
              << " time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << "  effective-bw=" << std::right << std::setw(7)
              << std::fixed << std::setprecision(2) << gbps << " GB/s"
              << std::endl;
}

int main() {
    const int N = 1 << 24;
    const int ITERS = 80;
    const int NUM_STREAMS = 4;
    const int block = 256;

    const size_t bytes_total = static_cast<size_t>(N) * sizeof(float);

    float* h_in = nullptr;
    float* h_out = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_in, bytes_total));
    CUDA_CHECK(cudaMallocHost(&h_out, bytes_total));
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes_total));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_total));

    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // Baseline: один stream, всё последовательно.
    {
        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes_total, cudaMemcpyHostToDevice));
        int grid = (N + block - 1) / block;
        busy_kernel<<<grid, block>>>(d_in, d_out, N, ITERS);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes_total, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        print_row("single stream (sequential)", ms, 2 * bytes_total);
    }

    // Pipeline: N stream'ов, каждый гонит свой chunk через H2D -> compute -> D2H.
    // Разные stream'ы работают параллельно: пока один делает D2H, другой уже
    // считает, третий качает следующий H2D.
    {
        std::vector<cudaStream_t> streams(NUM_STREAMS);
        for (int s = 0; s < NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaStreamCreate(&streams[s]));
        }

        const int chunk = N / NUM_STREAMS;
        const size_t chunk_bytes = chunk * sizeof(float);

        CUDA_CHECK(cudaEventRecord(t0));
        for (int s = 0; s < NUM_STREAMS; ++s) {
            int offset = s * chunk;
            CUDA_CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset, chunk_bytes,
                                       cudaMemcpyHostToDevice, streams[s]));
            int grid = (chunk + block - 1) / block;
            busy_kernel<<<grid, block, 0, streams[s]>>>(
                d_in + offset, d_out + offset, chunk, ITERS);
            CUDA_CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset, chunk_bytes,
                                       cudaMemcpyDeviceToHost, streams[s]));
        }
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        print_row("multi-stream (overlap)", ms, 2 * bytes_total);

        for (int s = 0; s < NUM_STREAMS; ++s) {
            CUDA_CHECK(cudaStreamDestroy(streams[s]));
        }
    }

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    return 0;
}
