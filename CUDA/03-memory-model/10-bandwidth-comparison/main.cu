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

// Сравнение пропускной способности (bandwidth) при разных типах памяти:
//
//   1. Pageable (обычный malloc) — ОС может переместить страницы,
//      поэтому драйвер CUDA копирует через промежуточный pinned-буфер.
//
//   2. Pinned (cudaMallocHost) — страницы заблокированы в физической
//      памяти, DMA-контроллер копирует напрямую.
//
//   3. Pinned + Write-Combining (cudaHostAllocWriteCombined) — запись
//      с хоста идёт без L1/L2 кеширования, оптимально для H→D потока.
//      Чтение с хоста будет медленным (нет кеша).
//
//   4. Unified Memory без prefetch (cudaMallocManaged) — страницы
//      мигрируют по page-fault при первом обращении GPU.
//
//   5. Unified Memory + cudaMemPrefetchAsync — явная миграция страниц
//      на GPU до запуска ядра, без page-faults.
//
// Для типов 1-3 замеряем bandwidth cudaMemcpy (H→D и D→H).
// Для типов 4-5 замеряем эффективную bandwidth через ядро-читатель,
// потому что unified memory используется без явного cudaMemcpy.
//
// Ссылки:
//   Хабр: Работа с памятью в GPU — https://habr.com/ru/articles/55461/
//   CUDA Best Practices Guide, Memory Optimizations
//   03-host-alloc-benchmarks — детальное сравнение pageable vs pinned

__global__ void read_kernel(const float* data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += data[i];
    }
    if (idx == 0) {
        *out = sum;
    }
}

static void print_bw(const char* label, double bytes, float ms) {
    double gbps = bytes / (ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);
    std::cout << std::left << std::setw(42) << label
              << "  time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << "  BW=" << std::setw(7) << std::setprecision(2)
              << gbps << " GB/s" << std::endl;
}

int main() {
    const unsigned int N = 64 * 1024 * 1024;
    const size_t bytes = N * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Transfer size: " << (bytes >> 20) << " MB\n" << std::endl;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    float* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, bytes));

    // ============================
    // 1. Pageable memory
    // ============================
    {
        float* h = (float*)malloc(bytes);
        for (unsigned int i = 0; i < N; ++i) h[i] = (float)i;

        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(d_buf, h, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        print_bw("1. Pageable H->D", bytes, ms);

        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(h, d_buf, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        print_bw("   Pageable D->H", bytes, ms);

        free(h);
    }

    // ============================
    // 2. Pinned memory
    // ============================
    {
        float* h;
        CUDA_CHECK(cudaMallocHost(&h, bytes));
        for (unsigned int i = 0; i < N; ++i) h[i] = (float)i;

        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(d_buf, h, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        print_bw("2. Pinned H->D", bytes, ms);

        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(h, d_buf, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        print_bw("   Pinned D->H", bytes, ms);

        CUDA_CHECK(cudaFreeHost(h));
    }

    // ============================
    // 3. Pinned + Write-Combining
    // ============================
    {
        float* h;
        CUDA_CHECK(cudaHostAlloc(&h, bytes, cudaHostAllocWriteCombined));
        for (unsigned int i = 0; i < N; ++i) h[i] = (float)i;

        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(d_buf, h, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        print_bw("3. Pinned+WC H->D", bytes, ms);

        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(h, d_buf, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        print_bw("   Pinned+WC D->H (slow read!)", bytes, ms);

        CUDA_CHECK(cudaFreeHost(h));
    }

    CUDA_CHECK(cudaFree(d_buf));

    // Для unified memory замеряем эффективную bandwidth через ядро,
    // потому что unified memory не требует явного cudaMemcpy.
    const int block = 256;
    const int grid = 256;
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    // ============================
    // 4. Unified Memory (no prefetch) — page-fault при обращении GPU
    // ============================
    {
        float* u;
        CUDA_CHECK(cudaMallocManaged(&u, bytes));
        for (unsigned int i = 0; i < N; ++i) u[i] = 1.0f;

        CUDA_CHECK(cudaEventRecord(t0));
        read_kernel<<<grid, block>>>(u, d_out, N);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        print_bw("4. Unified (lazy, kernel read)", bytes, ms);

        CUDA_CHECK(cudaFree(u));
    }

    // ============================
    // 5. Unified Memory + prefetch — миграция ДО обращения GPU
    // ============================
    {
        float* u;
        CUDA_CHECK(cudaMallocManaged(&u, bytes));
        for (unsigned int i = 0; i < N; ++i) u[i] = 1.0f;

        CUDA_CHECK(cudaMemPrefetchAsync(u, bytes, 0));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(t0));
        read_kernel<<<grid, block>>>(u, d_out, N);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        print_bw("5. Unified + prefetch (kernel read)", bytes, ms);

        CUDA_CHECK(cudaFree(u));
    }

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return 0;
}
