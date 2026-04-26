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

// Inclusive scan внутри warp'а через __shfl_up_sync.
// После вызова лейн L содержит сумму элементов [0..L].
// Аналог Хиллиса-Стила, но без shared memory — данные
// передаются через регистры (shuffle instructions).
//
// Сложность: O(N log N) работы, log N шагов.
// Для warp'а (N=32) это 5 шагов по 32 лейна = 160 операций.
//
// Ссылка: Mark Harris, "Parallel Prefix Sum (Scan) with CUDA"
// GPU Gems 3, Chapter 39 — warp-level primitives.
__device__ int warp_inclusive_scan(int val) {
    const unsigned FULL = 0xffffffffu;
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(FULL, val, offset);
        if ((threadIdx.x & 31) >= offset) {
            val += n;
        }
    }
    return val;
}

// Block-level inclusive scan через warp-level scan + shared memory.
//
// Алгоритм (3 фазы):
// 1) Каждый warp делает inclusive scan через __shfl_up_sync.
// 2) Последний лейн каждого warp'а (сумма warp'а) записывается в shared memory.
//    Warp 0 сканирует эти warp-суммы (их <= 32, помещаются в один warp).
// 3) Каждый поток прибавляет к своему результату сумму предыдущих warp'ов.
//
// Преимущества перед алгоритмом Блеллоха на shared memory:
//   - Меньше обращений к shared memory (только warp-суммы, не весь массив)
//   - Нет bank conflicts при scan внутри warp'а (shuffle работает через регистры)
//   - Меньше __syncthreads() (всего 2 вместо 2*log(N))
//
// ВАЖНО: это scan внутри одного блока. Для полного scan большого массива
// нужна дополнительная координация между блоками (см. 01.75-recursive).
__global__ void BlockScanWarp(const int* in, int* out, int n) {
    extern __shared__ int warp_sums[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    int val = (idx < n) ? in[idx] : 0;

    // Фаза 1: inclusive scan внутри каждого warp'а
    int scanned = warp_inclusive_scan(val);

    // Фаза 2: собираем warp-суммы и сканируем их
    if (lane == 31) {
        warp_sums[warp_id] = scanned;
    }
    __syncthreads();

    if (warp_id == 0 && lane < num_warps) {
        warp_sums[lane] = warp_inclusive_scan(warp_sums[lane]);
    }
    __syncthreads();

    // Фаза 3: прибавляем prefix суммы предыдущих warp'ов
    if (warp_id > 0) {
        scanned += warp_sums[warp_id - 1];
    }

    if (idx < n) {
        out[idx] = scanned;
    }
}

int main() {
    const int block_size = 1024;
    const int array_size = 1 << 20;

    int* h_in = new int[array_size];
    for (int i = 0; i < array_size; ++i) {
        h_in[i] = 1;
    }

    int* d_in = nullptr;
    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, array_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, array_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, array_size * sizeof(int),
                          cudaMemcpyHostToDevice));

    int num_blocks = (array_size + block_size - 1) / block_size;
    int num_warps = block_size / 32;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    BlockScanWarp<<<num_blocks, block_size, num_warps * sizeof(int)>>>(
        d_in, d_out, array_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    int* h_out = new int[array_size];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Проверка: inclusive scan единиц => 1, 2, ..., block_size внутри блока
    bool ok = true;
    for (int b = 0; b < num_blocks && ok; ++b) {
        for (int t = 0; t < block_size && ok; ++t) {
            int idx = b * block_size + t;
            if (idx >= array_size) break;
            int expected = t + 1;
            if (h_out[idx] != expected) {
                std::cerr << "Mismatch at block=" << b << " tid=" << t
                          << ": got " << h_out[idx]
                          << ", expected " << expected << std::endl;
                ok = false;
            }
        }
    }

    std::cout << "Block-level inclusive scan (__shfl_up_sync)" << std::endl;
    std::cout << "N=" << array_size << "  block=" << block_size
              << "  check=" << (ok ? "OK" : "FAIL")
              << "  kernel_time=" << std::fixed << std::setprecision(3)
              << ms << " ms" << std::endl;

    std::cout << "block 0: out[0..7]   = ";
    for (int i = 0; i < 8; ++i) std::cout << h_out[i] << " ";
    std::cout << std::endl;

    std::cout << "block 0: out["
              << (block_size - 4) << ".." << (block_size - 1) << "] = ";
    for (int i = block_size - 4; i < block_size; ++i)
        std::cout << h_out[i] << " ";
    std::cout << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;
    return 0;
}
