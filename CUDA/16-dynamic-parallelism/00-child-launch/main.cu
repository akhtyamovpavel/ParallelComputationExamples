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

// Dynamic parallelism: ядро может запускать другие ядра прямо из device-кода.
// Доступно с sm_35+; требует раздельной компиляции (nvcc -rdc=true) и линковки
// с cudadevrt — в CMake это выставляется свойствами target'а:
//   CUDA_SEPARABLE_COMPILATION ON
//   CUDA_RESOLVE_DEVICE_SYMBOLS ON
//
// Начиная с CUDA 12, вызов cudaDeviceSynchronize() из device-кода запрещён
// (он был в CDP1, убран в CDP2). Остался режим fire-and-forget: parent
// запускает child'ов, host ждёт всё дерево через cudaDeviceSynchronize.
//
// Реальная польза Dynamic Parallelism — в data-dependent launch'ах:
// например, adaptive mesh refinement (запускать child только там, где
// нужно уточнение), BFS на графе и т.п. Здесь — минимальный learning-stub:
// parent с N потоков, каждый поток запускает child, который пишет i*i.

__global__ void child(int i, int* out) {
    out[i] = i * i;
}

__global__ void parent(int* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Launch configuration задаётся прямо в device-коде тем же синтаксисом
        // <<<...>>>. Аргументы передаются как обычно — по значению.
        child<<<1, 1>>>(idx, out);
    }
}

int main() {
    // CDP-launch'и несут заметный overhead (десятки микросекунд на child),
    // поэтому демо-масштаб небольшой. Для реального перфа используют
    // data-dependent паттерны, а не "по child'у на тред".
    const int N = 256;

    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(int)));

    parent<<<1, N>>>(d_out, N);
    // host'овая синхронизация ждёт завершения parent и всех его child'ов.
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_out[N];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int),
                          cudaMemcpyDeviceToHost));

    int max_err = 0;
    for (int i = 0; i < N; ++i) {
        int diff = h_out[i] - i * i;
        if (diff < 0) {
            diff = -diff;
        }
        if (diff > max_err) {
            max_err = diff;
        }
    }

    std::cout << std::left << std::setw(28) << "dynamic parallelism"
              << " N=" << N
              << "  out[0]=" << h_out[0]
              << "  out[" << (N - 1) << "]=" << h_out[N - 1]
              << "  max_err=" << max_err << std::endl;

    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
