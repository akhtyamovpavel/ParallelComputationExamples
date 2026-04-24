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

// Маленькое ядро: данных мало, считает быстро — launch overhead оказывается
// сопоставим с самим временем работы. Именно в этом режиме CUDA Graphs
// дают измеримое ускорение.
__global__ void saxpy_step(float* data, int n, float a, float b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        data[i] = a * data[i] + b;
    }
}

// Запускает один "проход" пайплайна: 5 коротких ядер подряд на одном stream'е.
// Выделен в функцию, чтобы ниже её можно было использовать и напрямую
// (baseline), и внутри stream capture — код один и тот же.
static void pipeline(float* d_data, int n, int grid, int block, cudaStream_t s) {
    saxpy_step<<<grid, block, 0, s>>>(d_data, n, 1.0f, 0.5f);
    saxpy_step<<<grid, block, 0, s>>>(d_data, n, 0.999f, 0.0f);
    saxpy_step<<<grid, block, 0, s>>>(d_data, n, 1.001f, -0.1f);
    saxpy_step<<<grid, block, 0, s>>>(d_data, n, 1.0f, 0.0f);
    saxpy_step<<<grid, block, 0, s>>>(d_data, n, 1.0f, 0.2f);
}

int main() {
    const int N = 1 << 14;
    const int ITERS = 1000;
    const int block = 256;
    const int grid = (N + block - 1) / block;

    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // Warm-up.
    pipeline(d_data, N, grid, block, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // --- Baseline: каждую итерацию гоняем 5 launch'ей напрямую. ---
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < ITERS; ++i) {
        pipeline(d_data, N, grid, block, stream);
    }
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms_direct = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_direct, t0, t1));

    // --- Graph: капчуем ровно один проход пайплайна в граф, ---
    // --- дальше M раз запускаем уже собранный graphExec.   ---
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    // Capture mode "Global" — любые операции с stream'ом должны попасть в граф;
    // всё, что вне capture окна — не трогаем.
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    pipeline(d_data, N, grid, block, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    // Warm-up graph launch.
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    }
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms_graph = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_graph, t0, t1));

    std::cout << std::left << std::setw(28) << "direct launches"
              << " iters=" << ITERS
              << "  time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms_direct << " ms"
              << std::endl;
    std::cout << std::left << std::setw(28) << "graph launches"
              << " iters=" << ITERS
              << "  time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms_graph << " ms"
              << "  speedup=" << std::fixed << std::setprecision(2)
              << (ms_direct / ms_graph) << "x" << std::endl;

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
