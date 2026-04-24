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

// То же ядро, что в 00-stream-capture. Здесь собираем граф вручную —
// cudaGraph_t как набор узлов (nodes) и рёбер-зависимостей (edges).
// Это то, во что stream capture разворачивается за кадром.
__global__ void saxpy_step(float* data, int n, float a, float b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        data[i] = a * data[i] + b;
    }
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

    // Все 5 узлов используют одно и то же ядро, но с разными параметрами.
    // Параметры хранятся в void*[], а сам указатель на этот массив — в
    // cudaKernelNodeParams::kernelParams. Соответственно aN/bN должны жить
    // до момента cudaGraphInstantiate (т.е. пока граф не "запечён" в exec).
    int n_int = N;
    float a0 = 1.0f;   float b0 = 0.5f;
    float a1 = 0.999f; float b1 = 0.0f;
    float a2 = 1.001f; float b2 = -0.1f;
    float a3 = 1.0f;   float b3 = 0.0f;
    float a4 = 1.0f;   float b4 = 0.2f;

    void* args0[] = {&d_data, &n_int, &a0, &b0};
    void* args1[] = {&d_data, &n_int, &a1, &b1};
    void* args2[] = {&d_data, &n_int, &a2, &b2};
    void* args3[] = {&d_data, &n_int, &a3, &b3};
    void* args4[] = {&d_data, &n_int, &a4, &b4};

    auto make_params = [&](void** args) {
        cudaKernelNodeParams p = {};
        p.func = reinterpret_cast<void*>(saxpy_step);
        p.gridDim = dim3(grid);
        p.blockDim = dim3(block);
        p.sharedMemBytes = 0;
        p.kernelParams = args;
        p.extra = nullptr;
        return p;
    };

    cudaKernelNodeParams p0 = make_params(args0);
    cudaKernelNodeParams p1 = make_params(args1);
    cudaKernelNodeParams p2 = make_params(args2);
    cudaKernelNodeParams p3 = make_params(args3);
    cudaKernelNodeParams p4 = make_params(args4);

    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    // Добавляем 5 kernel-узлов с линейной цепочкой зависимостей:
    // n0 -> n1 -> n2 -> n3 -> n4.
    cudaGraphNode_t n0;
    cudaGraphNode_t n1;
    cudaGraphNode_t n2;
    cudaGraphNode_t n3;
    cudaGraphNode_t n4;

    CUDA_CHECK(cudaGraphAddKernelNode(&n0, graph, nullptr, 0, &p0));
    CUDA_CHECK(cudaGraphAddKernelNode(&n1, graph, &n0, 1, &p1));
    CUDA_CHECK(cudaGraphAddKernelNode(&n2, graph, &n1, 1, &p2));
    CUDA_CHECK(cudaGraphAddKernelNode(&n3, graph, &n2, 1, &p3));
    CUDA_CHECK(cudaGraphAddKernelNode(&n4, graph, &n3, 1, &p4));

    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    // Warm-up.
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    }
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms_graph = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_graph, t0, t1));

    std::cout << std::left << std::setw(28) << "explicit-graph launches"
              << " iters=" << ITERS
              << "  time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms_graph << " ms"
              << "  (5 kernel nodes, linear chain)" << std::endl;

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
