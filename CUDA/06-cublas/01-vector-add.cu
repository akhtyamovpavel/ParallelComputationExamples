#include <iostream>
#include <cublas_v2.h>

int main() {
    cudaError_t cuda_stat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    int array_size = (1 << 22);
    float* h_x = new float[array_size];

    for (int i = 0; i < array_size; ++i) {
        h_x[i] = i * 1.0f;
    }

    float* d_x;

    cuda_stat = cudaMalloc(&d_x, sizeof(float) * array_size);

    stat = cublasCreate(&handle);
    stat = cublasSetVector(
        array_size,
        sizeof(*h_x),
        h_x,
        /* space by host */ 1,
        d_x,
        /* space by device */ 1
    );
    int result;

    stat = cublasIsamax(handle, array_size, d_x, 3, &result);

    // 0, 3, 6, 9, ...
    // 1, 2, 3, 4, 5, ...
    // max element located at (1398102 - 1) * 3

    std::cout << result << std::endl;

    return 0;
}