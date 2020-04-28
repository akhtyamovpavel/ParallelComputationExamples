#include <iostream>
#include <cublas_v2.h>

int main() {
    cudaError_t cuda_stat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    int array_size = (1 << 22);
    float* h_x = new float[array_size];
    float* h_y = new float[array_size];

    for (int i = 0; i < array_size; ++i) {
        h_x[i] = i * 1.0f;
        h_y[i] = i * 1.0f;
    }

    float* d_x;
    float* d_y;

    cuda_stat = cudaMalloc(&d_x, sizeof(float) * array_size);
    cuda_stat = cudaMalloc(&d_y, sizeof(float) * array_size);

    stat = cublasCreate(&handle);
    stat = cublasSetVector(
        array_size,
        sizeof(*h_x),
        h_x,
        /* space by host */ 1,
        d_x,
        /* space by device */ 1
    );

    stat = cublasSetVector(
        array_size,
        sizeof(*h_y),
        h_y,
        1,
        d_y,
        1
    );

    float alpha = 1.0;

    float scalar_product;
    float norm_x;
    float norm_y;

    stat = cublasSdot(
        handle,
        array_size,
        d_x, 1,
        d_y, 1,
        &scalar_product
    );

    stat = cublasSnrm2(
        handle,
        array_size,
        d_x, 1,
        &norm_x
    );

    stat = cublasSnrm2(
        handle,
        array_size,
        d_y, 1,
        &norm_y
    );

    std::cout << scalar_product / norm_x / norm_y << std::endl;
    cublasDestroy(handle);


    return 0;

    // 6 * 5

    // 0 1 2 3 4
    // 5 6 7 8 9
    // 10 11 12 13 14
    // ...
    
}