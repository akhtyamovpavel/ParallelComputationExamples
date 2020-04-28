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

    // cudaMemcpy(trgt, src, ..., cudaMemcpyH2D / cudaMemcpyD2H)

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

    int result;

    stat = cublasSaxpy(
        handle, array_size / 2,
        &alpha,
        d_x, 2,
        d_y, 2
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Error occurred" << std::endl;
    } else {
        std::cerr << "Error not occured" << std::endl;
    }

    // 0 -> x[0] + y[0]
    // 1 -> y[1]
    // 2 -> x[2] + y[2]
    // ...

    stat = cublasGetVector(
        array_size,
        sizeof(*d_y),
        d_y,
        1,
        h_y,
        1
    );


    for (int i = 0; i < 20; ++i) {
        std::cout << h_x[i] << " " << h_y[i] << " " << h_y[i] - h_x[i] << std::endl;
    }
    cublasDestroy(handle);


    return 0;
}