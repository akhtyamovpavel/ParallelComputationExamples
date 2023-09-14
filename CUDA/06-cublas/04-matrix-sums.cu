#include <iostream>
#include <cublas_v2.h>

int main() {
    cudaError_t cuda_stat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    int array_size = 12;
    float* h_x = new float[array_size];

    float* h_zeros = new float[array_size];
    for (int i = 0; i < array_size; ++i) {
        h_x[i] = i * 2.0f + 1;
        h_zeros[i] = 0.0f;
    }

    float* d_x;

    cuda_stat = cudaMalloc(&d_x, sizeof(float) * array_size);

    // cuda_stat = cudaMemcpy(d_x, h_zeros, sizeof(float) * 12, cudaMemcpyHostToDevice);

    stat = cublasCreate(&handle);
    stat = cublasSetMatrix(
        4, // nrows
        3, // ncols
        sizeof(*h_x),
        h_x,
        4,
        d_x,
        4
    );

    float sum_2nd_column = 0.0;
    float sum_2nd_row = 0.0;

    int ncols = 4;
    int nrows = 3;
    cublasSasum(handle, ncols, d_x + ncols * 1, 1, &sum_2nd_column);
    cublasSasum(handle, nrows, d_x + 1, ncols, &sum_2nd_row);

    std::cout << sum_2nd_column << std::endl;
    std::cout << sum_2nd_row << std::endl;
    
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "ERROR" << std::endl;
    }
    cuda_stat = cudaMemcpy(h_x, d_x, sizeof(float) * 12, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 12; ++i) {
        std::cout << i << " " << h_x[i] << std::endl;
    }

    cudaFree(d_x);
    return 0;
}
