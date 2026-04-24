#include <iostream>
#include <iomanip>
#include <cmath>

#include <cusparse.h>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

#define CUSPARSE_CHECK(call) do {                                             \
    cusparseStatus_t st = (call);                                             \
    if (st != CUSPARSE_STATUS_SUCCESS) {                                      \
        std::cerr << "cuSPARSE error " << st << " at "                        \
                  << __FILE__ << ":" << __LINE__ << std::endl;                \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// SpMV (sparse matrix - dense vector) y = A * x на маленькой матрице 4x4.
// Формат — CSR (Compressed Sparse Row): три массива описывают все ненулевые
// элементы.
//
// A =
//   1 0 2 0        nnz      = 8
//   0 3 0 4        values   = [1, 2, 3, 4, 5, 6, 7, 8]
//   5 0 6 0        colInd   = [0, 2, 1, 3, 0, 2, 1, 3]
//   0 7 0 8        rowPtr   = [0, 2, 4, 6, 8]
//
// x = [1, 2, 3, 4]  =>  A * x = [7, 22, 23, 46].
int main() {
    const int rows = 4;
    const int cols = 4;
    const int nnz = 8;

    int h_rowPtr[5]  = {0, 2, 4, 6, 8};
    int h_colInd[8]  = {0, 2, 1, 3, 0, 2, 1, 3};
    float h_values[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_x[4] = {1, 2, 3, 4};
    float h_y[4] = {0, 0, 0, 0};
    float expected[4] = {7.0f, 22.0f, 23.0f, 46.0f};

    int* d_rowPtr = nullptr;
    int* d_colInd = nullptr;
    float* d_values = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colInd, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, h_rowPtr, (rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colInd, h_colInd, nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values, nnz * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, cols * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Generic cuSPARSE API (CUDA 11+): отделяет описание матрицы/векторов
    // (descriptor'ы) от конкретной операции. Сначала строим descriptor'ы,
    // потом вызываем cusparseSpMV.
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, rows, cols, nnz,
                                     d_rowPtr, d_colInd, d_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseDnVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, rows, d_y, CUDA_R_32F));

    float alpha = 1.0f;
    float beta = 0.0f;

    // cuSPARSE требует отдельного workspace-буфера для SpMV; его размер
    // запрашивается через bufferSize-call, потом аллоцируется руками.
    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecX,
                                           &beta, vecY,
                                           CUDA_R_32F,
                                           CUSPARSE_SPMV_ALG_DEFAULT,
                                           &buffer_size));
    void* d_buffer = nullptr;
    if (buffer_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));
    }

    // y = alpha * A * x + beta * y.
    CUSPARSE_CHECK(cusparseSpMV(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX,
                                &beta, vecY,
                                CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT,
                                d_buffer));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, rows * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < rows; ++i) {
        float e = std::fabs(h_y[i] - expected[i]);
        if (e > max_err) {
            max_err = e;
        }
    }

    std::cout << "cuSPARSE SpMV (4x4 CSR)  y = ["
              << h_y[0] << ", " << h_y[1] << ", "
              << h_y[2] << ", " << h_y[3] << "]"
              << "  expected=[7, 22, 23, 46]"
              << "  max_err=" << std::scientific << std::setprecision(2)
              << max_err << std::endl;

    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    if (d_buffer != nullptr) {
        CUDA_CHECK(cudaFree(d_buffer));
    }
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colInd));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
