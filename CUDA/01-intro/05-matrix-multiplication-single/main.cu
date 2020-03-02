#include <iostream>
#include <cmath>

#include <random>
#include <ctime>
#include <functional>
#include <cstdio>

typedef struct {
    int width;
	int height;
	float* elements;
} Matrix;



int GetBlockSize() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    return roundl(sqrtl(deviceProp.maxThreadsPerBlock));
}


__global__ void MatMulKernel(Matrix *A, Matrix *B, Matrix *C);

void MatMul(Matrix* A, Matrix* B, Matrix* C) {
	Matrix d_A;
	d_A.width = A->width;
    d_A.height = A->height;

    size_t size = A->width * A->height * sizeof(float);

    cudaMalloc(&d_A.elements, size);

    cudaMemcpy(d_A.elements, A->elements, size, cudaMemcpyHostToDevice);
	
    Matrix d_B;
    d_B.width = B->width;
    d_B.height = B->height;

    size = B->width * B->height * sizeof(float);

    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B->elements, size, cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = B->width;
    d_C.height = A->height;

    size = d_C.width * d_C.height * sizeof(float);

    cudaMalloc(&d_C.elements, size);

    int block_size = GetBlockSize();

    dim3 dim_block(block_size, block_size);
    dim3 dim_grid(B->width / dim_block.x, A->height / dim_block.y);

    std::cout << dim_block.x << " " << dim_block.y << " " << dim_grid.x << " " << dim_grid.y << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    MatMulKernel<<<dim_grid, dim_block>>>(&d_A, &d_B, &d_C);
    float milliseconds = 0;

    cudaEventRecord(stop);
    cudaMemcpy(C->elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop); 
    std::cout << "Calculation in " << milliseconds << std::endl;

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void MatMulKernel(Matrix *A, Matrix *B, Matrix *C) {
    float value = 0;

    printf("%d %d %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    printf("%d, %d, %d\n", row, column, A->width);

    for (int index = 0; index < A->width; ++index) {
        value += A->elements[row * A->width + index] * B->elements[index * B->width + column];
    }

    C->elements[row * C->width + column] = value;
}


Matrix* CreateMatrix(int size) {
    Matrix* matrix = new Matrix();
    
    matrix->width = size;
    matrix->height = size;
    
    std::mt19937 gen(time(0));
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    auto generate = std::bind(distribution, gen);

    matrix->elements = new float[size * size];

    for (int index = 0; index < size * size; ++index) {
        matrix->elements[index] = generate();
    }

    return matrix;
}

void FreeMatrix(Matrix* matrix) {
    delete[] matrix->elements;
    delete matrix;
}

int main(int argc, char** argv) {
    Matrix *A = CreateMatrix(32);
    Matrix *B = CreateMatrix(32);
    Matrix *C = CreateMatrix(32);

    MatMul(A, B, C);
    
    FreeMatrix(A);
    FreeMatrix(B);
    FreeMatrix(C);

}
