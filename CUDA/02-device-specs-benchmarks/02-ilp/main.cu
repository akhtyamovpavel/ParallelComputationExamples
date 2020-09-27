#include <iostream>


__global__ void SumV0(int* x, int* y, int* result) {
    // 1x registers are available here
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;
    result[tid] = x[tid] + y[tid];

}

__global__ void SumV1(int *x, int* y, int* result) {
    // 2x registers are available here
    int double_tid = threadIdx.x + 2 * blockDim.x * blockIdx.x;

    result[double_tid] = x[double_tid] + y[double_tid];
    result[double_tid + blockDim.x] = x[double_tid + blockDim.x] + y[double_tid + blockDim.x]; 
}

int main() {
    int array_size = 1 << 28;
    
    int *h_x = new int[array_size];
    int *h_y = new int[array_size];
    
    for (int i = 0; i < array_size; ++i) {
        h_x[i] = i;
        h_y[i] = 2 * i;
    }

    int* d_x;
    int* d_y;
    int* d_result;
    
    int num_bytes = sizeof(*h_x) * array_size;
    cudaMalloc(&d_x, num_bytes);
    cudaMalloc(&d_y, num_bytes);
    cudaMalloc(&d_result, num_bytes);

    cudaMemcpy(d_x, h_x, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, num_bytes, cudaMemcpyHostToDevice);

    int block_size = 512;

    int num_blocks = (array_size + block_size - 1) / block_size;
    
    cudaEvent_t start0;
    cudaEvent_t start1;

    cudaEvent_t end0;
    cudaEvent_t end1;
    
    cudaEventCreate(&start0);
    cudaEventCreate(&start1);
    cudaEventCreate(&end0);
    cudaEventCreate(&end1);

    cudaEventRecord(start0);    
    SumV1<<<num_blocks, block_size / 2>>>(d_x, d_y, d_result);
    cudaEventRecord(end0);
    cudaEventSynchronize(end0);
    float millis0 = 0.0;
    cudaEventElapsedTime(&millis0, start0, end0);

    cudaEventRecord(start1);
    SumV0<<<num_blocks, block_size>>>(d_x, d_y, d_result);
    cudaEventRecord(end1);
    cudaEventSynchronize(end1);
    float millis1 = 0.0;
    cudaEventElapsedTime(&millis1, start1, end1);

    std::cout << "ILP 2: " << millis0 << " ILP 1: " << millis1 << std::endl;

    int *h_result = new int[array_size];

    cudaMemcpy(h_result, d_result, num_bytes, cudaMemcpyDeviceToHost);
   
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    delete[] h_x;
    delete[] h_y;
    delete[] h_result;
    return 0;
}
