#include <iostream>

#define BLOCKSIZE 256

__global__ void StupidSumArray(int* array, int* result) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    result[tid] = 0;
    for (int i = tid * 1024; i < (tid + 1) * 1024; ++i) {
        result[tid] += array[i];
    }
}


int main() {
    int N = 1 << 18;
    int *h_x = new int[N];

    for (int i = 0; i < N; ++i) {
        h_x[i] = 1;
    }
    int *d_x;
    int size = sizeof(int) * N;
    cudaMalloc(&d_x, size);

    int* h_result = new int[256];
    for (int i = 0; i < BLOCKSIZE; ++i) {
        h_result[i] = 0;
    }
    int *d_result;
    cudaMalloc(&d_result, sizeof(int) * 256); 

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    StupidSumArray<<<1, 256>>>(d_x, d_result);

    cudaEventRecord(stop);

    cudaMemcpy(h_result, d_result, sizeof(int) * 256, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    for (int i = 0; i < 256; ++i) {
        std::cout << i << " " << h_result[i] << std::endl;
    }

    std::cout << ms << std::endl;
    cudaFree(d_x);
    cudaFree(d_result);
    delete[] h_result;
    delete[] h_x;

}
