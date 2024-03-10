#include <iostream>
#include <cstdio>

#define BLOCKSIZE 256
#define CNT 4

__global__ void NormalSumArray(int* array, int* result, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("%d %d %d %d\n", tid, blockIdx.x, blockDim.x, threadIdx.x);
    result[tid] = 0;
    
    
    int stride = gridDim.x * blockDim.x;
    // if (tid == 0) {
    //     printf("%d %d\n", stride, tid);
    // }

    for (int i = tid; i < N; i += stride) {
        if (tid == 0) {
            printf("%d %d\n", i, tid);
        }
        result[tid] += array[i];
    }
}


int main() {
    int N = 1 << 28;
    int *h_x = new int[N];

    for (int i = 0; i < N; ++i) {
        h_x[i] = 1;
    }
    int *d_x;
    int size = sizeof(int) * N;
    cudaMalloc(&d_x, size);

    cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

    int multiprocess_count = deviceProp.multiProcessorCount;

    int output_elements = BLOCKSIZE * multiprocess_count;

    int* h_result = new int[output_elements];
    for (int i = 0; i < output_elements; ++i) {
        h_result[i] = 0;
    }
    int *d_result;
    
    cudaMalloc(&d_result, sizeof(int) * output_elements); 

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, sizeof(int) * output_elements, cudaMemcpyHostToDevice);

    
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    std::cout << multiprocess_count << std::endl;
    

    NormalSumArray<<<multiprocess_count, BLOCKSIZE>>>(d_x, d_result, N);

    cudaEventRecord(stop);

    cudaMemcpy(h_result, d_result, sizeof(int) * output_elements, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    int result = 0;
    for (int i = 0; i < output_elements; ++i) {
        result += h_result[i];
        if (i % 100 == 0) {
            std::cout << result << std::endl;
        }
    }

    std::cout << result << std::endl;
    // for (int i = 0; i < output_elements; ++i) {
    //     std::cout << i << " " << h_result[i] << std::endl;
    // }

    // std::cout << ms << std::endl;
    cudaFree(d_x);
    cudaFree(d_result);
    delete[] h_result;
    delete[] h_x;

}
