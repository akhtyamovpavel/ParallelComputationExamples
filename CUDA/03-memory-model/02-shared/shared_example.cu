#include <iostream>
#include <cmath>
#include <cassert>


#define BLOCKSIZE 512

__global__ void ComputeThreeSum(int n, int* input, int* result) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int local_tid = threadIdx.x;
    __shared__ int s_data[BLOCKSIZE + 2]; // unique for every block!
    if (local_tid == 0 && tid > 0) {
        s_data[0] = input[tid - 1];
    } else if (local_tid == blockDim.x - 1 && tid + 1 < n) {
        s_data[BLOCKSIZE + 1] = input[tid + 1];
    }
    s_data[local_tid + 1] = input[tid]; // copy data to shared memory
    
    __syncthreads();

    result[tid] = s_data[local_tid] + s_data[local_tid + 1] + s_data[local_tid + 2]; 

}


int main() {
    int N = 1 << 28;

    int* h_array = new int[N];
    int* h_diff = new int[N];
    for (int i = 0; i < N; ++i) {
        h_array[i] = 1;
    }
    
    int* d_array;
    int* d_diff;
    unsigned int size = N * sizeof(int);
    cudaMalloc(&d_array, size);
    cudaMalloc(&d_diff, size);

    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
    
    int num_blocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    ComputeThreeSum<<<num_blocks, BLOCKSIZE>>>(N, d_array, d_diff);
    cudaEventRecord(stop);


    cudaMemcpy(h_diff, d_diff, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds;


    cudaEventElapsedTime(&milliseconds, start, stop);

    for (int i = 1; i < N - 1; ++i) {
        if (h_diff[i] != 3) {
            std::cout << i << " " << h_diff[i] << std::endl;    
        }
        assert(h_diff[i] == 3);
    }

    std::cout << milliseconds << " elapsed" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_array);
    cudaFree(d_diff);
    delete[] h_array;
    delete[] h_diff;

}
