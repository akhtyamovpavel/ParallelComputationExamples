#include <iostream>
#include <cstdio>


__global__ void Scan(int* in_data, int* out_data) {
    // in_data ->  [1 2 3 4 5 6 7 8], block_size 4
    // block_idx -> [0 0 0 0 1 1 1 1 ]
    
    extern __shared__ int shared_data[];
    // block_idx = 0

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = in_data[index];

    // shared_data -> [1, 2, 3, 4]
    __syncthreads();
    
    // shift = 2^(d - 1)
    for (unsigned int shift = 1; shift < blockDim.x; shift <<= 1 ) {
        int tmp;
        if (tid >= shift) {
            tmp = shared_data[tid - shift];
        }

        __syncthreads();
        if (tid >= shift) {
            shared_data[tid] += tmp;
        }

        // shift = 1
        // [1, 2, 3, 4] -> [1, 1 + 2, 2 + 3, 3 + 4] = [1, 3, 5, 7]
        // shift = 2
        // [1, 3, 5, 7] -> [1, 3, 1 + 5, 3 + 7] = [1, 3, 6, 10]
        __syncthreads();
    }
    //if (blockIdx.x == 16383) {
        //printf("%d %d %d\n", tid, shared_data[tid], index);
        // std::cout << shared_data[tid] << std::endl;
    //}
    // block_idx = 0 -> [a0, a1, a2, a3]
    // block_idx = 1 -> [a4, a5, a6, a7]
    out_data[index] = shared_data[tid];

    __syncthreads();

    // out_data[block_idx == 0] = [1, 3, 6, 10]

    // out_data[block_idx == 1] = [5, 11, 18, 26]

}


int main() {
    const int block_size = 1024;

    const int array_size = 1 << 22;
    int* h_array = new int[array_size];
    for (int i = 0; i < array_size; ++i) {
        h_array[i] = 1;
    }

    // int* output = new int[array_size];

    int* d_array;
    cudaMalloc(&d_array, sizeof(int) * array_size);

    cudaMemcpy(d_array, h_array, sizeof(int) * array_size, cudaMemcpyHostToDevice);


    int num_blocks = array_size / block_size;

    int* d_localscan;
    cudaMalloc(&d_localscan, sizeof(int) * array_size);
    int* h_localscan = new int[array_size];

    cudaEvent_t start;
    cudaEvent_t stop;

    // Creating event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);
    Scan<<<num_blocks, block_size, sizeof(int) * block_size>>>(d_array, d_localscan);


    cudaEventRecord(stop);

    cudaMemcpy(h_localscan, d_localscan, sizeof(int) * array_size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << " elapsed" << std::endl;

    std::cout << h_localscan[array_size - 1] << std::endl;

    delete[] h_array;
    delete[] h_localscan;


}
