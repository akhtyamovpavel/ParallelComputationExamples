#include <iostream>
#include <cstdio>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#define GET_OFFSET(idx) (idx >> LOG_NUM_BANKS)

__global__ void Scan(int* in_data, int* out_data) {
    // in_data ->  [1 2 3 4 5 6 7 8], block_size 4
    // block_idx -> [0 0 0 0 1 1 1 1 ]
    
    extern __shared__ int shared_data[];
    // block_idx = 0

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid + GET_OFFSET(tid)] = in_data[index];

    // shared_data[tid + (tid >> LOG_NUM_BANKS)] = in_data[index];

    // shared_data -> [1, 2, 3, 4]
    __syncthreads();
    
    // shift = 2^(d - 1)
    for (unsigned int shift = 1; shift < blockDim.x; shift <<= 1 ) {
        int ai = shift * (2 * tid + 1) - 1; // tid = 0, shift = 1, ai = 0; // tid = 16, shift = 1, ai = 32 = 0
        int bi = shift * (2 * tid + 2) - 1;

        if (bi < blockDim.x) {
            shared_data[bi + GET_OFFSET(bi)] += shared_data[ai + GET_OFFSET(ai)];
        }

        __syncthreads();
    }

    if (tid == 0) {
        shared_data[blockDim.x - 1 + GET_OFFSET(blockDim.x - 1)] = 0;
    }

    __syncthreads();

    int temp;
    for (unsigned int shift = blockDim.x / 2; shift > 0; shift >>= 1) {
        int bi = shift * (2 * tid + 2) - 1;
        int ai = shift * (2 * tid + 1) - 1;
        int ai_offset = ai + GET_OFFSET(ai);
        int bi_offset = bi + GET_OFFSET(bi);
        if (bi < blockDim.x) {
            temp = shared_data[ai_offset]; // blue in temp

            // temp = 4
            shared_data[ai_offset] = shared_data[bi_offset]; // orange

            // 1 2 1 0 1 2 1 0 // temp = 4
            shared_data[bi_offset] = temp + shared_data[bi_offset];
        }
        __syncthreads();

    }
    // if (blockIdx.x == 16383) {
    //     printf("%d %d %d %d\n", tid, tid + GET_OFFSET(tid), shared_data[tid + GET_OFFSET(tid)], index);
    //     // std::cout << shared_data[tid] << std::endl;
    // }
    // block_idx = 0 -> [a0, a1, a2, a3]
    // block_idx = 1 -> [a4, a5, a6, a7]
    out_data[index] = shared_data[tid + GET_OFFSET(tid)];

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
    Scan<<<num_blocks, block_size, sizeof(int) * (block_size + GET_OFFSET(block_size))>>>(d_array, d_localscan);


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
