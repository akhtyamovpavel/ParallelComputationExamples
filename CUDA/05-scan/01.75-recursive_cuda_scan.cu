#include <iostream>
#include <cstdio>


__global__ void Scan(int* in_data, int* out_data, int* blockSums) {
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
        if (tid >= shift) {
            shared_data[tid + blockDim.x] = shared_data[tid - shift] + shared_data[tid];
        }
        __syncthreads();
        if (tid >= shift) {
            shared_data[tid] = shared_data[tid + blockDim.x];
        }
        __syncthreads();

        // shift = 1
        // [1, 2, 3, 4] -> [1, 1 + 2, 2 + 3, 3 + 4] = [1, 3, 5, 7]
        // shift = 2
        // [1, 3, 5, 7] -> [1, 3, 1 + 5, 3 + 7] = [1, 3, 6, 10]
        
    }
    
    // block_idx = 0 -> [a0, a1, a2, a3]
    // block_idx = 1 -> [a4, a5, a6, a7]
    out_data[index] = shared_data[tid];
    if (tid == blockDim.x - 1) {
        if (blockIdx.x < gridDim.x - 1) {
            blockSums[blockIdx.x + 1] = out_data[index];
        } else {
            blockSums[0] = 0;
        }
    }

    //__syncthreads();

    // out_data[block_idx == 0] = [1, 3, 6, 10]

    // out_data[block_idx == 1] = [5, 11, 18, 26]

}

__global__ void InjectSums(int* elements, int* prefix_sums) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    elements[tid] += prefix_sums[blockIdx.x];
}


int main() {
    const int block_size = 1024;

    const int array_size = 1 << 20;
    int* h_array;
    // cudaMallocHost(&h_array, sizeof(int) * array_size);
    int* h_array = new int[array_size];
    for (int i = 0; i < array_size; ++i) {
        h_array[i] = 1;
    }

    int* h_localscan = new int[array_size];

    cudaEvent_t start;
    cudaEvent_t stop;

    // Creating event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);
    int* d_array;
    cudaMalloc(&d_array, sizeof(int) * array_size);

    cudaMemcpy(d_array, h_array, sizeof(int) * array_size, cudaMemcpyHostToDevice);


    int num_blocks = array_size / block_size;

    int* d_localscan;
    cudaMalloc(&d_localscan, sizeof(int) * array_size);

    int* d_blocksums;
    cudaMalloc(&d_blocksums, sizeof(int) * num_blocks);

    int* d_blocksums_prefix;
    cudaMalloc(&d_blocksums_prefix, sizeof(int) * num_blocks);

    int* d_block_block_sums;
    cudaMalloc(&d_block_block_sums, sizeof(int));

    // int *h_blocksums = new int[num_blocks];
    // int *h_blocksprefix = new int[num_blocks];

    Scan<<<num_blocks, block_size, sizeof(int) * block_size * 2>>>(
        d_array, d_localscan,
        d_blocksums
    );

    Scan<<<num_blocks / block_size, block_size, sizeof(int) * block_size * 2>>>(
        d_blocksums, 
        d_blocksums_prefix,
        d_block_block_sums
    );

    InjectSums<<<num_blocks, block_size>>>(
        d_localscan,
        d_blocksums_prefix
    );

    cudaMemcpy(h_localscan, d_localscan, sizeof(int) * array_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    // cudaMemcpy(h_blocksprefix, d_blocksums_prefix, sizeof(int) * num_blocks, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << " elapsed" << std::endl;

    // for (int i = 0; i < 1048576; ++i) {
    //     std::cout << h_localscan[i] << std::endl;
    // }

    delete[] h_array;
    delete[] h_localscan;

    // delete[] h_blocksums;
}
