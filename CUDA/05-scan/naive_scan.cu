#include <iostream>

__device__ void WarpReduce(volatile int* shared_data, int tid) {
    shared_data[tid] += shared_data[tid + 32];
    shared_data[tid] += shared_data[tid + 16];
    shared_data[tid] += shared_data[tid + 8];
    shared_data[tid] += shared_data[tid + 4];
    shared_data[tid] += shared_data[tid + 2];
    shared_data[tid] += shared_data[tid + 1];
}

__global__ void Reduce(int* in_data, int* out_data) {
    extern __shared__ int shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    shared_data[tid] = in_data[index] + in_data[index + blockDim.x];
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduce(shared_data, tid);
    }
    
    if (tid == 0) {
        out_data[blockIdx.x] = shared_data[0];
    }
}


int main() {
    const int block_size = 256;
    // __shared__ int shared_data[];

    const int array_size = 1 << 26;
    int* h_array = new int[array_size];
    for (int i = 0; i < array_size; ++i) {
        h_array[i] = 1;
    }

    int* output = new int[array_size];

    // int* d_array;
    // cudaMalloc(&d_array, sizeof(int) * array_size);

    // cudaMemcpy(d_array, h_array, sizeof(int) * array_size, cudaMemcpyHostToDevice);

    // int num_blocks = array_size / block_size / 2;

    // int* d_blocksum;
    // cudaMalloc(&d_blocksum, sizeof(int) * num_blocks);
    // int* h_blocksum = new int[num_blocks];

    cudaEvent_t start;
    cudaEvent_t stop;

    // Creating event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);

    output[0] = h_array[0];
    for (int i = 1; i < array_size; ++i) {
        output[i] = output[i - 1] + h_array[i];
    }

    // Reduce<<<num_blocks, block_size, sizeof(int) * block_size>>>(d_array, d_blocksum);

    cudaEventRecord(stop);

    // cudaMemcpy(h_blocksum, d_blocksum, sizeof(int) * num_blocks, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << " elapsed" << std::endl;

    std::cout << output[array_size - 1] << std::endl;

    delete[] h_array;
    delete[] output;


}
