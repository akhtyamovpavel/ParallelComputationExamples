#include <iostream>

__device__ void WarpReduce(volatile int* shared_data, int tid) {
    
    shared_data[tid] += shared_data[tid + 32];
	int val = shared_data[tid];
	val += __shfl_down_sync(-1, val, 16);
	val += __shfl_down_sync(-1, val, 8);
	val += __shfl_down_sync(-1, val, 4);
	val += __shfl_down_sync(-1, val, 2);
	val += __shfl_down_sync(-1, val, 1);
	shared_data[tid] = val;
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

    const int array_size = 1 << 22;
    int* h_array = new int[array_size];
    for (int i = 0; i < array_size; ++i) {
        h_array[i] = 1;
    }

    int* d_array;
    cudaMalloc(&d_array, sizeof(int) * array_size);

    cudaMemcpy(d_array, h_array, sizeof(int) * array_size, cudaMemcpyHostToDevice);

    int num_blocks = array_size / block_size / 2;

    int* d_blocksum;
    cudaMalloc(&d_blocksum, sizeof(int) * num_blocks);
    int* h_blocksum = new int[num_blocks];

    cudaEvent_t start;
    cudaEvent_t stop;

    // Creating event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);
    // 0

    Reduce<<<num_blocks, block_size, sizeof(int) * block_size>>>(d_array, d_blocksum);
    // device - some time - t1: time - t1
    // host - immediately: time - 0

    cudaEventRecord(stop);

    cudaMemcpy(h_blocksum, d_blocksum, sizeof(int) * num_blocks, cudaMemcpyDeviceToHost);

    // device - t2: time t1 + t2
    // host - t1 + t2 : time t1 + t2

    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << " elapsed" << std::endl;
    
    int sum = 0;
    for (int i = 0; i < num_blocks; ++i) {
        sum += h_blocksum[i];
    }

    std::cout << sum << std::endl;

    cudaFree(d_blocksum);
    cudaFree(d_array);
    delete[] h_array;
    delete[] h_blocksum;

}
