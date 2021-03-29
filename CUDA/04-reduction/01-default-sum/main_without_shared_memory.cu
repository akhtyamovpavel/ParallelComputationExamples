#include <iostream>

__global__ void Reduce(int* in_data, int* tmp_data, int* out_data) {
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    tmp_data[index] = in_data[index];

    // __syncthreads();    
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            tmp_data[index] += tmp_data[index + s];
        }
        // __syncthreads();
    }

    if (tid == 0) {
        out_data[blockIdx.x] = tmp_data[index];
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
    int* tmp_array;
    cudaMalloc(&d_array, sizeof(int) * array_size);
    cudaMalloc(&tmp_array, sizeof(int) * array_size);

    cudaMemcpy(d_array, h_array, sizeof(int) * array_size, cudaMemcpyHostToDevice);

    int num_blocks = array_size / block_size;

    int* d_blocksum;
    cudaMalloc(&d_blocksum, sizeof(int) * num_blocks);
    int* h_blocksum = new int[num_blocks];

    cudaEvent_t start;
    cudaEvent_t stop;

    // Creating event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);

    Reduce<<<num_blocks, block_size, sizeof(int) * block_size>>>(d_array, tmp_array, d_blocksum);

    cudaEventRecord(stop);

    cudaMemcpy(h_blocksum, d_blocksum, sizeof(int) * num_blocks, cudaMemcpyDeviceToHost);

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
