#include <iostream>
#include <cmath>
#include <cstdio>

__global__
void add(int n, float* x, float* y, float* z) {
    int double_tid = threadIdx.x + 2 * blockDim.x * blockIdx.x;
    z[double_tid] = 2.0f * x[double_tid] + y[double_tid];
    z[double_tid + blockDim.x] = 2.0f * x[double_tid + blockDim.x] + y[double_tid + blockDim.x]; 
}

__global__
void stupid_add(int n, float* x, float* y, float* z) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int double_index = 2 * index;
    int double_plus_one = 2 * index + 1;
    z[double_index] = 2.0f * x[double_index] + y[double_index];
    z[double_plus_one] = 2.0f * x[double_plus_one] + y[double_plus_one];
}


int main() {
	int N = 1 << 28;
	size_t size = N * sizeof(float);
	float *x = (float*)malloc(size);
	float *y = (float*)malloc(size);
    float *z = (float*)malloc(size);

	float *d_x, *d_y, *d_z;

	cudaMalloc(&d_x, size);
	cudaMalloc(&d_y, size);
    cudaMalloc(&d_z, size);


	for (int i = 0; i < N; ++i) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}


	cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

	int blockSize = 256;

	int numBlocks = (N + blockSize - 1) / blockSize;

    cudaEvent_t start;
    cudaEvent_t stop;

    // Creating event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);
	add<<<numBlocks / 2, blockSize>>>(N, d_x, d_y, d_z);
    cudaEventRecord(stop);

	cudaMemcpy(z, d_z, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    cudaEvent_t start1;
    cudaEvent_t stop1;
    
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1);
    stupid_add<<<numBlocks / 2, blockSize>>>(N, d_x, d_y, d_z);
    cudaEventRecord(stop1);

    cudaMemcpy(z, d_z, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop1);
    
    float milliseconds = 0;
    float milliseconds_stupid = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventElapsedTime(&milliseconds_stupid, start1, stop1);
    std::cout << milliseconds << " elapsed normal" << std::endl;
    std::cout << milliseconds_stupid << " elapsed stupid" << std::endl;

	cudaFree(d_x);
	cudaFree(d_y);
    cudaFree(d_z);
	free(x);
	free(y);
    free(z);
	return 0;
}
