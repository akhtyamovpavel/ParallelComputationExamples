#include <iostream>
#include <cmath>

__global__
void add(int n, float* x, float* y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i += stride) {
		y[i] = x[i] + y[i];
	}	
}


int main() {
	int N = 1 << 28;
	size_t size = N * sizeof(float);
	float *x = (float*)malloc(size);
	float *y = (float*)malloc(size);

	float *d_x, *d_y;

	cudaMalloc(&d_x, size);
	cudaMalloc(&d_y, size);


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
	add<<<numBlocks, blockSize>>>(N, d_x, d_y);

    // cudaEventRecord(stop);

	cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << " elapsed" << std::endl;

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
	return 0;
}
