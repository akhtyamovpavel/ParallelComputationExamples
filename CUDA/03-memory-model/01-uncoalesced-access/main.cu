#include <iostream>
#include <cmath>

__global__
void add(int n, float* x, float* y, float* z) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i += stride) {
        int iindex = (i / 32) * 32 + (i) % 32; // call for coalescing 
		z[i] = 2.0f * x[iindex] + y[iindex];
	}	
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
	add<<<numBlocks, blockSize>>>(N, d_x, d_y, d_z);

    cudaEventRecord(stop);

	cudaMemcpy(z, d_z, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << " elapsed" << std::endl;

	cudaFree(d_x);
	cudaFree(d_y);
    cudaFree(d_z);
	free(x);
	free(y);
    free(z);
	return 0;
}
