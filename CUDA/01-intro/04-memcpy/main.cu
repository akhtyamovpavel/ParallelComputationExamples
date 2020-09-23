#include <iostream>
#include <cmath>
#include <cstdio>


__constant__ int device_n;


__global__
void add(int n, float* x, float* y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    
//    if (threadIdx.x == 0) {
//        printf("%d %d %d\n", blockIdx.x, gridDim.x, blockDim.x);
//    }

	for (int i = index; i < n; i += stride) {
		y[i] = x[i] + y[i];
	}	
}


int main() {
	int N = 1 << 28;
	size_t size = N * sizeof(float);
	float *h_x = (float*)malloc(size);
	float *h_y = (float*)malloc(size);

	float *d_x, *d_y;

	cudaMalloc(&d_x, size);
	cudaMalloc(&d_y, size);


	for (int i = 0; i < N; ++i) {
		h_x[i] = 1.0f;
		h_y[i] = 2.0f;
	}


	cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

	int blockSize = 256;

	int numBlocks = (N + blockSize - 1) / blockSize;

	add<<<numBlocks, blockSize>>>(N, d_x, d_y);

	// cudaDeviceSynchronize();	
	cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(h_y[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	cudaFree(d_x);
	cudaFree(d_y);
	free(h_x);
	free(h_y);
	return 0;
}
