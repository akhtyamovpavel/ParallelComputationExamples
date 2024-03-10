#include <iostream>

#define BLOCK_SIZE 1024


void FillMatrix(float* matrix, int height, int width) {
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (i == j) {
				matrix[i * width + j] = 1;
			} else {
				matrix[i * width + j] = 0;
			}
		}
	}
}

void PrintMatrix(float *matrix, int height, int width) {

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (i == j) {
				std::cout << i << " " << j << " " << matrix[i * width + j] << "\n";
			}
		}
	}
}


__global__
void MatrixMul(float* A, float* B, float* C, int mid_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int height = blockDim.x * gridDim.x;
    int width = blockDim.y * gridDim.y;

    C[i * width + j] = .0f;

    for (int k = 0; k < mid_size; ++k) {
        C[i * width + j] += A[i * mid_size + k] * B[j * width + k];
    }
}


int main() {

	float *h_A;
	float *h_B;
	float *h_C;
	// h_A 1024 * 1024,
	// h_B 1024 * 1024
	// h_C 1024 * 1024

	h_A = new float[1024 * 1024];
	h_B = new float[1024 * 1024];
	h_C = new float[1024 * 1024];

	FillMatrix(h_A, 1024, 1024);
	FillMatrix(h_B, 1024, 1024);

    // PrintMatrix(h_A, 1024, 1024);


	float* d_A;
	float* d_B;
	float* d_C;

	cudaMalloc(&d_A, sizeof(float) * 1024 * 1024);
	cudaMalloc(&d_B, sizeof(float) * 1024 * 1024);
	cudaMalloc(&d_C, sizeof(float) * 1024 * 1024);

    cudaMemcpy(d_A, h_A, sizeof(float) * 1024 * 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * 1024 * 1024, cudaMemcpyHostToDevice);

    // kernel call
    dim3 num_blocks(32, 32);
    dim3 block_size(32, 32);

    MatrixMul<<<num_blocks, block_size>>>(d_A, d_B, d_C, 1024);

    cudaMemcpy(h_C, d_C, sizeof(float) * 1024 * 1024, cudaMemcpyDeviceToHost);
    // PrintMatrix(h_C, 1024, 1024);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	return 0;
}
