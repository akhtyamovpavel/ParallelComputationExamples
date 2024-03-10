#include <iostream>


#define BLOCK_SIZE 1024
__global__
void Increment(int N, int* x, int d) {

    __shared__ int elements[1024];


    int index = blockIdx.x * blockDim.x + threadIdx.x;

    elements[threadIdx.x] = x[index];
    // __syncthreads();
    if (index >= N) {
        return;
    }
        
    for (int i = 1; i <= d; ++i) {
        elements[threadIdx.x] += i;
    }

    x[index] = elements[threadIdx.x];
}

__global__
void AddIncrement(int N, int* a, int* b, int d) {

    extern __shared__ int elements[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int shifted = threadIdx.x + BLOCK_SIZE;

    elements[threadIdx.x] = a[index];
    elements[shifted] = b[index];
    // __syncthreads();
    if (index >= N) {
        return;
    }
        
    for (int i = 1; i <= d; ++i) {
        elements[threadIdx.x] += elements[shifted] + i;
    }

    a[index] = elements[threadIdx.x];
}

__global__
void AddIncrementWoShared(int N, int* a, int* b, int d) {


    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // __syncthreads();
    if (index >= N) {
        return;
    }
        
    for (int i = 1; i <= d; ++i) {
        a[index] += b[index] + i;
    }

}

int main(int argc, char** argv ) {
    int d = std::atoi(argv[1]);
    int N = 1 << 28;


    int* d_a;
    int* d_b;

    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N* sizeof(int));

    int* h_a = new int[N];
    int* h_b = new int[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    AddIncrement<<<
        N / BLOCK_SIZE,
        BLOCK_SIZE,
        2 * BLOCK_SIZE * sizeof(int),
        0
    >>>(
        N, d_a, d_b, d
    );
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << ms << std::endl;

    cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;
}

