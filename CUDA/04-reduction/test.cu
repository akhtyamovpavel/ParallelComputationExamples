#include <stdio.h>
#include <stdlib.h>

__global__ void kernel() {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  printf("Hello from thread.\n");
}



int main(){
  int host_a, host_b, host_c;
  int *dev_a, *dev_b, *dev_c;

  int size = sizeof (int);

  cudaMalloc((void**) &dev_a, size);
  cudaMalloc((void**) &dev_b, size);
  cudaMalloc((void**) &dev_c, size);

  host_a = 2;
  host_b = 7;

  cudaMemcpy(dev_a, &host_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, &host_b, size, cudaMemcpyHostToDevice);

  kernel <<< 1, 1 >>> ();

  cudaDeviceSynchronize();

  cudaMemcpy(&host_c, dev_c, size, cudaMemcpyDeviceToHost);

  printf("C = %d \n", host_c);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  printf("Hello, CUDA! \n");
}
