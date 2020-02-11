#include<cstdio>

using namespace std;

__global__ void add(const int *a, const int *b, int *c)
{
	int i = threadIdx.x;
	c[i] = a[i] * *b;
}

int main(void)
{
	int count = 100;
	int size = sizeof(int) * count;
	int *cpu_a = (int *)malloc(size);	int *gpu_a; cudaMalloc((void**)&gpu_a, size);
	int  cpu_b = 5;						int *gpu_b; cudaMalloc((void**)&gpu_b, sizeof(int));
	int *cpu_c = (int *)malloc(size);	int *gpu_c; cudaMalloc((void**)&gpu_c, size);

	for(int i=0; i<count; i++) cpu_a[i]=i;

	cudaMemcpy(gpu_a,  cpu_a, size,			cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, &cpu_b, sizeof(int),	cudaMemcpyHostToDevice);

	add<<<1, count>>>(gpu_a, gpu_b, gpu_c);

	cudaMemcpy(cpu_c, gpu_c, size, cudaMemcpyDeviceToHost);

	for(int i=0; i<count; i++)
		printf("%d * %d = %dn", cpu_a[i], cpu_b, cpu_c[i]);
	free(cpu_a);	cudaFree(gpu_a);
					cudaFree(gpu_b);
	free(cpu_c);	cudaFree(gpu_c);
}

