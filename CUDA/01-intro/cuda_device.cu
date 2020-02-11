#include <cstdio>

int main() {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Device name: %s\n", deviceProp.name);
	printf("Total global memory: %ld\n", deviceProp.totalGlobalMem);
	printf("Shared memory per block: %ld\n", deviceProp.sharedMemPerBlock);
	printf("Registers per block: %ld\n", deviceProp.regsPerBlock);
	printf("Warp size: %ld\n", deviceProp.warpSize);
	printf("Memory pitch: %ld\n", deviceProp.memPitch);
	printf("Max threads per block: %ld\n", deviceProp.maxThreadsPerBlock);

	printf("Max threads dimensions: x = %ld, y = %ld, z = %ld\n",
		deviceProp.maxThreadsDim[0],
		deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);

	printf("Max grid size: x = %ld, y = %ld, z = %ld\n",
		deviceProp.maxGridSize[0],
		deviceProp.maxGridSize[1],
		deviceProp.maxGridSize[2]);

	printf("Clock rate: %ld\n", deviceProp.clockRate);
	printf("Total constant memory: %ld\n", deviceProp.totalConstMem);
	printf("Compute capability: %ld.%ld\n", deviceProp.major, deviceProp.minor);
	printf("Texture alignment: %ld\n", deviceProp.textureAlignment);
	printf("Device overlap: %ld\n", deviceProp.deviceOverlap);
	printf("Multiprocessor count: %ld\n", deviceProp.multiProcessorCount);

	printf("Kernel execution timeout enabled: %s\n",
		deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
	scanf("");
}
