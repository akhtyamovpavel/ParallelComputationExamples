import pycuda.driver as cuda
import pycuda.autoinit

from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

import numpy as np


kernel_code = SourceModule(
"""
__global__ void AddTwoArrays(int N, float* x, float* y, float* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        result[i] = x[i] + y[i];
    }
}
"""
)

func = kernel_code.get_function('AddTwoArrays')


def main():
    start = cuda.Event()
    end = cuda.Event()
    
    array_size = int(2 ** 22)
    number_of_blocks = array_size // 256
    
    x = np.asarray(np.random.randn(array_size), np.float32)
    y = np.random.randn(array_size).astype(np.float32)

    z = np.zeros(array_size, dtype=np.float32)

    x_gpu = gpuarray.to_gpu(x)
    y_gpu = gpuarray.to_gpu(y)

    z_gpu = gpuarray.to_gpu(z)


    start.record()
    
    func(np.uint32(array_size), x_gpu.gpudata, y_gpu.gpudata, z_gpu.gpudata, block=(256, 1, 1), grid=(number_of_blocks, 1, 1));
    end.record()

    end.synchronize()

    secs = start.time_till(end) * 1e-3

    print(secs)

    print(x[:10])
    print(y[:10])

    print(z_gpu.get()[:10])


if __name__ == '__main__':
    main()
