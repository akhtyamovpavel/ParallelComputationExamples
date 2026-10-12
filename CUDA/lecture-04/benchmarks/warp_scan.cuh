#pragma once
#include <cuda_runtime.h>

namespace warp_scan {

// Every lane in a full 32-thread warp calls every shuffle.
__device__ __forceinline__ int warp_inclusive_scan(int value) {
    constexpr unsigned int full_mask = 0xffffffffu;
    int lane = threadIdx.x % 32;
    for (int offset = 1; offset < 32; offset *= 2) {
        int earlier = __shfl_up_sync(full_mask, value, offset);
        if (lane >= offset) {
            value += earlier;
        }
    }
    return value;
}

// Contract: blockDim.x is a multiple of 32, between 32 and 1024.
// Missing input elements become zero; no thread returns before the barriers.
__global__ void block_exclusive_scan(const int* input, int* output, int size) {
    __shared__ int warp_sums[32];
    __shared__ int warp_offsets[32];
    int local = threadIdx.x;
    int lane = local % 32;
    int warp = local / 32;
    int warp_count = blockDim.x / 32;
    int index = blockIdx.x * blockDim.x + local;
    int value = 0;
    if (index < size) {
        value = input[index];
    }
    int inclusive = warp_inclusive_scan(value);
    if (lane == 31) {
        warp_sums[warp] = inclusive;
    }
    __syncthreads();

    if (warp == 0) {
        int sum = 0;
        if (lane < warp_count) {
            sum = warp_sums[lane];
        }
        int scanned = warp_inclusive_scan(sum);
        warp_offsets[lane] = scanned - sum;
    }
    __syncthreads();

    if (index < size) {
        output[index] = warp_offsets[warp] + inclusive - value;
    }
}

}
