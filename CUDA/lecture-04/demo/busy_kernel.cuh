// Exact author kernel; SHA and original source recorded by run_lecture_demos.py.
__global__ void busy_kernel(const float* in, float* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float x = in[i];
        for (int k = 0; k < iters; ++k) {
            x = x * 1.0001f + 0.5f;
        }
        out[i] = x;
    }
}

