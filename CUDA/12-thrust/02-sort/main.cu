#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// thrust::sort на device_vector — одна строчка. Внутри это radix sort,
// не quicksort (поэтому с ДЗ tasks/03-quick-sort это не пересекается —
// там студенты сами реализуют quicksort через prefix scan).
//
// Для входных данных используем thrust::tabulate с функтором-хешем,
// чтобы получить псевдослучайную перестановку без отдельного cuRAND.

struct hash_fn {
    __host__ __device__ int operator()(int i) const {
        unsigned x = static_cast<unsigned>(i);
        x = x * 2654435761u + 1013904223u;
        x ^= (x >> 16);
        x = x * 2654435761u;
        return static_cast<int>(x & 0x7fffffffu);
    }
};

int main() {
    const int N = 1 << 22;

    thrust::device_vector<int> d(N);
    thrust::tabulate(d.begin(), d.end(), hash_fn());

    cudaEvent_t t0;
    cudaEvent_t t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    thrust::sort(d.begin(), d.end());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    bool sorted = thrust::is_sorted(d.begin(), d.end());
    double mkeys_per_s = N / (ms * 1e-3) / 1e6;

    std::cout << std::left << std::setw(24) << "thrust::sort(int)"
              << " N=" << N
              << "  time=" << std::right << std::setw(8)
              << std::fixed << std::setprecision(3) << ms << " ms"
              << "  " << std::right << std::setw(7)
              << std::fixed << std::setprecision(2) << mkeys_per_s << " Mkeys/s"
              << "  sorted=" << (sorted ? "yes" : "no") << std::endl;

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return 0;
}
