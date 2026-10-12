#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

constexpr int threads = 256;
constexpr int tile_size = 2 * threads;

__global__ void make_mask(const int* input, int* mask, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        mask[index] = input[index] > 3;
    }
}

// Two elements per thread. Every thread reaches every barrier, including the tail.
__global__ void scan_blocks(const int* input, int* output, int* sums, int size) {
    __shared__ int values[tile_size];
    int local = threadIdx.x;
    int first = blockIdx.x * tile_size + local;
    int second = first + threads;
    values[local] = 0;
    values[local + threads] = 0;
    if (first < size) {
        values[local] = input[first];
    }
    if (second < size) {
        values[local + threads] = input[second];
    }
    __syncthreads();

    for (int offset = 1; offset < tile_size; offset *= 2) {
        int right = (local + 1) * 2 * offset - 1;
        if (right < tile_size) {
            values[right] += values[right - offset];
        }
        __syncthreads();
    }
    if (local == 0) {
        sums[blockIdx.x] = values[tile_size - 1];
        values[tile_size - 1] = 0;
    }
    __syncthreads();

    for (int offset = tile_size / 2; offset > 0; offset /= 2) {
        int right = (local + 1) * 2 * offset - 1;
        if (right < tile_size) {
            int left_sum = values[right - offset];
            values[right - offset] = values[right];
            values[right] += left_sum;
        }
        __syncthreads();
    }
    if (first < size) {
        output[first] = values[local];
    }
    if (second < size) {
        output[second] = values[local + threads];
    }
}

__global__ void add_offsets(int* output, const int* offsets, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] += offsets[index / tile_size];
    }
}

__global__ void scatter(const int* input, const int* mask, const int* prefix,
                        int* output, int* count, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size && mask[index] != 0) {
        output[prefix[index]] = input[index];
    }
    if (index == size - 1) {
        *count = prefix[index] + mask[index];
    }
}

struct ScanLevel {
    int* sums = nullptr;
    int* offsets = nullptr;
};

static std::vector<ScanLevel> make_workspace(int size) {
    std::vector<ScanLevel> levels;
    while (size > 0) {
        int blocks = (size + tile_size - 1) / tile_size;
        ScanLevel level;
        check(cudaMalloc(&level.sums, blocks * sizeof(int)));
        check(cudaMalloc(&level.offsets, blocks * sizeof(int)));
        levels.push_back(level);
        if (blocks == 1) {
            break;
        }
        size = blocks;
    }
    return levels;
}

static void exclusive_scan(const int* input, int* output, int size, int depth,
                           std::vector<ScanLevel>& levels, cudaStream_t stream) {
    int blocks = (size + tile_size - 1) / tile_size;
    ScanLevel& level = levels[depth];
    scan_blocks<<<blocks, threads, 0, stream>>>(input, output, level.sums, size);
    check(cudaGetLastError());
    if (blocks > 1) {
        exclusive_scan(level.sums, level.offsets, blocks, depth + 1, levels, stream);
        int grid = (size + threads - 1) / threads;
        add_offsets<<<grid, threads, 0, stream>>>(output, level.offsets, size);
        check(cudaGetLastError());
    }
}

static void fill_input(int* input, int size, const std::string& pattern) {
    const int sample[] = {3, 1, 7, 0, 4, 1, 6, 3};
    for (int index = 0; index < size; ++index) {
        if (pattern == "none") {
            input[index] = 3;
        } else if (pattern == "all") {
            input[index] = index % 17 + 4;
        } else if (pattern == "sample") {
            input[index] = sample[index % 8];
        } else {
            input[index] = index % 19 - 9;
        }
    }
}

int main(int argc, char** argv) {
    int size = 1048579;
    int warmup = 5;
    int repetitions = 3;
    std::string pattern = "mixed";
    for (int argument = 1; argument < argc; argument += 2) {
        if (argument + 1 == argc) {
            return 2;
        }
        std::string name = argv[argument];
        std::string value = argv[argument + 1];
        if (name == "--size") {
            size = std::stoi(value);
        } else if (name == "--warmup") {
            warmup = std::stoi(value);
        } else if (name == "--repetitions") {
            repetitions = std::stoi(value);
        } else if (name == "--pattern") {
            pattern = value;
        } else {
            return 2;
        }
    }
    if (size < 0 || size > (1 << 26) || warmup < 0 || repetitions < 1) {
        std::cerr << "Require 0 <= size <= 2^26, warmup >= 0, repetitions >= 1\n";
        return 2;
    }
    if (pattern != "mixed" && pattern != "all" && pattern != "none" && pattern != "sample") {
        return 2;
    }

    cudaDeviceProp property;
    check(cudaGetDeviceProperties(&property, 0));
    std::cout << "{\"record\":\"device\",\"name\":\"" << property.name;
    std::cout << "\",\"async_engine_count\":" << property.asyncEngineCount;
    std::cout << ",\"concurrent_kernels\":" << property.concurrentKernels << "}\n";

    size_t bytes = std::max(size, 1) * sizeof(int);
    int* host_input = nullptr;
    int* host_output = nullptr;
    int* host_prefix = nullptr;
    int* host_count = nullptr;
    check(cudaMallocHost(&host_input, bytes));
    check(cudaMallocHost(&host_output, bytes));
    check(cudaMallocHost(&host_prefix, bytes));
    check(cudaMallocHost(&host_count, sizeof(int)));
    fill_input(host_input, size, pattern);
    std::vector<int> reference;
    std::vector<int> reference_prefix(size);
    for (int index = 0; index < size; ++index) {
        reference_prefix[index] = reference.size();
        if (host_input[index] > 3) {
            reference.push_back(host_input[index]);
        }
    }

    int* input = nullptr;
    int* mask = nullptr;
    int* prefix = nullptr;
    int* output = nullptr;
    int* count = nullptr;
    check(cudaMalloc(&input, bytes));
    check(cudaMalloc(&mask, bytes));
    check(cudaMalloc(&prefix, bytes));
    check(cudaMalloc(&output, bytes));
    check(cudaMalloc(&count, sizeof(int)));
    auto workspace = make_workspace(size);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t start;
    cudaEvent_t stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    std::vector<float> times;

    for (int iteration = -warmup; iteration < repetitions; ++iteration) {
        check(cudaEventRecord(start, stream));
        if (size > 0) {
            check(cudaMemcpyAsync(input, host_input, bytes, cudaMemcpyHostToDevice, stream));
            int grid = (size + threads - 1) / threads;
            make_mask<<<grid, threads, 0, stream>>>(input, mask, size);
            check(cudaGetLastError());
            exclusive_scan(mask, prefix, size, 0, workspace, stream);
            scatter<<<grid, threads, 0, stream>>>(input, mask, prefix, output, count, size);
            check(cudaGetLastError());
            check(cudaMemcpyAsync(host_output, output, bytes, cudaMemcpyDeviceToHost, stream));
        } else {
            check(cudaMemsetAsync(count, 0, sizeof(int), stream));
        }
        check(cudaMemcpyAsync(host_count, count, sizeof(int), cudaMemcpyDeviceToHost, stream));
        check(cudaEventRecord(stop, stream));
        check(cudaEventSynchronize(stop));
        float milliseconds = 0;
        check(cudaEventElapsedTime(&milliseconds, start, stop));
        if (iteration >= 0) {
            times.push_back(milliseconds);
        }
        if (*host_count != static_cast<int>(reference.size())) {
            std::cerr << "Compaction count mismatch\n";
            return 1;
        }
        if (!std::equal(reference.begin(), reference.end(), host_output)) {
            std::cerr << "Compaction values or stable order mismatch\n";
            return 1;
        }
    }
    if (size > 0) {
        check(cudaMemcpyAsync(host_prefix, prefix, bytes, cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        if (!std::equal(reference_prefix.begin(), reference_prefix.end(), host_prefix)) {
            std::cerr << "Global exclusive prefixes mismatch\n";
            return 1;
        }
    }

    std::cout << "{\"record\":\"measurement\",\"size\":" << size;
    std::cout << ",\"pattern\":\"" << pattern << "\",\"count\":" << *host_count;
    std::cout << ",\"correctness\":true,\"prefix_correctness\":true,\"warmup\":" << warmup;
    std::cout << ",\"milliseconds\":[";
    for (int index = 0; index < repetitions; ++index) {
        if (index > 0) {
            std::cout << ',';
        }
        std::cout << times[index];
    }
    std::cout << "]}\n";
    if (size <= 16) {
        std::cout << "input:";
        for (int index = 0; index < size; ++index) {
            std::cout << ' ' << host_input[index];
        }
        std::cout << "\nprefix:";
        for (int index = 0; index < size; ++index) {
            std::cout << ' ' << host_prefix[index];
        }
        std::cout << "\noutput:";
        for (int index = 0; index < *host_count; ++index) {
            std::cout << ' ' << host_output[index];
        }
        std::cout << '\n';
    }

    for (auto& level : workspace) {
        check(cudaFree(level.sums));
        check(cudaFree(level.offsets));
    }
    check(cudaFree(input));
    check(cudaFree(mask));
    check(cudaFree(prefix));
    check(cudaFree(output));
    check(cudaFree(count));
    check(cudaFreeHost(host_input));
    check(cudaFreeHost(host_output));
    check(cudaFreeHost(host_prefix));
    check(cudaFreeHost(host_count));
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaStreamDestroy(stream));
}
