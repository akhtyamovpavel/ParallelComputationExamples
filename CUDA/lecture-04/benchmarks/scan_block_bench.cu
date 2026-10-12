#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "scan_block_kernels.cuh"
#include "warp_scan.cuh"

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

static void launch(const std::string& variant, int* input, int* output,
                   int size, int block, cudaStream_t stream) {
    int grid = (size + block - 1) / block;
    if (variant == "hillis_steele") {
        original_hillis_steele::Scan<<<grid, block, 2 * block * sizeof(int), stream>>>(input, output);
    } else if (variant == "warp_shuffle") {
        warp_scan::block_exclusive_scan<<<grid, block, 0, stream>>>(input, output, size);
    } else {
        original_blelloch::Scan<<<grid, block, block * sizeof(int), stream>>>(input, output);
    }
    check(cudaGetLastError());
}

static void validate(const std::vector<int>& input, const std::vector<int>& output, int block) {
    int prefix = 0;
    for (size_t index = 0; index < input.size(); ++index) {
        if (index % block == 0) {
            prefix = 0;
        }
        if (output[index] != prefix) {
            std::cerr << "Local exclusive prefix mismatch at " << index;
            std::cerr << ": expected " << prefix << ", got " << output[index] << '\n';
            std::exit(1);
        }
        prefix += input[index];
    }
}

int main(int argc, char** argv) {
    int size = 1 << 22;
    int warmup = 10;
    int repetitions = 5;
    int batch = 25;
    std::string selected_block = "all";
    std::string selected_variant = "all";
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
        } else if (name == "--batch") {
            batch = std::stoi(value);
        } else if (name == "--block") {
            selected_block = value;
        } else if (name == "--variant") {
            selected_variant = value;
        } else {
            return 2;
        }
    }
    if (size < 1 || size > (1 << 24) || (selected_variant != "warp_shuffle" && size % 1024 != 0)) {
        std::cerr << "Require positive N <= 16777216; author variants also need N divisible by 1024\n";
        return 2;
    }
    if (warmup < 0 || repetitions < 1 || batch < 1) {
        return 2;
    }
    if (selected_variant != "all" && selected_variant != "hillis_steele" && selected_variant != "blelloch" && selected_variant != "warp_shuffle") {
        return 2;
    }
    std::vector<int> blocks = {32, 64, 128, 256, 512, 1024};
    if (selected_block != "all") {
        int chosen = std::stoi(selected_block);
        if (std::find(blocks.begin(), blocks.end(), chosen) == blocks.end()) {
            return 2;
        }
        blocks = {chosen};
    }
    cudaDeviceProp property;
    check(cudaGetDeviceProperties(&property, 0));
    std::cout << "{\"record\":\"device\",\"name\":\"" << property.name;
    std::cout << "\",\"multiprocessors\":" << property.multiProcessorCount;
    std::cout << ",\"warp_size\":" << property.warpSize << "}\n";
    std::vector<int> host_input(size);
    std::vector<int> host_output(size);
    for (int index = 0; index < size; ++index) {
        host_input[index] = (index * 17 + index / 29) % 23 - 11;
    }
    size_t bytes = static_cast<size_t>(size) * sizeof(int);
    int* input = nullptr;
    int* output = nullptr;
    check(cudaMalloc(&input, bytes));
    check(cudaMalloc(&output, bytes));
    check(cudaMemcpy(input, host_input.data(), bytes, cudaMemcpyHostToDevice));
    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t stop;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    std::vector<std::string> variants = {"hillis_steele", "blelloch", "warp_shuffle"};
    for (int block : blocks) {
        for (const auto& variant : variants) {
            if (selected_variant != "all" && variant != selected_variant) {
                continue;
            }
            std::vector<float> batch_times;
            for (int iteration = -warmup; iteration < repetitions; ++iteration) {
                check(cudaEventRecord(start, stream));
                for (int invocation = 0; invocation < batch; ++invocation) {
                    launch(variant, input, output, size, block, stream);
                }
                check(cudaEventRecord(stop, stream));
                check(cudaEventSynchronize(stop));
                float milliseconds = 0;
                check(cudaEventElapsedTime(&milliseconds, start, stop));
                if (iteration >= 0) {
                    batch_times.push_back(milliseconds);
                    check(cudaMemcpy(host_output.data(), output, bytes, cudaMemcpyDeviceToHost));
                    validate(host_input, host_output, block);
                }
            }
            std::cout << "{\"record\":\"measurement\",\"variant\":\"" << variant << "\"";
            std::cout << ",\"size\":" << size << ",\"block\":" << block;
            std::cout << ",\"grid\":" << (size + block - 1) / block << ",\"warmup_batches\":" << warmup;
            std::cout << ",\"kernels_per_batch\":" << batch << ",\"correctness\":true";
            std::cout << ",\"batch_milliseconds\":[";
            for (int index = 0; index < repetitions; ++index) {
                if (index > 0) {
                    std::cout << ',';
                }
                std::cout << batch_times[index];
            }
            std::cout << "],\"per_kernel_milliseconds\":[";
            for (int index = 0; index < repetitions; ++index) {
                if (index > 0) {
                    std::cout << ',';
                }
                std::cout << batch_times[index] / batch;
            }
            std::cout << "]}\n";
        }
    }
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaStreamDestroy(stream));
    check(cudaFree(input));
    check(cudaFree(output));
}
