#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "warp_scan.cuh"

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    int size = 1024;
    int block = 1024;
    std::string pattern = "ones";
    try {
        for (int argument = 1; argument < argc; argument += 2) {
            if (argument + 1 == argc) {
                throw std::invalid_argument("A value is missing");
            }
            std::string name = argv[argument];
            std::string value = argv[argument + 1];
            if (name == "--size") {
                size = std::stoi(value);
            } else if (name == "--block") {
                block = std::stoi(value);
            } else if (name == "--pattern") {
                pattern = value;
            } else {
                throw std::invalid_argument("Unknown argument");
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    if (size < 0 || size > 16777216 || block < 32 || block > 1024 || block % 32 != 0) {
        std::cerr << "Require 0 <= N <= 16777216 and 32 <= B <= 1024, B divisible by 32\n";
        return 2;
    }
    if (pattern != "ones" && pattern != "signed") {
        std::cerr << "Use --pattern ones or --pattern signed\n";
        return 2;
    }
    std::vector<int> host_input(size, 1);
    std::vector<int> host_output(size);
    if (pattern == "signed") {
        for (int index = 0; index < size; ++index) {
            host_input[index] = (index * 17 + index / 29) % 23 - 11;
        }
    }
    // Empty input performs no allocations and launches no kernel.
    if (size > 0) {
        size_t bytes = static_cast<size_t>(size) * sizeof(int);
        int* input = nullptr;
        int* output = nullptr;
        check(cudaMalloc(&input, bytes));
        check(cudaMalloc(&output, bytes));
        check(cudaMemcpy(input, host_input.data(), bytes, cudaMemcpyHostToDevice));
        int grid = (size + block - 1) / block;
        warp_scan::block_exclusive_scan<<<grid, block>>>(input, output, size);
        check(cudaGetLastError());
        check(cudaMemcpy(host_output.data(), output, bytes, cudaMemcpyDeviceToHost));
        check(cudaFree(input));
        check(cudaFree(output));
    }
    int prefix = 0;
    for (int index = 0; index < size; ++index) {
        if (index % block == 0) {
            prefix = 0;
        }
        if (host_output[index] != prefix) {
            std::cerr << "Local exclusive prefix mismatch at " << index;
            std::cerr << ": expected " << prefix << ", got " << host_output[index] << '\n';
            return 1;
        }
        prefix += host_input[index];
    }
    std::cout << "{\"size\":" << size << ",\"block\":" << block;
    std::cout << ",\"pattern\":\"" << pattern << "\",\"correctness\":true";
    std::cout << ",\"checked_prefixes\":" << size;
    std::cout << ",\"kernel_launched\":" << (size > 0 ? "true" : "false");
    std::cout << ",\"output_preview\":[";
    bool first = true;
    std::vector<int> preview = {0, 1, 30, 31, 32, 33, 241, 511, 512, 1023, 1024, size - 1};
    std::sort(preview.begin(), preview.end());
    preview.erase(std::unique(preview.begin(), preview.end()), preview.end());
    for (int index : preview) {
        if (index < 0 || index >= size) {
            continue;
        }
        if (!first) {
            std::cout << ',';
        }
        std::cout << "{\"index\":" << index << ",\"value\":" << host_output[index] << '}';
        first = false;
    }
    std::cout << "]}\n";
}
