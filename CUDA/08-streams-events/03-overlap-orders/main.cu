#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "busy_kernel.cuh"

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    int size = 1 << 24;
    int iterations = 80;
    int warmup = 5;
    int repetitions = 3;
    std::string selected = "all";
    for (int argument = 1; argument < argc; argument += 2) {
        if (argument + 1 == argc) {
            return 2;
        }
        std::string name = argv[argument];
        std::string value = argv[argument + 1];
        if (name == "--size") {
            size = std::stoi(value);
        } else if (name == "--iterations") {
            iterations = std::stoi(value);
        } else if (name == "--warmup") {
            warmup = std::stoi(value);
        } else if (name == "--repetitions") {
            repetitions = std::stoi(value);
        } else if (name == "--order") {
            selected = value;
        } else {
            return 2;
        }
    }
    if (size < 1 || size > (1 << 26) || iterations < 1 || iterations > 1000) {
        return 2;
    }
    if (warmup < 0 || repetitions < 1) {
        return 2;
    }
    if (selected != "all" && selected != "baseline" && selected != "interleaved" && selected != "batched") {
        return 2;
    }
    const int stream_count = 4;
    const int block = 256;
    size_t bytes = static_cast<size_t>(size) * sizeof(float);
    float* host_input = nullptr;
    float* host_output = nullptr;
    float* device_input = nullptr;
    float* device_output = nullptr;
    check(cudaMallocHost(&host_input, bytes));
    check(cudaMallocHost(&host_output, bytes));
    check(cudaMalloc(&device_input, bytes));
    check(cudaMalloc(&device_output, bytes));
    std::vector<float> reference(31);
    for (int index = 0; index < 31; ++index) {
        float value = 1.0f + index / 31.0f;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            value = std::fma(value, 1.0001f, 0.5f);
        }
        reference[index] = value;
    }
    for (int index = 0; index < size; ++index) {
        host_input[index] = 1.0f + (index % 31) / 31.0f;
    }

    cudaDeviceProp property;
    check(cudaGetDeviceProperties(&property, 0));
    std::cout << "{\"record\":\"device\",\"name\":\"" << property.name;
    std::cout << "\",\"async_engine_count\":" << property.asyncEngineCount;
    std::cout << ",\"concurrent_kernels\":" << property.concurrentKernels << "}\n";

    cudaStream_t coordinator;
    cudaStream_t workers[stream_count];
    cudaEvent_t done[stream_count];
    cudaEvent_t start;
    cudaEvent_t stop;
    check(cudaStreamCreateWithFlags(&coordinator, cudaStreamNonBlocking));
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    for (int index = 0; index < stream_count; ++index) {
        check(cudaStreamCreateWithFlags(&workers[index], cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&done[index], cudaEventDisableTiming));
    }
    const std::vector<std::string> orders = {"baseline", "interleaved", "batched"};
    for (const auto& order : orders) {
        if (selected != "all" && selected != order) {
            continue;
        }
        int active = stream_count;
        if (order == "baseline") {
            active = 1;
        }
        std::vector<float> times;
        std::vector<double> wall_times;
        for (int iteration = -warmup; iteration < repetitions; ++iteration) {
            auto wall_start = std::chrono::steady_clock::now();
            check(cudaEventRecord(start, coordinator));
            for (int index = 0; index < active; ++index) {
                check(cudaStreamWaitEvent(workers[index], start, 0));
            }
            auto copy_input = [&](int index) {
                int begin = static_cast<long long>(size) * index / active;
                int end = static_cast<long long>(size) * (index + 1) / active;
                size_t chunk_bytes = static_cast<size_t>(end - begin) * sizeof(float);
                if (end > begin) {
                    check(cudaMemcpyAsync(device_input + begin, host_input + begin, chunk_bytes, cudaMemcpyHostToDevice, workers[index]));
                }
            };
            auto compute = [&](int index) {
                int begin = static_cast<long long>(size) * index / active;
                int end = static_cast<long long>(size) * (index + 1) / active;
                if (end > begin) {
                    int grid = (end - begin + block - 1) / block;
                    busy_kernel<<<grid, block, 0, workers[index]>>>(device_input + begin, device_output + begin, end - begin, iterations);
                    check(cudaGetLastError());
                }
            };
            auto copy_output = [&](int index) {
                int begin = static_cast<long long>(size) * index / active;
                int end = static_cast<long long>(size) * (index + 1) / active;
                size_t chunk_bytes = static_cast<size_t>(end - begin) * sizeof(float);
                if (end > begin) {
                    check(cudaMemcpyAsync(host_output + begin, device_output + begin, chunk_bytes, cudaMemcpyDeviceToHost, workers[index]));
                }
            };
            if (order == "batched") {
                for (int index = 0; index < active; ++index) {
                    copy_input(index);
                }
                for (int index = 0; index < active; ++index) {
                    compute(index);
                }
                for (int index = 0; index < active; ++index) {
                    copy_output(index);
                }
            } else {
                for (int index = 0; index < active; ++index) {
                    copy_input(index);
                    compute(index);
                    copy_output(index);
                }
            }
            for (int index = 0; index < active; ++index) {
                check(cudaEventRecord(done[index], workers[index]));
                check(cudaStreamWaitEvent(coordinator, done[index], 0));
            }
            check(cudaEventRecord(stop, coordinator));
            check(cudaEventSynchronize(stop));
            auto wall_stop = std::chrono::steady_clock::now();
            float milliseconds = 0;
            check(cudaEventElapsedTime(&milliseconds, start, stop));
            double wall_ms = std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
            if (iteration >= 0) {
                times.push_back(milliseconds);
                wall_times.push_back(wall_ms);
            }
            for (int index = 0; index < size; ++index) {
                float expected = reference[index % 31];
                float difference = std::abs(host_output[index] - expected);
                if (!std::isfinite(host_output[index]) || difference > 1e-5f * std::abs(expected)) {
                    std::cerr << "CPU reference mismatch at " << index << '\n';
                    return 1;
                }
            }
        }
        std::cout << "{\"record\":\"measurement\",\"order\":\"" << order << "\"";
        std::cout << ",\"size\":" << size << ",\"iterations\":" << iterations;
        std::cout << ",\"correctness\":true,\"warmup\":" << warmup << ",\"milliseconds\":[";
        for (int index = 0; index < repetitions; ++index) {
            if (index > 0) {
                std::cout << ',';
            }
            std::cout << times[index];
        }
        std::cout << "],\"wall_milliseconds\":[";
        for (int index = 0; index < repetitions; ++index) {
            if (index > 0) {
                std::cout << ',';
            }
            std::cout << wall_times[index];
        }
        std::cout << "]}\n";
    }
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    for (int index = 0; index < stream_count; ++index) {
        check(cudaEventDestroy(done[index]));
        check(cudaStreamDestroy(workers[index]));
    }
    check(cudaStreamDestroy(coordinator));
    check(cudaFree(device_input));
    check(cudaFree(device_output));
    check(cudaFreeHost(host_input));
    check(cudaFreeHost(host_output));
}
