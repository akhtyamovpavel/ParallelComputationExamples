#include <iostream>

int main() {
    const int block_size = 1024;

    const int array_size = 1 << 20;
    int* h_array = new int[array_size];
    for (int i = 0; i < array_size; ++i) {
        h_array[i] = 1;
    }

    int* output = new int[array_size];

    cudaEvent_t start;
    cudaEvent_t stop;

    // Creating event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);

    output[0] = h_array[0];
    for (int i = 1; i < array_size; ++i) {
        output[i] = output[i - 1] + h_array[i];
    }


    cudaEventRecord(stop);


    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << " elapsed" << std::endl;

    std::cout << output[array_size - 1] << std::endl;

    delete[] h_array;
    delete[] output;


}
