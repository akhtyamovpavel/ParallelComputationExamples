#include <iostream>
#include <chrono>

int main() {
    const int block_size = 1024;

    const int array_size = 1 << 20;
    int* h_array = new int[array_size];
    for (int i = 0; i < array_size; ++i) {
        h_array[i] = 1;
    }

    int* output = new int[array_size];

    const auto start = std::chrono::steady_clock::now();

    output[0] = h_array[0];
    for (int i = 1; i < array_size; ++i) {
        output[i] = output[i - 1] + h_array[i];
    }


    const auto stop = std::chrono::steady_clock::now();
    const double milliseconds =
        std::chrono::duration<double, std::milli>(stop - start).count();

    std::cout << milliseconds << " elapsed" << std::endl;

    std::cout << output[array_size - 1] << std::endl;

    // Check outside the timed interval; every prefix of ones is i + 1.
    bool correct = true;
    for (int i = 0; i < array_size; ++i) {
        if (output[i] != i + 1) {
            std::cerr << "Incorrect prefix at " << i << std::endl;
            correct = false;
            break;
        }
    }

    delete[] h_array;
    delete[] output;
    return correct ? 0 : 1;


}
