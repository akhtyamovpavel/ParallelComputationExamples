#include <iostream>
int main() {
    int width = 129;
    int height = 518;

    size_t pitch;
    int* a;

    cudaMallocPitch(&a, &pitch, width * sizeof(int), height * sizeof(int));

    std::cout << pitch << std::endl;

    cudaFree(a);
}