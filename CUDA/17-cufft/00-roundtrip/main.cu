#include <iostream>
#include <iomanip>
#include <cmath>

#include <cufft.h>

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

#define CUFFT_CHECK(call) do {                                                \
    cufftResult st = (call);                                                  \
    if (st != CUFFT_SUCCESS) {                                                \
        std::cerr << "cuFFT error " << st << " at "                           \
                  << __FILE__ << ":" << __LINE__ << std::endl;                \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// Roundtrip-проверка: BЧ-преобразование входного сигнала, затем обратное,
// нормировка на N и сравнение с оригиналом. cuFFT не нормирует обратное
// преобразование сам — это ответственность пользователя (стандартное
// поведение всех «быстрых» FFT-библиотек, включая FFTW).
int main() {
    const int N = 1 << 14;
    const double TWO_PI = 6.283185307179586;

    // cufftComplex — это float2 с {x, y} под real/imag. Инициализируем
    // синусоидой — содержательный сигнал, удобнее видеть в отладке FFT-вывода.
    cufftComplex* h_in = new cufftComplex[N];
    for (int i = 0; i < N; ++i) {
        h_in[i].x = static_cast<float>(std::sin(TWO_PI * i / 16.0));
        h_in[i].y = 0.0f;
    }

    cufftComplex* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemcpy(d_data, h_in, N * sizeof(cufftComplex),
                          cudaMemcpyHostToDevice));

    // План создаётся под конкретный размер / тип / batch — переиспользуется
    // для множества exec'ов и должен быть живым, пока идут вызовы.
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
    CUDA_CHECK(cudaDeviceSynchronize());

    cufftComplex* h_out = new cufftComplex[N];
    CUDA_CHECK(cudaMemcpy(h_out, d_data, N * sizeof(cufftComplex),
                          cudaMemcpyDeviceToHost));

    // Нормировка и сравнение.
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float re = h_out[i].x / N;
        float im = h_out[i].y / N;
        float e_re = std::fabs(re - h_in[i].x);
        float e_im = std::fabs(im - h_in[i].y);
        float e = e_re + e_im;
        if (e > max_err) {
            max_err = e;
        }
    }

    std::cout << std::left << std::setw(28) << "cuFFT C2C roundtrip"
              << " N=" << N
              << "  max_err=" << std::scientific << std::setprecision(2)
              << max_err << std::endl;

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_data));
    delete[] h_in;
    delete[] h_out;
    return 0;
}
