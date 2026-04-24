#include <iostream>
#include <iomanip>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>

// transform_reduce объединяет две операции в один проход по данным:
//  - unary_op применяется к каждому элементу (тут x -> x*x);
//  - binary_op склеивает результаты (тут plus<float>).
// На выходе получаем sum(x_i^2) без создания промежуточного массива.
// Это стандартная конструкция для L2-нормы, энергии сигнала и т.п.
struct square {
    __host__ __device__ float operator()(const float& x) const {
        return x * x;
    }
};

int main() {
    const int N = 1 << 20;

    // Заполняем 1, 2, 3, ..., N — так есть закрытая формула для проверки:
    // sum_{i=1..N} i^2 = N*(N+1)*(2N+1)/6.
    thrust::device_vector<float> d(N);
    thrust::sequence(d.begin(), d.end(), 1.0f);

    float sum_sq = thrust::transform_reduce(d.begin(), d.end(),
                                            square(),
                                            0.0f,
                                            thrust::plus<float>());
    float l2 = std::sqrt(sum_sq);

    double expected_sum_sq =
        static_cast<double>(N) * (N + 1) * (2.0 * N + 1) / 6.0;
    double expected_l2 = std::sqrt(expected_sum_sq);
    double rel_err = std::fabs(l2 - expected_l2) / expected_l2;

    std::cout << std::left << std::setw(24) << "thrust::transform_reduce"
              << "  sum(x^2)=" << std::scientific << std::setprecision(4) << sum_sq
              << "  L2=" << std::fixed << std::setprecision(2) << l2
              << "  rel_err=" << std::scientific << std::setprecision(2) << rel_err
              << std::endl;
    return 0;
}
