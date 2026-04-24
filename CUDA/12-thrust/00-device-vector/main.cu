#include <iostream>
#include <iomanip>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

// Пример: vector-add без единого cudaMalloc / cudaMemcpy / kernel<<<...>>>.
// Thrust предоставляет RAII-обёртки device_vector / host_vector и
// STL-совместимые алгоритмы (transform, reduce, sort, ...). Под капотом это
// всё те же kernel'ы, но не нужно самому выбирать grid/block.
//
// thrust::plus<float>() — готовый functor-сумматор из <thrust/functional.h>.
// Можно заменить на свой struct с operator()(const T&, const T&) const.

int main() {
    const int N = 1 << 22;

    thrust::host_vector<float> h_x(N, 1.0f);
    thrust::host_vector<float> h_y(N, 2.0f);

    // Присваивание host_vector -> device_vector делает cudaMemcpy внутри.
    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;
    thrust::device_vector<float> d_z(N);

    // transform: d_z[i] = d_x[i] + d_y[i] для всего диапазона.
    thrust::transform(d_x.begin(), d_x.end(),
                      d_y.begin(),
                      d_z.begin(),
                      thrust::plus<float>());

    // Обратный copy — такое же присваивание.
    thrust::host_vector<float> h_z = d_z;

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float e = std::fabs(h_z[i] - 3.0f);
        if (e > max_err) {
            max_err = e;
        }
    }

    std::cout << "thrust::transform(vector_add)  N=" << N
              << "  max_err=" << std::scientific << std::setprecision(2)
              << max_err << std::endl;

    // cudaFree для device_vector и free() для host_vector вызывают деструкторы
    // автоматически — никаких ручных cudaFree в main нет.
    return 0;
}
