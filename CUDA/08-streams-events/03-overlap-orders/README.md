# Стенд порядков выдачи Streams

Сохраняем авторский busy_kernel из соседнего `02-overlap-compute-copy/main.cu` и сравниваем три порядка. Baseline обрабатывает весь массив в одном stream. Interleaved выдаёт H2D, ядро и D2H по каждой порции. Batched сначала выдаёт все H2D, затем все ядра и все D2H. Четыре порции занимают непересекающиеся диапазоны памяти.

Host-буферы закреплены через cudaMallocHost. Все streams созданы с cudaStreamNonBlocking. Coordinator записывает старт, workers ожидают его. После D2H каждый worker записывает done, coordinator ожидает все done перед stop. CPU читает результат только после ожидания stop. Завершение обработки не зависит от неявной синхронизации stream 0.

## Сборка через CMake

Из корня репозитория:

```bash
cmake -S . -B build-cuda -DBUILD_MPI_EXAMPLES=OFF -DBUILD_OPENMP_EXAMPLES=OFF -DBUILD_CUDA_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --target cuda_08-streams-events_03-overlap-orders
./CUDA/08-streams-events/03-overlap-orders/run.sh
```

Цели используют стандарт C++17. Архитектура 86 соответствует RTX 3090; для своего GPU выбираем подходящее значение.

## Прямая сборка

Из каталога стенда:

```bash
nvcc -O3 -std=c++17 -lineinfo -arch=sm_86 main.cu -o main
./run.sh
./main --size 4194307 --iterations 80 --warmup 5 --repetitions 3
./main --size 65537 --iterations 10 --order interleaved --warmup 3 --repetitions 3
```

Параметр order выбирает all, baseline, interleaved или batched. Размер от 1 до 67108864, iterations от 1 до 1000. `run.sh` без аргументов использует 65537 значений и десять арифметических итераций, три прогрева и три повторения.

CUDA events измеряют H2D, вычисления и D2H с ожиданием всех порций. Дополнительно выводится CPU wall-clock от постановки старта до завершения ожидания. Выделение памяти, заполнение входа и CPU-проверка находятся вне интервала. CPU сравнивает каждое значение с эталоном std::fma, относительный допуск 0,00001.

Возможность перекрытия зависит от устройства и свободных ресурсов. Ни interleaved, ни batched не объявлены универсальным победителем. Проверяем результат, выполняем прогрев и сравниваем медианы минимум трёх повторов. [Stream Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html) и [Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html) описывают используемый порядок ожиданий.
