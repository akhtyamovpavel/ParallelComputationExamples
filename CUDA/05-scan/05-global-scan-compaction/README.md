# Стенд глобального Scan и Compaction

Стенд продолжает авторское дерево из соседнего `02-two-layer-parallelism.cu`: сохраняет сумму каждого блока, строит exclusive Scan этих сумм и прибавляет смещения. Рекурсия на CPU ставит запуски в один stream; промежуточные массивы остаются на GPU. Один блок из 256 потоков обрабатывает 512 значений, хвост заполняется нулями. Все потоки выполняют каждый барьер.

Пример фильтрует значения строго больше трёх. Exclusive Scan маски задаёт уникальные места выбранных элементов, сохраняя порядок входа. CPU сравнивает каждый глобальный префикс, количество, значения и порядок выхода. Пустой вход проходит без запуска Scan.

## Сборка через CMake

Из корня репозитория:

```bash
cmake -S . -B build-cuda -DBUILD_MPI_EXAMPLES=OFF -DBUILD_OPENMP_EXAMPLES=OFF -DBUILD_CUDA_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --target cuda_05-scan_05-global-scan-compaction
./CUDA/05-scan/05-global-scan-compaction/run.sh
```

Цели используют стандарт C++17. Для своего устройства меняем архитектуру.

## Прямая сборка

Из каталога стенда:

```bash
nvcc -O3 -std=c++17 -lineinfo -arch=sm_86 main.cu -o main
./run.sh
./main --size 8 --pattern sample --warmup 3 --repetitions 3
./main --size 513 --pattern mixed --warmup 3 --repetitions 3
./main --size 1048579 --pattern mixed --warmup 3 --repetitions 3
./main --size 0 --pattern none --warmup 3 --repetitions 3
```

Шаблоны входа: sample, mixed, all и none. Допустим размер от 0 до 67108864; для первого запуска достаточно небольшого массива. `run.sh` без аргументов выбирает 513 элементов, три прогрева и три измерения; заданные аргументы передаются программе.

CUDA events охватывают H2D, Scan, scatter и D2H. Для простоты возвращается выходной буфер полной входной ёмкости; используются только первые count значений. Копирование префиксов для проверки находится вне таймера, как и выделение памяти с построением CPU-эталона. Из повторов сравниваем медианы, не отдельный минимум.
