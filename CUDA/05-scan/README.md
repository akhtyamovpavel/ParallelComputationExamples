# CUDA Scan

Раздел сохраняет последовательность авторских примеров и добавляет два самостоятельных стенда. Примеры в корне каталога показывают отдельные ступени алгоритма; результат внутри блока и результат всего массива различаются.

## Авторские примеры

- `00-naive_scan.cu`: последовательный inclusive Scan на CPU. Таймер `std::chrono::steady_clock` измеряет только цикл; заполнение и проверка всех префиксов находятся вне интервала.
- `01-naive_cuda_scan.cu`: намеренная учебная гонка. Потоки читают и перезаписывают один shared-массив в одном раунде. Этот файл оставлен для разбора ошибки.
- `01.5-correct_cuda_scan.cu`: два shared-буфера и два барьера на раунд. Inclusive Scan локален для каждого блока; смещения предыдущих блоков не добавляются.
- `01.75-recursive_cuda_scan.cu`: два уровня для конкретной конфигурации из 1048576 элементов и блока 1024. Объявление `step_cnt` восстановлено, финальная запись читает действующую половину shared memory при чётном и нечётном числе раундов. CPU-проверка сверяет все глобальные префиксы. Это не произвольная рекурсия для любых размеров или хвостов.
- `02-two-layer-parallelism.cu`: локальный exclusive Scan Blelloch с барьерами уровней. Полный блок и степень двойки являются условиями этого примера.
- `correct_scan.cu`: тот же учебный путь с барьером между уровнями down-sweep. Исправление не добавляет обработку хвоста и глобальные смещения.
- `03-scan-bank-conflicts.cu`: padding shared memory. Он меняет адреса для рассматриваемой модели банков, но сам по себе не гарантирует выигрыша времени.
- `04-warp-scan.cu`: локальный inclusive Scan через shuffle и суммы warp. В первом warp все 32 lane участвуют в каждом full-mask shuffle; отсутствующие суммы заменяются нулями. При ошибке CPU-проверки программа возвращает ненулевой код.

Работа Hillis–Steele внутри блока имеет порядок B log B, у дерева Blelloch порядок B; критический путь у обоих логарифмический. Барьеры, занятость и обращения к памяти влияют на фактическое время. Нет универсального обещания выигрыша или замедления на 10 процентов.

## Стенды

- [Глобальный Scan и стабильная Compaction](05-global-scan-compaction/README.md): хвост, рекурсивные суммы блоков, фильтрация значений строго больше трёх, CPU-проверка каждого префикса и порядка выхода.
- [Сравнение локальных Scan](06-local-scan-benchmark/README.md): Hillis–Steele, Blelloch и warp на одном входе, единая область таймера и повторные измерения. Отдельная программа проверяет локальный exclusive warp Scan.
- [Сравнение порядков Streams](../08-streams-events/03-overlap-orders/README.md): baseline, interleaved и batched с явным ожиданием завершения всех очередей.
- [Минимальные исправления и CPU-копия](corrections/README.md).

## Сборка

Из корня репозитория выбираем архитектуру своего GPU. Для RTX 3090 используем 86; для RTX 2080 Ti используем 75:

```bash
cmake -S . -B build-cuda -DBUILD_MPI_EXAMPLES=OFF -DBUILD_OPENMP_EXAMPLES=OFF -DBUILD_CUDA_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --target ScanNaiveCpu ScanRecursive ScanWarpShuffle ScanBarrierCorrected
./CUDA/05-scan/ScanNaiveCpu
./CUDA/05-scan/ScanRecursive
./CUDA/05-scan/ScanWarpShuffle
./CUDA/05-scan/ScanBarrierCorrected
```

Для CPU-примера достаточно обычного C++-компилятора. Из каталога `05-scan`:

```bash
g++ -O2 -std=c++17 -x c++ 00-naive_scan.cu -o corrections/scan_cpu
./corrections/scan_cpu
```

Перед сравнением времени проверяем каждый результат. Новые проверки корректности не заменяют прежние измерения лекции: устройство, компилятор, вход и границы таймера нужно фиксировать отдельно.

Правило участия lane описано в [CUDA Programming Guide: Warp Shuffle Functions](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#warp-shuffle-functions).
