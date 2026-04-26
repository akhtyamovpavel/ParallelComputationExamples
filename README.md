# Parallel Computation Examples

Учебный репозиторий с примерами параллельного программирования: **CUDA**, **MPI**, **OpenMP**, **OpenCL** и **Hadoop-стек** (MapReduce, Cassandra, Hive). Более **80 исходных файлов**, каждый из которых — самостоятельный, компилируемый пример с комментариями на русском языке.

Примеры упорядочены по возрастанию сложности и покрывают путь от первого «Hello World» на GPU до tensor cores, multi-GPU, CUDA Graphs и warp-level scan.

Репозиторий используется в курсе **«Параллельные вычисления»** в ФПМИ МФТИ.

---

## Быстрый старт

```bash
# MPI + OpenMP (по умолчанию)
mkdir build && cd build && cmake .. && make

# Включить CUDA и OpenCL
cmake .. -DBUILD_CUDA_EXAMPLES=ON -DBUILD_OPENCL_EXAMPLES=ON && make

# Запуск MPI
mpiexec -np 4 ./MPI/00-hello-world/bin/MpiHelloWorld

# Запуск CUDA / OpenCL
./CUDA/01-intro/01-hello-world-single-cuda-thread/main
```

### Зависимости

| Технология | Ubuntu | macOS |
|------------|--------|-------|
| MPI | `sudo apt install openmpi-bin libopenmpi-dev` | `brew install open-mpi` |
| OpenMP | Входит в GCC | `brew install libomp` |
| CUDA | [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | — |
| OpenCL | `sudo apt install opencl-headers ocl-icd-opencl-dev` | Входит в систему |

На кластере: `module add mpi/openmpi4-x86_64`

---

## Структура проекта

### CUDA (20 тем, 80+ файлов)

<details>
<summary><b>Введение и архитектура</b></summary>

| # | Директория | Описание |
|---|-----------|----------|
| 00 | [`00-how-to-run-task-on-cluster`](/CUDA/00-how-to-run-task-on-cluster) | Инструкция по запуску задач на кластере (SLURM) |
| 01 | [`01-intro`](/CUDA/01-intro) | Hello World, сложение массивов, memcpy, умножение матриц |
| 02 | [`02-device-specs-benchmarks`](/CUDA/02-device-specs-benchmarks) | Характеристики устройства, замер времени, bandwidth, ILP, occupancy |

</details>

<details>
<summary><b>Модель памяти</b></summary>

| # | Директория | Описание |
|---|-----------|----------|
| 03 | [`03-memory-model`](/CUDA/03-memory-model) | 10 подтем: coalesced/uncoalesced access, shared memory и bank conflicts, pinned memory, кеширование, pitching, constant memory, transpose (naive → tiled → padded), stencil с halo cells, **zero-copy (mapped) memory**, **сводный бенчмарк bandwidth** (pageable / pinned / WC / unified / prefetch) |
| 03.5 | [`03.5-matrix-multiplication-example`](/CUDA/03.5-matrix-multiplication-example) | Наивное умножение матриц на CUDA |

</details>

<details>
<summary><b>Reduction, Scan, Atomics, Warp-примитивы</b></summary>

| # | Директория | Описание |
|---|-----------|----------|
| 04 | [`04-reduction`](/CUDA/04-reduction) | 8 реализаций: наивная → shared memory → bank-conflict-free → warp-reduce (volatile + `__shfl_down_sync`) |
| 04.5 | [`04.5-atomics`](/CUDA/04.5-atomics) | `atomicAdd`, наивная и приватизированная гистограмма |
| 04.7 | [`04.7-warp-primitives`](/CUDA/04.7-warp-primitives) | `__shfl_sync`, `__shfl_down_sync` (warp reduce), `__ballot_sync` / `__any_sync` |
| 05 | [`05-scan`](/CUDA/05-scan) | 7 реализаций prefix sum: CPU baseline → naive (data race) → double-buffer → recursive multi-block → two-layer → bank-conflict-free Blelloch → **warp-level scan через `__shfl_up_sync`** |

</details>

<details>
<summary><b>Библиотеки CUDA Toolkit</b></summary>

| # | Директория | Описание |
|---|-----------|----------|
| 06 | [`06-cublas`](/CUDA/06-cublas) | cuBLAS: vector add, AXPY, cosine distance, matrix sums |
| 13 | [`13-curand`](/CUDA/13-curand) | cuRAND: Monte Carlo для числа π (host API + device API) |
| 17 | [`17-cufft`](/CUDA/17-cufft) | cuFFT: 1D complex FFT (forward → inverse roundtrip) |
| 18 | [`18-cusparse`](/CUDA/18-cusparse) | cuSPARSE: SpMV на CSR-матрице (generic API) |

</details>

<details>
<summary><b>Продвинутые возможности</b></summary>

| # | Директория | Описание |
|---|-----------|----------|
| 07 | [`07-pycuda`](/CUDA/07-pycuda) | PyCUDA: программирование GPU из Python |
| 08 | [`08-streams-events`](/CUDA/08-streams-events) | Потоки, события, 3 способа замера времени, перекрытие copy/compute |
| 09 | [`09-unified-memory`](/CUDA/09-unified-memory) | Unified memory: `cudaMallocManaged` и `cudaMemPrefetchAsync` |
| 10 | [`10-cooperative-groups`](/CUDA/10-cooperative-groups) | Cooperative groups: warp/block-группы, `tiled_partition` |
| 11 | [`11-cuda-graphs`](/CUDA/11-cuda-graphs) | CUDA Graphs: stream capture и explicit graph API |
| 12 | [`12-thrust`](/CUDA/12-thrust) | Thrust: `device_vector`, `transform_reduce`, `sort` |
| 14 | [`14-multi-gpu`](/CUDA/14-multi-gpu) | Multi-GPU: независимые устройства и peer access (>= 2 GPU) |
| 15 | [`15-tensor-cores`](/CUDA/15-tensor-cores) | Tensor Cores / WMMA (FP16 x FP16 -> FP32), требует sm_70+ |
| 16 | [`16-dynamic-parallelism`](/CUDA/16-dynamic-parallelism) | Dynamic parallelism: запуск ядер из ядра (`-rdc=true`) |
| 19 | [`19-profiling-tools`](/CUDA/19-profiling-tools) | **Профилирование: NVTX-аннотации** для Nsight Systems / Nsight Compute / nvprof, скрипт запуска |

</details>

<details>
<summary><b>Домашние задания</b></summary>

| # | Файл | Описание |
|---|------|----------|
| 1 | [`01-random-calculation.md`](/CUDA/tasks/01-random-calculation.md) | Случайные вычисления |
| 2 | [`02-matrix-multiplication.md`](/CUDA/tasks/02-matrix-multiplication.md) | Tiled matrix multiplication (shared memory) |
| 3 | [`03-quick-sort.md`](/CUDA/tasks/03-quick-sort.md) | Quick sort с parallel scan |

</details>

---

### MPI (14 примеров)

| # | Директория | Описание |
|---|-----------|----------|
| 00 | [`00-hello-world`](/MPI/00-hello-world) | Hello World с MPI |
| 01 | [`01-send_recv`](/MPI/01-send_recv) | Блокирующие `MPI_Send` / `MPI_Recv` |
| 02 | [`02-ping-pong`](/MPI/02-ping-pong) | Обмен сообщениями «Пинг-понг» |
| 03 | [`03-probe-message-status`](/MPI/03-probe-message-status) | `MPI_Probe` и получение статуса сообщений |
| 04 | [`04-isend-irecv`](/MPI/04-isend-irecv) | Неблокирующие `MPI_Isend` / `MPI_Irecv` |
| 05 | [`05-bcast`](/MPI/05-bcast) | `MPI_Bcast` и ручные реализации (линейная, биномиальное дерево) |
| 06 | [`06-scatter-gather`](/MPI/06-scatter-gather) | `MPI_Scatter` / `MPI_Gather` и аналоги на Send/Recv |
| 07 | [`07-reduce-allreduce`](/MPI/07-reduce-allreduce) | `MPI_Reduce` / `MPI_Allreduce` и ручная редукция |
| 08 | [`08-alltoall`](/MPI/08-alltoall) | `MPI_Alltoall` и реализация через `MPI_Sendrecv` |
| 09 | [`09-collective-benchmark`](/MPI/09-collective-benchmark) | Сводный бенчмарк коллективных операций (CSV) |
| 10 | [`10-comm-split`](/MPI/10-comm-split) | `MPI_Comm_split` (чётные vs нечётные) |
| 11 | [`11-comm-dup`](/MPI/11-comm-dup) | `MPI_Comm_dup`: изоляция тегов для библиотек |
| 12 | [`12-comm-group`](/MPI/12-comm-group) | Группы рангов: `MPI_Group_incl`, `MPI_Comm_create_group` |
| 13 | [`13-comm-cart`](/MPI/13-comm-cart) | Декартова 2D-топология: `MPI_Cart_create`, `MPI_Cart_shift` |

---

### OpenMP (7 примеров)

| # | Директория | Описание |
|---|-----------|----------|
| 00 | [`00-hello-world`](/OpenMP/00-hello-world) | Hello World с `#pragma omp parallel` |
| 01 | [`01-parallel-for`](/OpenMP/01-parallel-for) | Параллельные циклы |
| 02 | [`02-sections`](/OpenMP/02-sections) | Секции |
| 03 | [`03-master`](/OpenMP/03-master) | Директива `master` |
| 04 | [`04-parralel-sum`](/OpenMP/04-parralel-sum) | Параллельное суммирование |
| 05 | [`05-matrix-multiplication`](/OpenMP/05-matrix-multiplication) | Умножение матриц |
| 06 | [`06-critical-sections`](/OpenMP/06-critical-sections) | Критические секции |

---

### OpenCL (20 примеров)

<details>
<summary>Развернуть полный список</summary>

| # | Директория | Описание |
|---|-----------|----------|
| 00 | [`00-platform-device-info`](/OpenCL/00-platform-device-info) | Перечисление платформ и устройств |
| 01 | [`01-intro`](/OpenCL/01-intro) | Vector add |
| 02 | [`02-local-memory`](/OpenCL/02-local-memory) | `__local` память, транспоз матрицы naive vs local |
| 03 | [`03-reduction`](/OpenCL/03-reduction) | Дерево редукции с `barrier(CLK_LOCAL_MEM_FENCE)` |
| 04 | [`04-matrix-multiplication`](/OpenCL/04-matrix-multiplication) | Наивное умножение матриц |
| 05 | [`05-profiling-events`](/OpenCL/05-profiling-events) | Профилирование: `cl_event`, 4 стадии event'а, bandwidth |
| 06 | [`06-image-objects`](/OpenCL/06-image-objects) | Image objects: `image2d_t`, `sampler_t`, Box Blur |
| 07 | [`07-multi-queue`](/OpenCL/07-multi-queue) | Несколько очередей: перекрытие copy/compute |
| 08 | [`08-coalesced-access`](/OpenCL/08-coalesced-access) | Coalesced vs strided доступ, замер bandwidth |
| 09 | [`09-constant-memory`](/OpenCL/09-constant-memory) | `__constant` память: broadcast-оптимизация |
| 10 | [`10-pinned-memory`](/OpenCL/10-pinned-memory) | Pinned (mapped) память vs pageable |
| 11 | [`11-stencil`](/OpenCL/11-stencil) | 1D-стенсил с halo cells в `__local` памяти |
| 12 | [`12-atomics`](/OpenCL/12-atomics) | Атомарные операции, наивная и приватизированная гистограмма |
| 13 | [`13-scan`](/OpenCL/13-scan) | Prefix sum: **Хиллис-Стил** (inclusive) и **Блеллох** (exclusive, work-efficient) |
| 14 | [`14-kernel-from-file`](/OpenCL/14-kernel-from-file) | Загрузка ядра из внешнего `.cl` файла |
| 15 | [`15-build-options`](/OpenCL/15-build-options) | Опции компиляции: `-cl-fast-relaxed-math`, `-cl-mad-enable` |
| 16 | [`16-offline-compilation`](/OpenCL/16-offline-compilation) | Кеширование скомпилированного бинарника |
| 17 | [`17-sub-groups`](/OpenCL/17-sub-groups) | Sub-groups (аналог warp): `sub_group_reduce_add()` (OpenCL 2.0+) |
| 18 | [`18-svm`](/OpenCL/18-svm) | Shared Virtual Memory (аналог Unified Memory, OpenCL 2.0+) |
| 19 | [`19-multi-device`](/OpenCL/19-multi-device) | Разделение работы между CPU и GPU |
| 20 | [`20-pyopencl`](/OpenCL/20-pyopencl) | PyOpenCL: Python-обёртка |

</details>

---

### Hadoop-стек

| Директория | Описание |
|-----------|----------|
| [`MapReduce/01-one-stage`](/HadoopStackExamples/MapReduce/01-one-stage) | Одностадийный MapReduce (WordCount) |
| [`MapReduce/02-two-stages`](/HadoopStackExamples/MapReduce/02-two-stages) | Двухстадийный MapReduce (Top-слова) |
| [`Cassandra`](/HadoopStackExamples/Cassandra) | Apache Cassandra (Docker Compose) |
| [`Hive`](/HadoopStackExamples/Hive) | Apache Hive (Maven) |

---

## Сборка

Проект использует **CMake** со зонтичной сборкой. MPI и OpenMP включены по умолчанию, CUDA и OpenCL — по флагам:

```bash
mkdir build && cd build

# Только MPI + OpenMP
cmake .. && make

# Всё, включая CUDA и OpenCL
cmake .. -DBUILD_CUDA_EXAMPLES=ON -DBUILD_OPENCL_EXAMPLES=ON && make

# Переопределить CUDA-архитектуры (по умолчанию: Pascal—Ada Lovelace)
cmake .. -DBUILD_CUDA_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
```

Каждый пример можно собрать и отдельно:
```bash
cd MPI/00-hello-world && cmake . && make
cd CUDA/04-reduction/05-warp-reduce && nvcc main.cu -o main
```

---

## Запуск

### MPI
```bash
mpiexec -np 4 ./bin/MpiHelloWorld       # локально
sbatch bin/start_sbatch.sh              # на кластере (SLURM)
```

### CUDA / OpenCL
```bash
./main                                  # локально
sbatch run.sh                           # на кластере с GPU
```

### Профилирование CUDA
```bash
# Nsight Systems (timeline + NVTX-аннотации)
nsys profile --trace=cuda,nvtx -o report ./main

# Nsight Compute (метрики ядер)
ncu --set full ./main

# nvprof (legacy, CUDA <= 11)
nvprof --print-gpu-trace ./main
```

---

## Использование SLURM-кластера

| Команда | Описание |
|---------|----------|
| `sinfo` | Информация по нодам кластера |
| `sinfo -N -l` | Информация по каждой ноде |
| `squeue` | Очередь задач |
| `srun <command>` | Запуск команды на ноде |
| `sbatch <script>` | Запуск скрипта (начинается с `#!/bin/bash`) |

Пример: [`MPI/00-hello-world/bin/start_sbatch.sh`](/MPI/00-hello-world/bin/start_sbatch.sh). После `sbatch` появится `Submitted batch job <job_id>`, результат — в `slurm-<job_id>.out`.

---

## Полезные ссылки

### Документация и инструменты
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof) / [Visual Profiler](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#visual-profiler)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) / [Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [OpenMPI 4.0 Manual](https://www.open-mpi.org/doc/current/)
- [MPI Tutorial](https://mpitutorial.com)

### Курсы и лекции
- [GPU Programming — Caltech CS179](http://courses.cms.caltech.edu/cs179/)
- Лекции Евгения Перепёлкина (YouTube)
- Лекции Павла Ахтямова в Лектории ФПМИ МФТИ (YouTube)

### Книги
- Дж. Сандерс, Э. Кэррот. *Технология CUDA в примерах и задачах*, 2013
- [Боресков А. В., Харламов А. А. *Основы работы с технологией CUDA*](https://www.ozon.ru/product/osnovy-raboty-s-tehnologiey-cuda-boreskov-aleksey-viktorovich-1798385526/)
- [GPU Gems 3](https://developer.nvidia.com/gpugems/gpugems3/foreword) — особенно глава 39 (Parallel Prefix Sum)

### Статьи
- [Parallel Prefix Sum (Scan) with CUDA — Mark Harris, NVIDIA](http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf)
- [Лекция по Scan — Ben-Gurion University](https://www.cs.bgu.ac.il/~graph161/wiki.files/09f-GPU%20-%20Scans.pdf)
- [Хабр: Краткая история GPU](https://habr.com/ru/companies/itglobalcom/articles/746252/)
- [Хабр: Работа с памятью в GPU](https://habr.com/ru/articles/55461/)

---

## Лицензия

Учебный репозиторий для курса «Параллельные вычисления» ФПМИ МФТИ.
