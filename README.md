# ParallelComputationExamples
Примеры кодов с MPI, OpenMP, CUDA и OpenCL. Код собран при помощи CMake.

## Структура проекта

### MPI
* [`00-hello-world`](/MPI/00-hello-world) - Простой пример Hello World с использованием MPI
* [`01-send_recv`](/MPI/01-send_recv) - Примеры использования функций Send/Recv
* [`02-ping-pong`](/MPI/02-ping-pong) - Пример обмена сообщениями между процессами (задача "Пинг-понг")
* [`03-probe-message-status`](/MPI/03-probe-message-status) - Примеры использования Probe и получения статуса сообщений
* [`04-isend-irecv`](/MPI/04-isend-irecv) - Примеры неблокирующих операций ISend/IRecv
* [`05-bcast`](/MPI/05-bcast) - Коллективная рассылка `MPI_Bcast` и её ручные реализации (линейная и по биномиальному дереву)
* [`06-scatter-gather`](/MPI/06-scatter-gather) - Коллективы `MPI_Scatter`/`MPI_Gather` и их ручные аналоги на Send/Recv
* [`07-reduce-allreduce`](/MPI/07-reduce-allreduce) - Редукции `MPI_Reduce`/`MPI_Allreduce` и ручная редукция по биномиальному дереву
* [`08-alltoall`](/MPI/08-alltoall) - Обмен «все-со-всеми» `MPI_Alltoall` и ручная реализация через `MPI_Sendrecv`
* [`09-collective-benchmark`](/MPI/09-collective-benchmark) - Сводный бенчмарк коллективных операций с выводом в CSV
* [`10-comm-split`](/MPI/10-comm-split) - Разбиение коммуникатора через `MPI_Comm_split` (чётные vs нечётные)
* [`11-comm-dup`](/MPI/11-comm-dup) - `MPI_Comm_dup`: независимый «канал» для библиотек, изоляция тегов
* [`12-comm-group`](/MPI/12-comm-group) - Группы рангов через `MPI_Group_incl` и `MPI_Comm_create_group`
* [`13-comm-cart`](/MPI/13-comm-cart) - Декартова 2D-топология: `MPI_Cart_create`, координаты и соседи через `MPI_Cart_shift`

### OpenMP
* [`00-hello-world`](/OpenMP/00-hello-world) - Базовый пример с использованием OpenMP
* [`01-parallel-for`](/OpenMP/01-parallel-for) - Примеры параллельных циклов
* [`02-sections`](/OpenMP/02-sections) - Примеры использования секций в OpenMP
* [`03-master`](/OpenMP/03-master) - Примеры с директивой master
* [`04-parralel-sum`](/OpenMP/04-parralel-sum) - Примеры параллельного суммирования
* [`05-matrix-multiplication`](/OpenMP/05-matrix-multiplication) - Умножение матриц с использованием OpenMP
* [`06-critical-sections`](/OpenMP/06-critical-sections) - Примеры использования критических секций

### CUDA
* [`00-how-to-run-task-on-cluster`](/CUDA/00-how-to-run-task-on-cluster) - Инструкция по запуску задач на кластере
* [`01-intro`](/CUDA/01-intro) - Введение в CUDA
* [`02-device-specs-benchmarks`](/CUDA/02-device-specs-benchmarks) - Спецификации устройства и бенчмарки
* [`03-memory-model`](/CUDA/03-memory-model) - Модель памяти в CUDA
* [`03.5-matrix-multiplication-example`](/CUDA/03.5-matrix-multiplication-example) - Пример умножения матриц на CUDA
* [`04-reduction`](/CUDA/04-reduction) - Примеры редукции
* [`04.5-atomics`](/CUDA/04.5-atomics) - Атомарные операции и построение гистограммы
* [`04.7-warp-primitives`](/CUDA/04.7-warp-primitives) - Warp-уровневые примитивы: shuffle, reduce, vote
* [`05-scan`](/CUDA/05-scan) - Примеры сканирования
* [`06-cublas`](/CUDA/06-cublas) - Примеры использования библиотеки cuBLAS
* [`07-pycuda`](/CUDA/07-pycuda) - Примеры использования PyCUDA
* [`08-streams-events`](/CUDA/08-streams-events) - Потоки, события и перекрытие copy/compute
* [`09-unified-memory`](/CUDA/09-unified-memory) - Unified memory: managed-указатели и prefetch
* [`10-cooperative-groups`](/CUDA/10-cooperative-groups) - Cooperative groups: warp/block-группы и tiled_partition
* [`11-cuda-graphs`](/CUDA/11-cuda-graphs) - CUDA Graphs: stream capture и explicit graph
* [`12-thrust`](/CUDA/12-thrust) - Thrust: device_vector, transform_reduce, sort
* [`13-curand`](/CUDA/13-curand) - cuRAND: host и device API на задаче Monte Carlo π
* [`14-multi-gpu`](/CUDA/14-multi-gpu) - Multi-GPU: независимые устройства и peer access (≥ 2 GPU)
* [`15-tensor-cores`](/CUDA/15-tensor-cores) - Tensor Cores / WMMA (FP16 × FP16 → FP32), требует sm_70+
* [`16-dynamic-parallelism`](/CUDA/16-dynamic-parallelism) - Dynamic parallelism: запуск ядер из ядра (нужен -rdc=true)
* [`17-cufft`](/CUDA/17-cufft) - cuFFT: 1D complex FFT (forward/inverse roundtrip)
* [`18-cusparse`](/CUDA/18-cusparse) - cuSPARSE: SpMV на CSR-матрице (generic API)
* [`tasks`](/CUDA/tasks) - Упражнения для выполнения

### OpenCL
* [`00-platform-device-info`](/OpenCL/00-platform-device-info) - Перечисление платформ и устройств OpenCL
* [`01-intro`](/OpenCL/01-intro) - Введение в OpenCL (vector add)

### HadoopStackExamples
* [`MapReduce`](/HadoopStackExamples/MapReduce) - Примеры MapReduce
  * [`01-one-stage`](/HadoopStackExamples/MapReduce/01-one-stage) - Одностадийный MapReduce - задача WordCount
  * [`02-two-stages`](/HadoopStackExamples/MapReduce/02-two-stages) - Двухстадийный MapReduce для получения Top-слов
* [`Cassandra`](/HadoopStackExamples/Cassandra) - Примеры работы с Apache Cassandra
* [`Hive`](/HadoopStackExamples/Hive) - Примеры работы с Apache Hive

## Сборка проекта

Проект использует CMake для сборки примеров MPI и OpenMP. По умолчанию сборка CUDA- и OpenCL-примеров отключена — включается флагами `-DBUILD_CUDA_EXAMPLES=ON` / `-DBUILD_OPENCL_EXAMPLES=ON`.

Для сборки проекта:
```bash
mkdir build
cd build
cmake ..
make
```

Примеры HadoopStackExamples имеют отдельные системы сборки:
* MapReduce - сборка через Maven
* Cassandra - инфраструктура запускается через Docker
* Hive - сборка через Maven

## Инструкция по использованию MPI кластера

Компиляция программ происходит при помощи компиляторов `mpicc` и `mpic++`. Подключение происходит при помощи команды

```[bash]
module add mpi/openmpi4-x86_64
```

После этого mpicc и mpic++ подгрузятся в `$PATH`

### Запуск программ MPI

Для локального запуска можно использовать скрипт [`run_local.sh`](/MPI/00-hello-world/bin/run_local.sh). Опция -np используется для указания количества процессов.

MPI локально может быть установлен для следующих ОС:

Ubuntu: `sudo apt-get install openmpi-bin libopenmpi-dev`

Mac OS: `brew install open-mpi`

### Команды по использованию SLURM

* `sinfo` - посмотреть информацию по нодам кластера
* `sinfo -N -l` - посмотреть информацию по каждой ноде кластера
* `squeue` - посмотреть очередь задач
* `srun <command>` - запустить команду на ноде кластера
* `sbatch <script>` - запустить скрипт на нодах кластера. Каждый скрипт должен начинаться с `#!/bin/bash`.
Примеры запуска команд можно найти [здесь](/MPI/00-hello-world/bin/start_sbatch.sh).
После этого должно высветиться сообщение `Submitted batch job <job_id>`, результаты работы попадают в лог-файл `slurm-<job_id>.out`.

### Полезные ссылки:
* https://www.open-mpi.org/doc/current/ - мануал OpenMPI 4.0
* https://mpitutorial.com - хороший tutorial по MPI с примерами.



