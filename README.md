# ParallelComputationExamples
Примеры кодов с MPI, OpenMP и CUDA. Код собран при помощи CMake.

## Инструкция по использованию MPI кластера

Компиляция программ происходит при помощи компиляторов `mpicc` и `mpic++`. Подключение происходит при помощи команды

```[bash]
module add mpi/openmpi4-x86_64
```

После этого mpicc и mpic++ подгрузятся в `$PATH`

### Запуск программ MPI

Для локального запуска можно использовать скрипт [run_local.sh](/MPI/00-hello-world/bin/run_local.sh). Опция -np используется для указания количества процессов.

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



