# Стенд сравнения локальных Scan

Сравниваем локальные exclusive-префиксы каждой группы потоков. Это не время завершённого глобального Scan. Все варианты получают одинаковый знакопеременный int-вход, а CPU начинает эталонную сумму с нуля в каждом блоке и проверяет каждую выходную позицию.

Hillis–Steele сохраняет тело авторского `01.5-correct_cuda_scan.cu`; только финальная запись сдвинута из inclusive в exclusive. Blelloch использует точное тело `02-two-layer-parallelism.cu`, включая все барьеры down-sweep, без padding. Отдельный warp-вариант использует shuffle внутри полного warp и два барьера между warp. Файл `correct_scan.cu` не является источником показанного сравнения.

## Сборка через CMake

Из корня репозитория:

```bash
cmake -S . -B build-cuda -DBUILD_MPI_EXAMPLES=OFF -DBUILD_OPENMP_EXAMPLES=OFF -DBUILD_CUDA_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --target cuda_05-scan_06-local-scan-benchmark ScanWarpCheck
./CUDA/05-scan/06-local-scan-benchmark/run.sh
./CUDA/05-scan/06-local-scan-benchmark/ScanWarpCheck --size 513 --block 256 --pattern signed
```

Цели используют стандарт C++17. Для другого GPU выбираем его архитектуру.

## Прямая сборка

Из каталога стенда:

```bash
nvcc -O3 -std=c++17 -lineinfo -arch=sm_86 main.cu -o main
nvcc -O3 -std=c++17 -lineinfo -arch=sm_86 warp_scan_check.cu -o ScanWarpCheck
./run.sh
./main --size 4194304 --warmup 10 --repetitions 5 --batch 25
./ScanWarpCheck --size 0 --block 32 --pattern signed
./ScanWarpCheck --size 4097 --block 256 --pattern signed
```

`run.sh` без аргументов выбирает 65536 элементов, три прогревочные пачки и три измеряемые по три запуска. Для воспроизведения таблицы лекции используем параметры из [методики замеров](МЕТОДИКА-ЗАМЕРОВ.md). Блоки: 32, 64, 128, 256, 512 и 1024. Параметр variant выбирает all, hillis_steele, blelloch или warp_shuffle. Авторские варианты требуют размер, кратный 1024; warp-вариант отдельно умеет нейтральный хвост.

CUDA events измеряют всю сетку локальных Scan. Выделение памяти, копирования, CPU-проверка и межблочное завершение исключены. Программа проверяет результат каждой измеряемой пачки и выводит времена отдельных запусков; итогом сравнения служит медиана повторов.

`warp_scan_check.cu` проверяет каждый локальный exclusive-префикс, включая пустой вход, хвост и знакопеременный массив. Основной `04-warp-scan.cu` в соседнем каталоге остаётся авторским примером локального inclusive Scan. Различие вида результата здесь намеренное и явно проверяется.
