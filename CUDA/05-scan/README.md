# CUDA Scan

## Решение 00

## Решение 01 - naive CUDA Scan

Идея - вычисляем за $O(BS \cdot \log BS)$ сумму элементов вне блока.

![image](https://github.com/akhtyamovpavel/ParallelComputationExamples/assets/5366960/54f958f2-60c5-499a-81a1-450e4ef18477)

Однако, в наивной реализации нельзя делать операцию `+=` - получим Data Race!

## Решение 01.5 - correct CUDA Scan

Идея: выделяем дополнительный блок в Shared-памяти, в котором производим хранение элементов:
https://github.com/akhtyamovpavel/ParallelComputationExamples/blob/74c9a162b4ebc357a10048c2357545213cb64f75/CUDA/05-scan/01.5-correct_cuda_scan.cu#L22-L24

После общей синхронизации осуществляем обратное копирование элементов:
https://github.com/akhtyamovpavel/ParallelComputationExamples/blob/74c9a162b4ebc357a10048c2357545213cb64f75/CUDA/05-scan/01.5-correct_cuda_scan.cu#L26-L28

### Решение 01.75 - global CUDA Scan

Показывает, каким образом необходимо выполнять агрегацию локальных элементов массива. Код ядра возвращает:
* сумму чисел в блоках
* частичные суммы на префиксе

https://github.com/akhtyamovpavel/ParallelComputationExamples/blob/74c9a162b4ebc357a10048c2357545213cb64f75/CUDA/05-scan/01.75-recursive_cuda_scan.cu#L53-L59

## Решение 02 - O(N) операций
Идея - будем "пропускать" лишние элементы для сложения. В итоге, агрегация будет работать аналогично дереву Фенвика:

![image](https://github.com/akhtyamovpavel/ParallelComputationExamples/assets/5366960/abbc5884-31a3-4e68-9f17-0899e1077db8)

Однако, время работы алгоритма будет сильно медленнее по 2 причинам:
1. Bank Conflict.
2. Алгоритм имеет большую асимптотику по количеству warp-итераций.

## Решение 03 - решение Bank Conflict-ов

В данном решении производится сдвиг на определённый поток внутри warp-а. 

Вычисления:
* `tid=0`: `a[1] = a[0] + a[1]`
* `tid=16`: `a[33] = a[32] + a[33]`

Получаем Bank Conflict.

**Решение**. Сдвиг на 1 поток:
* `tid=0`: `a[1 + 1 // 32] = a[0 + 0 // 32] + a[1 + 1 // 32]`
* `tid=16`: `a[33 + 33//32] = a[34]`. Bank Conflict пропадает!

Однако, такое решение всё равно на 10% медленнее классического решения.

**Важно**. В классической реализации Scan в решении 01.5 считался по всему массиву (не только внутри блока). Поэтому в книге GPU Gems было получено ускорение.

## Решение 04 (TBA) - использование Warp Scan

Заметим, что $1024 = 32 \times 32$. По этой причине можно использовать `__shfl_up_sync` в двух стадиях.
