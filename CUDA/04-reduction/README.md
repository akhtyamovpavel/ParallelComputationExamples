# CUDA. Reduction

Код с примерами использования Reduction:
1. [Пример 0. Использование агрегации между блоками](/CUDA/04-reduction/00-stupid-sum)
2. [Пример 1. Реализация при помощи Shared Memory](/CUDA/04-reduction/01-default-sum)
3. [Пример 2. Меняем процесс математики](/CUDA/04-reduction/02-another-math)
4. [Пример 3. Затрагиваем вопрос Bank Conflict](/CUDA/04-reduction/03-improving-bank-conflicts)
5. [Пример 4. Первая операция перед загрузкой в Shared Memory](/CUDA/04-reduction/04-add-on-load)
6. [Пример 5. Операции внутри Warp-а](/CUDA/04-reduction/05-warp-reduce)
7. [Пример 6. Ассемблерные инструкции внутри Warp-а](/CUDA/04-reduction/06-warp-design-specific)

## Постановка задачи

В предыдущих блоках мы производили операции между массивами (сложение, перемножение). Однако, классическую задачу паралелльных вычислений, вычисление суммы элементов, мы не рассматривали. 

Рассмотрим решение суммы чисел массива. Такие задачи называются задачами Reduce, поскольку выполняют операцию агрегирования значений. 

**Важно:** Задача может быть решена для любой операции, которая удовлетворяет следующим условиям:
1. Ассоциативности (`a + (b + c) = (a + b) + c`)
2. Коммутативности (`a + b = b + a`)
3. Наличие "нейтрального" элемента: (`a + 0 = 0 + a = a`).

## Решение 0. Наивное решение

https://github.com/akhtyamovpavel/ParallelComputationExamples/blob/9867180abf08ea32bfb868fa292edfcf399d5060/CUDA/04-reduction/00-stupid-sum/main.cu#L5-L12

Мы разбиваем массив на блоки размером 1024. После этого каждый поток вычисляет суммы своего "блока" и получает результат.

https://github.com/akhtyamovpavel/ParallelComputationExamples/blob/9867180abf08ea32bfb868fa292edfcf399d5060/CUDA/04-reduction/00-stupid-sum/main.cu#L50-L52

После выполнения ядра код вычисляет сумму элементов в своём блоке.

В примере `StupidAdd` мы видели, что из-за проблем когерентности кэша такой способ вычисления является неэффективным, поэтому код работает очень медленно.

В итоге, сумма $2^{20}$ элементов массива вычисляется за 15 ms.

## Решение 0.5. Необходима ручная настройка

Поменяем формат взаимодействия элементов. Предположим, что в распоряжении имеется 1024 вычислительных ядра. 
Каждый элемент должен сохранять когерентность кэшей, поэтому поток с номером 0 должен вычислять сумму элементов $(0, 1024, 2048, ...)$, поток 1 - $(1, 1025, 2049, ...)$.

Таким образом, все потоки внутри одного warp-а будут брать один элемент из L1/L2-кэша:

https://github.com/akhtyamovpavel/ParallelComputationExamples/blob/9867180abf08ea32bfb868fa292edfcf399d5060/CUDA/04-reduction/00-stupid-sum/normal_sum.cu#L18-L23

Код запускается на количестве потоков, равное количеству Streaming Multiprocessor-ов: 

https://github.com/akhtyamovpavel/ParallelComputationExamples/blob/9867180abf08ea32bfb868fa292edfcf399d5060/CUDA/04-reduction/00-stupid-sum/normal_sum.cu#L67

В результате запуска кода получаем время работы, равное 10 ms. 

## Решение 1. Использование разделяемой памяти

Перейдем к основному алгоритму - вычисление суммы внутри блока. Первый подход - использование общей памяти для хранения промежуточных элементов. Главная идея - использование "турнирной сетки" (в первом раунде считаются соседние элементы, во втором раунде - считаются суммы в четных элементах (четверки), и так далее).

Однако, если реализовать поведение в общей памяти, то будет получен Data Race (каждый Warp будет вычислять элементы в разные моменты времени). Таким образом, сумма будет меньше ожидаемой.

Для решения задачи мы заводим Shared Memory Array, в котором будем хранить частичные суммы:

https://github.com/akhtyamovpavel/ParallelComputationExamples/blob/3e9a5c55725288d5e7c03d5fceff76695143ab4b/CUDA/04-reduction/01-default-sum/main.cu#L9-L17

Таким образом, после выполнения функции `__syncthreads()` все элементы в потоке будут синхронизированы.

## Решение 2. Изменяем математику процесса

Заметим, что если хотя бы один поток внутри warp-а выполняет операцию, то весь warp выполняет операцию. По этой причине в первой итерации цикла простаивает половина потоков, во второй итерации - 3/4 потоков. 

Поэтому лучше сделать так:
* нулевой поток складывает числа (0, 1)
* первый поток складывает числа (2, 3)

Тогда количество warp-ов будет прямо пропорциально количеству потоков для сложения

## Решение 3. Warp-conflict-ы

В предудыщем алгоритме возникает проблема в Shared Memory:
* нулевой поток складывает числа (0, 1), записывает результаты в элемент 0.
* поток 16 (находится в том же warp-е) складывает числа (32, 33), записывает результаты в элемент 32.

В Shared Memory возникает проблема: чтобы сохранять когерентность с кэшами, необходимо обеспечить эффективное сохранение данных внутри одного warp-а. Потоки с номерами 0 и 16 записывают результат одновременно в один bank 0 (индекс по модулю warp-size). Эта проблема именуется Bank Conflict. Запись переходит в режим "последовательной записи".

Главная задача в Shared Memory - избегать Bank Conflict-ов!

Воспользуемся коммутативностью:
* на первой итерации будем складывать первую половину элементов со второй половиной
* на второй итерации - первую четверть со второй четвертью

## Решение 4. Первая операция перед записью в Shared Memory

Заметим, что первой операцией в ядре является операция копирования элементов в разделяемую память.

Можно поступить эффективнее:
* уменьшить количество блоков в 2 раза (таким образом, на каждый поток будет приходиться 2 элемента массива)
* производить сложение первого элемента из блока и второго (который обрабатывается тем же потоком):

https://github.com/akhtyamovpavel/ParallelComputationExamples/blob/3e9a5c55725288d5e7c03d5fceff76695143ab4b/CUDA/04-reduction/04-add-on-load/main.cu#L6-L10

Таким образом, количество используемой разделяемой памяти уменьшается в 2 раза, сохраняя общее количество операций в регистре. Программа ускоряется практически в два раза.

## Решение 5. Используем операции в Warp-e

Через некоторое количество итераций алгоритма все операции выполняются внутри одного Warp-a. По этой причине можно убрать вызов функции `__syncthreads()`, что ускорит время работы программы.

## Решение 6. Используем инструкции в Warp-e

Для решения задачи можно использовать ассемблерные функции внутри ядра: `__shfl_down_sync`. Функция переносит значение внутри warp-а по битовой маске "вниз".