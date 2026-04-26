#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyOpenCL — аналог PyCUDA, обёртка над OpenCL C API для Python.
# В отличие от PyCUDA, работает не только с GPU NVIDIA, но и с любым
# OpenCL-совместимым устройством (CPU, GPU AMD/Intel, FPGA, ...).

import pyopencl as cl
import numpy as np
import time

# Исходный код ядра — такой же, как в C API (строка-литерал).
# В PyCUDA аналог — pycuda.compiler.SourceModule.
# В PyOpenCL — cl.Program(context, source).build().
KERNEL_SOURCE = """
__kernel void vector_add(__global const float* a,
                         __global const float* b,
                         __global float* c,
                         const int n) {
    int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"""


def main():
    # --- Создание контекста и очереди ---
    # cl.create_some_context() — интерактивный выбор платформы/устройства.
    # Если переменная окружения PYOPENCL_CTX задана, выберет автоматически.
    # В C API это: clGetPlatformIDs + clGetDeviceIDs + clCreateContext.
    # В PyCUDA: pycuda.autoinit делает то же самое для единственного GPU.
    ctx = cl.create_some_context()

    # Очередь команд с профилированием — аналог clCreateCommandQueue
    # с флагом CL_QUEUE_PROFILING_ENABLE. В PyCUDA — cuda.Stream().
    queue = cl.CommandQueue(ctx,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Выводим информацию об устройстве
    device = ctx.devices[0]
    print("Устройство: {} [{}]".format(
        device.name.strip(),
        cl.device_type.to_string(device.type)
    ))
    print()

    # --- Подготовка данных ---
    N = 1 << 22  # 4M элементов
    print("N = {} ({:.1f} M элементов)".format(N, N / 1e6))

    # Входные массивы на хосте — обычные NumPy-массивы
    h_a = np.random.randn(N).astype(np.float32)
    h_b = np.random.randn(N).astype(np.float32)

    # --- Создание буферов на устройстве ---
    # cl.Buffer — аналог clCreateBuffer в C API.
    # В PyCUDA: pycuda.gpuarray.to_gpu(array) или cuda.mem_alloc(nbytes).
    #
    # mf.READ_ONLY / WRITE_ONLY — подсказки драйверу для оптимизации.
    # COPY_HOST_PTR — скопировать данные из host-массива при создании буфера
    # (аналог CL_MEM_COPY_HOST_PTR в C API).
    mf = cl.mem_flags
    d_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_a)
    d_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_b)
    d_c = cl.Buffer(ctx, mf.WRITE_ONLY, h_a.nbytes)

    # --- Компиляция и запуск ядра ---
    # cl.Program(ctx, src).build() — JIT-компиляция OpenCL C кода.
    # Аналог clCreateProgramWithSource + clBuildProgram.
    # В PyCUDA: SourceModule(code) компилирует CUDA C через nvcc.
    program = cl.Program(ctx, KERNEL_SOURCE).build()

    # Запускаем ядро vector_add.
    # global_size = (N,) — общее число work-items (аналог gridDim*blockDim в CUDA).
    # local_size = (256,) — размер work-group (аналог blockDim в CUDA).
    # Можно передать local_size=None, тогда драйвер выберет сам.
    local_size = 256
    global_size = ((N + local_size - 1) // local_size) * local_size

    # Замер времени через события профилирования (аналог clGetEventProfilingInfo).
    # В PyCUDA для замера используют cuda.Event + start.time_till(end).
    event = program.vector_add(
        queue,
        (global_size,),       # global work size
        (local_size,),        # local work size (work-group)
        d_a, d_b, d_c,
        np.int32(N)
    )

    # Ждём завершения ядра
    event.wait()

    # Читаем время из события профилирования (наносекунды)
    kernel_time_ns = event.profile.end - event.profile.start
    kernel_time_ms = kernel_time_ns * 1e-6

    # --- Чтение результата обратно на хост ---
    # cl.enqueue_copy — аналог clEnqueueReadBuffer.
    # В PyCUDA: gpuarray.get() или cuda.memcpy_dtoh().
    h_c = np.empty_like(h_a)
    cl.enqueue_copy(queue, h_c, d_c).wait()

    # --- Проверка корректности ---
    expected = h_a + h_b
    max_diff = np.max(np.abs(h_c - expected))
    ok = max_diff < 1e-5

    print("Время ядра:    {:.3f} ms".format(kernel_time_ms))
    print("Макс. ошибка:  {:.2e}".format(max_diff))
    print("check={}".format("OK" if ok else "FAIL"))

    # Вывод первых элементов для наглядности
    print()
    print("a[:5] =", h_a[:5])
    print("b[:5] =", h_b[:5])
    print("c[:5] =", h_c[:5])


if __name__ == "__main__":
    main()
