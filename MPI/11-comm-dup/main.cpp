#include <mpi.h>
#include <iostream>
#include <vector>

// MPI_Comm_dup: создаёт дубликат коммуникатора с тем же составом рангов,
// но с независимым "контекстом" — собственное пространство тегов и
// отдельные каналы для коллективов.
//
// Главный кейс — изоляция внутренних коммуникаций библиотеки: если
// библиотека шлёт сообщения через MPI_COMM_WORLD с каким-то тегом, они
// могут случайно совпасть с пользовательскими send/recv. Работа через
// свой dup полностью исключает пересечение.
//
// Демонстрация: rank 1 шлёт rank-у 0 два сообщения с одним и тем же
// tag=7 — одно по MPI_COMM_WORLD, другое по lib_comm (dup). На rank 0
// мы сначала принимаем сообщение именно из lib_comm, потом из WORLD.
// Значения разные, и порядок приёма не зависит от порядка отправки —
// сообщения в разных коммуникаторах не смешиваются.

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            std::cout << "нужно минимум 2 процесса" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    MPI_Comm lib_comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &lib_comm);

    const int TAG = 7;
    if (rank == 0) {
        std::cout << "=== MPI_Comm_dup: изоляция тегов между WORLD и dup ==="
                  << std::endl;
    }

    if (rank == 1) {
        int world_msg = 100;
        int lib_msg   = 999;
        // Отправляем в "перепутанном" порядке: сначала WORLD, потом lib.
        MPI_Send(&world_msg, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD);
        MPI_Send(&lib_msg,   1, MPI_INT, 0, TAG, lib_comm);
    } else if (rank == 0) {
        int lib_received   = -1;
        int world_received = -1;
        // Принимаем в обратном порядке — сначала из lib_comm, потом из WORLD.
        // Если бы dup не создавал независимый контекст, мы получили бы из
        // lib_comm сообщение, отправленное в WORLD (оно пришло раньше).
        MPI_Recv(&lib_received,   1, MPI_INT, 1, TAG, lib_comm,
                 MPI_STATUS_IGNORE);
        MPI_Recv(&world_received, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        std::cout << "  из lib_comm (ожидается 999): " << lib_received
                  << std::endl;
        std::cout << "  из WORLD    (ожидается 100): " << world_received
                  << std::endl;
    }

    // Дополнительно — коллектив на lib_comm считает сумму независимо.
    int world_rank_sum = 0;
    MPI_Allreduce(&rank, &world_rank_sum, 1, MPI_INT, MPI_SUM, lib_comm);
    if (rank == 0) {
        int expected = size * (size - 1) / 2;
        std::cout << "  Allreduce на lib_comm: sum(rank)=" << world_rank_sum
                  << " (ожидается " << expected << ")" << std::endl;
    }

    MPI_Comm_free(&lib_comm);
    MPI_Finalize();
    return 0;
}
