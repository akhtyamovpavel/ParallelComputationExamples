#include <mpi.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// MPI_Comm_split: разбиваем MPI_COMM_WORLD на несколько независимых
// подкоммуникаторов. Все процессы вызывают функцию коллективно, каждый
// передаёт свой color — ранги с одинаковым color попадают в общий
// подкоммуникатор; внутри него порядок задаётся параметром key.
//
// В этом примере чётные и нечётные ранги образуют две независимые группы.
// Коллективы (Bcast / Reduce / Allreduce) на подкоммуникаторе затрагивают
// только "своих" — это основной приём для работы только с частью рангов.

// Собираем строки всех рангов к rank 0 внутри переданного коммуникатора,
// чтобы порядок вывода был детерминированным.
static void print_ordered(MPI_Comm comm, const std::string& line) {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (rank == 0) {
        std::cout << line << std::endl;
        for (int r = 1; r < size; ++r) {
            MPI_Status st;
            MPI_Probe(r, 0, comm, &st);
            int nbytes = 0;
            MPI_Get_count(&st, MPI_CHAR, &nbytes);
            std::vector<char> buf(nbytes);
            MPI_Recv(buf.data(), nbytes, MPI_CHAR, r, 0, comm, MPI_STATUS_IGNORE);
            std::cout << std::string(buf.begin(), buf.end()) << std::endl;
        }
    } else {
        MPI_Send(line.data(), static_cast<int>(line.size()), MPI_CHAR,
                 0, 0, comm);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        std::cout << "=== MPI_Comm_split: чётные vs нечётные, world_size="
                  << world_size << " ===" << std::endl;
    }

    int color = world_rank % 2;   // 0 — чётные, 1 — нечётные
    int key   = world_rank;       // упорядочим внутри подгруппы по world_rank
    MPI_Comm sub;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &sub);

    int sub_rank = 0;
    int sub_size = 0;
    MPI_Comm_rank(sub, &sub_rank);
    MPI_Comm_size(sub, &sub_size);

    // Суммируем world_rank только внутри своей группы.
    int local = world_rank;
    int group_sum = 0;
    MPI_Allreduce(&local, &group_sum, 1, MPI_INT, MPI_SUM, sub);

    std::ostringstream oss;
    oss << "  world=" << world_rank << "/" << world_size
        << "  color=" << color
        << "  sub=" << sub_rank << "/" << sub_size
        << "  sum_in_group=" << group_sum;
    print_ordered(MPI_COMM_WORLD, oss.str());

    MPI_Comm_free(&sub);
    MPI_Finalize();
    return 0;
}
