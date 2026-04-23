#include <mpi.h>
#include <iostream>
#include <vector>

// MPI_Group — это просто набор рангов без способности обмениваться
// сообщениями. Из него создаётся коммуникатор. Схема такая:
//   1) MPI_Comm_group(comm, &g)          — достаём группу коммуникатора;
//   2) MPI_Group_incl / excl / union / ...— собираем новую группу из рангов;
//   3) MPI_Comm_create_group(comm, g, ...) — создаём коммуникатор из группы.
//
// Отличие от MPI_Comm_split: split — коллективная операция на ВСЕХ
// рангах comm; create_group вызывают только те, кто входит в группу.
// У остальных рангов коммуникатор просто не появляется (их это не волнует),
// и внутри функции demo_group() они в ветку if не попадают.
//
// В этом примере берём "первую половину" мировых рангов и делаем на ней
// Reduce(MPI_MAX). Ранги из второй половины в create_group не заходят
// и не участвуют в редукции — это важно: иначе коллектив повиснет.

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        std::cout << "=== MPI_Group_incl + MPI_Comm_create_group, world_size="
                  << world_size << " ===" << std::endl;
    }

    int half = (world_size + 1) / 2;
    bool in_group = (world_rank < half);

    if (in_group) {
        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);

        std::vector<int> ranks;
        ranks.reserve(half);
        for (int r = 0; r < half; ++r) {
            ranks.push_back(r);
        }

        MPI_Group first_half;
        MPI_Group_incl(world_group, static_cast<int>(ranks.size()),
                       ranks.data(), &first_half);

        // create_group — неколлективен на MPI_COMM_WORLD: вызывают только
        // участники новой группы. Второй аргумент (tag) должен совпадать.
        MPI_Comm half_comm = MPI_COMM_NULL;
        MPI_Comm_create_group(MPI_COMM_WORLD, first_half, 0, &half_comm);

        int hr = 0;
        int hs = 0;
        MPI_Comm_rank(half_comm, &hr);
        MPI_Comm_size(half_comm, &hs);

        int max_world = 0;
        MPI_Reduce(&world_rank, &max_world, 1, MPI_INT, MPI_MAX, 0, half_comm);

        if (hr == 0) {
            std::cout << "  в группе " << hs << " рангов, max(world_rank)="
                      << max_world << std::endl;
        }

        MPI_Comm_free(&half_comm);
        MPI_Group_free(&first_half);
        MPI_Group_free(&world_group);
    }

    // Барьер на WORLD просто чтобы вывод второй группы не смешивался
    // с финальным сообщением.
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank >= half) {
        // Процессы "второй половины" сюда попадают и молчат: у них своего
        // коммуникатора нет, они просто не участвуют в коллективе выше.
    }

    MPI_Finalize();
    return 0;
}
