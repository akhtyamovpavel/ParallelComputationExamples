#include <mpi.h>
#include <unistd.h>
#include <iostream>


int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (rank == 0) {
		int num_tasks = 100;
		for (int local_rank = 1; local_rank < world_size; ++local_rank) {
			MPI_Send(&num_tasks, 1, MPI_INT, local_rank, 0, MPI_COMM_WORLD);
		}

        int counts[world_size];
        for (int i = 0; i < world_size; ++i) {
            counts[i] = 0;
        }
		for (int i = 0; i < num_tasks * (world_size - 1); ++i) {
			int flag;
			MPI_Request request;
			MPI_Irecv(&flag, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request);
			MPI_Status status;
			MPI_Wait(&request, &status);
            
            counts[status.MPI_SOURCE]++;

            if (i % 25 == 24) {
			    std::cout << i + 1 << " tasks completed " << std::endl;
                for (int j = 1; j < world_size; ++j) {
                    std::cout << counts[j] << " ";
                }
                std::cout << std::endl;
            }
		}
	} else {
		int num_tasks;
		MPI_Recv(&num_tasks, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		for (int task_id = 0; task_id < num_tasks; ++task_id) {
            // usleep(100000);
			int completed = 1;
			MPI_Request request;
			MPI_Isend(&completed, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
		}
	}
	
	MPI_Finalize();
} 
