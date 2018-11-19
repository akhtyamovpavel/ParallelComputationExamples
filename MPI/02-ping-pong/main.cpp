#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
	
	MPI_Init(NULL, NULL);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int world_rank;

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	
	int number;

	int ping_pong_count = 0;
	
	int partner_rank = (world_rank + 1) % 2;
	int PING_PONG_LIMIT = 10;

	while (ping_pong_count < PING_PONG_LIMIT) {
		if (ping_pong_count % 2 == world_rank) {
			++ping_pong_count;
			MPI_Send(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
			std::cout << world_rank << " sent ping_pong_count " << ping_pong_count << " to " << partner_rank << std::endl;
		} else {
			MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			std::cout << world_rank << " received ping_pong_count " << ping_pong_count << " from " << partner_rank << std::endl;
		}
	}

	
	MPI_Finalize();
	return 0;
}
