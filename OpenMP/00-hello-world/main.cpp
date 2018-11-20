#include <iostream>
#include <omp.h>

int main(int argc, char** argv) {
	int num_threads;
	int thread_id;
#pragma omp parallel private(thread_id)
	{
		thread_id = omp_get_thread_num();
		#pragma omp critical
		{
			std::cout << "Hello world from thread " << thread_id << std::endl;
		}
		
		if (thread_id == 0) {
			num_threads = omp_get_num_threads();

			std::cout << "Number of threads " << num_threads << std::endl;
		}
	}
}
