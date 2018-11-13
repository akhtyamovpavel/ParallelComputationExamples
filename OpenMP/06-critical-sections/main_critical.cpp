#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <ctime>


int main(int argc, char** argv) {
	int num_elements = atoi(argv[1]);
	long long *a = new long long[num_elements];
	long long *b = new long long[num_elements];

	long long result = 0;
	for (int index = 0; index < num_elements; ++index) {
		a[index] = index;
		b[index] = 2 * index;
	}

	int parallel_index;


	double start_time = omp_get_wtime();

	#pragma omp parallel
	{
		int num_threads = omp_get_num_threads();
		int thread_id = omp_get_thread_num();
		int items_per_thread = num_elements / num_threads;

		int left_bound = thread_id * items_per_thread;

		int right_bound = (thread_id == num_threads - 1) ? (num_elements - 1) : (left_bound + items_per_thread - 1);

		for (int index = left_bound; index <= right_bound; ++index) {
			#pragma omp critical
			{
				result += a[index] + b[index];
			}
		}
	}

	double end_time = omp_get_wtime();

	std::cout << "Time: " << end_time - start_time << std::endl;

	std::cout << result << std::endl;
	delete[] a;
	delete[] b;
	return 0;
}
