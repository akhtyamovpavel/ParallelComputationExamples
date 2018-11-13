#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <ctime>


int main(int argc, char** argv) {
	int num_elements = atoi(argv[1]);
	int *a = new int[num_elements];
	int *b = new int[num_elements];
	int *c = new int[num_elements];
	for (int index = 0; index < num_elements; ++index) {
		a[index] = index;
		b[index] = 2 * index;
	}

	int parallel_index;


	double start_time = omp_get_wtime();

	int threads_counter[10];
	for (int index = 0; index < 10; ++index) {
		threads_counter[index] = 0;
	}
	int thread_id;

	int num_threads = omp_get_num_threads();
	std::cout << num_threads << std::endl;
#pragma omp parallel shared(a, b, c, num_elements, threads_counter) private(parallel_index, thread_id)
	{
		#pragma omp for schedule(static, 10000) nowait
		for (parallel_index = 0; parallel_index < num_elements; ++parallel_index) {
			thread_id = omp_get_thread_num();
			c[parallel_index] = a[parallel_index] + b[parallel_index];
			
			threads_counter[thread_id]++;
		}
	}

	double end_time = omp_get_wtime();

	for (int index = 0; index < 10; ++index) {
		std::cout << threads_counter[index] << " ";
	}
	std::cout << std::endl;
	std::cout << "Time: " << end_time - start_time  << std::endl;

	std::cout << c[10000] << std::endl;
	delete[] a;
	delete[] b;
	delete[] c;
	return 0;
}
