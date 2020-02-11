#include <iostream>
#include <cmath>

void add(int n, float* x, float* y) {
	for (int i = 0; i < n; ++i) {
		y[i] = x[i] + y[i];
	}	
}


int main() {
	int N = 1 << 28;
	float* x = new float[N];
	float* y = new float[N];

	for (int i = 0; i < N; ++i) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	add(N, x, y);
	
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(y[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;
	delete [] x;
	delete [] y;
	return 0;
}
