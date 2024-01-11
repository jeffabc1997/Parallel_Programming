#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]); // radius
	unsigned long long k = atoll(argv[2]); // % answer
	unsigned long long pixels = 0;

	# pragma omp parallel shared(pixels) 
	{
		unsigned long long radius_sq = r * r, y = 0;
		#pragma omp for schedule(static) nowait
		for(unsigned long long x = 0; x < r; x++) {
			y += ceil(sqrtl(radius_sq - x*x));
			if(y & 1024 == 1) {
				y %= k;
			}
		}
		#pragma omp critical
			pixels += (y % k);
	}
	printf("%llu\n", (4 * pixels) % k);
}
