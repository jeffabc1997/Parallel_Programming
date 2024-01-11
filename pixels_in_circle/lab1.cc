#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long radius = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long result = 0;
    unsigned long long square_radius = radius * radius;
	int numtasks, rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	
	for(unsigned long long int r = rank; r < radius; r += numtasks) {
		unsigned long long y = ceil(sqrtl(square_radius - r * r));
		pixels += y;
		pixels %= k;
	}
	// printf("Hello. I am %d of %d. I have %d pixels\n", rank, numtasks, pixels);
	
	/* should Add up all results here */
	MPI_Reduce(&pixels, &result, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	
    MPI_Finalize();
	if(rank == 0) {
		printf("%llu\n", (4 * result) % k); // mutiply by 4
	}
    return 0;
} 

/*
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	for (unsigned long long x = 0; x < r; x++) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
		pixels %= k;
	}
	printf("%llu\n", (4 * pixels) % k);
} */
// int main(int argc, char *argv[])
// {
// 	if (argc != 3) {
// 		fprintf(stderr, "must provide exactly 2 arguments!\n");
// 		return 1;
// 	}
// 	unsigned long long radius = atoll(argv[1]);
// 	unsigned long long k = atoll(argv[2]);
// 	unsigned long long pixels = 0;
// 	unsigned long long result = 0;
//     int rank, numtasks;
// 	int rootID = 0;
//     MPI_Init(&argc, &argv);

//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	
// 	for(unsigned long long r = rank; r < radius; r += numtasks) {
// 		unsigned long long y = ceil(sqrtl(radius * radius - r * r));
// 		pixels += y;
// 		pixels %= k;
// 		// int testR = 10090;
// 		// if((rank == 3) && (r > testR) && (r < (testR+50))) {
// 		// 	cout << y << endl;
// 		// }
// 	}
// 	// printf("Hello. I am %d of %d. I have %d pixels\n", rank, numtasks, pixels);
	
// 	/* should Add up all results here */
// 	if(rank == rootID) {
// 		int* buffer = (int*)malloc(sizeof(int) * numtasks);
// 		MPI_Gather(&pixels, 1, MPI_INT, buffer, 1, MPI_INT, rootID, MPI_COMM_WORLD);
// 		// printf("Values collected on process %d: %d, %d, %d, %d.\n", rank, buffer[0], buffer[1], buffer[2], buffer[3]);
// 		for(int i = 0; i < numtasks; i++) {
// 			printf("Values collected on process %d: %d\n", i, buffer[i]);
// 			result += buffer[i];
// 			result %= k;
// 		}
// 		printf("%llu\n", (4 * result) % k);
// 	} else {
// 		MPI_Gather(&pixels, 1, MPI_INT, NULL, 0, MPI_INT, rootID, MPI_COMM_WORLD);
// 	}
// 	// MPI_Reduce(&pixels, &result, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	
//     MPI_Finalize();
// 	// if(rank == 0) {
// 	// 	printf("%llu\n", (4 * result) % k); // mutiply by 4
// 	// }
//     return 0;
// } 