#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <utility>
#include <iostream>

void* calculate(void* info) {
    unsigned long long* id_radius = (unsigned long long*) info;
	unsigned long long thread_id = id_radius[0];
	unsigned long long radius = id_radius[1];
	unsigned long long ncpus = id_radius[2];
	unsigned long long k = id_radius[3];
	unsigned long long radius_sq = radius*radius;
    
	unsigned long long total = 0;
	
	unsigned long long *startInd = new unsigned long long[ncpus];
	unsigned long long *endInd = new unsigned long long[ncpus];
	startInd[0] = 0;
	endInd[0] = (radius%ncpus > 0) ? radius / ncpus: radius / ncpus - 1; 
	for(unsigned long long i = 1; i < ncpus; i++) {
		startInd[i] = endInd[i-1] + 1;
		endInd[i] = (radius%ncpus > i) ? radius / ncpus + startInd[i]: radius / ncpus - 1 + startInd[i]; 
	}
	// if(thread_id == 0) {
	// 	for(unsigned long long i = 0; i < ncpus; i++) {
	// 		cout << startInd[i] << ", " << endInd[i] << endl;
	// 	}
	// 		// cout << x << endl;
	// }
	for (unsigned long long x = startInd[thread_id], y; x <= endInd[thread_id]; x++) {
		y = ceil(sqrtl(radius_sq - x*x));
		
		total += y;
		if(x & 64 == 1) {
			total %= k;
		}
		
	}
	delete []startInd;
	delete []endInd;
	unsigned long long* res = new unsigned long long;
	*res = total;
	

	// std::cout << thread_id << ", " << total << std::endl;
    pthread_exit((void *) res);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long radius = atoll(argv[1]); // radius
	unsigned long long k = atoll(argv[2]); // % answer

	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);

	unsigned long long pixels = 0;
	pthread_t threads[ncpus];
	unsigned long long ID[ncpus];
	void* ret[ncpus];
	unsigned long long id_radius[ncpus][4];
	int rc;
	
	for (unsigned long long t = 0; t < ncpus; t++) {
        ID[t] = t;
		id_radius[t][0] = ID[t];
		id_radius[t][1] = radius;
		id_radius[t][2] = ncpus;
		id_radius[t][3] = k;
        rc = pthread_create(&threads[t], NULL, calculate, (void*)&id_radius[t]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }	
    }

	for(int i = 0; i < ncpus; i++) {
		pthread_join(threads[i], &ret[i]);
	}
	for(int i = 0; i < ncpus; i++) {
		unsigned long long* res = (unsigned long long*) ret[i];
		pixels += (*res % k);
	}

	printf("%llu\n", (4 * pixels) % k);
}
