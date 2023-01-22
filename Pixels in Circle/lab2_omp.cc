#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int getSubNum(int N, int omp_threads, int omp_thread) {
    if(N % omp_threads == 0) {
        return N / omp_threads;
    } else if (omp_thread < N % omp_threads) {
        return N / omp_threads + 1;
    } else {
        return N / omp_threads;
    }
}

int main(int argc, char** argv) {

	int omp_threads, omp_thread;
	int chunk = 100;
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	
    omp_lock_t lock;
    omp_init_lock(&lock);
	#pragma omp parallel private(omp_thread) shared(pixels,r,k)
    {
        omp_threads = omp_get_num_threads();
        omp_thread = omp_get_thread_num();
		#pragma omp for schedule(dynamic,chunk) reduction(+:pixels) nowait
		for(unsigned long long i=0; i<r ;i++){
			unsigned long long y = ceil(sqrtl(r*r - i*i));
			pixels += y;
		}
	}
	pixels %= k;
	omp_destroy_lock(&lock);
	printf("%llu\n", (4 * pixels) % k);
    return 0;
}
