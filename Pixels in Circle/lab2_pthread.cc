#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#define MAX_THREAD 10000

unsigned long long pixels = 0;
unsigned long long num_thread;
unsigned long long r = 0;
unsigned long long k = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

struct part {
   int tid;
   int threadnum;
   unsigned long long idx;
   unsigned long long subnum;
};

void* calculate(void* tmp) {
	struct part* thisPart = (struct part*)tmp;
    unsigned long long partSum = 0;
    for(int i = 0; i < thisPart->subnum; i++){
		unsigned long long y = ceil(sqrtl(r*r - ((thisPart->idx + i) * (thisPart->idx + i))));
		partSum += y;
		
	}
	partSum %= k;
	pthread_mutex_lock(&mutex);
    pixels += partSum;
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
	
	r = atoll(argv[1]);
	k = atoll(argv[2]);
	
	int num_thread = MAX_THREAD;

    pthread_t threads[MAX_THREAD];
    int rc;
	struct part all_threads[MAX_THREAD];
	
	//cpu_set_t cpuset;
	//sched_getaffinity(0, sizeof(cpuset), &cpuset);
	//unsigned long long ncpus = CPU_COUNT(&cpuset);
	
	if(r < MAX_THREAD){
		num_thread = r;
	}
    for (int t = 0; t < num_thread; t++) {
		
		if(r > MAX_THREAD){
			all_threads[t].subnum = (t==MAX_THREAD-1) ? (unsigned long long) r/num_thread + r%num_thread : (unsigned long long) r/num_thread;
			
		}
		else{
			all_threads[t].subnum = 1;
		}
		all_threads[t].idx = (unsigned long long) t*(r/num_thread);
		all_threads[t].tid = t;
		all_threads[t].threadnum = num_thread;

        rc = pthread_create(&threads[t], NULL, calculate, (void*)&all_threads[t]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
	
	for (unsigned long long i=0; i<num_thread; i++) 
		pthread_join(threads[i], NULL);
	
	printf("%llu\n", (4 * pixels) % k);
	
    pthread_exit(NULL);
	
	return 0;
}

