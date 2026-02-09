#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 256
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

/* Declare  statically two arrays of ARRAY_SIZE each */
unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];

__global__
void what_is_my_id(unsigned int * block, unsigned int * thread, unsigned int n)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(thread_idx < n)
	{
		block[thread_idx] = blockIdx.x;
		thread[thread_idx] = threadIdx.x;
	}
}

void run_case(unsigned int num_blocks, unsigned int num_threads)
{
	/* Declare pointers for GPU based params */
	unsigned int *gpu_block;
	unsigned int *gpu_thread;

	/* Initialize CPU arrays so output is clean */
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		cpu_block[i] = 0;
		cpu_thread[i] = 0;
	}

	cudaMalloc((void **)&gpu_block, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);

	/* Correct direction: Host -> Device */
	cudaMemcpy(gpu_block, cpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_thread, cpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	/* Execute our kernel */
	what_is_my_id<<<num_blocks, num_threads>>>(gpu_block, gpu_thread, ARRAY_SIZE);
	cudaDeviceSynchronize();

	/* Copy back results */
	cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(gpu_block);
	cudaFree(gpu_thread);

	printf("\nBLOCKS=%u THREADS=%u TOTAL_THREADS=%u\n", num_blocks, num_threads, num_blocks * num_threads);

	/* Iterate through the arrays and print */
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("Thread: %2u - Block: %2u\n", cpu_thread[i], cpu_block[i]);
	}
}

int main()
{
	run_case(16, 16);
	run_case(32, 8);
	run_case(8, 32);
	run_case(4, 64);
	run_case(2, 128);

	return EXIT_SUCCESS;
}
