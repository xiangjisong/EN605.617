// Modification of Ingemar Ragnemalm "Real Hello World!" program
// To compile execute below:
// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world

#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE_IN_BYTES(n) (sizeof(int) * (n))

__global__
void hello(int * block, int n)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_idx < (unsigned int)n)
        block[thread_idx] = (int)threadIdx.x;
}

void main_sub(int n, int block_size)
{
    int *gpu_block;
    int *cpu_block = (int*)malloc(sizeof(int) * n);

    for (int i = 0; i < n; i++) cpu_block[i] = -1;

    cudaMalloc((void **)&gpu_block, ARRAY_SIZE_IN_BYTES(n));
    cudaMemcpy(gpu_block, cpu_block, ARRAY_SIZE_IN_BYTES(n), cudaMemcpyHostToDevice);

    int num_blocks = (n + block_size - 1) / block_size;
    hello<<<num_blocks, block_size>>>(gpu_block, n);

    cudaDeviceSynchronize();

    cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES(n), cudaMemcpyDeviceToHost);
    cudaFree(gpu_block);

    printf("\nN=%d, BLOCK_SIZE=%d, NUM_BLOCKS=%d\n", n, block_size, num_blocks);
    for (int i = 0; i < n; i++)
        printf("Calculated Thread: - Block: %2d\n", cpu_block[i]);

    free(cpu_block);
}

int main()
{
    main_sub(16, 16);
    main_sub(32, 16);
    main_sub(64, 32);
    main_sub(100, 32);
    main_sub(256, 64);

    return EXIT_SUCCESS;
}
