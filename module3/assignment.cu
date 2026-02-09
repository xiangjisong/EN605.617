//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

__global__ void kernel_nobranch(const float *a, const float *b, float *out, int n)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = tid; i < n; i += stride)
		out[i] = a[i] * 1.618f + b[i] * 0.414f;
}

__global__ void kernel_branch(const float *a, const float *b, float *out, int n)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for (int i = tid; i < n; i += stride)
	{
		float x = a[i];
		if (x > 0.0f) out[i] = x * 1.618f + b[i];
		else          out[i] = x * 0.414f - b[i];
	}
}

static void cpu_nobranch(const float *a, const float *b, float *out, int n)
{
	for (int i = 0; i < n; i++)
		out[i] = a[i] * 1.618f + b[i] * 0.414f;
}

static void cpu_branch(const float *a, const float *b, float *out, int n)
{
	for (int i = 0; i < n; i++)
	{
		float x = a[i];
		if (x > 0.0f) out[i] = x * 1.618f + b[i];
		else          out[i] = x * 0.414f - b[i];
	}
}

static void fill_data(float *a, float *b, int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = (float)((i % 100) - 50) * 0.01f;
		b[i] = (float)((i % 200) - 100) * 0.005f;
	}
}

static float run_gpu(int numBlocks, int blockSize, int n, int use_branch,
					 const float *h_a, const float *h_b, float *h_out)
{
	float *d_a, *d_b, *d_out;
	cudaMalloc((void**)&d_a, n * sizeof(float));
	cudaMalloc((void**)&d_b, n * sizeof(float));
	cudaMalloc((void**)&d_out, n * sizeof(float));

	cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	if (use_branch) kernel_branch<<<numBlocks, blockSize>>>(d_a, d_b, d_out, n);
	else            kernel_nobranch<<<numBlocks, blockSize>>>(d_a, d_b, d_out, n);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);

	cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);

	return ms;
}

static long long run_cpu(int n, int use_branch, const float *h_a, const float *h_b, float *h_out)
{
	auto t0 = std::chrono::high_resolution_clock::now();
	if (use_branch) cpu_branch(h_a, h_b, h_out, n);
	else            cpu_nobranch(h_a, h_b, h_out, n);
	auto t1 = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;

	if (argc >= 2) totalThreads = atoi(argv[1]);
	if (argc >= 3) blockSize = atoi(argv[2]);

	if (blockSize <= 0) blockSize = 256;
	if (totalThreads <= 0) totalThreads = (1 << 20);

	int numBlocks = totalThreads / blockSize;

	// validate command line arguments (keep sample behavior)
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks * blockSize;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	if (numBlocks < 1) numBlocks = 1;

	int n = 1 << 22; // ~4 million floats

	float *h_a = (float*)malloc(n * sizeof(float));
	float *h_b = (float*)malloc(n * sizeof(float));
	float *h_out_gpu = (float*)malloc(n * sizeof(float));
	float *h_out_cpu = (float*)malloc(n * sizeof(float));

	fill_data(h_a, h_b, n);

	printf("blocks=%d blockSize=%d totalThreads=%d n=%d\n", numBlocks, blockSize, totalThreads, n);

	float gpu_nb_ms = run_gpu(numBlocks, blockSize, n, 0, h_a, h_b, h_out_gpu);
	long long cpu_nb_us = run_cpu(n, 0, h_a, h_b, h_out_cpu);

	float gpu_br_ms = run_gpu(numBlocks, blockSize, n, 1, h_a, h_b, h_out_gpu);
	long long cpu_br_us = run_cpu(n, 1, h_a, h_b, h_out_cpu);

	printf("GPU_nobranch_ms=%.4f CPU_nobranch_us=%lld\n", gpu_nb_ms, cpu_nb_us);
	printf("GPU_branch_ms=%.4f CPU_branch_us=%lld\n", gpu_br_ms, cpu_br_us);

	free(h_a);
	free(h_b);
	free(h_out_gpu);
	free(h_out_cpu);

	return 0;
}
