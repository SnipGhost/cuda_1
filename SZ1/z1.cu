/*
 * Solve-1 by SnipGhost 22.03.2017
 */
#include <stdio.h>
#include <stdlib.h>

#define INCREMENT 5
#define BLOCK_SIZE 8

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


__global__ void kernel_simulate(float *data, float *buff, const int size, const float Hx, const float Ht)
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid == 0) {
		buff[tid] = 0;
		return;
	} else if (tid == size-1) {
		buff[tid] += INCREMENT;
		return;
	}
	buff[tid] += ((data[tid+1] - 2*data[tid] + data[tid-1]) * Ht) / (Hx * Hx);
}

void initialize(float *d, int size, float init = 0)
{
	for (int i = 0; i < size; ++i)
		d[i] = init;
}

void save_data(FILE *f, float *d, int size, int z)
{
	 if (!f)
		 printf("File output error\n");
	 else
		 for (int i = 0; i < size; ++i)
			 fprintf(f, "%4d %4d %20.4f\n", i, z, d[i]);
}

void print_data(float *d, int size, int z)
{
	for (int i = 0; i < size; ++i)
		printf("%4d %4d %10.4f\n", i, z, d[i]);
}

int main(void)
{
	const int S_LENGTH = 32;
	const int T_LENGTH = 16;

	const float Hx = 0.5;
	const float Ht = 0.05;

	const int NODES = S_LENGTH / Hx;
	const int TICKS = T_LENGTH / Ht;
	const int numBytes = sizeof(float) * NODES;

	FILE *f = fopen("out","w");

	dim3 threads(BLOCK_SIZE);
	dim3 blocks(NODES/BLOCK_SIZE);

	float *data_dev, *buff_dev, *data_host;

	data_host = (float*)malloc(numBytes);

	initialize(data_host, NODES);

	CUDA_CHECK_RETURN(cudaMalloc((void**)&data_dev, numBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&buff_dev, numBytes));
	CUDA_CHECK_RETURN(cudaMemcpy(data_dev, data_host, numBytes, cudaMemcpyHostToDevice));

	for (int t = 0; t < TICKS; ++t)
	{
		kernel_simulate <<< blocks, threads >>> (data_dev, buff_dev, NODES, Hx, Ht);
		CUDA_CHECK_RETURN(cudaGetLastError());
		CUDA_CHECK_RETURN(cudaMemcpy(data_dev, buff_dev, numBytes, cudaMemcpyDeviceToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(data_host, buff_dev, numBytes, cudaMemcpyDeviceToHost));
		save_data(f, data_host, NODES, t);
	}

	CUDA_CHECK_RETURN(cudaFree((void*)data_dev));
	CUDA_CHECK_RETURN(cudaFree((void*)buff_dev));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	free(data_host);

	fclose(f);

	return 0;
}
