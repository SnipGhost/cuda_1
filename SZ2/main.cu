/*
 * Solve-2 by SnipGhost 22.03.2017
 */
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 8

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


__device__ float f(const int x, const int y, const int t)
{
	if (t > 256) return 0;
	if (x == 0 && y == 0) return 5000;
	if (x*x+y*y <= 10) return 4000;
	else return 0;
}

__global__ void kernel_simulate(const float *z, float *b, const int t,
		                        const int xsize, const int ysize,
		                        const float Hx, const float Hy, const float Ht)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned idl = idy * xsize + idx;
	float top = 0, bottom = 0, left = 0, right = 0;
	if (idx+1 < xsize) right = z[idl+1];
	if (idx-1 >= 0) left = z[idl-1];
	if (idy+1 < ysize) bottom = z[idl+xsize];
	if (idy-1 >= 0) top = z[idl-xsize];
	float dx = (right - 2*z[idl] + left) / (Hx * Hx);
	float dy = (bottom - 2*z[idl] + top) / (Hy * Hy);
	b[idl] += (5*(dx+dy) + f(idx-xsize/2, idy-ysize/2, t)) * Ht;
}

void initialize(float *d, int size, float init = 0)
{
	for (int i = 0; i < size; ++i)
		d[i] = init;
}

void save_data(FILE *file, float *d, int xsize, int ysize, int t)
{
	 if (!file)
		 printf("File output error\n");
	 else
		 for (int i = 0; i < xsize*ysize; ++i)
			 fprintf(file, "%4d %4d %20.14f %4d\n", i/xsize, i%xsize, d[i], t);
}

int main(void)
{

	const float Hx = 1;
	const float Hy = 1;
	const float Ht = 0.05;

	const int XNODES = 64;
	const int YNODES = 64;
	const int TICKS = 800;
	const int numBytes = sizeof(float) * XNODES * YNODES;

	FILE *file = fopen("out","w");

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(XNODES/BLOCK_SIZE, YNODES/BLOCK_SIZE);

	float *data_dev, *buff_dev, *data_host;

	data_host = (float*)malloc(numBytes);

	initialize(data_host, XNODES*YNODES);

	CUDA_CHECK_RETURN(cudaMalloc((void**)&data_dev, numBytes));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&buff_dev, numBytes));
	CUDA_CHECK_RETURN(cudaMemcpy(data_dev, data_host, numBytes, cudaMemcpyHostToDevice));

	for (int t = 0; t < TICKS; ++t)
	{
		kernel_simulate <<< blocks, threads >>> (data_dev, buff_dev, t, XNODES, YNODES, Hx, Hy, Ht);
		CUDA_CHECK_RETURN(cudaGetLastError());
		CUDA_CHECK_RETURN(cudaMemcpy(data_dev, buff_dev, numBytes, cudaMemcpyDeviceToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(data_host, buff_dev, numBytes, cudaMemcpyDeviceToHost));
		save_data(file, data_host, XNODES, YNODES, t);
	}

	CUDA_CHECK_RETURN(cudaFree((void*)data_dev));
	CUDA_CHECK_RETURN(cudaFree((void*)buff_dev));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	free(data_host);

	fclose(file);

	return 0;
}
