#include <reduce5/reduce5.cuh>
#include <stdio.h>


__device__ void warpReduce(volatile float* sdata, int tid) 
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// template <typename T>
__global__ void reduce5(float *g_idata, float *g_odata) 
{
  extern __shared__ float sdata[];
  
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>32; s>>=1) 
  {
    if (tid < s)
    sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid < 32) warpReduce(sdata, tid);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}



// template <typename T>
void call_reduce5(float *g_idata, float *g_odata, int dataSize)
{

  int TPB = 512;
  int BSIZE = (dataSize + TPB - 1) / TPB;
  reduce5<<<BSIZE, TPB, TPB * sizeof(float)>>>(g_idata, g_odata);
  cudaDeviceSynchronize();

}
