#include <reduce6/reduce6.cuh>
#include <stdio.h>

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* cache, unsigned int tid)
{
    if (blockSize >= 64)cache[tid]+=cache[tid+32];
    if (blockSize >= 32)cache[tid]+=cache[tid+16];
    if (blockSize >= 16)cache[tid]+=cache[tid+8];
    if (blockSize >= 8)cache[tid]+=cache[tid+4];
    if (blockSize >= 4)cache[tid]+=cache[tid+2];
    if (blockSize >= 2)cache[tid]+=cache[tid+1];
}

template <unsigned int blockSize>
__global__ void reduce6(float *g_idata, float *g_odata) 
{
  extern __shared__ float sdata[];
  
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
  __syncthreads();

   // do reduction in shared mem
    if (blockSize >= 512) {
        if (tid < 256) { 
            sdata[tid] += sdata[tid + 256]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) {
        if (tid < 128) { 
            sdata[tid] += sdata[tid + 128]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) {
        if (tid < 64) { 
            sdata[tid] += sdata[tid + 64]; 
        } 
        __syncthreads(); 
    }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// template <typename T>
void call_reduce6(float *g_idata, float *g_odata, int dataSize)
{

  int TPB = 512;
  int BSIZE = (dataSize + TPB - 1) / TPB;
  reduce6<512><<<BSIZE, TPB, TPB * sizeof(float)>>>(g_idata, g_odata);
  cudaDeviceSynchronize();

}
