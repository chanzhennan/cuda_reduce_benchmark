#include <reduce4/reduce4.cuh>
#include <stdio.h>


// template <typename T>
__global__ void reduce4(float *g_idata, float *g_odata) 
{
  extern __shared__ float sdata[];
  
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
  {
    if (tid < s)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}



// template <typename T>
void call_reduce4(float *g_idata, float *g_odata, int dataSize)
{

  int TPB = 512;
  int BSIZE = (dataSize + TPB - 1) / TPB;
  reduce4<<<BSIZE, TPB, TPB * sizeof(float)>>>(g_idata, g_odata);
  cudaDeviceSynchronize();

}
