#include <reduce2/reduce2.cuh>
#include <stdio.h>


// template <typename T>
__global__ void reduce2(float *g_idata, float *g_odata) 
{
  extern __shared__ float sdata[];
  
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  sdata[tid] = g_idata[i];
  __syncthreads();
  // do reduction in shared mem
  for (unsigned int s=1; s < blockDim.x; s *= 2) 
  {
    int index = 2 * s * tid;
    if (index < blockDim.x) 
    {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}



// template <typename T>
void call_reduce2(float *g_idata, float *g_odata, int dataSize)
{

  int TPB = 512;
  int BSIZE = (dataSize + TPB - 1) / TPB;
  reduce2<<<BSIZE, TPB, TPB * sizeof(float)>>>(g_idata, g_odata);
  cudaDeviceSynchronize();

}
