#include "add_during_load/add_during_load.cuh"


template <size_t blockSize, typename T>
__global__ void reducebase4(T *g_idata, T *g_odata, size_t size)
{
 __shared__ T sdata[blockSize];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = 0;

    if(i < size)
      sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// PRE:
// dA is an array allocated on the GPU
// N <= len(dA) is a power of two (N >= BLOCKSIZE)
// POST: the sum of the first N elements of dA is returned
template<size_t blockSize, typename T>
T GPUReduction4(T* dA, size_t N)
{
   int size = N;
   // thrust::host_vector<int> data_h_i(size, 1);

   int threadsPerBlock = 256;
   int totalBlocks = (size + (threadsPerBlock - 1)) / (2 * threadsPerBlock);

   T *output;
   cudaMalloc((void **)&output, sizeof(T) * totalBlocks);

   bool turn = true;

   while (true)
   {
      if (turn)
      {
         reducebase4<blockSize><<<totalBlocks, threadsPerBlock>>>(dA, output, size);
         turn = false;
       }
       else{
         reducebase4<blockSize><<<totalBlocks, threadsPerBlock>>>(output, dA, size);
         turn = true;
       }

       if(totalBlocks == 1) break;
       size = totalBlocks;
       totalBlocks = ceil((double)totalBlocks/( 2 * threadsPerBlock));
     }
     cudaDeviceSynchronize();

     T tot = 0.;

     if(turn)
     {
       cudaMemcpy(&tot, dA, sizeof(T), cudaMemcpyDeviceToHost);
     }
     else
     {
       cudaMemcpy(&tot, output, sizeof(T), cudaMemcpyDeviceToHost);
     }
     cudaFree(output);
     //  std::cout << tot << std::endl;

     return tot;


}


template float GPUReduction4<1024, float>(float *dA, size_t N);
template int GPUReduction4<1024, int>(int *dA, size_t N);
