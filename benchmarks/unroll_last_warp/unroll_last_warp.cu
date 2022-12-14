#include "unroll_last_warp/unroll_last_warp.cuh"




template <typename T>
__device__ void warpReduce(volatile T* cache, unsigned int tid){
    cache[tid]+=cache[tid+32];
    //__syncthreads();
    cache[tid]+=cache[tid+16];
    //__syncthreads();
    cache[tid]+=cache[tid+8];
    //__syncthreads();
    cache[tid]+=cache[tid+4];
    //__syncthreads();
    cache[tid]+=cache[tid+2];
    //__syncthreads();
    cache[tid]+=cache[tid+1];
    //__syncthreads();
}


template <size_t blockSize, typename T>
__global__ void reducebase(T *g_idata, T *g_odata, size_t size)
{
    __shared__ T sdata[blockSize];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

// PRE:
// dA is an array allocated on the GPU
// N <= len(dA) is a power of two (N >= BLOCKSIZE)
// POST: the sum of the first N elements of dA is returned
template<size_t blockSize, typename T>
T GPUReduction5(T* dA, size_t N)
{
   int size = N;
   // thrust::host_vector<int> data_h_i(size, 1);

   int threadsPerBlock = 256;
   int totalBlocks = (size + (threadsPerBlock - 1)) / threadsPerBlock;

   T *output;
   cudaMalloc((void **)&output, sizeof(T) * totalBlocks);

   bool turn = true;

   while (true)
   {
      if (turn)
      {
         reducebase<blockSize><<<totalBlocks, threadsPerBlock>>>(dA, output, size);
         turn = false;
       }
       else{
         reducebase<blockSize><<<totalBlocks, threadsPerBlock>>>(output, dA, size);
         turn = true;
       }

       if(totalBlocks == 1) break;
       size = totalBlocks;
       totalBlocks = ceil((double)totalBlocks/threadsPerBlock);
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


template float GPUReduction5<1024, float>(float *dA, size_t N);
template int GPUReduction5<1024, int>(int *dA, size_t N);
