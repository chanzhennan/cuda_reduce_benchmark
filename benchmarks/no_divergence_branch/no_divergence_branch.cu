#include "no_divergence_branch/no_divergence_branch.cuh"


template <size_t blockSize, typename T>
__global__ void reducebase(T *g_idata, T *g_odata, size_t size)
{
  __shared__ float sdata[blockSize];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
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
T GPUReduction2(T* dA, size_t N)
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


template float GPUReduction2<1024, float>(float *dA, size_t N);
template int GPUReduction2<1024, int>(int *dA, size_t N);
