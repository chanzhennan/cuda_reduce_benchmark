#include "reduce_baseline/reduce_baseline.cuh"

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(EXIT_FAILURE); 
    }
}

template <size_t blockSize, typename T> __device__ void warpReduce(volatile T *sdata, size_t tid)
{
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
    if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
    if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
    if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
}

template <size_t blockSize, typename T> 
__global__ void reduceCUDA(T* g_idata, T* g_odata, size_t n)
{
    __shared__ T sdata[blockSize];

    size_t tid = threadIdx.x;
    //size_t i = blockIdx.x*(blockSize*2) + tid;
    //size_t gridSize = blockSize*2*gridDim.x;
    size_t i = blockIdx.x*(blockSize) + tid;
    size_t gridSize = blockSize*gridDim.x;
    sdata[tid] = 0;

    while (i < n) { sdata[tid] += g_idata[i]; i += gridSize; }
    //while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >=  512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >=  256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >=  128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <size_t blockSize, typename T>
__global__ void reduce1(T *g_idata, T *g_odata, size_t n) 
{
    __shared__ T sdata[blockSize];
  
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t gridSize = blockSize*gridDim.x;
  
//   sdata[tid] = g_idata[i];
    while (i < n) { sdata[tid] += g_idata[i]; i += gridSize; }

  __syncthreads();
  // do reduction in shared mem
  for(unsigned int s=1; s < blockDim.x; s *= 2)
  {
    if (tid % (2*s) == 0) 
    {
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
T GPUReduction(T* dA, size_t N)
{
    T tot = 0.;
    size_t n = N;
    size_t blocksPerGrid = std::ceil((1.*n) / blockSize);

    T* tmp;
    cudaMalloc(&tmp, sizeof(T) * blocksPerGrid); checkCUDAError("Error allocating tmp [GPUReduction]");

    T* from = dA;

    do
    {
        blocksPerGrid   = std::ceil((1.*n) / blockSize);
        reduce1<blockSize><<<blocksPerGrid, blockSize>>>(from, tmp, n);
        from = tmp;
        n = blocksPerGrid;
        std::cout << "n " << n << std::endl;
    } while (n > blockSize);

    if (n > 1)
        reduce1<blockSize><<<1, blockSize>>>(tmp, tmp, n);

    cudaDeviceSynchronize();
    checkCUDAError("Error launching kernel [GPUReduction]");

    cudaMemcpy(&tot, tmp, sizeof(T), cudaMemcpyDeviceToHost); checkCUDAError("Error copying result [GPUReduction]");
    cudaFree(tmp);
    return tot;
}

// template unsigned int GPUReduction<1024, unsigned int>(unsigned int *dA, size_t N);
template float GPUReduction<1024, float>(float *dA, size_t N);
template int GPUReduction<1024, int>(int *dA, size_t N);



// // template <typename T>
// void call_reduce1(float *g_idata, float *g_odata, int dataSize)
// {

//   int BSIZE = (dataSize + TPB - 1) / TPB;
//   reduce1<<<BSIZE, TPB, TPB * sizeof(float)>>>(g_idata, g_odata);
//   cudaDeviceSynchronize();

// }


// void checkCUDAError(const char *msg)
// {
//     cudaError_t err = cudaGetLastError();
//     if( cudaSuccess != err)
//     {
//         fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err) );
//         exit(EXIT_FAILURE); 
//     }
// }