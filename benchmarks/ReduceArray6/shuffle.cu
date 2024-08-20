#include "ReduceArray6/shuffle.cuh"

#define WARP_SIZE 32

template <unsigned int blockSize, typename T>
__device__ __forceinline__ T warpReduceSum(T sum) {
  if (blockSize >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (blockSize >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (blockSize >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (blockSize >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (blockSize >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}

template <size_t blockSize, typename T>
__global__ void reducebase6(T *g_idata, T *g_odata, size_t size) {
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  T sum = i < size ? g_idata[i] : 0;
  __syncthreads();

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ T warpLevelSums[WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;

  sum = warpReduceSum<blockSize>(sum);

  if (laneId == 0) warpLevelSums[warpId] = sum;
  __syncthreads();

  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum<blockSize / WARP_SIZE>(sum);

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sum;
}

// PRE:
// dA is an array allocated on the GPU
// N <= len(dA) is a power of two (N >= BLOCKSIZE)
// POST: the sum of the first N elements of dA is returned
template <size_t blockSize, typename T>
T GPUReduction6(T *dA, size_t N) {
  int size = N;

  int totalBlocks = (size + (TPB - 1)) / (TPB);

  T *output;
  cudaMalloc((void **)&output, sizeof(T) * totalBlocks);

  T *tmp;
  cudaMallocHost((void **)&tmp, sizeof(T) * totalBlocks);

  bool turn = true;
  int iter = 0;
  while (true) {
    if (turn) {
      reducebase6<blockSize><<<totalBlocks, TPB>>>(dA, output, size);
      turn = false;
    } else {
      reducebase6<blockSize><<<totalBlocks, TPB>>>(output, dA, size);
      turn = true;
    }

    if (totalBlocks == 1) break;
    size = totalBlocks;
    totalBlocks = ceil((double)totalBlocks / (TPB));
    iter++;
  }
  cudaDeviceSynchronize();

  T tot = 0.;

  if (turn) {
    cudaMemcpy(&tot, dA, sizeof(T), cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(&tot, output, sizeof(T), cudaMemcpyDeviceToHost);
  }
  cudaFree(output);

  return tot;
}

template float GPUReduction6<TPB, float>(float *dA, size_t N);
template int GPUReduction6<TPB, int>(int *dA, size_t N);
