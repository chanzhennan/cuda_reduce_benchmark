#include "unroll_last_warp/unroll_last_warp.cuh"

template <typename T>
__device__ void warpReduce5(volatile T *cache, unsigned int tid) {
  cache[tid] += cache[tid + 32];
  //__syncthreads();
  cache[tid] += cache[tid + 16];
  //__syncthreads();
  cache[tid] += cache[tid + 8];
  //__syncthreads();
  cache[tid] += cache[tid + 4];
  //__syncthreads();
  cache[tid] += cache[tid + 2];
  //__syncthreads();
  cache[tid] += cache[tid + 1];
  //__syncthreads();
}

template <size_t blockSize, typename T>
__global__ void reducebase5(T *g_idata, T *g_odata, size_t size) {
  __shared__ T sdata[blockSize];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  sdata[tid] = 0;

  /* consider this case
   *                                       size = 1600
   *   reduce [-----------------------------------]
   *                                              |
   *           256         256        256         |    threadMax = 2048
   *   thread [------_____------_____------_____------_____]
   *                                            ##|
   *                                            ##|
   *          this section should not add 'g_idata[i + blockDim.x]'
   */
  T adder = i + blockDim.x < size ? g_idata[i + blockDim.x] : 0;

  if (i < size) sdata[tid] = g_idata[i] + adder;
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid < 32) warpReduce5(sdata, tid);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// PRE:
// dA is an array allocated on the GPU
// N <= len(dA) is a power of two (N >= BLOCKSIZE)
// POST: the sum of the first N elements of dA is returned
template <size_t blockSize, typename T>
T GPUReduction5(T *dA, size_t N) {
  int size = N;
  // thrust::host_vector<int> data_h_i(size, 1);

  int totalBlocks = (size + (TPB - 1)) / (2 * TPB);

  T *output;
  cudaMalloc((void **)&output, sizeof(T) * totalBlocks);

  bool turn = true;

  while (true) {
    if (turn) {
      reducebase5<blockSize><<<totalBlocks, TPB>>>(dA, output, size);
      turn = false;
    } else {
      reducebase5<blockSize><<<totalBlocks, TPB>>>(output, dA, size);
      turn = true;
    }

    if (totalBlocks == 1) break;
    size = totalBlocks;
    totalBlocks = ceil((double)totalBlocks / (2 * TPB));
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

template float GPUReduction5<TPB, float>(float *dA, size_t N);
template int GPUReduction5<TPB, int>(int *dA, size_t N);
