#include "completely_unroll/completely_unroll.cuh"

template <unsigned int blockSize, typename T>
__device__ void warpReduce6(volatile T *cache, unsigned int tid) {
  if (blockSize >= 64) cache[tid] += cache[tid + 32];
  if (blockSize >= 32) cache[tid] += cache[tid + 16];
  if (blockSize >= 16) cache[tid] += cache[tid + 8];
  if (blockSize >= 8) cache[tid] += cache[tid + 4];
  if (blockSize >= 4) cache[tid] += cache[tid + 2];
  if (blockSize >= 2) cache[tid] += cache[tid + 1];
}

template <size_t blockSize, typename T>
__global__ void reducebase6(T *g_idata, T *g_odata, size_t size) {
  __shared__ float sdata[blockSize];

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
  if (tid < 32) warpReduce6<blockSize>(sdata, tid);

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// PRE:
// dA is an array allocated on the GPU
// N <= len(dA) is a power of two (N >= BLOCKSIZE)
// POST: the sum of the first N elements of dA is returned
template <size_t blockSize, typename T>
T GPUReduction6(T *dA, size_t N) {
  int size = N;
  // thrust::host_vector<int> data_h_i(size, 1);

  int threadsPerBlock = 256;
  int totalBlocks = (size + (threadsPerBlock - 1)) / (2 * threadsPerBlock);

  T *output;
  cudaMalloc((void **)&output, sizeof(T) * totalBlocks);

  bool turn = true;

  while (true) {
    if (turn) {
      reducebase6<blockSize>
          <<<totalBlocks, threadsPerBlock>>>(dA, output, size);
      turn = false;
    } else {
      reducebase6<blockSize>
          <<<totalBlocks, threadsPerBlock>>>(output, dA, size);
      turn = true;
    }

    if (totalBlocks == 1) break;
    size = totalBlocks;
    totalBlocks = ceil((double)totalBlocks / (2 * threadsPerBlock));
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

template float GPUReduction6<1024, float>(float *dA, size_t N);
template int GPUReduction6<1024, int>(int *dA, size_t N);
