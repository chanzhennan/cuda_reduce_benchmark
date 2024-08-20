#include "ReduceArray0/baseline.cuh"

template <size_t blockSize, typename T>
__global__ void reducebase1(T *g_idata, T *g_odata, size_t size) {
  __shared__ T sdata[blockSize];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = 0;
  if (i < size) sdata[tid] = g_idata[i];
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  __syncthreads();

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// PRE:
// dA is an array allocated on the GPU
// N <= len(dA) is a power of two (N >= BLOCKSIZE)
// POST: the sum of the first N elements of dA is returned
template <size_t blockSize, typename T>
T GPUReduction(T *dA, size_t N) {
  int size = N;
  int totalBlocks = (size + (TPB - 1)) / TPB;

  T *output;
  cudaMalloc((void **)&output, sizeof(T) * totalBlocks);

  bool turn = true;

  while (true) {
    if (turn) {
      reducebase1<blockSize><<<totalBlocks, TPB>>>(dA, output, size);
      turn = false;

    } else {
      reducebase1<blockSize><<<totalBlocks, TPB>>>(output, dA, size);
      turn = true;
    }

    if (totalBlocks == 1) break;
    size = totalBlocks;
    totalBlocks = ceil((double)totalBlocks / TPB);
  }

  T tot = 0.;
  if (turn) {
    cudaMemcpy(&tot, dA, sizeof(T), cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy(&tot, output, sizeof(T), cudaMemcpyDeviceToHost);
  }
  cudaFree(output);

  return tot;
}

template float GPUReduction<TPB, float>(float *dA, size_t N);
template int GPUReduction<TPB, int>(int *dA, size_t N);
