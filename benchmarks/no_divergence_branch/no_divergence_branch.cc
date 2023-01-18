// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "no_divergence_branch/no_divergence_branch.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/benchmark_base.h"
#include "bm_lib/utils.h"

#define BLOCKSIZE 1024

template <typename T>
class NoDivergenceBranch : public cudabm::BenchmarkBase {
 public:
  NoDivergenceBranch() : cudabm::BenchmarkBase(/*enableMonitor=*/true) {}

  void callKernel(benchmark::State &state) {
    dataSize = state.range(0) * state.range(0) * 100;
    // Populate array

    cudaMallocHost(&array, sizeof(T) * dataSize);
    for (size_t i = 0; i < dataSize; i++) array[i] = 1;

    cudaMalloc((void **)&d_array, sizeof(T) * dataSize);
    cudaMemcpy(d_array, array, sizeof(T) * dataSize, cudaMemcpyHostToDevice);

    // call kernel
    result = GPUReduction2<BLOCKSIZE>(d_array, dataSize);

    if (dataSize != (long int)result) {
      std::cout << "dataSize : " << dataSize << '\n';
      std::cout << "result : " << (long int)result << '\n';
      // throw std::invalid_argument("Results are different.");
    }

    cudaFree(d_array);
    cudaFreeHost(array);
  }

  double getDataSize() override { return (double)dataSize; }

 private:
  T *d_array, *array;
  T result = (T)0.;
  long int dataSize;
};

#define BENCHMARK_REDUCE2_OP(name, dType)                              \
  BENCHMARK_TEMPLATE_DEFINE_F(NoDivergenceBranch, name, dType)         \
  (benchmark::State & st) {                                            \
    for (auto _ : st) {                                                \
      callKernel(st);                                                  \
    }                                                                  \
    st.counters["DATASIZE"] = getDataSize();                           \
    st.counters["FLOPS"] = benchmark::Counter{                         \
        getDataSize(), benchmark::Counter::kIsIterationInvariantRate}; \
  }                                                                    \
  BENCHMARK_REGISTER_F(NoDivergenceBranch, name)                       \
      ->Unit(benchmark::kMillisecond)                                  \
      ->RangeMultiplier(2)                                             \
      ->Range(1024, 2048);

#define BENCHMARK_REDUCE2_OP_TYPE(dType) \
  BENCHMARK_REDUCE2_OP(Reduce_##dType, dType)

BENCHMARK_REDUCE2_OP_TYPE(float)
BENCHMARK_REDUCE2_OP_TYPE(int)
