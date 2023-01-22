// Copyright (c) 2022 Zhennanc Ltd. All rights reserved.
#include "add_during_load/add_during_load.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/utils.h"

template <typename T>
class AddDuringLoad : public benchmark::Fixture {
 public:
  void callKernel(benchmark::State &state) {
    cudaMemcpy(d_array, array, sizeof(T) * dataSize, cudaMemcpyHostToDevice);

    // call kernel
    result = GPUReduction4<TPB>(d_array, dataSize);

    if (dataSize != (long int)result) {
      std::cout << "dataSize : " << dataSize << '\n';
      std::cout << "result : " << (long int)result << '\n';
      // throw std::invalid_argument("Results are different.");
    }
  }
  void SetUp(const ::benchmark::State &state) BENCHMARK_OVERRIDE {
    dataSize = state.range(0) * state.range(0) * 100;
    // Populate array
    cudaMallocHost(&array, sizeof(T) * dataSize);
    for (size_t i = 0; i < dataSize; i++) array[i] = 1;

    cudaMalloc((void **)&d_array, sizeof(T) * dataSize);
  }

  void TearDown(const ::benchmark::State &st) BENCHMARK_OVERRIDE {
    cudaFree(d_array);
    cudaFreeHost(array);
  }
  double getDataSize() { return (double)dataSize; }

 private:
  T *d_array, *array;
  T result = (T)0.;
  long int dataSize;
};

#define BENCHMARK_REDUCE4_OP(name, dType)                              \
  BENCHMARK_TEMPLATE_DEFINE_F(AddDuringLoad, name, dType)              \
  (benchmark::State & st) {                                            \
    for (auto _ : st) {                                                \
      callKernel(st);                                                  \
    }                                                                  \
    st.counters["DATASIZE"] = getDataSize();                           \
    st.counters["FLOPS"] = benchmark::Counter{                         \
        getDataSize(), benchmark::Counter::kIsIterationInvariantRate}; \
  }                                                                    \
  BENCHMARK_REGISTER_F(AddDuringLoad, name)                            \
      ->Unit(benchmark::kMillisecond)                                  \
      ->RangeMultiplier(2)                                             \
      ->Range(1024, 2048);

#define BENCHMARK_REDUCE4_OP_TYPE(dType) \
  BENCHMARK_REDUCE4_OP(Reduce_##dType, dType)

BENCHMARK_REDUCE4_OP_TYPE(float)
BENCHMARK_REDUCE4_OP_TYPE(int)
