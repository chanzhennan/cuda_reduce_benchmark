// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/benchmark_base.h"
#include "bm_lib/utils.h"
#include "reduce1/reduce1.cuh"


namespace bm = benchmark;

enum class DataType {
  INT,
  FLOAT,
};

template <typename T>
class Reduce1 : public cudabm::BenchmarkBase {
 public:
  Reduce1() : cudabm::BenchmarkBase(/*numIpus=*/1, /*enableMonitor=*/true) {}

  void callKernel(size_t data_size) {

    T *h_data, *d_data;
    T *h_res, *d_res;
    
    cudaMallocHost((void **)&h_data, sizeof(T) * data_size);
    cudaMallocHost((void **)&h_res, sizeof(T));
    cudaMalloc((void **)&d_data, sizeof(T) * data_size);
    cudaMalloc((void **)&d_res, sizeof(T));

    cudabm::genRandom(h_data, data_size);
    // cudabm::Print(h_data, data_size);

    cudaMemcpy(d_data, h_data, data_size * sizeof(T), cudaMemcpyHostToDevice);
    call_reduce1(d_data, d_res, data_size);

    cudaMemcpy(h_res, d_res,  sizeof(T), cudaMemcpyDeviceToHost);
    float sum = cudabm::Sum(h_data, data_size);
    printf("11111 %f %f\n", h_res[0], sum);
    cudaFree(d_data);
    cudaFree(d_res);
    cudaFreeHost(h_res);
    cudaFreeHost(h_data);

  }

  double dataSize(const bm::State& st) override {
    size_t sizePerTile = st.range(0);
    uint64_t numOperations = numTilesPerIpu() * sizePerTile;
    return (double)numOperations;
  }
};

#define BENCHMARK_REDUCE1_OP(name, dType)                   \
  BENCHMARK_TEMPLATE_DEFINE_F(Reduce1, name, dType)(bm::State & st) {              \
    callKernel(st.range(0));                      \
    runBenchmark(st);                                                       \
    st.counters["FLOPS"] = bm::Counter{                                     \
        dataSize(st), bm::Counter::kIsIterationInvariantRate}; \
  }                                                                         \
  BENCHMARK_REGISTER_F(Reduce1, name)                              \
      ->Unit(bm::kMillisecond)                                              \
      ->RangeMultiplier(2)                                                  \
      ->Range(32, 32);


#define BENCHMARK_REDUCE1_OP_TYPE(dType)                    \
  BENCHMARK_REDUCE1_OP(Reduce_##dType, dType)

BENCHMARK_REDUCE1_OP_TYPE(float)
// BENCHMARK_REDUCE1_OP_TYPE(int)
