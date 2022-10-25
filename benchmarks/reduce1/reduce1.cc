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

  void callKernel(size_t sizePerTile) {

    T *h_data, d_data;
    T *h_res, d_res;
    cudaMalloc((void **)&h_data, sizeof(T) * sizePerTile);
    cudaMalloc((void **)&h_res, sizeof(T));
    cudaMallocHost((void **)&d_data, sizeof(T) * sizePerTile);
    cudaMallocHost((void **)&d_res, sizeof(T));
    // cudaMemcpy()


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
      ->Range(8 * 1024, 32 * 1024);


#define BENCHMARK_REDUCE1_OP_TYPE(dType)                    \
  BENCHMARK_REDUCE1_OP(Reduce_##dType, dType)

BENCHMARK_REDUCE1_OP_TYPE(float)
BENCHMARK_REDUCE1_OP_TYPE(int)
