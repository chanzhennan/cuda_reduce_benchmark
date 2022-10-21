// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/benchmark_base.h"
#include "bm_lib/utils.h"

namespace bm = benchmark;

class Reduce1 : public cudabm::BenchmarkBase {
 public:
  Reduce1() : cudabm::BenchmarkBase(/*numIpus=*/1, /*enableMonitor=*/true) {}

  void callKernel(size_t sizePerTile, ) {
  }

  double dataSize(const bm::State& st) override {
    size_t sizePerTile = st.range(0);
    uint64_t numOperations = numTilesPerIpu() * sizePerTile;
    return (double)numOperations;
  }
};

#define BENCHMARK_REDUCE1_OP(name, dType)                   \
  BENCHMARK_DEFINE_F(IpuCastBenchmark, name)(bm::State & st) {              \
    callKernel(st.range(0), dType);                      \
    runBenchmark(st);                                                       \
    st.counters["FLOPS"] = bm::Counter{                                     \
        dataSize(st), bm::Counter::kIsIterationInvariantRate}; \
  }                                                                         \
  BENCHMARK_REGISTER_F(IpuCastBenchmark, name)                              \
      ->Unit(bm::kMillisecond)                                              \
      ->RangeMultiplier(2)                                                  \
      ->Range(8 * 1024, 32 * 1024);


#define BENCHMARK_REDUCE1_OP_TYPE(dType)                    \
  BENCHMARK_REDUCE1_OP(Reduce_##dType, dType)

BENCHMARK_REDUCE1_OP_TYPE(FLOAT)
BENCHMARK_REDUCE1_OP_TYPE(INT)
