// Copyright (c) 2022 Zhennanc Ltd. All rights reserved.
#include "ReduceArray5/completely_unroll.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/basereduce.h"
#include "bm_lib/utils.h"

template <typename T>
class CompletelyUnroll : public BaseReduce<T> {
 public:
  void callKernel(benchmark::State &state) {
    BaseReduce<T>::shuffle(state);

    auto len = BaseReduce<T>::getDataSize(state);
    auto result = GPUReduction5<TPB>(BaseReduce<T>::getDeviceArray(), len);

    if (len != (long int)result) {
      std::cout << "dataSize : " << len << '\n';
      std::cout << "result : " << (long int)result << '\n';
      // throw std::invalid_argument("Results are different.");
    }
  }
};

#define BENCHMARK_REDUCE5_OP(name, dType)                              \
  BENCHMARK_TEMPLATE_DEFINE_F(CompletelyUnroll, name, dType)           \
  (benchmark::State & st) {                                            \
    for (auto _ : st) {                                                \
      callKernel(st);                                                  \
    }                                                                  \
    double iter = st.iterations();                                     \
    st.counters["DATASIZE"] = getDataSize(st);                         \
    st.counters["TFlops"] = benchmark::Counter(                        \
        (getDataSize(st) * iter / 1e12), benchmark::Counter::kIsRate); \
  }                                                                    \
  BENCHMARK_REGISTER_F(CompletelyUnroll, name)                         \
      ->Unit(benchmark::kMillisecond)                                  \
      ->RangeMultiplier(2)                                             \
      ->Range(1024, 2048);

#define BENCHMARK_REDUCE5_OP_TYPE(dType) \
  BENCHMARK_REDUCE5_OP(Reduce_##dType, dType)

BENCHMARK_REDUCE5_OP_TYPE(float)
BENCHMARK_REDUCE5_OP_TYPE(int)