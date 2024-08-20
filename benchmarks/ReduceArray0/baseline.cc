// Copyright (c) 2022 Zhennanc Ltd. All rights reserved.
#include "ReduceArray0/baseline.cuh"

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
class Baseline : public BaseReduce<T> {
 public:
  void callKernel(benchmark::State &state) {
    // call kernel
    BaseReduce<T>::shuffle(state);

    auto len = BaseReduce<T>::getDataSize(state);
    auto result = GPUReduction<TPB>(BaseReduce<T>::getDeviceArray(), len);

    if (len != (long int)result) {
      std::cout << "dataSize : " << len << '\n';
      std::cout << "result : " << (long int)result << '\n';
      // throw std::invalid_argument("Results are different.");
    }
  }
};

#define BENCHMARK_REDUCE0_OP(name, dType)                              \
  BENCHMARK_TEMPLATE_DEFINE_F(Baseline, name, dType)                   \
  (benchmark::State & st) {                                            \
    for (auto _ : st) {                                                \
      callKernel(st);                                                  \
    }                                                                  \
    double iter = st.iterations();                                     \
    st.counters["DATASIZE"] = getDataSize(st);                         \
    st.counters["TFlops"] = benchmark::Counter(                        \
        (getDataSize(st) * iter / 1e12), benchmark::Counter::kIsRate); \
  }                                                                    \
  BENCHMARK_REGISTER_F(Baseline, name)                                 \
      ->Unit(benchmark::kMillisecond)                                  \
      ->RangeMultiplier(2)                                             \
      ->Range(1024, 2048);
// ->Iterations(1);

#define BENCHMARK_REDUCE0_OP_TYPE(dType) \
  BENCHMARK_REDUCE0_OP(Reduce_##dType, dType)

// BENCHMARK_REDUCE0_OP_TYPE(float)
BENCHMARK_REDUCE0_OP_TYPE(int)
