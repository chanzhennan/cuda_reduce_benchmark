// Copyright (c) 2022 Zhennanc Ltd. All rights reserved.
#include "ReduceArray5/completely_unroll.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_base/basereduce.h"
#include "bmlib/utils.h"

template <typename T>
class CompletelyUnroll : public BaseReduce<T> {
 public:
  void callKernel(benchmark::State &state) {
    BaseReduce<T>::shuffle(state);

    auto len = BaseReduce<T>::getDataSize(state);
    auto result = GPUReduction5<TPB>(BaseReduce<T>::getDeviceArray(), len);
  }
};

RegisterBenchmark(CompletelyUnroll, int);
RegisterBenchmark(CompletelyUnroll, float);
