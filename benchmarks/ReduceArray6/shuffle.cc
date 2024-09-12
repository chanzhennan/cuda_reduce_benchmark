// Copyright (c) 2022 Zhennanc Ltd. All rights reserved.
#include "ReduceArray6/shuffle.cuh"

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
class Shuffle : public BaseReduce<T> {
 public:
  void callKernel(benchmark::State &state) {
    BaseReduce<T>::shuffle(state);

    auto len = BaseReduce<T>::getDataSize(state);
    auto result = GPUReduction6<TPB>(BaseReduce<T>::getDeviceArray(), len);
  }
};

RegisterBenchmark(Shuffle, int);
RegisterBenchmark(Shuffle, float);
