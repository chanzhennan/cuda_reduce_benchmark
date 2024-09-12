// Copyright (c) 2022 Zhennanc Ltd. All rights reserved.
#include "ReduceArray0/baseline.cuh"

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
class Baseline : public BaseReduce<T> {
 public:
  void callKernel(benchmark::State &state) {
    // call kernel
    BaseReduce<T>::shuffle(state);

    auto len = BaseReduce<T>::getDataSize(state);
    auto result = GPUReduction<TPB>(BaseReduce<T>::getDeviceArray(), len);
  }
};

// 使用示例
RegisterBenchmark(Baseline, int);
RegisterBenchmark(Baseline, float);
