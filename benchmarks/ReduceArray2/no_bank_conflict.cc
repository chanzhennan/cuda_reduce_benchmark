// Copyright (c) 2022 Zhennanc Ltd. All rights reserved.
#include "ReduceArray2/no_bank_conflict.cuh"

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
class NoBankConflict : public BaseReduce<T> {
 public:
  void callKernel(benchmark::State &state) {
    BaseReduce<T>::shuffle(state);

    auto len = BaseReduce<T>::getDataSize(state);
    auto result = GPUReduction2<TPB>(BaseReduce<T>::getDeviceArray(), len);
  }
};

RegisterBenchmark(NoBankConflict, int);
