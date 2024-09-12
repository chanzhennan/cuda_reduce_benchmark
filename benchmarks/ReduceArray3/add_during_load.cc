// Copyright (c) 2022 Zhennanc Ltd. All rights reserved.
#include "ReduceArray3/add_during_load.cuh"

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
class AddDuringLoad : public BaseReduce<T> {
 public:
  void callKernel(benchmark::State &state) {
    BaseReduce<T>::shuffle(state);

    auto len = BaseReduce<T>::getDataSize(state);
    auto result = GPUReduction3<TPB>(BaseReduce<T>::getDeviceArray(), len);
  }
};

RegisterBenchmark(AddDuringLoad, int);
