// Copyright (c) 2022 Zhennanc Ltd. All rights reserved.

#include "benchmark_base.h"

// #include <poplar/IPUModel.hpp>

#include <iostream>

#include "utils.h"
namespace cudabm {

void BenchmarkBase::engineRun() { return; /*engine_->run(0);*/ }

void BenchmarkBase::releaseEngine() {}

void BenchmarkBase::runBenchmark(benchmark::State& state) {
  for (auto _ : state) {
    engineRun();
  }
}

}  // namespace cudabm
