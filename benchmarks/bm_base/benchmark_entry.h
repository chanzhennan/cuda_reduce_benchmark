// Copyright (c) 2022 Zhennanc Ltd. All rights reserved.
#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <iostream>
// Helper macro to create a main routine in a test that runs the benchmarks

#include "bmlib/gpu_info.h"

#define CUDA_BENCHMARK_MAIN()                                           \
  int main(int argc, char** argv) {                                     \
    GPUInfo();                                                          \
    ::benchmark::Initialize(&argc, argv);                               \
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1; \
    ::benchmark::RunSpecifiedBenchmarks();                              \
    ::benchmark::Shutdown();                                            \
    copyRight();                                                        \
    return 0;                                                           \
  }                                                                     \
  int main(int, char**)
