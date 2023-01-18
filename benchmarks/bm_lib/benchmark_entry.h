// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include <benchmark/benchmark.h>

// Helper macro to create a main routine in a test that runs the benchmarks
#define CUDA_BENCHMARK_MAIN()                                           \
  int main(int argc, char** argv) {                                     \
    ::benchmark::Initialize(&argc, argv);                               \
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1; \
    ::benchmark::RunSpecifiedBenchmarks();                              \
    ::benchmark::Shutdown();                                            \
    return 0;                                                           \
  }                                                                     \
  int main(int, char**)
