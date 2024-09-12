
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bmlib/utils.h"

#define RegisterBenchmark(BenchmarkClass, dType)                     \
  BENCHMARK_TEMPLATE_DEFINE_F(BenchmarkClass, Reduce_##dType, dType) \
  (benchmark::State & st) {                                          \
    for (auto _ : st) {                                              \
      callKernel(st);                                                \
    }                                                                \
    setBenchmarkCounters(st);                                        \
  }                                                                  \
  BENCHMARK_REGISTER_F(BenchmarkClass, Reduce_##dType)               \
      ->Unit(benchmark::kMillisecond)                                \
      ->Range(20480000, 40960000);

template <typename T>
class BaseReduce : public benchmark::Fixture {
 public:
  virtual void callKernel(benchmark::State &state);

  void SetUp(const ::benchmark::State &state) override;

  void verify(const ::benchmark::State &st, T len, T result);

  void TearDown(const ::benchmark::State &st) override;

  void shuffle(const ::benchmark::State &st);

  double getDataSize(const ::benchmark::State &state);

  double getFlops(const ::benchmark::State &state);

  void setBenchmarkCounters(benchmark::State &st);

  T *getDeviceArray();

 private:
  T *d_array, *array;
  long int dataSize;
  long int flops;
};
