
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "utils.h"
template <typename T>
class BaseReduce : public benchmark::Fixture {
 public:
  virtual void callKernel(benchmark::State &state);

  void SetUp(const ::benchmark::State &state) override;

  void verify(const ::benchmark::State &st);

  void TearDown(const ::benchmark::State &st) override;

  void shuffle(const ::benchmark::State &st);

  double getDataSize(const ::benchmark::State &state);

  double getFlops(const ::benchmark::State &state);

  T *getDeviceArray();

 private:
  T *d_array, *array;
  long int dataSize;
  long int flops;
};
