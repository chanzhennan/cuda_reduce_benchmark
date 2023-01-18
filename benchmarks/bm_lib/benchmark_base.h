// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include <benchmark/benchmark.h>
#include <unistd.h>

#include <deque>
#include <memory>
#include <stdexcept>
#include <thread>

namespace std {

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace std

namespace cudabm {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct MonitorItems {
  float power;
  float temperature;
  float clock;
};

class BenchmarkBase : public benchmark::Fixture {
 public:
  BenchmarkBase(bool enableMonitor = false)
      : benchmark::Fixture(),
        enableMonitor_(enableMonitor),
        monitorQueueMaxLen_(0) {}

  // interface
  virtual void engineRun();

  // methods
  void SetUp(const ::benchmark::State &state) BENCHMARK_OVERRIDE {
    // device_ = createDevice(numIpus_, useIpuModel_);
  }

  void TearDown(const ::benchmark::State &st) BENCHMARK_OVERRIDE {
    // detachDevice(device_);
    // releaseEngine();
  }

  /**
   * \brief The amount of data that the benchmark operates on, which is
   * displayed on the X-axis of the chart, all benchmarks must provide this
   * value.
   */
  virtual double getDataSize() = 0;

 protected:
  // monitor thread function
  void monitorFunc();

  void runBenchmark(benchmark::State &state);

  void releaseEngine();

  bool enableMonitor_;
  bool shutDownMonitor_;
  std::deque<MonitorItems> monitorQueue_;
  size_t monitorQueueMaxLen_;
  std::thread monitorThread_;
};

}  // namespace cudabm
