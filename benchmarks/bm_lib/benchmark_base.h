// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include <benchmark/benchmark.h>
// #include <graphcore_target_access/gcipuinfo/IPUAttributeLabels.h>
// #include <graphcore_target_access/gcipuinfo/gcipuinfo.h>
#include <unistd.h>

#include <deque>
#include <memory>
// #include <poplar/DeviceManager.hpp>
// #include <poplar/Engine.hpp>
// #include <poplar/Graph.hpp>
#include <stdexcept>
#include <thread>

namespace std {

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args) {
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
  BenchmarkBase(unsigned numIpus = 1, bool enableMonitor = false,
                bool useIpuModel = false)
      : benchmark::Fixture(),
        numIpus_(numIpus),
        enableMonitor_(enableMonitor),
        monitorQueueMaxLen_(0),
        useIpuModel_(useIpuModel) {}

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

  size_t numTiles() { return device_.getTarget().getNumTiles(); }

  size_t numIpus() { return numIpus_; }

  size_t numTilesPerIpu() { return numTiles() / numIpus(); }

  /**
   * \brief The amount of data that the benchmark operates on, which is
   * displayed on the X-axis of the chart, all benchmarks must provide this
   * value.
   */
  virtual double dataSize(const ::benchmark::State &state) = 0;

 protected:
  // monitor thread function
  void monitorFunc();

  void startMonitor() {
    shutDownMonitor_ = false;
    monitorThread_ = std::thread(&BenchmarkBase::monitorFunc, this);
  }

  void stopMonitor() {
    if (enableMonitor_) {
      shutDownMonitor_ = true;
      monitorThread_.join();
    }
  }

  void runBenchmark(benchmark::State &state);

  // poplar::Device createDevice(unsigned numIpus, bool useIpuModel = false);

  void detachDevice(poplar::Device &device);

  void releaseEngine();

  // /**
  //  * \brief poplar device
  //  */
  // poplar::Device device_;

  // /**
  //  * \brief poplar execution engine
  //  */
  // std::unique_ptr<poplar::Engine> engine_;

  unsigned numIpus_;
  bool useIpuModel_;
  bool enableMonitor_;
  bool shutDownMonitor_;
  std::deque<MonitorItems> monitorQueue_;
  size_t monitorQueueMaxLen_;
  std::thread monitorThread_;
};

}  // namespace ipubm
