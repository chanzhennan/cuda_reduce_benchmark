// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "benchmark_base.h"

// #include <poplar/IPUModel.hpp>

#include "utils.h"
#include <iostream>
namespace cudabm {

void BenchmarkBase::engineRun() { return;/*engine_->run(0);*/ }

// void BenchmarkBase::detachDevice(poplar::Device& device) {
//   if (device.getTarget().getTargetType() == poplar::TargetType::IPU_MODEL) {
//     return;
//   }
//   device.detach();
// }

void BenchmarkBase::releaseEngine() {
  // if (engine_ != nullptr) {
  //   engine_.reset();
  // }
}

// void BenchmarkBase::monitorFunc() {
//   // DeviceDiscoveryMode discoveryMode = DiscoverActivePartitionIPUs;
//   // // gcipuinfo ipuinfo(discoveryMode);
//   // // ipuinfo.setUpdateMode(true);
//   // std::map<std::string, std::string> results;

//   while (!shutDownMonitor_) {
//     // results = ipuinfo.getAttributesForDevice(devId);
//     // MonitorItems items = {
//     //     std::stof(results[IPUAttributeLabels::IpuPower]),
//     //     std::stof(results[IPUAttributeLabels::AverageDieTemp]),
//     //     std::stof(results[IPUAttributeLabels::ClockFrequency]),
//     // };
//     // monitorQueue_.push_back(items);

//     // Only keep last `monitorQueueMaxLen_` values
//     if ((monitorQueueMaxLen_ != 0) &&
//         (monitorQueue_.size() > monitorQueueMaxLen_)) {
//       monitorQueue_.pop_front();
//     }
//     usleep(100);
//   }
// }

void BenchmarkBase::runBenchmark(benchmark::State& state) {
  // if (enableMonitor_) {
  //   startMonitor();
  // }
  // std::cout << enableMonitor_ << state.max_iterations << std::endl;

  for (auto _ : state) {
    engineRun();
  }

  // if (enableMonitor_) {
  //   stopMonitor();
  // }


  // if (enableMonitor_) {
  //   for (int i = 0; i < monitorQueue_.size(); i++) {
  //     state.counters[cudabm::strFormat("power_%d", i)] =
  //         monitorQueue_.at(i).power;
  //   }
  // }
}

}  // namespace ipubm
