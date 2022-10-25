// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "benchmark_base.h"

// #include <poplar/IPUModel.hpp>

#include "utils.h"

namespace cudabm {

void BenchmarkBase::engineRun() { return;/*engine_->run(0);*/ }

// poplar::Device BenchmarkBase::createDevice(unsigned numIpus, bool useIpuModel) {
//   if (useIpuModel) {
//     poplar::IPUModel ipuModel;
//     ipuModel.numIPUs = numIpus;
//     return ipuModel.createDevice();
//   } else {
//     poplar::DeviceManager manager =
//         poplar::DeviceManager::createDeviceManager();

//     for (auto& device : manager.getDevices(poplar::TargetType::IPU, numIpus)) {
//       if (device.attach()) {
//         return std::move(device);
//       }
//     }
//   }

//   throw ipubm::Error("Failed to attach any IPU device.");
// }

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

void BenchmarkBase::monitorFunc() {
  // unsigned devId = device_.getId();
  // DeviceDiscoveryMode discoveryMode = DiscoverActivePartitionIPUs;
  // gcipuinfo ipuinfo(discoveryMode);
  // ipuinfo.setUpdateMode(true);
  // std::map<std::string, std::string> results;

  // while (!shutDownMonitor_) {
  //   results = ipuinfo.getAttributesForDevice(devId);
  //   MonitorItems items = {
  //       std::stof(results[IPUAttributeLabels::IpuPower]),
  //       std::stof(results[IPUAttributeLabels::AverageDieTemp]),
  //       std::stof(results[IPUAttributeLabels::ClockFrequency]),
  //   };
  //   monitorQueue_.push_back(items);

  //   // Only keep last `monitorQueueMaxLen_` values
  //   if ((monitorQueueMaxLen_ != 0) &&
  //       (monitorQueue_.size() > monitorQueueMaxLen_)) {
  //     monitorQueue_.pop_front();
  //   }
  //   usleep(100);
  // }
}

void BenchmarkBase::runBenchmark(benchmark::State& state) {
  if (enableMonitor_) {
    startMonitor();
  }

  for (auto _ : state) {
    engineRun();
  }

  if (enableMonitor_) {
    stopMonitor();
  }

  state.counters["data_size"] = dataSize(state);

  if (enableMonitor_) {
    for (int i = 0; i < monitorQueue_.size(); i++) {
      state.counters[ipubm::strFormat("power_%d", i)] =
          monitorQueue_.at(i).power;
    }
  }
}

}  // namespace ipubm
