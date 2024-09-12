#include "basereduce.h"

template <typename T>
void BaseReduce<T>::callKernel(benchmark::State &state) {
  throw std::runtime_error("callKernel need implement");
}

template <typename T>
void BaseReduce<T>::SetUp(const ::benchmark::State &state) {
  dataSize = state.range(0);
  // Populate array
  cudaMalloc((void **)&d_array, sizeof(T) * dataSize);
  cudaMallocManaged((void **)&array, sizeof(T) * dataSize);
  for (size_t i = 0; i < dataSize; i++) array[i] = (T)1;
}

template <typename T>
T *BaseReduce<T>::getDeviceArray() {
  return d_array;
}

template <typename T>
void BaseReduce<T>::shuffle(const ::benchmark::State &st) {
  dataSize = st.range(0);
  cudaMemcpy(d_array, array, sizeof(T) * dataSize, cudaMemcpyDeviceToDevice);
}

template <typename T>
void BaseReduce<T>::verify(const ::benchmark::State &st, T len, T result) {
  if ((long int)len != (long int)result) {
    std::cout << "dataSize : " << len << '\n';
    std::cout << "result : " << (long int)result << '\n';
    // throw std::invalid_argument("Results are different.");
  }
}

template <typename T>
void BaseReduce<T>::TearDown(const ::benchmark::State &st) {
  cudaFree(d_array);
  cudaFree(array);
}

template <typename T>
double BaseReduce<T>::getDataSize(const ::benchmark::State &state) {
  return (double)(state.range(0));
}

template <typename T>
double BaseReduce<T>::getFlops(const ::benchmark::State &state) {
  return (double)(state.range(0));
}

template <typename T>
void BaseReduce<T>::setBenchmarkCounters(
    benchmark::State &state) {  // 确保这里是非const引用
  double iter = state.iterations();
  state.counters["DATASIZE"] =
      benchmark::Counter(getDataSize(state), benchmark::Counter::kIsRate);
  state.counters["TFlops"] = benchmark::Counter(
      (getDataSize(state) * iter / 1e12), benchmark::Counter::kIsRate);
}

template class BaseReduce<float>;
template class BaseReduce<int>;
