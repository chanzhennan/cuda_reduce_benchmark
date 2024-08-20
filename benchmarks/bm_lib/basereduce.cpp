#include "basereduce.h"

template <typename T>
void BaseReduce<T>::callKernel(benchmark::State &state) {
  throw std::runtime_error("callKernel need implement");
}

template <typename T>
void BaseReduce<T>::SetUp(const ::benchmark::State &state) {
  dataSize = state.range(0) * state.range(0) * 100;
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
  dataSize = st.range(0) * st.range(0) * 100;
  cudaMemcpy(d_array, array, sizeof(T) * dataSize, cudaMemcpyDeviceToDevice);
}

template <typename T>
void BaseReduce<T>::verify(const ::benchmark::State &st) {
  // for test M, N, K = state.range(0)
  // cudabm::Gemm<T>(dA, dB, testC, st.range(0), st.range(1), st.range(2));
  // cudabm::Equal<T>(st.range(0) * st.range(1), dC, testC, 1e-2);
  // // if (!)
  //   throw std::runtime_error("Value diff occur in Dense");
}

template <typename T>
void BaseReduce<T>::TearDown(const ::benchmark::State &st) {
  cudaFree(d_array);
  cudaFree(array);
}

template <typename T>
double BaseReduce<T>::getDataSize(const ::benchmark::State &state) {
  return (double)(state.range(0) * state.range(0) * 100);
}

template <typename T>
double BaseReduce<T>::getFlops(const ::benchmark::State &state) {
  return (double)(state.range(0) * state.range(0) * 100);
}

template class BaseReduce<float>;
template class BaseReduce<int>;
