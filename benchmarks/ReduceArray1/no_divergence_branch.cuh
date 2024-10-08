#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#define TPB 128

template <size_t blockSize, typename T>
T GPUReduction1(T *dA, size_t N);
