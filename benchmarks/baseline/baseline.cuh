#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

template <size_t blockSize, typename T>
T GPUReduction(T *dA, size_t N);
