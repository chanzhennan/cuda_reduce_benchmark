#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <size_t blockSize, typename T>
T GPUReduction6(T *dA, size_t N);
