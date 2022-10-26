#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// template <typename T>
void call_reduce4(float *g_idata, float *g_odata, int dataSize);

// template <> void call_reduce1<float>(float *g_idata, float *g_odata, int dataSize);
// template <> void call_reduce1<int>(int *g_idata, int *g_odata, int dataSize);