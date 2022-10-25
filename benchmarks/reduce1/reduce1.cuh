#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void call_reduce1(int *g_idata, int *g_odata);

__global__ void reduce1(int *g_idata, int *g_odata);