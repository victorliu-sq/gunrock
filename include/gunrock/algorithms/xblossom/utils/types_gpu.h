// Vendored from X-Blossom (src/include/utils/types_gpu.h) for the Gunrock X-Blossom app.
#ifndef TYPES_GPU_H
#define TYPES_GPU_H

#include "utils/types_cpu.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/system/cuda/memory_resource.h>
#include <ostream>

#define DFLAG_FALSE 0
#define DFLAG_TRUE 1

// Host-side pinned vector (accelerates H2D transfer)
using MrPinned = thrust::system::cuda::universal_host_pinned_memory_resource;

// template<typename T>
// using HVector = thrust::host_vector<T, thrust::mr::stateless_resource_allocator<T, MrPinned> >;

// Device-side CUDA vector
template<typename T>
using DVector = thrust::device_vector<T>;

template<typename T1, typename T2>
using DPair = thrust::pair<T1, T2>;

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const thrust::pair<T1, T2>& p) {
  os << "(" << p.first << ", " << p.second << ")";
  return os;
}

// Device-side CUDA vector explicitly holding a single element
// template<typename T>
// using DValue = thrust::device_vector<T>;

// Device-side boolean type
using DFlag = int;

__device__ __forceinline__ void SetDFlag(DFlag *flag) {
  *flag = DFLAG_TRUE;
}

__device__ __forceinline__ void ResetDFlag(DFlag *flag) {
  *flag = DFLAG_FALSE;
}

__device__ __forceinline__ bool IsDFlagSet(DFlag *flag) {
  return *flag == DFLAG_TRUE;
}

// Device-side atomic boolean type
using DAtomicFlag = int;

__device__ __forceinline__ bool SetDAtomicFlag(DAtomicFlag *flag) {
  return atomicCAS(flag, DFLAG_FALSE, DFLAG_TRUE) == DFLAG_FALSE;
}

__device__ __forceinline__ void ResetDAtomicFlag(DAtomicFlag *flag) {
  atomicExch(flag, DFLAG_FALSE);
}

__device__ __forceinline__ bool IsDAtomicFlagSet(DAtomicFlag *flag) {
  return *flag == DFLAG_TRUE;
}

#endif //TYPES_GPU_H
