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
#define DFLAG_TENTATIVE 1  // GACGE tree claims: held while its holder tries the pair's other tree
#define DFLAG_COMMITTED 2  // GACGE tree claims: both trees of a pair claimed; final for the search

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

// XB-GAP's TryClaim: a plain load first, the compare-and-swap only on a free
// flag (test-and-test-and-set); volatile, so the load is not an L1 copy
__device__ __forceinline__ bool TryClaimDAtomicFlag(DAtomicFlag *flag) {
  if (*reinterpret_cast<volatile DAtomicFlag *>(flag) != DFLAG_FALSE) return false;
  return atomicCAS(flag, DFLAG_FALSE, DFLAG_TRUE) == DFLAG_FALSE;
}

// GACGE (2026-10-16) tentative tree claims for the augment step. XB-GAP's
// protocol holds a tree while trying the pair's other tree and releases it on
// failure; a thread that meets the held tree gives up at once. On a CPU that
// window is rarely hit; on a GPU, where the candidate edges of a frontier
// run at once, it costs the tree its pairing for the search. Here the state
// tells a tentative hold from a committed pair: a thread meeting a tentative
// tree waits it out (bounded, __nanosleep backoff) and claims the tree if it
// is released; a committed tree is final. Deadlock-free: a holder waits only
// on trees of higher id (the pair's lower root is claimed first), so every
// wait chain is strictly increasing and ends at a thread that holds nothing
// or commits; and every wait is bounded by spin.
__device__ __forceinline__ bool ClaimTreeWait(DAtomicFlag *flag, int spin) {
  for (int i = 0;; ++i) {
    const DAtomicFlag s = *reinterpret_cast<volatile DAtomicFlag *>(flag);
    if (s == DFLAG_COMMITTED) return false;
    if (s == DFLAG_FALSE && atomicCAS(flag, DFLAG_FALSE, DFLAG_TENTATIVE) == DFLAG_FALSE) return true;
    if (i >= spin) return false;
    __nanosleep(i < 6 ? (32u << i) : 1024u);
  }
}

#endif //TYPES_GPU_H
