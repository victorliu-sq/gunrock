// Vendored from X-Blossom (src/include/utils/cuda_algo.h) for the Gunrock X-Blossom app.
#ifndef CUDA_ALGO_H
#define CUDA_ALGO_H
#include <cub/thread/thread_search.cuh>
#include "dev_array.h"
#include "utils_gpu.h"

DEV_INLINE index_t FindRightmostLTEQ(const DArrayView<int>& array_dview, size_t n, int target) {
  int left = 0, right = n - 1;
  while (left < right) {
    index_t mid = left + (right - left) / 2;
    index_t presum = array_dview[mid + 1];
    if (presum <= target) {
      left = mid + 1;
    }
    else {
      right = mid;
    }
  }
  return left;
}

DEV_INLINE index_t RightmostLTEQ(const DArrayView<int>& array_dview, size_t n, int target) {
  int lowEnough = 0, tooHigh = n + 1;

  while (lowEnough + 1 < tooHigh) {
    int mid = lowEnough + (tooHigh - lowEnough) / 2;
    int value = array_dview[mid];
    if (value <= target) {
      // mid satisfies the condition; move boundary up
      lowEnough = mid;
    }
    else {
      // mid is too high; shrink the upper boundary
      tooHigh = mid;
    }
  }
  return lowEnough;
}

DEV_INLINE int ReadOnlyLoad(const int* ptr) {
#if defined(__CUDA_ARCH__)
  return __ldg(ptr);
#else
  return *ptr;
#endif
}

DEV_INLINE index_t FindRightmostLTEQFast(const DArrayView<int>& array_dview, size_t n, int target) {
  const int* data = &array_dview[0];
  int left = 0;
  int right = static_cast<int>(n) - 1;
  while (left < right) {
    const int mid = left + ((right - left) >> 1);
    const int presum = ReadOnlyLoad(data + mid + 1);
    const bool move_right = presum <= target;
    left = move_right ? mid + 1 : left;
    right = move_right ? right : mid;
  }
  return left;
}

DEV_INLINE index_t FindRightmostLTEQCub(const DArrayView<int>& array_dview, size_t n, int target) {
  const int* data = &array_dview[0];
  const int pos = cub::UpperBound(data, static_cast<int>(n), target);
  return static_cast<index_t>(pos - 1);
}

DEV_INLINE index_t RightmostLTEQFast(const DArrayView<int>& array_dview, size_t n, int target) {
  const int* data = &array_dview[0];
  int low_enough = 0;
  int too_high = static_cast<int>(n) + 1;
  while (low_enough + 1 < too_high) {
    const int mid = low_enough + ((too_high - low_enough) >> 1);
    const int value = ReadOnlyLoad(data + mid);
    const bool move_right = value <= target;
    low_enough = move_right ? mid : low_enough;
    too_high = move_right ? too_high : mid;
  }
  return low_enough;
}

#endif //CUDA_ALGO_H
