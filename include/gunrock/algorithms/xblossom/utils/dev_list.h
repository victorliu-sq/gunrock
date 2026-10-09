// Vendored from X-Blossom (src/include/utils/dev_list.h) for the Gunrock X-Blossom app.
#ifndef DEV_LIST_H
#define DEV_LIST_H
#include <thrust/host_vector.h>
#include <cooperative_groups.h>

#include "dev_array.h"
#include "types_gpu.h"
#include "dev_value.h"
#include "utils/utils_gpu.h"

template<typename T>
using HList = HVector<T>;

template<typename T>
class DListView {
public:
  DEV_HOST_INLINE DListView(DArrayView<T> array_view, DValueView<index_t> counter)
    : array_(array_view), counter_(counter) {
  }

  DEV_INLINE index_t Append(T val) {
    auto idx = atomicAdd(&counter_, 1);
    array_[idx] = val;
    return idx;
  }

  DEV_INLINE index_t AppendWarp(T val) {
    auto g = cooperative_groups::coalesced_threads();

    // 1. No need to gather, number of active thread is already provided.
    // Thread 0 only needs to perform atomicAdd to get the base offset
    index_t base_offset;
    index_t thread_offset;
    const index_t first_active_thread_rank = 0;
    if (g.thread_rank() == first_active_thread_rank) {
      base_offset = atomicAdd(&counter_, g.size());
    }

    // 2. Scatter base offset to all active threads in the warp.
    base_offset = g.shfl(base_offset, first_active_thread_rank);
    thread_offset = base_offset + g.thread_rank();

    array_[thread_offset] = val;
    return thread_offset;
  }

  DEV_INLINE index_t Reserve(size_t size) {
    index_t offset = atomicAdd(&counter_, size);
    return offset;
  }

  DEV_INLINE index_t ReserveWarp(size_t size) {
    auto warp = cooperative_groups::coalesced_threads();

    // 1. Thread 0 in the warp gathers all sizes and perform atomicAdd
    const index_t first_active_thread_rank = 0;

    index_t total_size = 0;

    const index_t thread_rank = warp.thread_rank();

    // printf("[Warp] Thread %d: Starting ReserveWarp with size %d\n", warp.thread_rank(), thread_size);

    // Each thread has its own thread_size, we sum them across the warp
    size_t thread_size = size;
    #pragma unroll
    for (int i = 0; i < warp.size(); i++) {
      total_size += warp.shfl(thread_size, i);
    }

    index_t base_offset = 0;
    if (thread_rank == first_active_thread_rank) {
      base_offset = atomicAdd(&counter_, total_size);
    }

    // 2. Scatter the base offset to all active threads
    base_offset = warp.shfl(base_offset, first_active_thread_rank);
    // 3. Compute individual offset by accumulation for each thread
    index_t thread_offset = base_offset;

    #pragma unroll
    for (int i = 0; i < warp.size(); i++) {
      index_t val = warp.shfl(thread_size, i);
      thread_offset += (i < thread_rank) ? val : 0;
    }

    assert(thread_offset < this->array_.size());

    return thread_offset;
  }

  DEV_INLINE T &operator[](index_t idx) {
    // assert(idx < Size());
    assert(idx < this->array_.size());
    return array_[idx];
  }

  DEV_INLINE const T &operator[](index_t idx) const {
    // assert(idx < Size());
    assert(idx < this->array_.size());
    return array_[idx];
  }

  // DEV_INLINE index_t Size() const {
  //   return atomicAdd(const_cast<index_t*>(&counter_), 0);
  //   // return *counter_;
  // }

private:
  DArrayView<T> array_;
  DValueView<index_t> counter_;
};

template<typename T>
class DList {
public:
  explicit DList(size_t capacity)
    : array_(capacity), counter_(0) {
  }

  inline void Clear() {
    counter_.SetH2D(0);
  }

  inline void GetFromDevice(HList<T> &host_list) {
    size_t size = counter_.GetD2H();
    host_list.resize(size);
    thrust::copy_n(array_.begin(), size, host_list.begin());
  }

  inline DListView<T> DeviceView() {
    return {array_.DeviceView(), counter_.DeviceView()};
  }

  auto begin() const { return array_.begin(); }

  auto end() const { return array_.begin() + counter_.GetD2H(); }

  size_t size() const { return counter_.GetD2H(); }

private:
  DArray<T> array_;
  DValue<index_t> counter_;
};

#endif //DEV_LIST_H
