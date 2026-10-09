// Vendored from X-Blossom (src/include/enode_queue/enode_queue.h) for the Gunrock X-Blossom app.
#ifndef ENODE_QUEUE_H
#define ENODE_QUEUE_H

#include "utils/types_gpu.h"
#include "utils/dev_array.h"
#include "utils/dev_value.h"
#include <cooperative_groups.h>

class TestENodeQueueFixture;

namespace zblossom {
  template<typename XBlossomEngine>
  class XbsmGpuEngineTester;

  // class XBlossomGpuEngineTester;
  // class XBlossomGpuEngineParEdgeTester;

  class ENodeQueueDeviceView {
  public:
    explicit ENodeQueueDeviceView(DArrayView<index_t> enodes_buffer,
                                  DValueView<index_t> enodes1_begin,
                                  DValueView<index_t> enodes1_end,
                                  DValueView<index_t> enodes2_begin,
                                  DValueView<index_t> enodes2_end
    ): enodes_buffer_(enodes_buffer),
       enodes1_begin_(enodes1_begin),
       enodes1_end_(enodes1_end),
       enodes2_begin_(enodes2_begin),
       enodes2_end_(enodes2_end) {
    }

    // DEV_INLINE index_t Begin() {
    //   return MIN(enodes1_begin_.Value(), enodes2_begin_.Value());
    // }
    //
    // DEV_INLINE index_t End() {
    //   return MAX(enodes1_end_.Value(), enodes2_end_.Value());
    // }

    DEV_INLINE void AppendENode1(index_t enode1) {
      index_t enodes1_offset = atomicAdd(&enodes1_end_, 1);
      enodes_buffer_[enodes1_offset] = enode1;
      __threadfence();
    }

    DEV_INLINE index_t AppendENode1Warp(index_t enode1) {
      auto g = cooperative_groups::coalesced_threads();

      // 1. No need to gather, number of active thread is already provided.
      // Thread 0 only needs to perform atomicAdd to get the base offset
      index_t base_offset;
      index_t thread_offset;
      const index_t first_active_thread_rank = 0;
      if (g.thread_rank() == first_active_thread_rank) {
        base_offset = atomicAdd(&enodes1_end_, g.size());
      }

      // 2. Scatter base offset to all active threads in the warp.
      base_offset = g.shfl(base_offset, first_active_thread_rank);
      thread_offset = base_offset + g.thread_rank();

      enodes_buffer_[thread_offset] = enode1;
      return thread_offset;
    }

    DEV_INLINE void AppendENode2(index_t enode2) {
      index_t enodes2_offset = atomicAdd(&enodes2_end_, 1);
      enodes_buffer_[enodes2_offset] = enode2;
      __threadfence();
    }

    DEV_INLINE index_t AppendENode2Warp(index_t enode2) {
      auto g = cooperative_groups::coalesced_threads();

      // 1. No need to gather, number of active thread is already provided.
      // Thread 0 only needs to perform atomicAdd to get the base offset
      index_t base_offset;
      index_t thread_offset;
      const index_t first_active_thread_rank = 0;
      if (g.thread_rank() == first_active_thread_rank) {
        base_offset = atomicAdd(&enodes2_end_, g.size());
      }

      // 2. Scatter base offset to all active threads in the warp.
      base_offset = g.shfl(base_offset, first_active_thread_rank);
      thread_offset = base_offset + g.thread_rank();

      enodes_buffer_[thread_offset] = enode2;
      return thread_offset;
    }

    DEV_INLINE index_t &operator[](index_t idx) {
      return this->enodes_buffer_[idx];
    }

  private:
    DArrayView<index_t> enodes_buffer_;

    DValueView<index_t> enodes1_begin_;
    DValueView<index_t> enodes1_end_;
    DValueView<index_t> enodes2_begin_;
    DValueView<index_t> enodes2_end_;
  };

  class ENodeQueue {
    friend class ENodeQueueTester;
    friend class ::TestENodeQueueFixture;
    // friend class zblossom::XBlossomGpuEngineTester;
    // friend class zblossom::XBlossomGpuEngineParEdgeTester;

    template<typename XBlossomEngine>
    friend class XbsmGpuEngineTester;

  public:
    ENodeQueue(size_t n)
      : enodes_buffer_(n),
        enodes1_begin_(0),
        enodes1_end_(0),
        enodes2_begin_(0),
        enodes2_end_(0),
        capacity_(n) {
    }

    ENodeQueueDeviceView DeviceView() {
      return ENodeQueueDeviceView(enodes_buffer_.DeviceView(),
                                  enodes1_begin_.DeviceView(),
                                  enodes1_end_.DeviceView(),
                                  enodes2_begin_.DeviceView(),
                                  enodes2_end_.DeviceView());
    }

    inline index_t ENnode1Begin() {
      return enodes1_begin_.GetD2H();
    }

    inline index_t ENnode1End() {
      return enodes1_end_.GetD2H();
    }

    inline index_t ENnode2Begin() {
      return enodes2_begin_.GetD2H();
    }

    inline index_t ENnode2End() {
      return enodes2_end_.GetD2H();
    }

    inline index_t Begin() {
      return MIN(enodes1_begin_.GetD2H(), enodes2_begin_.GetD2H());
    }

    inline index_t End() {
      return MAX(enodes1_end_.GetD2H(), enodes2_end_.GetD2H());
    }

    inline size_t Size() {
      return this->End() - this->Begin();
    }

    inline void PrepareForAppendingENode1() {
      enodes1_begin_.SetD2D(enodes2_end_);
      enodes1_end_.SetD2D(enodes1_begin_);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    inline void PrepareForAppendingENode2() {
      enodes2_begin_.SetD2D(enodes1_end_);
      enodes2_end_.SetD2D(enodes2_begin_);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    inline void Clear() {
      enodes1_begin_.SetH2D(0);
      enodes1_end_.SetH2D(0);
      enodes2_begin_.SetH2D(0);
      enodes2_end_.SetH2D(0);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // for sorting
    inline auto BeginIterator() {
      return enodes_buffer_.begin();
    }


  private:
    DArray<index_t> enodes_buffer_;

    DValue<index_t> enodes1_begin_;
    DValue<index_t> enodes1_end_;
    DValue<index_t> enodes2_begin_;
    DValue<index_t> enodes2_end_;

    size_t capacity_;
  };
}


#endif //ENODE_QUEUE_H
