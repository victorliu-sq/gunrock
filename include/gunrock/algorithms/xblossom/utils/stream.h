// Vendored from X-Blossom (src/include/utils/stream.h) for the Gunrock X-Blossom app.
#ifndef STREAM_H
#define STREAM_H

#include <cuda_runtime.h>
#include "utils/utils_gpu.h"

enum class StreamPriority { kDefault, kHigh, kLow };

class CudaStream {
public:
  explicit CudaStream(StreamPriority priority = StreamPriority::kDefault)
    : priority_(priority), cuda_stream_() {
    if (priority_ == StreamPriority::kDefault) {
      CUDA_CHECK(
        cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));
    }
    else {
      int leastPriority, greatestPriority;
      CUDA_CHECK(
        cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
      CUDA_CHECK(cudaStreamCreateWithPriority(
        &cuda_stream_, cudaStreamNonBlocking,
        priority_ == StreamPriority::kHigh ? greatestPriority
        : leastPriority));
    }
  }

  // Copy Operations are deleted
  CudaStream(const CudaStream& other) = delete;

  CudaStream& operator=(const CudaStream& other) = delete;

  CudaStream(CudaStream&& other) noexcept
    : priority_(other.priority_), cuda_stream_(other.cuda_stream_) {
    other.cuda_stream_ = nullptr;
  }


  // Move Operations
  CudaStream& operator=(CudaStream&& other) noexcept {
    if (this != &other) {
      this->cuda_stream_ = other.cuda_stream_;
      other.cuda_stream_ = nullptr;
    }
    return *this;
  }

  ~CudaStream() {
    this->Destroy();
  }

  void Destroy() {
    if (cuda_stream_) {
      CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
      cuda_stream_ = nullptr;
    }
  }

  void Sync() const {
    if (cuda_stream_) {
      CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
    }
  }

  cudaStream_t cuda_stream() const {
    return cuda_stream_;
  }

private:
  StreamPriority priority_;
  cudaStream_t cuda_stream_;
};


#endif //STREAM_H
