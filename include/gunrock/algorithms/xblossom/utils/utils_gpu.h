#pragma once
// Vendored from X-Blossom (src/include/utils/utils_gpu.h) for the Gunrock X-Blossom app.

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/memory_resource.h>

#include "utils/types_gpu.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#define DEV_HOST __device__ __host__
#define DEV_HOST_INLINE __device__ __host__ __forceinline__
#define DEV_INLINE __device__ __forceinline__
#define CONST_STATIC_INIT(...)
#else
#define DEV_HOST
#define DEV_HOST_INLINE
#define DEV_INLINE
#define CONST_STATIC_INIT(...) = __VA_ARGS__
#endif

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define MAX_GRID_SIZE (1024)
// #define MAX_GRID_SIZE (128)
#define MAX_BLOCK_SIZE (256)

// #define MAX_GRID_SIZE (16)
// #define MAX_BLOCK_SIZE (1024)

// #define MAX_GRID_SIZE (1)
// #define MAX_BLOCK_SIZE (8)

#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TOTAL_THREADS_1D (gridDim.x * blockDim.x)

#define PARTITION_SIZE(n, num_gpus, gpu_id) ((n) / (num_gpus) + ((gpu_id) < ((n) % (num_gpus)) ? 1 : 0))
#define PARTITION_START(n, num_gpus, gpu_id) (((n) / (num_gpus)) * (gpu_id) + min((gpu_id), (n) % (num_gpus)))

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__                                 \
                << " with err msg: " << cudaGetErrorString(rt) << std::endl;   \
      throw;                                                                   \
    }                                                                          \
  } while (0);

static DEV_HOST_INLINE size_t round_up(size_t numerator,
                                       size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

static uint64_t getNanoSecond() {
  return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

// Raw pointer extraction macro
#define CastRawPtr(v) thrust::raw_pointer_cast((v).data())

// Vendored for Gunrock: X-Blossom's glog/gflags plumbing is dropped. The SM cap
// (X-Blossom's --max_cuda_sms) is a plain global the app sets.
inline uint64_t xb_max_cuda_sms = 0;

inline void ApplyCudaLaunchCaps(int& block_num, int& block_size) {
  if (xb_max_cuda_sms > 0) {
    const int capped_sms = static_cast<int>(
        std::min<uint64_t>(xb_max_cuda_sms,
                           static_cast<uint64_t>(std::numeric_limits<int>::max())));
    block_num = std::max(1, std::min(block_num, capped_sms));
  }
}

inline void KernelSizing(int &block_num, int &block_size, size_t work_size) {
  block_size = MAX_BLOCK_SIZE;
  block_num = std::min(MAX_GRID_SIZE, (int) round_up(work_size, block_size));
  ApplyCudaLaunchCaps(block_num, block_size);
}


#define SYNC_ALL_STREAMS(streams) \
for (auto &stream : (streams)) { \
stream.Sync(); \
}

#define FOR_EACH_GPU(num_gpus, gpu_id) \
for (int gpu_id = 0; gpu_id < (num_gpus); ++gpu_id) if ((cudaSetDevice(gpu_id), true))

// template<typename T>
// void SafeCudaMemcpyAsync(DVector<T> &dst, const T *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream,
//                                 const char *error_context) {
//   cudaError_t status = cudaMemcpyAsync(CastRawPtr(dst), src, count * sizeof(T), kind, stream);
//   CHECK_EQ(status, cudaSuccess) << "cudaMemcpyAsync (" << error_context << ") error: "
//                                 << cudaGetErrorString(status);
// }

#define SLEEP_MILLISECONDS(us) std::this_thread::sleep_for(std::chrono::milliseconds(us))

// Overflow Problem of index
#define IDX_MUL(a, b) (static_cast<size_t>(a) * static_cast<size_t>(b))
#define SIZE_MUL(a, b) (static_cast<size_t>(a) * static_cast<size_t>(b))
#define IDX_ADD(a, b) (static_cast<size_t>(a) + static_cast<size_t>(b))
#define IDX_MUL_ADD(a, b, c)  (static_cast<size_t>(a) * static_cast<size_t>(b) + static_cast<size_t>(c))

// Compute Thread Task Range
// using index_t = uint64_t;
// using index_t = unsigned long long;
// using index_t = int64_t;
static inline void ComputeTaskletRange(size_t num_tasks,
                                       size_t num_threads,
                                       index_t thread_id,
                                       index_t& start,
                                       index_t& end) {
  const size_t num_tasks_per_thread = (num_tasks + num_threads - 1) / num_threads;
  start = thread_id * num_tasks_per_thread;
  end = std::min(start + num_tasks_per_thread, num_tasks);
}

// Profiling Statement: only enabled in profiling mode
#ifdef MM_PROFILE
  #define PROFILE_STATEMENT(stmt) stmt
#else
  #define PROFILE_STATEMENT(stmt)
#endif

#define SET_MIN_MAX(v, w) \
  do {                    \
    if ((v) > (w)) {      \
      index_t temp = (v); \
      (v) = (w);          \
      (w) = temp;         \
    }                     \
  } while (0)


template <typename InputIterator, typename OutputIterator>
OutputIterator simple_exclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
{
  return thrust::exclusive_scan(first, last + 1, result);
}
