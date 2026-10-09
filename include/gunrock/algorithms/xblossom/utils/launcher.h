// Vendored from X-Blossom (src/include/utils/launcher.h) for the Gunrock X-Blossom app.
#ifndef LAUNCHER_H
#define LAUNCHER_H

#include "utils/utils_gpu.h"
#include "utils/stream.h"
#include <cuda_runtime.h>

template <typename F, typename... Args>
__global__ void KernelWrapper(F f, Args... args) {
  f(args...);
}

template <typename F, typename... Args>
__global__ void KernelWrapperForEach(size_t size, F f, Args... args) {
  for (size_t i = TID_1D; i < size; i += TOTAL_THREADS_1D) {
    f(i, args...);
  }
}

inline void CheckKernelLaunch(const char* func_name, const char* file, int line) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Error Launching kernel for function '%s' at %s:%d %s\n",
            func_name, file, line, cudaGetErrorString(err));
    std::terminate();
  }
}

#define CUDA_CHECK_KERNEL(func) \
  CheckKernelLaunch(#func, __FILE__, __LINE__)

#ifndef XB_MAX_TASKLET_BLOCK_SIZE
#define XB_MAX_TASKLET_BLOCK_SIZE 512
#endif

template <typename F, typename... Args>
void LaunchKernel(const CudaStream& stream, dim3 grid_size, dim3 block_size, F f,
                  Args&&... args) {
  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
    f, std::forward<Args>(args)...);

  CUDA_CHECK_KERNEL(f);
}

template <typename F, typename... Args>
void LaunchKernel(const CudaStream& stream, F f, Args&&... args) {
  int grid_size, block_size;

  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
    &grid_size, &block_size, KernelWrapper<F, Args...>, 0,
    reinterpret_cast<int>(MAX_BLOCK_SIZE)));
  ApplyCudaLaunchCaps(grid_size, block_size);

  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
    f, std::forward<Args>(args)...);

  CUDA_CHECK_KERNEL(f);
}

template <typename F, typename... Args>
void LaunchKernel(const CudaStream& stream, size_t size, F f, Args&&... args) {
  int grid_size, block_size;

  KernelSizing(grid_size, block_size, size);
  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
    f, std::forward<Args>(args)...);

  CUDA_CHECK_KERNEL(f);
}

// For Testing
template <typename F, typename... Args>
void LaunchKernelSingleThread(const CudaStream& stream, F f, Args&&... args) {
  KernelWrapper<<<1, 1>>>(f, std::forward<Args>(args)...);
  CUDA_CHECK_KERNEL(f);
}


template <typename F, typename... Args>
void LaunchKernelForEach(const CudaStream& stream, size_t size, F f, Args&&... args) {
  int grid_size, block_size;

  KernelSizing(grid_size, block_size, size);
  KernelWrapperForEach<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
    size, f, std::forward<Args>(args)...);

  stream.Sync();

  CUDA_CHECK_KERNEL(f);
}

template <typename F, typename... Args>
void LaunchKernelForEachMax(const CudaStream& stream, size_t size, F f, Args&&... args) {
  int grid_size, block_size;

  // KernelSizing(grid_size, block_size, size);
  block_size = XB_MAX_TASKLET_BLOCK_SIZE;
  grid_size = CEIL_DIV(size, block_size);
  ApplyCudaLaunchCaps(grid_size, block_size);

  KernelWrapperForEach<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
    size, f, std::forward<Args>(args)...);

  stream.Sync();

  CUDA_CHECK_KERNEL(f);
}

template <typename F, typename... Args>
void LaunchKernelFix(const CudaStream& stream, size_t size, F f, Args&&... args) {
  int grid_size = 256;
  int block_size = 256;
  ApplyCudaLaunchCaps(grid_size, block_size);

  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
    f, std::forward<Args>(args)...);

  CUDA_CHECK_KERNEL(f);
}

class DefaultLauncher {
public:
  template <typename F, typename... Arg>
  static void ExecuteOneTasklet(F f, Arg... args) {
    auto& s = GetDefaultStream();
    LaunchKernelSingleThread(s, f, args...);
  }

  template <typename F, typename... Arg>
  static void ExecuteNTasklet(size_t n, F f, Arg... arg) {
    auto& s = GetDefaultStream();
    LaunchKernelForEach(s, n, f, arg...);
  }

private:
  static CudaStream& GetDefaultStream() {
  static CudaStream s{}; // Initialize s at first method call
    return s;
  }
};

#endif //LAUNCHER_H
