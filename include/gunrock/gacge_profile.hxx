// GACGE profile region (not part of upstream Gunrock): with
// GACGE_PROFILE_TRIAL=<k>, the timed region of trial k (0-based, the warm-up
// included) runs between cudaProfilerStart and cudaProfilerStop, so
// `ncu --profile-from-start off` profiles only its kernels: graph loading,
// problem setup and the other trials are not profiled. Unset: nothing happens.
// The drivers number the trials (SetTrial); the timed region is the enactor's
// enact() loop (framework/enactor.hxx), TC's relabelling plus its count, and
// X-Blossom's FindMaximumMatch.
#pragma once

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdlib>

namespace gacge {
namespace profile {

inline int target() {
  static const int t = [] {
    const char* e = std::getenv("GACGE_PROFILE_TRIAL");
    return e ? std::atoi(e) : -1;
  }();
  return t;
}
inline int& trial() {
  static int t = -1;
  return t;
}
inline void SetTrial(int i) { trial() = i; }
inline bool Active() { return target() >= 0 && trial() == target(); }
// Begin twice (TC's relabel, then its enact) or End twice is harmless.
inline void Begin() {
  if (!Active()) return;
  cudaDeviceSynchronize();
  cudaProfilerStart();
}
inline void End() {
  if (!Active()) return;
  cudaDeviceSynchronize();
  cudaProfilerStop();
}

}  // namespace profile
}  // namespace gacge
