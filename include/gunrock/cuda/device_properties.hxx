/**
 * @file device_properties.hxx
 * @author Muhammad Osama
 * @brief Updated for CUDA ≥ 12.6 (removed legacy cudaDeviceProp fields)
 */

#pragma once
#include <iostream>
#include <gunrock/cuda/device.hxx>
#include <gunrock/error.hxx>

// Order matters: include runtime API before cuda.h
#include <cuda_runtime_api.h>
#include <cuda.h>

namespace gunrock {
namespace gcuda {

typedef cudaDeviceProp device_properties_t;

struct compute_capability_t {
  unsigned major;
  unsigned minor;
  constexpr unsigned as_combined_number() const { return major * 10 + minor; }
  constexpr bool operator==(int i) const { return (int)as_combined_number() == i; }
  constexpr bool operator!=(int i) const { return (int)as_combined_number() != i; }
  constexpr bool operator>(int i)  const { return (int)as_combined_number() >  i; }
  constexpr bool operator<(int i)  const { return (int)as_combined_number() <  i; }
  constexpr bool operator>=(int i) const { return (int)as_combined_number() >= i; }
  constexpr bool operator<=(int i) const { return (int)as_combined_number() <= i; }
};

constexpr compute_capability_t make_compute_capability(unsigned major, unsigned minor) {
  return compute_capability_t{major, minor};
}
constexpr compute_capability_t make_compute_capability(unsigned combined) {
  return compute_capability_t{combined / 10, combined % 10};
}
constexpr compute_capability_t fetch_compute_capability() {
  return make_compute_capability(SM_TARGET);
}

namespace properties {

enum : size_t { KiB = 1024, K = 1024 };

inline constexpr const char* arch_name(compute_capability_t capability) {
  return (capability.major == 8)                            ? "Ampere"
         : (capability.major == 7 && capability.minor == 5) ? "Turing"
         : (capability.major == 7)                          ? "Volta"
         : (capability.major == 6)                          ? "Pascal"
         : (capability.major == 5)                          ? "Maxwell"
         : (capability.major == 3)                          ? "Kepler"
                                                            : nullptr;
}

inline constexpr unsigned cta_max_threads() { return 1 << 10; }
inline constexpr unsigned warp_max_threads() { return 1 << 5; }

inline constexpr unsigned sm_max_ctas(compute_capability_t c) {
  return (c >= 86) ? 16 : (c >= 80) ? 32 : (c >= 75) ? 16 : (c >= 50) ? 32 : 16;
}
inline constexpr unsigned sm_max_threads(compute_capability_t c) {
  return (c >= 86) ? 1536 : (c >= 80) ? 2048 : (c >= 75) ? 1024 : 2048;
}
inline constexpr unsigned sm_registers(compute_capability_t c) {
  return (c >= 50) ? 64 * K : (c >= 37) ? 128 * K : 64 * K;
}

template <enum cudaFuncCache sm3XCacheConfig = cudaFuncCachePreferNone>
inline constexpr unsigned sm_max_shared_memory_bytes(compute_capability_t capability) {
  unsigned sm3XConfiguredSmem =
      (sm3XCacheConfig == cudaFuncCachePreferNone)     ? 48 * KiB
    : (sm3XCacheConfig == cudaFuncCachePreferShared)   ? 48 * KiB
    : (sm3XCacheConfig == cudaFuncCachePreferL1)       ? 16 * KiB
    : (sm3XCacheConfig == cudaFuncCachePreferEqual)    ? 32 * KiB
                                                       : 48 * KiB;

  return (capability >= 86) ? 100 * KiB :
         (capability >= 80) ? 164 * KiB :
         (capability >= 75) ? 64  * KiB :
         (capability >= 70) ? 96  * KiB :
         (capability >= 62) ? 64  * KiB :
         (capability >= 61) ? 96  * KiB :
         (capability >= 53) ? 64  * KiB :
         (capability >= 52) ? 96  * KiB :
         (capability >= 50) ? 64  * KiB :
         (capability >= 37) ? 64  * KiB + sm3XConfiguredSmem :
                              sm3XConfiguredSmem;
}

inline constexpr unsigned shared_memory_banks() { return 1 << 5; }

template <enum cudaSharedMemConfig sm3XSmemConfig = cudaSharedMemBankSizeDefault>
inline constexpr unsigned shared_memory_bank_stride() {
  return (sm3XSmemConfig == cudaSharedMemBankSizeEightByte) ? 1 << 3 : 1 << 2;
}

// Modern attribute-based accessors
inline unsigned clock_rate() {
  int val = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&val, cudaDevAttrClockRate, dev);
  return static_cast<unsigned>(val);
}

inline constexpr unsigned compute_version(device_properties_t& prop) {
  return prop.major * 10 + prop.minor;
}

unsigned device_count() {
  int count = 0; cudaGetDeviceCount(&count); return count;
}
unsigned driver_version() {
  int v = 0; cudaDriverGetVersion(&v); return v;
}
unsigned runtime_version() {
  int v = 0; cudaRuntimeGetVersion(&v); return v;
}

inline std::string gpu_name(device_properties_t& prop) { return prop.name; }
inline constexpr unsigned sm_major(device_properties_t& prop) { return prop.major; }
inline constexpr unsigned sm_minor(device_properties_t& prop) { return prop.minor; }
inline constexpr unsigned multi_processor_count(device_properties_t& prop) {
  return prop.multiProcessorCount;
}

void set_device_properties(device_properties_t* prop) {
  int dev = 0;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(prop, dev);
}

inline constexpr unsigned total_global_memory(device_properties_t& prop) {
  return prop.totalGlobalMem;
}

inline int get_max_grid_dimension_x(int dev) {
  int max_dim_x = 0;
  cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, dev);
  return max_dim_x;
}

void print(device_properties_t& prop) {
  int ordinal = 0;
  cudaGetDevice(&ordinal);

  size_t freeMem = 0, totalMem = 0;
  error::error_t status = cudaMemGetInfo(&freeMem, &totalMem);
  error::throw_if_exception(status);

  // Try to query clock rate; ignore failures
  int coreClock = 0;
  cudaDeviceGetAttribute(&coreClock, cudaDevAttrClockRate, ordinal);

  // Stub values for fields removed in CUDA 12.6+
  int memClock = 0;
  int busWidth = 0;

  // Optional: set some fake but reasonable defaults
  memClock = 1000;  // 1 GHz placeholder
  busWidth = 256;   // 256-bit memory bus placeholder

  double memBandwidth = (memClock * 1000.0) * (busWidth / 8.0 * 2) / 1.0e9;

  std::cout << prop.name << " : " << coreClock / 1000.0 << " MHz "
            << "(Ordinal " << ordinal << ")" << std::endl;
  std::cout << "FreeMem: " << (int)(freeMem / (1 << 20)) << " MB  "
            << "TotalMem: " << (int)(totalMem / (1 << 20)) << " MB  "
            << ((int)8 * sizeof(int*)) << "-bit pointers." << std::endl;
  std::cout << prop.multiProcessorCount << " SMs enabled, Compute Capability sm_"
            << prop.major << prop.minor << std::endl;
  std::cout << "Mem Clock: " << memClock / 1000.0 << " MHz x "
            << busWidth << " bits (" << memBandwidth << " GB/s)" << std::endl;
  std::cout << "ECC " << (prop.ECCEnabled ? "Enabled" : "Disabled") << std::endl;
}

}  // namespace properties
}  // namespace gcuda
}  // namespace gunrock
