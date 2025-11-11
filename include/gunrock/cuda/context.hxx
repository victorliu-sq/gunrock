#pragma once

#include <gunrock/cuda/device.hxx>
#include <gunrock/cuda/device_properties.hxx>
#include <gunrock/cuda/event_management.hxx>
#include <gunrock/cuda/stream_management.hxx>
#include <gunrock/cuda/function.hxx>
#include <gunrock/error.hxx>
#include <gunrock/util/timer.hxx>
#include <gunrock/container/array.hxx>
#include <gunrock/container/vector.hxx>

#include <moderngpu/context.hxx>
#include <thrust/execution_policy.h>
#include <iostream>

namespace gunrock {
namespace gcuda {

template <int dummy_arg>
__global__ void dummy_k() {}

struct context_t {
  context_t() = default;
  context_t(const context_t&) = delete;
  context_t& operator=(const context_t&) = delete;

  virtual const gcuda::device_properties_t& props() const = 0;
  virtual void print_properties() = 0;
  virtual gcuda::compute_capability_t ptx_version() const = 0;
  virtual gcuda::stream_t stream() = 0;
  virtual mgpu::standard_context_t* mgpu() = 0;
  virtual void synchronize() = 0;
  virtual gcuda::event_t event() = 0;
  virtual util::timer_t& timer() = 0;
};

class standard_context_t : public context_t {
 protected:
  gcuda::device_properties_t _props{};
  gcuda::compute_capability_t _ptx_version{0, 0};
  gcuda::device_id_t _ordinal{0};
  gcuda::stream_t _stream{};
  gcuda::event_t _event{};
  mgpu::standard_context_t* _mgpu_context{nullptr};
  util::timer_t _timer{};

  template <int dummy_arg = 0>
  void init() {
    gcuda::function_attributes_t attr{};
    cudaFuncGetAttributes(&attr, dummy_k<0>);
    _ptx_version = gcuda::make_compute_capability(attr.ptxVersion);

    cudaSetDevice(_ordinal);
    cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&_event, cudaEventDisableTiming);
    cudaGetDeviceProperties(&_props, _ordinal);

    _mgpu_context = new mgpu::standard_context_t(false, _stream);
  }

 public:
  explicit standard_context_t(gcuda::device_id_t device = 0)
      : _ordinal(device) { init(); }

  standard_context_t(cudaStream_t stream, gcuda::device_id_t device = 0)
      : _ordinal(device), _stream(stream) { init(); }

  ~standard_context_t() { cudaEventDestroy(_event); }  // no override keyword

  const gcuda::device_properties_t& props() const override { return _props; }

  void print_properties() override {
    gcuda::device::set(_ordinal);
    std::cout << "Device [" << _ordinal << "]: "
              << _props.name << " (stub print, CUDA 13.0 build)\n";
  }

  gcuda::compute_capability_t ptx_version() const override { return _ptx_version; }
  gcuda::stream_t stream() override { return _stream; }
  mgpu::standard_context_t* mgpu() override { return _mgpu_context; }

  void synchronize() override {
    if (_stream) cudaStreamSynchronize(_stream);
    else cudaDeviceSynchronize();
  }

  gcuda::event_t event() override { return _event; }
  util::timer_t& timer() override { return _timer; }

  gcuda::device_id_t ordinal() { return _ordinal; }

  auto execution_policy() { return thrust::cuda::par_nosync.on(this->stream()); }
};

class multi_context_t {
 public:
  thrust::host_vector<standard_context_t*> contexts;
  thrust::host_vector<gcuda::device_id_t> devices;
  static constexpr std::size_t MAX_NUMBER_OF_GPUS = 1024;

  explicit multi_context_t(thrust::host_vector<gcuda::device_id_t> _devices)
      : devices(std::move(_devices)) {
    for (auto& d : devices) contexts.push_back(new standard_context_t(d));
  }

  multi_context_t(thrust::host_vector<gcuda::device_id_t> _devices,
                  cudaStream_t stream)
      : devices(std::move(_devices)) {
    for (auto& d : devices)
      contexts.push_back(new standard_context_t(stream, d));
  }

  explicit multi_context_t(gcuda::device_id_t _device)
      : devices(1, _device) {
    contexts.push_back(new standard_context_t(_device));
  }

  multi_context_t(gcuda::device_id_t _device, cudaStream_t stream)
      : devices(1, _device) {
    contexts.push_back(new standard_context_t(stream, _device));
  }

  ~multi_context_t() = default;

  standard_context_t* get_context(gcuda::device_id_t i) {
    return contexts[i];
  }

  std::size_t size() { return contexts.size(); }

  void enable_peer_access() {
    int n = static_cast<int>(size());
    for (int i = 0; i < n; ++i) {
      auto ctx = get_context(i);
      cudaSetDevice(ctx->ordinal());
      for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        cudaDeviceEnablePeerAccess(get_context(j)->ordinal(), 0);
      }
    }
    if (n > 0) cudaSetDevice(get_context(0)->ordinal());
  }
};

}  // namespace gcuda
}  // namespace gunrock
