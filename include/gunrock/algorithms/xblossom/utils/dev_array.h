// Vendored from X-Blossom (src/include/utils/dev_array.h) for the Gunrock X-Blossom app.
#ifndef ARRAY_VIEW_H
#define ARRAY_VIEW_H

#include "dev_array.h"
#include "utils/utils_gpu.h"

template<typename T>
using HArray = HVector<T>;

template<typename T>
class DArrayView {
public:
  DEV_HOST_INLINE DArrayView(T *data, size_t size)
    : data_(data), size_(size) {
  }

  DEV_INLINE T &operator[](size_t idx) {
    assert(idx < size_);
    return data_[idx];
  }

  DEV_INLINE const T &operator[](size_t idx) const {
    assert(idx < size_);
    return data_[idx];
  }

  DEV_INLINE T *data() const { return data_; }

  DEV_INLINE size_t size() const { return size_; }

private:
  T *data_;
  size_t size_;
};

template<typename T>
class DArray : public DVector<T> {
public:
  // Default constructor
  DArray() : DVector<T>() {
  }

  // Constructor with size
  explicit DArray(size_t size) : DVector<T>(size) {
  }

  // Constructor with size and default value
  DArray(size_t size, const T &val) : DVector<T>(size, val) {
  }

  auto GetRawPtr() -> T* {
    return thrust::raw_pointer_cast(this->data());
  }

  DArrayView<T> DeviceView() {
    return DArrayView<T>(thrust::raw_pointer_cast(this->data()), this->size());
  }

  void Fill(T val) {
    thrust::fill(this->begin(), this->end(), val);
  }


  void GetFromDevice(HArray<T> &host_array) const {
    // Ensure host array size matches device array size clearly
    host_array.resize(this->size());
    // Explicitly copy data from device (DArray) to host (HArray)
    thrust::copy(this->begin(), this->end(), host_array.begin());
  }

  void SetToDevice(const HArray<T> &host_array) {
    // Ensure host array size matches device array size clearly
    this->resize(host_array.size());
    // Explicitly copy data from device (DArray) to host (HArray)
    thrust::copy(host_array.begin(), host_array.end(), this->begin());
  }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const DArray<T>& darray) {
  // Copy data from device to host
  HArray<T> host_copy;
  darray.GetFromDevice(host_copy);

  os << "[";
  for (size_t i = 0; i < host_copy.size(); ++i) {
    os << host_copy[i];
    if (i + 1 < host_copy.size())
      os << ", ";
  }
  os << "]";
  return os;
}

#endif //ARRAY_VIEW_H
