// Vendored from X-Blossom (src/include/utils/dev_value.h) for the Gunrock X-Blossom app.
#ifndef VALUE_VIEW_H
#define VALUE_VIEW_H

#include "utils/types_gpu.h"

// A reference to a value in device memory used by gpu kernel or device function
template<typename T>
class DValueView {
public:
  DEV_HOST_INLINE explicit DValueView(T *data)
    : data_(data) {
  }

  // DEV_INLINE operator T &() { return *data_; }

  // DEV_INLINE operator const T &() const { return *data_; }

  // Explicit Assignment operator
  DEV_INLINE T &operator=(const T &new_data) {
    *data_ = new_data;
    // return *this;
    return *data_;
  }

  DEV_INLINE T *operator&() const { return data_; }

  DEV_INLINE T operator*() const { return *data_; }

  DEV_INLINE T Value() const { return *data_; }

  // <<< Add volatile read for frequent termination checks >>>
  DEV_INLINE T VolatileValue() const { return *(volatile T*)data_; }

private:
  T *data_;
};


// A reference to a value in device memory used by host
template<typename T>
class DValue {
public:
  explicit DValue() : vector_(1) {
  }

  explicit DValue(T val) : vector_(1) {
    vector_[0] = val;
  }

  inline void SetH2D(T val) {
    vector_[0] = val;
  }

  inline void SetD2D(DValue<T> &val) {
    // Incorrect Implementation, the value is read from device to host, then host to device, which is inefficient/
    // vector_[0] = val.vector_[0];
    thrust::copy(val.vector_.begin(), val.vector_.end(), vector_.begin());
  }

  inline T GetD2H() const {
    return vector_[0];
  }

  inline DValueView<T> DeviceView() {
    return DValueView<T>(CastRawPtr(vector_));
  }


private:
  DVector<T> vector_;
};

#endif //VALUE_VIEW_H
