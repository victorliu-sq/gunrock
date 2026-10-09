// Vendored from X-Blossom (src/include/utils/cpp_types.h) for the Gunrock X-Blossom app.
#ifndef CPP_TYPES_H
#define CPP_TYPES_H

#include <memory>
#include <thread>
#include <vector>
#include <mutex>

// Standard vector (host-side standard memory)
template<typename T>
using Vector = std::vector<T>;

using String = std::string;

template<typename T>
using UPtr = std::unique_ptr<T>;

template<typename T>
using SPtr = std::shared_ptr<T>;

#define Move(x) std::move(x)
#define Ref(x) std::ref(x)

template<typename T, typename... Args>
static inline auto MakeUPtr(Args &&... args) -> UPtr<T> {
  return std::make_unique<T>(std::forward<Args>(args)...);
}

template<typename T, typename... Args>
static inline auto MakeSPtr(Args &&... args) -> SPtr<T> {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

// standard threads
using SThread = std::thread;

template<typename Function, typename... Args>
static inline auto LaunchThread(Function&& f, Args &&... args) -> SThread {
  return std::thread(std::forward<Function>(f), std::forward<Args>(args)...);
}

using LockGuard = std::lock_guard<std::mutex>;

template<typename P1, typename P2>
using Pair = std::pair<P1, P2>;

using ns_t = uint64_t;
using ms_t = double;
using sec_t = double;

#endif //CPP_TYPES_H
