// GACGE (2026-10-16): task-length distributions of the GPU primitives,
// GACGE_TASK_STATS=1 only (off: every record call returns at once, nothing is
// allocated). A task is a primitive's tile: a frontier vertex's edges (BFS
// top-down), an unvisited vertex's search for a frontier parent (BFS
// bottom-up), a bin vertex's edges (SSSP), a vertex's in-edges in an iteration
// (PageRank), an edge's intersection (TC), a frontier edge (X-Blossom); its
// length is the inner loop's iterations as executed. The lengths go to a
// histogram with eight buckets per power of two (exact below 8), with the
// count, the total, the maximum and the sum of squares; a warp's lanes with the
// same bucket add up first (one atomic per bucket and warp). Printed as
//   XBTask: prim=<p> kind=<k> tasks=<n> work=<sum> max=<m> sumsq=<s> extra=<a,b,c,d,e> buckets=<b:n:w;...>
#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thrust/device_vector.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace gacge {
namespace task {

constexpr int kBuckets = 320;
constexpr int kExtra = 5;

inline bool enabled() {
  static const bool on = std::getenv("GACGE_TASK_STATS") != nullptr;
  return on;
}

struct view_t {
  unsigned long long* cnt = nullptr;   // per bucket: the tasks
  unsigned long long* work = nullptr;  // per bucket: their total length
  unsigned long long* tot = nullptr;   // [0] tasks, [1] work, [2] max, [3..] extra counters
  double* sq = nullptr;                // the sum of the squared lengths
};

__host__ __device__ inline int bucket(unsigned long long len) {
  if (len < 8)
    return (int)len;
  int e = 63;
#ifdef __CUDA_ARCH__
  e = 63 - __clzll((long long)len);
#else
  while (!((len >> e) & 1ULL))
    --e;
#endif
  return 8 + (e - 3) * 8 + (int)((len >> (e - 3)) & 7ULL);
}

// a bucket's smallest length
inline unsigned long long bucket_low(int b) {
  if (b < 8)
    return (unsigned long long)b;
  const int e = (b - 8) / 8 + 3, m = (b - 8) % 8;
  return (8ULL + (unsigned long long)m) << (e - 3);
}

__device__ __forceinline__ void record(const view_t& s, unsigned long long len) {
  if (s.cnt == nullptr)
    return;
  namespace cg = cooperative_groups;
  const int b = bucket(len);
  auto g = cg::labeled_partition(cg::coalesced_threads(), b);
  const unsigned long long w = cg::reduce(g, len, cg::plus<unsigned long long>());
  const double q = cg::reduce(g, (double)len * (double)len, cg::plus<double>());
  const unsigned long long mx = cg::reduce(g, len, cg::greater<unsigned long long>());
  if (g.thread_rank() == 0) {
    const unsigned long long n = g.size();
    atomicAdd(s.cnt + b, n);
    atomicAdd(s.work + b, w);
    atomicAdd(s.tot + 0, n);
    atomicAdd(s.tot + 1, w);
    atomicMax(s.tot + 2, mx);
    atomicAdd(s.sq, q);
  }
}

// an extra counter k (0..kExtra-1), a warp's lanes adding up first
__device__ __forceinline__ void count(const view_t& s, int k, unsigned long long v) {
  if (s.cnt == nullptr)
    return;
  namespace cg = cooperative_groups;
  auto g = cg::coalesced_threads();
  const unsigned long long t = cg::reduce(g, v, cg::plus<unsigned long long>());
  if (g.thread_rank() == 0)
    atomicAdd(s.tot + 3 + k, t);
}

// The warp-internal measure: called by all 32 lanes of every warp at one point
// (the kernel's end, no lane returned earlier), w the lane's iterations (0 for
// a lane without work). The full warp synchronizes there and reduces over all
// 32 lanes, so each physical warp adds exactly once: to extra[0] the sum of
// its lanes' w, to extra[1] 32 x the largest w, to extra[2] one (a warp with
// no work adds nothing). extra[0] / extra[1] is the share of the lane slots its
// work fills when every iteration costs the same.
__device__ __forceinline__ void record_warp(const view_t& s, unsigned long long w) {
  if (s.cnt == nullptr)
    return;
  __syncwarp(0xffffffffu);
  const unsigned w32 = (unsigned)(w < 0xffffffffULL ? w : 0xffffffffULL);
  const unsigned sum = __reduce_add_sync(0xffffffffu, w32);
  const unsigned mx = __reduce_max_sync(0xffffffffu, w32);
  if ((threadIdx.x & 31) == 0 && mx > 0) {
    atomicAdd(s.tot + 3, (unsigned long long)sum);
    atomicAdd(s.tot + 4, 32ULL * mx);
    atomicAdd(s.tot + 5, 1ULL);
  }
}

// record_warp's self-test: block 0, warp 0 all lanes 10 (100%); warp 1 31 lanes
// 1 and one lane 128 (159 / 4096 = 3.9%); each warp in its own view
static __global__ void record_warp_selftest(view_t uniform, view_t skewed) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane & 1) {  // a divergent region before the call, as in the kernels
    __nanosleep(100);
  }
  if (warp == 0)
    record_warp(uniform, 10);
  else
    record_warp(skewed, lane == 7 ? 128 : 1);
}

// one task per listed vertex (all vertices without a list), its length the degree
template <typename graph_t, typename vertex_t>
__global__ void degree_tasks(graph_t G, const vertex_t* list, long long n, view_t s) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    record(s, (unsigned long long)G.get_number_of_neighbors(list ? list[i] : (vertex_t)i));
}

struct stats_t {
  thrust::device_vector<unsigned long long> cnt_, work_, tot_;
  thrust::device_vector<double> sq_;

  view_t view() {
    if (!enabled())
      return {};
    selftest();
    if (cnt_.empty()) {
      cnt_.assign(kBuckets, 0ULL);
      work_.assign(kBuckets, 0ULL);
      tot_.assign(3 + kExtra, 0ULL);
      sq_.assign(1, 0.0);
    }
    return {thrust::raw_pointer_cast(cnt_.data()), thrust::raw_pointer_cast(work_.data()),
            thrust::raw_pointer_cast(tot_.data()), thrust::raw_pointer_cast(sq_.data())};
  }

  unsigned long long tasks() { return cnt_.empty() ? 0ULL : (unsigned long long)tot_[0]; }

  static void selftest() {
    static bool done = false;
    if (done)
      return;
    done = true;
    thrust::device_vector<unsigned long long> t(2 * (3 + kExtra), 0ULL);
    unsigned long long* p = thrust::raw_pointer_cast(t.data());
    thrust::device_vector<unsigned long long> dummy(kBuckets, 0ULL);
    thrust::device_vector<double> dq(2, 0.0);
    view_t a{thrust::raw_pointer_cast(dummy.data()), thrust::raw_pointer_cast(dummy.data()), p,
             thrust::raw_pointer_cast(dq.data())};
    view_t b{a.cnt, a.work, p + 3 + kExtra, a.sq + 1};
    record_warp_selftest<<<1, 64>>>(a, b);
    cudaDeviceSynchronize();
    thrust::host_vector<unsigned long long> h = t;
    std::printf("XBTaskSelfTest: uniform=%.1f%% (100) skewed=%.1f%% (3.9) warps=%llu,%llu\n",
                h[4] ? 100.0 * h[3] / h[4] : -1.0, h[3 + kExtra + 4] ? 100.0 * h[3 + kExtra + 3] / h[3 + kExtra + 4] : -1.0,
                h[5], h[3 + kExtra + 5]);
  }

  // n tasks of one length, added on the host (X-Blossom's plain edge checks)
  void add(unsigned long long n, unsigned long long len) {
    if (!enabled() || n == 0)
      return;
    view();
    const int b = bucket(len);
    cnt_[b] = cnt_[b] + n;
    work_[b] = work_[b] + n * len;
    tot_[0] = tot_[0] + n;
    tot_[1] = tot_[1] + n * len;
    if (len > (unsigned long long)tot_[2])
      tot_[2] = len;
    sq_[0] = sq_[0] + (double)n * (double)len * (double)len;
  }

  void print(const char* prim, const char* kind, const std::string& note = "") {
    if (!enabled())
      return;
    view();
    cudaDeviceSynchronize();
    thrust::host_vector<unsigned long long> c = cnt_, w = work_, t = tot_;
    thrust::host_vector<double> q = sq_;
    std::printf("XBTask: prim=%s kind=%s tasks=%llu work=%llu max=%llu sumsq=%.6e extra=", prim, kind, t[0], t[1],
                t[2], q[0]);
    for (int k = 0; k < kExtra; ++k)
      std::printf(k ? ",%llu" : "%llu", t[3 + k]);
    std::printf("%s%s buckets=", note.empty() ? "" : " ", note.c_str());
    bool first = true;
    for (int b = 0; b < kBuckets; ++b)
      if (c[b]) {
        std::printf(first ? "%d:%llu:%llu" : ";%d:%llu:%llu", b, c[b], w[b]);
        first = false;
      }
    std::printf("\n");
    thrust::fill(cnt_.begin(), cnt_.end(), 0ULL);
    thrust::fill(work_.begin(), work_.end(), 0ULL);
    thrust::fill(tot_.begin(), tot_.end(), 0ULL);
    thrust::fill(sq_.begin(), sq_.end(), 0.0);
  }
};

}  // namespace task
}  // namespace gacge
