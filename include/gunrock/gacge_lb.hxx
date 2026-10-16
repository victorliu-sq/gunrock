/**
 * @file gacge_lb.hxx
 * @brief GACGE's load balancing: the only GPU-specific choice of the five
 * primitives (BFS, SSSP, PageRank, TC, X-Blossom), whose algorithms follow
 * their CPU counterparts (GAP, XB-GAP) step for step. A step's work is a list
 * of tiles (a frontier's vertices, or every vertex), tile i being vertex u with
 * len(i) atoms (edges) from start(i). The schedules are those of Osama,
 * Porumbescu and Owens, "A Programming Model for GPU Load Balancing" (PPoPP
 * 2023; the gunrock/loops library):
 *
 *   thread  thread-mapped: one thread per tile walks its atoms;
 *   warp    group-mapped with groups of 32 (warp-mapped),
 *   group   group-mapped with groups of GACGE_LB_GROUP (64 or 128, default 64),
 *   block   group-mapped with groups of 256 (block-mapped): a group takes as
 *           many tiles as threads, scans their lengths in shared memory, and
 *           its threads share the atoms, an atom's tile found by a binary
 *           search in the scan;
 *   merge   merge-path (Merrill and Garland, as CUB's SpMV): tiles plus atoms
 *           are the work, cut evenly into chunks of GACGE_LB_GRAIN units
 *           (default 8); a thread finds its chunk's start by a binary search on
 *           the merge-path diagonal over the tiles' prefix sum and walks it.
 *
 * GACGE_LB picks one for every primitive (each has its default). f(i, u, e)
 * handles atom e of tile i (vertex u); true skips the rest of tile i's atoms
 * in that thread (group: in that group), e.g. BFS's bottom-up step at the first
 * neighbor in the front.
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace gacge {
namespace lb {

enum class mode_t { thread, warp, group, block, merge };
constexpr int kBlock = 256;

// a schedule at compile time: 1 thread-mapped, 0 merge-path, else the group size
template <int G>
using tag_t = std::integral_constant<int, G>;

// set by a driver to use these schedules in place of Gunrock's advance (xblossom -l gacge)
inline bool& enabled() {
  static bool v = false;
  return v;
}

inline int group_size() {
  static const int v = [] {
    const char* e = std::getenv("GACGE_LB_GROUP");
    return (e && std::atoi(e) == 128) ? 128 : 64;
  }();
  return v;
}

inline const char* name(mode_t m) {
  switch (m) {
    case mode_t::thread: return "thread";
    case mode_t::warp: return "warp";
    case mode_t::group: return group_size() == 128 ? "group128" : "group64";
    case mode_t::block: return "block";
    default: return "merge";
  }
}

// GACGE_LB, else the primitive's default
inline mode_t mode(mode_t def) {
  const char* e = std::getenv("GACGE_LB");
  if (e == nullptr)
    return def;
  if (std::strcmp(e, "thread") == 0) return mode_t::thread;
  if (std::strcmp(e, "warp") == 0) return mode_t::warp;
  if (std::strcmp(e, "group") == 0) return mode_t::group;
  if (std::strcmp(e, "merge") == 0) return mode_t::merge;
  return mode_t::block;
}

inline int grain() {
  static const int v = [] {
    const char* e = std::getenv("GACGE_LB_GRAIN");
    return e ? std::max(1, std::atoi(e)) : 8;
  }();
  return v;
}

// merge: a persistent grid of 8 blocks per SM
inline unsigned int merge_grid() {
  static const unsigned int v = [] {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    return (unsigned int)(sms * 8);
  }();
  return v;
}

// the grid of a step over n tiles (a block holds kBlock tiles, except merge)
template <int G>
inline unsigned int grid(long long n) {
  if (G == 0)
    return merge_grid();
  return (unsigned int)std::max<long long>(1, (n + kBlock - 1) / kBlock);
}

// f(tag_t<G>{}) for the schedule m
template <typename func_t>
void dispatch(mode_t m, func_t&& f) {
  switch (m) {
    case mode_t::thread: f(tag_t<1>{}); break;
    case mode_t::warp: f(tag_t<32>{}); break;
    case mode_t::group:
      if (group_size() == 128)
        f(tag_t<128>{});
      else
        f(tag_t<64>{});
      break;
    case mode_t::block: f(tag_t<256>{}); break;
    default: f(tag_t<0>{}); break;
  }
}

// tiles_t: __device__ void item(long long i, vertex_t& u, long long& start) const;
//          __device__ long long len(long long i) const;

template <typename tiles_t, typename func_t>
__device__ __forceinline__ void thread_mapped(const tiles_t& it, long long n, func_t f) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  typename tiles_t::vertex_type u;
  long long start;
  it.item(i, u, start);
  const long long last = start + it.len(i);
  for (long long e = start; e < last; ++e)
    if (f(i, u, e))
      break;
}

// Group-mapped over tiles [first, first + k), k <= kBlock: the block's
// kBlock / G groups each take G consecutive tiles. Every thread of the block
// calls it (it synchronizes the block).
template <int G, typename tiles_t, typename func_t>
__device__ __forceinline__ void group_tiles(const tiles_t& it, long long first, int k, func_t f) {
  static_assert(G >= 2 && G <= kBlock && (kBlock % G) == 0, "group size");
  using vertex_t = typename tiles_t::vertex_type;
  __shared__ vertex_t s_u[kBlock];
  __shared__ long long s_start[kBlock];
  __shared__ long long s_off[kBlock];  // inclusive scan within each group
  __shared__ int s_stop[kBlock];
  const int t = threadIdx.x;
  const int lane = t % G;
  const int base = t - lane;  // the group's first slot
  long long len = 0;
  if (t < k) {
    vertex_t u;
    long long start;
    it.item(first + t, u, start);
    s_u[t] = u;
    s_start[t] = start;
    len = it.len(first + t);
  }
  s_off[t] = len;
  s_stop[t] = 0;
  __syncthreads();
  for (int d = 1; d < G; d <<= 1) {  // the groups' scans (Hillis-Steele)
    const long long add = lane >= d ? s_off[t - d] : 0;
    __syncthreads();
    s_off[t] += add;
    __syncthreads();
  }
  const int kg = min(G, k - base);  // the group's tiles
  const long long total = kg > 0 ? s_off[base + kg - 1] : 0;
  for (long long x = lane; x < total; x += G) {
    int lo = 0, hi = kg - 1;  // the first tile whose inclusive end exceeds x
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (s_off[base + mid] > x)
        hi = mid;
      else
        lo = mid + 1;
    }
    const int s = base + lo;
    if (s_stop[s])
      continue;
    const long long tile_first = lo > 0 ? s_off[s - 1] : 0;  // the tile's first atom in the group
    if (f(first + s, s_u[s], s_start[s] + (x - tile_first)))
      s_stop[s] = 1;
  }
  __syncthreads();
}

template <int G, typename tiles_t, typename func_t>
__device__ __forceinline__ void group_mapped(const tiles_t& it, long long n, func_t f) {
  const long long first = (long long)blockIdx.x * kBlock;
  group_tiles<G>(it, first, (int)min((long long)kBlock, n - first), f);
}

// Merge-path: unit d < n + total is a tile end (x advances) or an atom (y
// advances); offsets[0..n] is the tiles' exclusive prefix sum.
template <typename offset_t>
__device__ __forceinline__ void merge_path_search(long long d, const offset_t* offsets, long long n, long long total,
                                                  long long& x, long long& y) {
  long long x_min = max(d - total, 0LL), x_max = min(d, n);
  while (x_min < x_max) {
    const long long pivot = (x_min + x_max) >> 1;
    if ((long long)offsets[pivot + 1] <= d - pivot - 1)
      x_min = pivot + 1;
    else
      x_max = pivot;
  }
  x = min(x_min, n);
  y = d - x;
}

template <typename tiles_t, typename offset_t, typename func_t>
__device__ __forceinline__ void merge_path(const tiles_t& it, long long n, const offset_t* offsets, int grain,
                                           func_t f) {
  const long long total = (long long)offsets[n];
  const long long work = n + total;
  const long long chunks = (work + grain - 1) / grain;
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < chunks; c += stride) {
    const long long d0 = c * grain, d1 = min(d0 + grain, work);
    long long x, y;
    merge_path_search(d0, offsets, n, total, x, y);
    typename tiles_t::vertex_type u{};
    long long start = 0, cached = -1;
    bool stopped = false;
    for (long long d = d0; d < d1; ++d) {
      if (x < n && y < (long long)offsets[x + 1]) {  // an atom of tile x
        if (!stopped) {
          if (cached != x) {
            it.item(x, u, start);
            cached = x;
          }
          stopped = f(x, u, start + (y - (long long)offsets[x]));
        }
        ++y;
      } else {  // tile x ends
        ++x;
        stopped = false;
      }
    }
  }
}

// the schedule G over n tiles (offsets and grain for merge-path)
template <int G, typename tiles_t, typename offset_t, typename func_t>
__device__ __forceinline__ void for_edges(const tiles_t& it, long long n, const offset_t* offsets, int grain,
                                          func_t f) {
  if constexpr (G == 1)
    thread_mapped(it, n, f);
  else if constexpr (G == 0)
    merge_path(it, n, offsets, grain, f);
  else
    group_mapped<G>(it, n, f);
}

// merge: offsets[0..n] of the tiles' lengths, on stream
template <typename tiles_t>
void prefix(const tiles_t& it, long long n, unsigned long long* offsets, cudaStream_t stream) {
  thrust::transform_exclusive_scan(
      thrust::cuda::par_nosync.on(stream), thrust::counting_iterator<long long>(0),
      thrust::counting_iterator<long long>(n + 1), offsets,
      [it, n] __device__(long long i) -> unsigned long long { return i < n ? (unsigned long long)it.len(i) : 0ULL; },
      0ULL, thrust::plus<unsigned long long>());
}

// a growable device buffer of offsets (merge), sized outside the timing
struct offsets_t {
  unsigned long long* data = nullptr;
  long long cap = 0;  // tiles
  offsets_t() = default;
  offsets_t(const offsets_t&) = delete;
  offsets_t& operator=(const offsets_t&) = delete;
  ~offsets_t() {
    if (data)
      cudaFree(data);
  }
  void reserve(long long n) {
    if (n <= cap)
      return;
    if (data)
      cudaFree(data);
    cap = std::max<long long>(n, 2 * cap);
    cudaMalloc(&data, (cap + 1) * sizeof(unsigned long long));
  }
};

// a frontier's vertices with all their edges (X-Blossom, BFS top-down)
template <typename graph_t>
struct frontier_tiles_t {
  using vertex_type = typename graph_t::vertex_type;
  graph_t G;
  const vertex_type* frontier;
  __device__ void item(long long i, vertex_type& u, long long& start) const {
    u = frontier[i];
    start = G.get_starting_edge(u);
  }
  __device__ long long len(long long i) const { return G.get_number_of_neighbors(frontier[i]); }
};

// the general advance: op(u, neighbor, edge, weight) over a frontier's edges
template <int G, typename graph_t, typename op_t>
__global__ void __launch_bounds__(kBlock)
    advance_kernel(graph_t g, op_t op, frontier_tiles_t<graph_t> tiles, long long n, const unsigned long long* offsets,
                   int grain) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  for_edges<G>(tiles, n, offsets, grain, [&](long long, vertex_t u, long long e) {
    op(u, g.get_destination_vertex((edge_t)e), (edge_t)e, g.get_edge_weight((edge_t)e));
    return false;
  });
}

template <typename graph_t, typename op_t>
void advance(mode_t m, graph_t& g, op_t op, const typename graph_t::vertex_type* frontier, long long n,
             offsets_t& offs, cudaStream_t stream) {
  if (n <= 0)
    return;
  frontier_tiles_t<graph_t> tiles{g, frontier};
  dispatch(m, [&](auto tag) {
    constexpr int G = decltype(tag)::value;
    if constexpr (G == 0) {
      offs.reserve(n);
      prefix(tiles, n, offs.data, stream);
    }
    advance_kernel<G><<<grid<G>(n), kBlock, 0, stream>>>(g, op, tiles, n, G == 0 ? offs.data : nullptr, grain());
  });
}

}  // namespace lb
}  // namespace gacge
