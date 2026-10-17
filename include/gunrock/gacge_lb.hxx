/**
 * @file gacge_lb.hxx
 * @brief GACGE's load balancing: the only GPU-specific choice of the five
 * primitives (BFS, SSSP, PageRank, TC, X-Blossom), whose algorithms follow
 * their CPU counterparts (GAP, XB-GAP) step for step. A step's work is a list
 * of tiles (a frontier's vertices, or every vertex), tile i being vertex u with
 * len(i) atoms (edges) from start(i). The schedules are gunrock/loops' (Osama,
 * Porumbescu and Owens, "A Programming Model for GPU Load Balancing", PPoPP
 * 2023; src/loops), with loops' launch box for sm_89 (128 threads a block):
 *
 *   thread  thread-mapped (loops::schedule thread_mapped): a thread per tile;
 *   warp    group-mapped with groups of 32 (loops' warp_mapped),
 *   block   group-mapped with groups of 128 (loops' block_mapped): a group
 *           takes as many tiles as threads, scans their sizes, and its
 *           threads share the atoms;
 *   merge   merge-path (loops' merge_path_flat): tiles plus atoms cut evenly,
 *           7 units a thread, a block per merge tile; the merge tiles'
 *           coordinates from loops' generate_search_coordinates.
 *
 * GACGE_LB picks one for every primitive (default merge, since 2026-10-16: on
 * the eight graphs it was the fastest of the four or within 25% of it in 36
 * of 40 primitive-graph cases, at worst 1.82x). f(i, u, e)
 * handles atom e of tile i (vertex u); true skips the rest of tile i's atoms
 * in that thread (group: in that group), e.g. BFS's bottom-up step at the first
 * neighbor in the front.
 */
#pragma once

#include <loops/schedule.hxx>
#include <gunrock/gacge_task_stats.hxx>

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
constexpr int kBlock = 128;           // loops' launch box (sm_89)
constexpr std::size_t kItems = 7;     // merge-path: units a thread (loops' launch box, 4-byte values)
constexpr long long kMergeTile = kBlock * (long long)kItems;

// a schedule at compile time: 1 thread-mapped, 0 merge-path, else the group size
template <int G>
using tag_t = std::integral_constant<int, G>;

// set by a driver to use these schedules in place of Gunrock's advance (xblossom -l gacge)
inline bool& enabled() {
  static bool v = false;
  return v;
}

inline const char* name(mode_t m) {
  switch (m) {
    case mode_t::thread: return "thread";
    case mode_t::warp: return "warp";
    case mode_t::group: return "group64";
    case mode_t::block: return "block";
    default: return "merge";
  }
}

// a schedule's name, else def
inline mode_t parse(const char* e, mode_t def) {
  if (e == nullptr)
    return def;
  if (std::strcmp(e, "thread") == 0) return mode_t::thread;
  if (std::strcmp(e, "warp") == 0) return mode_t::warp;
  if (std::strcmp(e, "group") == 0) return mode_t::group;
  if (std::strcmp(e, "merge") == 0) return mode_t::merge;
  return mode_t::block;
}

// GACGE_LB, else the primitive's default
inline mode_t mode(mode_t def) { return parse(std::getenv("GACGE_LB"), def); }

// a kernel's own variable (e.g. GACGE_LB_TD), else GACGE_LB, else its default
inline mode_t mode(const char* var, mode_t def) { return parse(std::getenv(var), mode(def)); }

// f(tag_t<G>{}) for the schedule m
template <typename func_t>
void dispatch(mode_t m, func_t&& f) {
  switch (m) {
    case mode_t::thread: f(tag_t<1>{}); break;
    case mode_t::warp: f(tag_t<32>{}); break;
    case mode_t::group: f(tag_t<64>{}); break;
    case mode_t::block: f(tag_t<kBlock>{}); break;
    default: f(tag_t<0>{}); break;
  }
}

// the grid of the thread- and group-mapped schedules over n tiles (merge-path: merge_t::tiles)
inline unsigned int grid(long long n) {
  return (unsigned int)std::max<long long>(1, (n + kBlock - 1) / kBlock);
}

// tiles_t: __device__ void item(long long i, vertex_t& u, long long& start) const;
//          __device__ long long len(long long i) const;

// a loops layout over tiles_t: tile t's atoms are its edges [start, start + len)
template <typename tiles_t>
struct edge_layout_t {
  using tile_id_t = int;
  using atom_id_t = int;
  using tile_end_iterator_t = const int*;
  tiles_t it;
  int n;
  __host__ __device__ int num_tiles() const { return n; }
  __host__ __device__ int num_atoms() const { return 0; }
  __device__ int tile_size(int t) const { return (int)it.len(t); }
  __device__ int tile_begin(int t) const {
    typename tiles_t::vertex_type u;
    long long start;
    it.item(t, u, start);
    return (int)start;
  }
  __device__ int tile_end(int t) const { return tile_begin(t) + tile_size(t); }
};

// thread-mapped: loops' setup<thread_mapped>
template <typename tiles_t, typename func_t>
__device__ __forceinline__ void thread_mapped(const tiles_t& it, long long n, func_t f) {
  using layout_t = edge_layout_t<tiles_t>;
  using setup_t =
      loops::schedule::setup<loops::schedule::algorithms_t::thread_mapped, 1, 1, int, int, int, int, layout_t>;
  setup_t config(layout_t{it, (int)n});
  for (auto tile : config.tiles()) {
    typename tiles_t::vertex_type u;
    long long start;
    it.item(tile, u, start);
    for (auto e : config.atoms(tile))
      if (f((long long)tile, u, (long long)e))
        break;
  }
}

// group-mapped (groups of G, a block of kBlock): loops' setup<group_mapped>
template <int G, typename tiles_t, typename func_t>
__device__ __forceinline__ void group_mapped(const tiles_t& it, long long n, func_t f) {
  using layout_t = edge_layout_t<tiles_t>;
  using setup_t =
      loops::schedule::setup<loops::schedule::algorithms_t::group_mapped, kBlock, G, int, int, int, int, layout_t>;
  using vertex_t = typename tiles_t::vertex_type;
  __shared__ typename setup_t::storage_t temporary_storage;
  __shared__ vertex_t s_u[kBlock];  // the block's tiles' vertices, by slot
  __shared__ int s_stop[kBlock];
  const long long first = (long long)blockIdx.x * kBlock;
  if (first + threadIdx.x < n) {
    long long start;
    it.item(first + threadIdx.x, s_u[threadIdx.x], start);
  }
  s_stop[threadIdx.x] = 0;
  __syncthreads();
  setup_t config(temporary_storage, layout_t{it, (int)n});
  auto p = config.partition();
  for (auto virtual_atom : config.atom_accessor(p)) {
    auto virtual_tile = config.tile_accessor(virtual_atom, p);
    if (!config.is_valid_accessor(virtual_tile, p))
      continue;
    const int slot = (int)virtual_tile + (int)(p.meta_group_rank() * p.size());
    if (s_stop[slot])
      continue;
    auto tile = config.tile_id(virtual_tile, p);
    auto e = config.atom_id(virtual_atom, tile, virtual_tile, p);
    if (f((long long)tile, s_u[slot], (long long)e))
      s_stop[slot] = 1;
  }
  __syncthreads();
}

// Group-mapped over tiles [first, first + k), k <= kBlock, inside one block
// (SSSP's bin fusion rounds, a block's own vertices): the block's kBlock / G
// groups each take G consecutive tiles. Every thread of the block calls it.
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

// merge-path: loops' setup<merge_path_flat> over a CSR layout of the tiles'
// prefix sum (offsets[0..n], int), its merge tiles' coordinates in meta
using coord_t = loops::schedule::coord_t;
using meta_t = loops::schedule::merge_path::preprocess_t<kBlock, kItems, int, int, std::size_t, std::size_t>;
using merge_setup_t = loops::schedule::setup<loops::schedule::algorithms_t::merge_path_flat, kBlock, kItems, int, int,
                                             std::size_t, std::size_t>;

struct merge_args_t {
  const int* offsets = nullptr;
  meta_t* meta = nullptr;
};

struct no_done_t {
  __device__ void operator()() const {}
};

// f(i, u, e) for this thread's atoms; done() by every thread of a merge tile
// after them (converged: e.g. a warp reduction of a thread's partial)
template <typename tiles_t, typename func_t, typename done_t = no_done_t>
__device__ __forceinline__ void merge_path(const tiles_t& it, long long n, merge_args_t m, func_t f,
                                           done_t done = done_t{}) {
  __shared__ typename merge_setup_t::storage_t temporary_storage;
  const std::size_t nnz = (std::size_t)m.offsets[n];
  merge_setup_t config(*m.meta, temporary_storage, const_cast<int*>(m.offsets), (std::size_t)n, nnz);
  auto map = config.init();
  if (!config.is_valid_accessor(map))
    return;
  long long cur = -1, row_begin = 0, start = 0;
  typename tiles_t::vertex_type u{};
  bool stopped = false;
#pragma unroll
  for (auto item : config.virtual_idx()) {
    auto nz = config.atom_idx(item, map);
    auto row = config.tile_idx(map);
    if (config.atoms_counting_it[map.y] < temporary_storage.tile_end_offset[map.x]) {
      if (row != cur) {
        it.item(row, u, start);
        row_begin = m.offsets[row];
        cur = row;
        stopped = false;
      }
      if (!stopped)
        stopped = f((long long)row, u, start + ((long long)nz - row_begin));
      map.y++;
    } else {
      map.x++;
    }
  }
  done();
}

// the schedule G over n tiles (merge-path: m)
template <int G, typename tiles_t, typename func_t>
__device__ __forceinline__ void for_edges(const tiles_t& it, long long n, merge_args_t m, func_t f) {
  if constexpr (G == 1)
    thread_mapped(it, n, f);
  else if constexpr (G == 0)
    merge_path(it, n, m, f);
  else
    group_mapped<G>(it, n, f);
}

// the merge-path side of a primitive: the tiles' prefix sum, the merge tiles'
// coordinates (loops' generate_search_coordinates) and a meta whose data()
// they are, in device memory; buffers sized outside the timing
struct merge_t {
  int* offsets = nullptr;
  long long cap = 0;  // tiles
  coord_t* coords = nullptr;
  long long coords_cap = 0;  // merge tiles + 1
  meta_t* d_meta = nullptr;
  long long tiles = 0;  // the last plan's merge tiles: the grid
  merge_t() = default;
  merge_t(const merge_t&) = delete;
  merge_t& operator=(const merge_t&) = delete;
  ~merge_t() {
    if (offsets) cudaFree(offsets);
    if (coords) cudaFree(coords);
    if (d_meta) cudaFree(d_meta);
  }
  void reserve(long long n) {
    if (n <= cap)
      return;
    if (offsets)
      cudaFree(offsets);
    cap = std::max<long long>(n, 2 * cap);
    cudaMalloc(&offsets, (cap + 1) * sizeof(int));
  }
  // room for the merge tiles of up to `work` units (tiles + atoms)
  void reserve_work(long long work) {
    const long long need = work / kMergeTile + 2;
    if (need <= coords_cap)
      return;
    if (coords)
      cudaFree(coords);
    coords_cap = std::max<long long>(need, 2 * coords_cap);
    cudaMalloc(&coords, coords_cap * sizeof(coord_t));
    if (!d_meta)
      cudaMalloc(&d_meta, sizeof(meta_t));
    // preprocess_t's data(): its total work, merge tiles, then the coordinates' pointer
    struct {
      std::size_t total_work, num_merge_tiles;
      coord_t* coordinates;
    } head{0, 0, coords};
    static_assert(sizeof(head) <= sizeof(meta_t), "loops' preprocess_t layout");
    cudaMemcpy(d_meta, &head, sizeof(head), cudaMemcpyHostToDevice);
  }
  // offsets[0..n] of the tiles' lengths, on stream
  template <typename tiles_t>
  void prefix(const tiles_t& it, long long n, cudaStream_t stream) {
    thrust::transform_exclusive_scan(
        thrust::cuda::par_nosync.on(stream), thrust::counting_iterator<long long>(0),
        thrust::counting_iterator<long long>(n + 1), offsets,
        [it, n] __device__(long long i) -> int { return i < n ? (int)it.len(i) : 0; }, 0, thrust::plus<int>());
  }
  // the merge tiles over offs[0..n] (this prefix sum, or a static one such as
  // the row offsets), on stream: sets tiles (the grid) and their coordinates
  merge_args_t plan(const int* offs, long long n, cudaStream_t stream) {
    int nnz = 0;
    cudaMemcpyAsync(&nnz, offs + n, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    tiles = (n + nnz + kMergeTile - 1) / kMergeTile;
    reserve_work(n + nnz);
    using layout_t = loops::layout::csr<int, int>;
    loops::schedule::merge_path::generate_search_coordinates<kBlock, kItems, layout_t, std::size_t, std::size_t>
        <<<(unsigned int)((tiles + 1 + kBlock - 1) / kBlock), kBlock, 0, stream>>>(
            layout_t(offs, (int)n, nnz), (std::size_t)n, (std::size_t)nnz, (std::size_t)tiles, coords);
    return {offs, d_meta};
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
    advance_kernel(graph_t g, op_t op, frontier_tiles_t<graph_t> tiles, long long n, merge_args_t m) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  for_edges<G>(tiles, n, m, [&](long long, vertex_t u, long long e) {
    op(u, g.get_destination_vertex((edge_t)e), (edge_t)e, g.get_edge_weight((edge_t)e));
    return false;
  });
}

// GACGE_TASK_STATS: the advance with each thread's work added up (op returns its
// edge's work) and recorded once per thread that had any
template <int G, typename graph_t, typename op_t>
__global__ void __launch_bounds__(kBlock)
    advance_count_kernel(graph_t g, op_t op, frontier_tiles_t<graph_t> tiles, long long n, merge_args_t m,
                         gacge::task::view_t v) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  unsigned long long mine = 0;
  for_edges<G>(tiles, n, m, [&](long long, vertex_t u, long long e) {
    mine += (unsigned long long)op(u, g.get_destination_vertex((edge_t)e), (edge_t)e, g.get_edge_weight((edge_t)e));
    return false;
  });
  if (mine)
    gacge::task::record(v, mine);
  gacge::task::record_warp(v, mine);
}

template <typename graph_t, typename op_t>
void advance_count(mode_t md, graph_t& g, op_t op, const typename graph_t::vertex_type* frontier, long long n,
                   merge_t& mg, cudaStream_t stream, gacge::task::view_t v) {
  if (n <= 0)
    return;
  frontier_tiles_t<graph_t> tiles{g, frontier};
  dispatch(md, [&](auto tag) {
    constexpr int G = decltype(tag)::value;
    if constexpr (G == 0) {
      mg.reserve(n);
      mg.prefix(tiles, n, stream);
      const merge_args_t m = mg.plan(mg.offsets, n, stream);
      if (mg.tiles > 0)
        advance_count_kernel<0><<<(unsigned int)mg.tiles, kBlock, 0, stream>>>(g, op, tiles, n, m, v);
    } else {
      advance_count_kernel<G><<<grid(n), kBlock, 0, stream>>>(g, op, tiles, n, merge_args_t{}, v);
    }
  });
}

template <typename graph_t, typename op_t>
void advance(mode_t md, graph_t& g, op_t op, const typename graph_t::vertex_type* frontier, long long n, merge_t& mg,
             cudaStream_t stream) {
  if (n <= 0)
    return;
  frontier_tiles_t<graph_t> tiles{g, frontier};
  dispatch(md, [&](auto tag) {
    constexpr int G = decltype(tag)::value;
    if constexpr (G == 0) {
      mg.reserve(n);
      mg.prefix(tiles, n, stream);
      const merge_args_t m = mg.plan(mg.offsets, n, stream);
      if (mg.tiles > 0)
        advance_kernel<0><<<(unsigned int)mg.tiles, kBlock, 0, stream>>>(g, op, tiles, n, m);
    } else {
      advance_kernel<G><<<grid(n), kBlock, 0, stream>>>(g, op, tiles, n, merge_args_t{});
    }
  });
}

}  // namespace lb
}  // namespace gacge
