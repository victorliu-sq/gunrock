/**
 * @file bfs.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Breadth-First Search algorithm.
 * @date 2020-11-23
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/gacge_lb.hxx>

#include <cooperative_groups.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

namespace gunrock {
namespace bfs {

// GACGE_BFS_RULE: how a direction-optimized BFS picks a step's direction.
// gap (default since 2026-10-15, the GAP benchmark's DOBFS, alpha 15, beta 18):
// switch to pull when the edges of the vertices the last push found exceed the
// edges left unexamined / alpha; keep pulling while a pull wakes at least as
// many vertices as the one before or more than |V| / beta, then push again.
// ligra: pull when the frontier's vertices plus edges exceed 5% of the edges.
inline bool rule_gap() {
  static const bool v = [] {
    const char* e = std::getenv("GACGE_BFS_RULE");
    return e == nullptr || std::string(e) == "gap";
  }();
  return v;
}

// rule_gap and direction-optimized: GAP's DOBFS (src/gap/gapbs/src/bfs.cc)
// step for step. A top-down step (TDStep) relaxes the queue's out-edges, a
// vertex is claimed by a load and a compare-and-swap only while unvisited,
// claimed vertices are appended to the next queue and their degrees summed
// into the scout count; a bottom-up step (BUStep) lets every unvisited vertex
// scan its neighbors for one in the front bitmap, sets its bit in the next
// bitmap and counts it awake. QueueToBitmap / BitmapToQueue convert at the
// switches. One read of the counters per step.
namespace gap {

constexpr int kBlock = gacge::lb::kBlock;
constexpr int kAlpha = 15, kBeta = 18;  // GAP's defaults
// Both steps' edges are spread by the load balancing GACGE_LB (gacge_lb.hxx:
// thread, block or merge; the only GPU-specific choice of every GACGE
// primitive). A bottom-up vertex stops at its first neighbor in the front, as
// GAP's: its thread breaks (thread), its block skips its other edges (block),
// and any thread skips its edges once its distance is set (all three).

__device__ __forceinline__ bool get_bit(const uint32_t* bm, long long v) {
  return (bm[v >> 5] >> (v & 31)) & 1u;
}
__device__ __forceinline__ void set_bit_atomic(uint32_t* bm, long long v) {
  atomicOr(bm + (v >> 5), 1u << (v & 31));
}

// one atomic per coalesced group (GAP: QueueBuffer's push_back)
template <typename vertex_t>
__device__ void push_back(vertex_t* q, unsigned long long* tail, vertex_t v) {
  namespace cg = cooperative_groups;
  auto g = cg::coalesced_threads();
  unsigned long long base = 0;
  if (g.thread_rank() == 0)
    base = atomicAdd(tail, (unsigned long long)g.size());
  base = g.shfl(base, 0);
  q[base + g.thread_rank()] = v;
}

// GAP's claim: a load, then a compare-and-swap only while unvisited
template <typename vertex_t>
__device__ __forceinline__ bool claim(vertex_t* dist, vertex_t v, vertex_t level) {
  const vertex_t unvisited = std::numeric_limits<vertex_t>::max();
  return dist[v] == unvisited && atomicCAS(dist + v, unvisited, level + 1) == unvisited;
}

template <typename value_t>
__device__ __forceinline__ void block_add(value_t x, unsigned long long* to) {
  using reduce_t = cub::BlockReduce<unsigned long long, kBlock>;
  __shared__ typename reduce_t::TempStorage temp;
  const unsigned long long sum = reduce_t(temp).Sum((unsigned long long)x);
  if (threadIdx.x == 0 && sum)
    atomicAdd(to, sum);
}

// every vertex, an unvisited one with its edges (GAP's BUStep: parent[u] < 0)
template <typename graph_t>
struct bu_items_t {
  using vertex_type = typename graph_t::vertex_type;
  graph_t G;
  const vertex_type* dist;
  __device__ void item(long long i, vertex_type& u, long long& start) const {
    u = (vertex_type)i;
    start = G.get_starting_edge(u);
  }
  __device__ long long len(long long i) const {
    return dist[i] == std::numeric_limits<vertex_type>::max() ? (long long)G.get_number_of_neighbors((vertex_type)i) : 0;
  }
};

// counters: [0] the next queue's length, [1] the scout count, [2] the awake count
template <int Gs, typename graph_t, typename vertex_t>
__global__ void __launch_bounds__(kBlock) td_kernel(graph_t G, vertex_t* dist, vertex_t level,
                                                    gacge::lb::frontier_tiles_t<graph_t> items, long long n,
                                                    gacge::lb::merge_args_t m, vertex_t* next,
                                                    unsigned long long* counters) {
  unsigned long long scout = 0;
  gacge::lb::for_edges<Gs>(items, n, m, [&](long long, vertex_t, long long e) {
    const vertex_t v = G.get_destination_vertex(e);
    if (claim(dist, v, level)) {
      push_back(next, counters + 0, v);
      scout += G.get_number_of_neighbors(v);  // GAP: scout_count += -curr_val, v's degree
    }
    return false;
  });
  block_add(scout, counters + 1);
}

template <int Gs, typename graph_t, typename vertex_t>
__global__ void __launch_bounds__(kBlock) bu_kernel(graph_t G, vertex_t* dist, vertex_t level, bu_items_t<graph_t> items,
                                                    long long n, gacge::lb::merge_args_t m,
                                                    const uint32_t* front, uint32_t* next,
                                                    unsigned long long* counters) {
  const vertex_t unvisited = std::numeric_limits<vertex_t>::max();
  unsigned long long awake = 0;
  gacge::lb::for_edges<Gs>(items, n, m, [&](long long, vertex_t u, long long e) {
    if (dist[u] != unvisited)  // found: GAP's break
      return true;
    if (get_bit(front, G.get_destination_vertex(e))) {
      if (atomicCAS(dist + u, unvisited, level + 1) == unvisited) {
        set_bit_atomic(next, u);
        awake++;
      }
      return true;
    }
    return false;
  });
  block_add(awake, counters + 2);
}

// GACGE_BFS_WARP (default 128, 0 off): the bottom-up step's thread-mapped
// schedule with Merrill et al.'s warp level for the hubs: a thread scans its
// vertex and stops at the first neighbor in the front, as GAP's; a vertex of
// more than this many edges is scanned by its warp instead, 32 edges at a
// time, still stopping at the first found (the thread-warp-CTA scheme of
// Merrill, Garland and Grimshaw, PPoPP 2012). Without it a hub whose front
// neighbor comes late holds one thread, and its block, for its whole list.
inline long long bu_warp_degree() {
  static const long long v = [] {
    const char* e = std::getenv("GACGE_BFS_WARP");
    return e ? std::atoll(e) : 128LL;
  }();
  return v;
}

template <typename graph_t, typename vertex_t>
__global__ void __launch_bounds__(kBlock) bu_twc_kernel(graph_t G, vertex_t* dist, vertex_t level, vertex_t n,
                                                        const uint32_t* front, uint32_t* next,
                                                        unsigned long long* counters, long long warp_degree) {
  const vertex_t unvisited = std::numeric_limits<vertex_t>::max();
  const vertex_t u = blockIdx.x * kBlock + threadIdx.x;
  const int lane = threadIdx.x & 31;
  unsigned long long awake = 0;
  bool big = false;
  if (u < n && dist[u] == unvisited) {
    const auto first = G.get_starting_edge(u), last = G.get_starting_edge(u + 1);
    if ((long long)(last - first) > warp_degree) {
      big = true;
    } else {
      for (auto e = first; e < last; ++e) {
        if (get_bit(front, G.get_destination_vertex(e))) {
          dist[u] = level + 1;  // only u's thread writes u
          set_bit_atomic(next, u);
          awake++;
          break;
        }
      }
    }
  }
  // the warp scans its hubs one by one, 32 edges at a time, stopping at the first found
  unsigned big_lanes = __ballot_sync(0xffffffffu, big);
  while (big_lanes) {
    const int src_lane = __ffs(big_lanes) - 1;
    big_lanes &= big_lanes - 1;
    const vertex_t w = __shfl_sync(0xffffffffu, u, src_lane);
    const auto first = G.get_starting_edge(w), last = G.get_starting_edge(w + 1);
    for (auto e0 = first; e0 < last; e0 += 32) {
      const auto e = e0 + lane;
      const bool hit = e < last && get_bit(front, G.get_destination_vertex(e));
      if (__ballot_sync(0xffffffffu, hit)) {
        if (lane == src_lane) {
          dist[w] = level + 1;
          set_bit_atomic(next, w);
          awake++;
        }
        break;
      }
    }
  }
  block_add(awake, counters + 2);
}

template <typename vertex_t>
__global__ void queue_to_bitmap_kernel(const vertex_t* queue, unsigned long long n, uint32_t* bm) {
  const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    set_bit_atomic(bm, queue[i]);
}

template <typename vertex_t>
__global__ void bitmap_to_queue_kernel(const uint32_t* bm, vertex_t n, vertex_t* queue,
                                       unsigned long long* counters) {
  const vertex_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < n && get_bit(bm, v))
    push_back(queue, counters + 0, v);
}

}  // namespace gap

template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  // Direction-optimized (GACGE, as Ligra's edgeMap on BFS and the Gunrock
  // paper's push-pull BFS): a step whose frontier vertices + their edges exceed
  // 5% of the edges pulls (every unvisited vertex scans its neighbors until one
  // is in the current level; the graph is symmetric), any other step pushes.
  bool direction_optimized;
  param_t(vertex_t _single_source, bool _direction_optimized = false)
      : single_source(_single_source), direction_optimized(_direction_optimized) {}
};

template <typename vertex_t>
struct result_t {
  vertex_t* distances;
  vertex_t* predecessors;  /// @todo: implement this.
  result_t(vertex_t* _distances, vertex_t* _predecessors)
      : distances(_distances), predecessors(_predecessors) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  thrust::device_vector<vertex_t> visited;  /// @todo not used.

  // GAP's DOBFS (gap::): its queue (two buffers), bitmaps, counters and
  // switching state, allocated by init(), set by reset(), outside the timing
  bool gap_dobfs() const { return param.direction_optimized && rule_gap(); }
  vertex_t* queue[2] = {nullptr, nullptr};
  // the schedule of each of GAP's two step kernels (the mapping is chosen per
  // kernel): top-down merge-path (the frontier's edges spread evenly), bottom-up
  // thread-mapped (a thread scans its vertex and stops at the first neighbor
  // in the front, as GAP's); GACGE_LB_TD / GACGE_LB_BU, or GACGE_LB for both
  gacge::lb::mode_t lb_td = gacge::lb::mode("GACGE_LB_TD", gacge::lb::mode_t::merge);
  gacge::lb::mode_t lb_bu = gacge::lb::mode("GACGE_LB_BU", gacge::lb::mode_t::thread);
  gacge::lb::merge_t merge;  // merge-path: a step's prefix sum of edges and its merge tiles
  int cur = 0;
  unsigned long long queue_n = 0;
  uint32_t* front = nullptr;
  uint32_t* curr = nullptr;
  std::size_t words = 0;
  unsigned long long* d_counters = nullptr;
  unsigned long long* h_counters = nullptr;  // pinned
  long long edges_to_check = 0, scout = 0, awake = 0;
  bool pulling = false;
  vertex_t level = 0;

  cudaStream_t stream() { return this->context->get_context(0)->stream(); }
  void read_counters() {
    cudaMemcpyAsync(h_counters, d_counters, 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream());
    cudaStreamSynchronize(stream());
  }
  void clear_counters() {
    cudaMemsetAsync(d_counters, 0, 3 * sizeof(unsigned long long), stream());
  }

  void init() override {
    if (!gap_dobfs())
      return;
    const std::size_t n = this->get_graph().get_number_of_vertices();
    words = (n + 31) / 32;
    for (auto& q : queue)
      cudaMallocAsync(reinterpret_cast<void**>(&q), std::max<std::size_t>(n, 1) * sizeof(vertex_t), stream());
    cudaMallocAsync(reinterpret_cast<void**>(&front), words * sizeof(uint32_t), stream());
    cudaMallocAsync(reinterpret_cast<void**>(&curr), words * sizeof(uint32_t), stream());
    if (lb_td == gacge::lb::mode_t::merge || lb_bu == gacge::lb::mode_t::merge) {
      merge.reserve(n);
      merge.reserve_work((long long)n + this->get_graph().get_number_of_edges());
    }
    cudaMallocAsync(reinterpret_cast<void**>(&d_counters), 3 * sizeof(unsigned long long), stream());
    cudaMallocHost(reinterpret_cast<void**>(&h_counters), 3 * sizeof(unsigned long long));
  }

  void reset() override {
    auto n_vertices = this->get_graph().get_number_of_vertices();
    auto d_distances = thrust::device_pointer_cast(this->result.distances);
    thrust::fill(thrust::device, d_distances + 0, d_distances + n_vertices,
                 std::numeric_limits<vertex_t>::max());
    thrust::fill(thrust::device, d_distances + this->param.single_source,
                 d_distances + this->param.single_source + 1, 0);
    if (!gap_dobfs())
      return;
    // GAP's PrepareBFS: the source in the queue, cleared bitmaps; DOBFS's start:
    // edges_to_check = |E|, scout_count = the source's degree
    auto G = this->get_graph();
    const vertex_t source = this->param.single_source;
    cudaMemcpyAsync(queue[0], &source, sizeof(vertex_t), cudaMemcpyHostToDevice, stream());
    cur = 0;
    queue_n = 1;
    cudaMemsetAsync(front, 0, words * sizeof(uint32_t), stream());
    cudaMemsetAsync(curr, 0, words * sizeof(uint32_t), stream());
    clear_counters();
    edge_t offs[2];
    cudaMemcpyAsync(offs, G.get_row_offsets() + source, 2 * sizeof(edge_t), cudaMemcpyDeviceToHost, stream());
    cudaStreamSynchronize(stream());
    edges_to_check = (long long)G.get_number_of_edges();
    scout = (long long)(offs[1] - offs[0]);
    awake = 0;
    pulling = false;
    level = 0;
  }

  ~problem_t() {
    if (!d_counters)
      return;
    for (auto q : queue)
      cudaFreeAsync(q, stream());
    cudaFreeAsync(front, stream());
    cudaFreeAsync(curr, stream());
    cudaFreeAsync(d_counters, stream());
    cudaStreamSynchronize(stream());
    cudaFreeHost(h_counters);
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t())
      : gunrock::enactor_t<problem_t>(_problem, _context, _properties) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;

  void prepare_frontier(frontier_t* f,
                        gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    if (!P->gap_dobfs())
      f->push_back(P->param.single_source);
  }

  bool is_converged(gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    if (P->gap_dobfs())
      return !P->pulling && P->queue_n == 0;  // GAP: while (!queue.empty())
    return gunrock::enactor_t<problem_t>::is_converged(context);
  }

  // GAP's DOBFS loop body, one top-down or bottom-up step per call
  void gap_step() {
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto dist = P->result.distances;
    auto stream = P->stream();
    const vertex_t n = G.get_number_of_vertices();
    const unsigned int blocks_n = (unsigned int)(((long long)n + gap::kBlock - 1) / gap::kBlock);
    if (!P->pulling && P->scout > P->edges_to_check / gap::kAlpha) {
      // QueueToBitmap(queue, front); awake_count = queue.size()
      const unsigned int qb = (unsigned int)((P->queue_n + gap::kBlock - 1) / gap::kBlock);
      gap::queue_to_bitmap_kernel<<<qb, gap::kBlock, 0, stream>>>(P->queue[P->cur], P->queue_n, P->front);
      P->awake = (long long)P->queue_n;
      P->pulling = true;
    }
    if (P->pulling) {
      // awake_count = BUStep(g, parent, front, curr); front.swap(curr)
      const long long old_awake = P->awake;
      cudaMemsetAsync(P->curr, 0, P->words * sizeof(uint32_t), stream);
      P->clear_counters();
      {
        gap::bu_items_t<decltype(G)> items{G, dist};
        const long long nn = n;
        if (P->lb_bu == gacge::lb::mode_t::thread && gap::bu_warp_degree() > 0) {  // thread + warp for the hubs
          gap::bu_twc_kernel<<<gacge::lb::grid(nn), gap::kBlock, 0, stream>>>(
              G, dist, P->level, n, P->front, P->curr, P->d_counters, gap::bu_warp_degree());
        } else
        gacge::lb::dispatch(P->lb_bu, [&](auto tag) {
          constexpr int Gs = decltype(tag)::value;
          gacge::lb::merge_args_t m;
          unsigned int g = gacge::lb::grid(nn);
          if constexpr (Gs == 0) {
            P->merge.prefix(items, nn, stream);
            m = P->merge.plan(P->merge.offsets, nn, stream);
            g = (unsigned int)std::max<long long>(1, P->merge.tiles);
          }
          gap::bu_kernel<Gs><<<g, gap::kBlock, 0, stream>>>(
              G, dist, P->level, items, nn, m, P->front, P->curr,
              P->d_counters);
        });
      }
      P->read_counters();
      P->awake = (long long)P->h_counters[2];
      std::swap(P->front, P->curr);
      P->level++;
      if (!(P->awake >= old_awake || P->awake > (long long)n / gap::kBeta)) {
        // BitmapToQueue(g, front, queue); scout_count = 1
        P->clear_counters();
        gap::bitmap_to_queue_kernel<<<blocks_n, gap::kBlock, 0, stream>>>(P->front, n, P->queue[P->cur],
                                                                         P->d_counters);
        P->read_counters();
        P->queue_n = P->h_counters[0];
        P->scout = 1;
        P->pulling = false;
      }
      return;
    }
    // edges_to_check -= scout_count; scout_count = TDStep(g, parent, queue)
    P->edges_to_check -= P->scout;
    P->clear_counters();
    {
      gacge::lb::frontier_tiles_t<decltype(G)> items{G, P->queue[P->cur]};
      const long long nq = (long long)P->queue_n;
      vertex_t* next = P->queue[P->cur ^ 1];
      gacge::lb::dispatch(P->lb_td, [&](auto tag) {
        constexpr int Gs = decltype(tag)::value;
        gacge::lb::merge_args_t m;
        unsigned int g = gacge::lb::grid(nq);
        if constexpr (Gs == 0) {
          P->merge.prefix(items, nq, stream);
          m = P->merge.plan(P->merge.offsets, nq, stream);
          g = (unsigned int)std::max<long long>(1, P->merge.tiles);
        }
        gap::td_kernel<Gs><<<g, gap::kBlock, 0, stream>>>(G, dist, P->level, items, nq, m, next, P->d_counters);
      });
    }
    P->read_counters();
    P->cur ^= 1;
    P->queue_n = P->h_counters[0];
    P->scout = (long long)P->h_counters[1];
    P->level++;
  }

  // GAP's trace (XBStep): a top-down step's queue and scout count, a
  // bottom-up step's awake count and the degrees of the front bitmap
  void traversal_frontier_work(std::size_t& fv, std::size_t& fe, std::size_t& max_deg) override {
    auto P = this->get_problem();
    if (!P->gap_dobfs()) {
      gunrock::enactor_t<problem_t>::traversal_frontier_work(fv, fe, max_deg);
      return;
    }
    cudaStreamSynchronize(P->stream());
    auto G = P->get_graph();
    const bool pull = P->pulling || P->scout > P->edges_to_check / gap::kAlpha;
    if (pull && P->pulling) {
      const uint32_t* bm = P->front;
      auto deg = [G, bm] __device__(vertex_t v) -> std::size_t {
        return ((bm[v >> 5] >> (v & 31)) & 1u) ? (std::size_t)G.get_number_of_neighbors(v) : 0;
      };
      auto first = thrust::make_counting_iterator<vertex_t>(0);
      auto last = first + G.get_number_of_vertices();
      fv = (std::size_t)P->awake;
      fe = thrust::transform_reduce(thrust::device, first, last, deg, std::size_t(0), thrust::plus<std::size_t>());
      max_deg = thrust::transform_reduce(thrust::device, first, last, deg, std::size_t(0),
                                         thrust::maximum<std::size_t>());
      return;
    }
    const vertex_t* q = P->queue[P->cur];
    auto deg = [G, q] __device__(std::size_t i) -> std::size_t { return (std::size_t)G.get_number_of_neighbors(q[i]); };
    auto first = thrust::make_counting_iterator<std::size_t>(0);
    auto last = first + P->queue_n;
    fv = (std::size_t)P->queue_n;
    fe = pull ? thrust::transform_reduce(thrust::device, first, last, deg, std::size_t(0), thrust::plus<std::size_t>())
              : (std::size_t)P->scout;
    max_deg = thrust::transform_reduce(thrust::device, first, last, deg, std::size_t(0),
                                       thrust::maximum<std::size_t>());
  }

  // the input frontier's valid vertices and the sum of their degrees
  void frontier_size(std::size_t& n_f, std::size_t& m_f) {
    auto G = this->get_problem()->get_graph();
    auto policy = this->context->get_context(0)->execution_policy();
    auto* in = this->get_input_frontier();
    const std::size_t len = in->get_number_of_elements();
    const vertex_t* f = in->data();
    n_f = thrust::count_if(policy, f, f + len, [] __host__ __device__(vertex_t v) {
      return gunrock::util::limits::is_valid(v);
    });
    m_f = thrust::transform_reduce(
        policy, f, f + len,
        [G] __host__ __device__(vertex_t v) -> std::size_t {
          return gunrock::util::limits::is_valid(v) ? (std::size_t)G.get_number_of_neighbors(v) : 0;
        },
        std::size_t{0}, thrust::plus<std::size_t>());
  }

  // Direction-optimized BFS by Ligra's rule (GACGE_BFS_RULE=ligra): decides
  // the step's direction and, for a pull step, runs it and makes the next
  // frontier; false: push. (GAP's rule: gap_step.)
  bool pull_step(gcuda::multi_context_t& context) {
    auto G = this->get_problem()->get_graph();
    const std::size_t m = G.get_number_of_edges();
    std::size_t n_f = 0, m_f = 0;
    frontier_size(n_f, m_f);
    return n_f + m_f > m / 20 ? do_pull(context) >= 0 : false;
  }

  // one pull step; returns the vertices it woke (the next frontier's size)
  long long do_pull(gcuda::multi_context_t& context) {
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto distances = P->result.distances;
    auto policy = this->context->get_context(0)->execution_policy();
    const vertex_t level = this->iteration;
    const vertex_t unvisited = std::numeric_limits<vertex_t>::max();
    auto pull = [G, distances, level, unvisited] __host__ __device__(vertex_t const& v) {
      if (distances[v] != unvisited)
        return;
      auto e = G.get_starting_edge(v), ee = G.get_starting_edge(v + 1);
      for (; e < ee; ++e)
        if (distances[G.get_destination_vertex(e)] == level) {
          distances[v] = level + 1;  // only v's thread writes v: no atomic
          break;
        }
    };
    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(G, pull, context);
    // the next frontier: the vertices this step reached
    const vertex_t n = G.get_number_of_vertices();
    auto* out = this->get_output_frontier();
    out->reserve(n);
    vertex_t* o = out->data();
    auto end = thrust::copy_if(policy, thrust::counting_iterator<vertex_t>(0),
                               thrust::counting_iterator<vertex_t>(n), o,
                               [distances, level] __host__ __device__(vertex_t v) {
                                 return distances[v] == level + 1;
                               });
    out->set_number_of_elements(end - o);
    this->swap_frontier_buffers();
    return (long long)(end - o);
  }

  void loop(gcuda::multi_context_t& context) override {
    if (this->get_problem()->gap_dobfs()) {
      gap_step();
      return;
    }
    if (this->get_problem()->param.direction_optimized && pull_step(context))
      return;
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto single_source = P->param.single_source;
    auto distances = P->result.distances;
    auto visited = P->visited.data().get();

    auto iteration = this->iteration;

    auto search = [distances, single_source, iteration] __host__ __device__(
                      vertex_t const& source,    // ... source
                      vertex_t const& neighbor,  // neighbor
                      edge_t const& edge,        // edge
                      weight_t const& weight     // weight (tuple).
                      ) -> bool {
      // If the neighbor is not visited, update the distance. Returning false
      // here means that the neighbor is not added to the output frontier, and
      // instead an invalid vertex is added in its place. These invalides (-1 in
      // most cases) can be removed using a filter operator or uniquify.

      // if (distances[neighbor] != std::numeric_limits<vertex_t>::max())
      //   return false;
      // else
      //   return (math::atomic::cas(
      //               &distances[neighbor],
      //               std::numeric_limits<vertex_t>::max(), iteration + 1) ==
      //               std::numeric_limits<vertex_t>::max());

      // Simpler logic for the above.
      auto old_distance =
          math::atomic::min(&distances[neighbor], iteration + 1);
      return (iteration + 1 < old_distance);
    };

    auto remove_invalids =
        [] __host__ __device__(vertex_t const& vertex) -> bool {
      // Returning true here means that we keep all the valid vertices.
      // Internally, filter will automatically remove invalids and will never
      // pass them to this lambda function.
      return true;
    };

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::load_balance_t::block_mapped>(
        G, E, search, context);

    // Execute filter operator to remove the invalids.
    // @todo: Add CLI option to enable or disable this.
    // operators::filter::execute<operators::filter_algorithm_t::compact>(
    // G, E, remove_invalids, context);
  }

};  // struct enactor_t

/**
 * @brief Run Breadth-First Search algorithm on a given graph, G, starting from
 * the source node, single_source. The resulting distances are stored in the
 * distances pointer. All data must be allocated by the user, on the device
 * (GPU) and passed in to this function.
 *
 * @tparam graph_t Graph type.
 * @param G Graph object.
 * @param single_source A vertex in the graph (integral type).
 * @param distances Pointer to the distances array of size number of vertices.
 * @param predecessors Pointer to the predecessors array of size number of
 * vertices. (optional, wip)
 * @param context Device context.
 * @return float Time taken to run the algorithm.
 */
template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::vertex_type* distances,      // Output
          typename graph_t::vertex_type* predecessors,   // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0)),  // Context
          bool direction_optimized = false  // push-pull by Ligra's 5% rule
) {
  using vertex_t = typename graph_t::vertex_type;
  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t>;

  param_type param(single_source, direction_optimized);
  result_type result(distances, predecessors);

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  // GAP's DOBFS keeps its own queue and bitmaps: no |E|-sized frontiers
  enactor_properties_t props;
  props.self_manage_frontiers = problem.gap_dobfs();
  enactor_type enactor(&problem, context, props);
  return enactor.enact();
}

}  // namespace bfs
}  // namespace gunrock