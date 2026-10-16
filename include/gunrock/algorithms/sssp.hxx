/**
 * @file sssp.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Single-Source Shortest Path algorithm.
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/gacge_lb.hxx>
#include <gunrock/gacge_task_stats.hxx>

#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace gunrock {
namespace sssp {

// GACGE_SSSP: delta (default since 2026-10-15) is delta-stepping as the GAP
// benchmark's DeltaStep (src/gap/gapbs/src/sssp.cc), step for step:
//  - a step relaxes the out-edges of the current bin's frontier, skipping an
//    entry whose distance dropped below the bin (GAP's stale check), with a
//    load and a compare-and-swap only when the distance improves;
//  - every successful relaxation appends the vertex to the bin of its new
//    distance (no deduplication, as GAP);
//  - bin fusion (GACGE_SSSP_FUSE=0 off): a block, GAP's thread, relaxes its own
//    current-bin vertices at once while fewer than 1000 (GAP's
//    kBinSizeThreshold);
//  - the next frontier is the current bin again if it is non-empty, else the
//    lowest non-empty bin, taken whole (no copy).
// A relaxation from bin b lands at most max weight / delta + 1 bins ahead, so
// the bins are a ring of max weight / delta + 3 buffers; one kernel and one
// read of the bin sizes per step. The step's edges are spread by the load
// balancing GACGE_LB (gacge_lb.hxx: thread, block or merge; the only GPU-specific
// choice of every GACGE primitive). bf: Gunrock's frontier Bellman-Ford (every
// improved vertex is in the next frontier).
inline bool sssp_delta() {
  static const bool v = [] {
    const char* e = std::getenv("GACGE_SSSP");
    return e == nullptr || std::string(e) == "delta";
  }();
  return v;
}
inline double sssp_delta_width() {
  static const double v = [] {
    const char* e = std::getenv("GACGE_SSSP_DELTA");
    return e ? std::atof(e) : 1.0;
  }();
  return v;
}
inline bool sssp_fuse() {
  static const bool v = [] {
    const char* e = std::getenv("GACGE_SSSP_FUSE");
    return e == nullptr || std::atoi(e) != 0;
  }();
  return v;
}

namespace delta {

constexpr int kBlock = gacge::lb::kBlock;  // loops' launch box
constexpr int kBinSizeThreshold = 1000;  // GAP's
constexpr int kQueue = 1024;             // a block's current bin, >= the threshold

// The bins on the device: slots 0..R-1 hold bin b in slot b % R, slot R the
// current bin's next frontier. desc = [buffer pointers | capacities | tails]
// (R + 1 each) and an overflow flag: an append past a capacity marks the
// vertex instead, and the host grows the slot and re-adds the marked vertices.
template <typename vertex_t, typename weight_t>
struct step_t {
  weight_t* dist;
  weight_t delta;
  long long bin;
  int R;
  unsigned long long* desc;
  uint8_t* marked;
  unsigned long long* work;  // relaxed edges, GACGE_TRAVERSAL_STATS only
  bool fuse;
  gacge::task::view_t task{};    // GACGE_TASK_STATS: per bin vertex
  gacge::task::view_t thread{};  // GACGE_TASK_STATS: per thread and step

  __device__ vertex_t* buffer(int s) const {
    return reinterpret_cast<vertex_t*>(desc[s]);
  }
  __device__ unsigned long long capacity(int s) const { return desc[R + 1 + s]; }
  __device__ unsigned long long* tail(int s) const { return desc + 2 * (R + 1) + s; }
  __device__ unsigned long long* overflow() const { return desc + 3 * (R + 1); }
  __device__ int slot(long long b) const { return b == bin ? R : (int)(b % R); }
};

// one atomic per slot and coalesced group (GAP: local_bins[b].push_back(v))
template <typename vertex_t, typename weight_t>
__device__ void append(const step_t<vertex_t, weight_t>& p, int s, vertex_t v) {
  namespace cg = cooperative_groups;
  auto g = cg::labeled_partition(cg::coalesced_threads(), s);
  unsigned long long base = 0;
  if (g.thread_rank() == 0)
    base = atomicAdd(p.tail(s), (unsigned long long)g.size());
  base = g.shfl(base, 0);
  const unsigned long long pos = base + g.thread_rank();
  if (pos < p.capacity(s)) {
    p.buffer(s)[pos] = v;
  } else {
    p.marked[v] = 1;
    *p.overflow() = 1;
  }
}

// GAP's RelaxEdges: load, compare-and-swap only while the new distance is lower
template <typename weight_t, typename vertex_t>
__device__ bool relax(weight_t* dist, vertex_t v, weight_t nd) {
  weight_t old = dist[v];
  while (nd < old) {
    const int prev = atomicCAS(reinterpret_cast<int*>(&dist[v]), __float_as_int(old), __float_as_int(nd));
    if (prev == __float_as_int(old))
      return true;
    old = __int_as_float(prev);
  }
  return false;
}

template <typename vertex_t>
struct fusion_shared_t {
  vertex_t u[kBlock];
  vertex_t queue[kQueue];             // the block's current-bin vertices
  vertex_t fused[kBinSizeThreshold];  // the ones a fusion round relaxes
  int queued;                         // may exceed kQueue: the rest went to slot R
};

// GAP's RelaxEdges for one edge: a new current-bin vertex goes to the
// block's queue while fusing, any other to its bin
template <typename graph_t, typename vertex_t, typename weight_t>
__device__ __forceinline__ void relax_edge(const graph_t& G, const step_t<vertex_t, weight_t>& p,
                                           fusion_shared_t<vertex_t>& sh, vertex_t u, long long e) {
  const vertex_t v = G.get_destination_vertex(e);
  const weight_t nd = p.dist[u] + G.get_edge_weight(e);
  if (relax(p.dist, v, nd)) {
    gacge::task::count(p.task, 0, 1);  // a successful relaxation
    const long long b = (long long)(nd / p.delta);
    if (b == p.bin && p.fuse) {
      const int q = atomicAdd(&sh.queued, 1);
      if (q < kQueue) {
        sh.queue[q] = v;
        return;
      }
    }
    append(p, p.slot(b), v);
  }
}

// the step's items: frontier entries, a stale one (its distance dropped below
// the bin, GAP's check) with no edges
template <typename graph_t, typename weight_t>
struct step_items_t {
  using vertex_type = typename graph_t::vertex_type;
  graph_t G;
  const vertex_type* frontier;
  const weight_t* dist;
  weight_t floor_d;
  __device__ void item(long long i, vertex_type& u, long long& start) const {
    u = frontier[i];
    start = G.get_starting_edge(u);
  }
  __device__ long long len(long long i) const {
    const vertex_type u = frontier[i];
    return dist[u] >= floor_d ? (long long)G.get_number_of_neighbors(u) : 0;
  }
};

template <typename value_t>
__device__ __forceinline__ void block_add(value_t x, unsigned long long* to) {
  using reduce_t = cub::BlockReduce<unsigned long long, kBlock>;
  __shared__ typename reduce_t::TempStorage temp;
  const unsigned long long sum = reduce_t(temp).Sum((unsigned long long)x);
  if (threadIdx.x == 0 && sum)
    atomicAdd(to, sum);
}

// One step under the schedule Gs (gacge_lb.hxx), then GAP's bin fusion: a
// block (GAP's thread) relaxes its own current-bin vertices at once while
// fewer than 1000, a round's vertices spread as the step's (thread-mapped: one
// per thread; group-mapped: by its groups; merge-path: by the block).
template <int Gs, typename graph_t, typename vertex_t, typename weight_t>
__global__ void __launch_bounds__(kBlock)
    step_kernel(graph_t G, step_t<vertex_t, weight_t> p, step_items_t<graph_t, weight_t> items, long long n,
                gacge::lb::merge_args_t m) {
  __shared__ fusion_shared_t<vertex_t> sh;
  const int t = threadIdx.x;
  if (t == 0)
    sh.queued = 0;
  __syncthreads();
  unsigned long long work = 0;
  auto f = [&](long long, vertex_t u, long long e) {
    if (p.task.cnt && e == (long long)G.get_starting_edge(u))  // task: a bin vertex, its edges
      gacge::task::record(p.task, (unsigned long long)G.get_number_of_neighbors(u));
    work++;
    relax_edge(G, p, sh, u, e);
    return false;
  };
  gacge::lb::for_edges<Gs>(items, n, m, f);
  __syncthreads();
  // GAP's bin fusion: while the local current bin is non-empty and below the threshold
  while (p.fuse) {
    const int m = sh.queued;
    if (m == 0)
      break;
    if (m >= kBinSizeThreshold) {  // left to the next step, as GAP's shared frontier
      for (int i = t; i < min(m, kQueue); i += kBlock)
        append(p, p.R, sh.queue[i]);
      break;
    }
    for (int i = t; i < m; i += kBlock)
      sh.fused[i] = sh.queue[i];
    __syncthreads();
    if (t == 0)
      sh.queued = 0;
    __syncthreads();
    for (int c = 0; c < m; c += kBlock) {
      const int kk = min(kBlock, m - c);
      step_items_t<graph_t, weight_t> round{G, sh.fused + c, p.dist, items.floor_d};
      if constexpr (Gs == 1) {
        if (t < kk) {
          vertex_t u;
          long long start;
          round.item(t, u, start);
          const long long last = start + round.len(t);
          for (long long e = start; e < last; ++e)
            f(t, u, e);
        }
      } else {
        // the round's groups share its edges (merge-path: the whole block, one group)
        __syncthreads();
        gacge::lb::group_tiles<Gs == 0 ? gacge::lb::kBlock : Gs>(round, 0, kk, f);
      }
      __syncthreads();
    }
  }
  if (work)
    gacge::task::record(p.thread, work);  // this thread's edges in the step
  if (p.work)
    block_add(work, p.work);
}

// after growing the overflowed slots: the marked vertices join the bin of their distance
template <typename vertex_t, typename weight_t>
__global__ void readd_kernel(step_t<vertex_t, weight_t> p, vertex_t n) {
  for (vertex_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x) {
    if (p.marked[v]) {
      p.marked[v] = 0;
      append(p, p.slot((long long)(p.dist[v] / p.delta)), v);
    }
  }
}

}  // namespace delta

template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  param_t(vertex_t _single_source) : single_source(_single_source) {}
};

template <typename vertex_t, typename weight_t>
struct result_t {
  weight_t* distances;
  vertex_t* predecessors;
  result_t(weight_t* _distances, vertex_t* _predecessors, vertex_t n_vertices)
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
  using buffer_t = std::pair<vertex_t*, unsigned long long>;  // pointer, capacity
  gacge::task::stats_t task;    // GACGE_TASK_STATS: per bin vertex
  gacge::task::stats_t thread;  // GACGE_TASK_STATS: per thread and step
  long long thread_step = 0;

  thrust::device_vector<vertex_t> visited;

  // sssp_delta: the bins (delta::step_t), allocated by init(), outside the timing
  weight_t delta_w = 1;
  int R = 0;
  std::vector<buffer_t> slots;  // R + 1
  buffer_t front{nullptr, 0};
  unsigned long long front_n = 0;
  long long bin = 0;
  bool done = false;
  unsigned long long* d_desc = nullptr;
  unsigned long long* h_desc = nullptr;  // pinned
  uint8_t* marked = nullptr;
  unsigned long long* d_work = nullptr;
  gacge::lb::mode_t lb = gacge::lb::mode(gacge::lb::mode_t::block);  // GAP's DeltaStep: many steps of small frontiers
  gacge::lb::merge_t merge;  // merge-path: the step's prefix sum of edges and its merge tiles

  cudaStream_t stream() { return this->context->get_context(0)->stream(); }
  std::size_t desc_size() const { return 3 * (std::size_t)(R + 1) + 1; }
  unsigned long long& h_tail(int s) { return h_desc[2 * (R + 1) + s]; }
  unsigned long long& h_overflow() { return h_desc[3 * (R + 1)]; }

  vertex_t* allocate(unsigned long long n) {
    void* p = nullptr;
    cudaMallocAsync(&p, std::max<unsigned long long>(n, 1) * sizeof(vertex_t), stream());
    return static_cast<vertex_t*>(p);
  }

  delta::step_t<vertex_t, weight_t> step() {
    return {this->result.distances, delta_w, bin, R, d_desc, marked, d_work, sssp_fuse(), task.view(), thread.view()};
  }

  // the buffers' pointers and capacities to the descriptor, tails kept
  void put_buffers() {
    for (int s = 0; s <= R; s++) {
      h_desc[s] = reinterpret_cast<unsigned long long>(slots[s].first);
      h_desc[R + 1 + s] = slots[s].second;
    }
  }
  void upload() {
    cudaMemcpyAsync(d_desc, h_desc, desc_size() * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream());
  }
  void download_tails() {
    cudaMemcpyAsync(&h_tail(0), d_desc + 2 * (R + 1), (R + 2) * sizeof(unsigned long long),
                    cudaMemcpyDeviceToHost, stream());
    cudaStreamSynchronize(stream());
  }

  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    visited.resize(n_vertices);

    // Execution policy for a given context (using single-gpu).
    auto policy = this->context->get_context(0)->execution_policy();
    thrust::fill(policy, visited.begin(), visited.end(), -1);

    if (!sssp_delta())
      return;
    delta_w = (weight_t)sssp_delta_width();
    auto weight = [g] __host__ __device__(edge_t e) -> weight_t { return g.get_edge_weight(e); };
    const weight_t max_w = thrust::transform_reduce(
        policy, thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(g.get_number_of_edges()), weight, (weight_t)0,
        thrust::maximum<weight_t>());
    R = (int)std::floor(max_w / delta_w) + 3;
    // GAP sizes its frontier at |E| up front: every buffer holds n (an entry
    // per vertex) within 60% of the free memory, and grows if exceeded (the
    // buffers rotate between the bins, so a small one would overflow often)
    const unsigned long long n = n_vertices;
    std::size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const unsigned long long cap = std::max<unsigned long long>(
        4096, std::min<unsigned long long>(n, (unsigned long long)(0.6 * free_b / ((R + 2) * sizeof(vertex_t)))));
    for (int s = 0; s <= R; s++)
      slots.push_back({allocate(cap), cap});
    front = {allocate(cap), cap};
    cudaMallocAsync(&d_desc, desc_size() * sizeof(unsigned long long), stream());
    cudaMallocHost(&h_desc, desc_size() * sizeof(unsigned long long));
    cudaMallocAsync(&marked, n, stream());
    cudaMemsetAsync(marked, 0, n, stream());
    if (std::getenv("GACGE_TRAVERSAL_STATS"))
      cudaMallocAsync(&d_work, sizeof(unsigned long long), stream());
    if (lb == gacge::lb::mode_t::merge) {
      merge.reserve(cap);
      merge.reserve_work((long long)cap + g.get_number_of_edges());
    }
  }

  void reset() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();

    auto context = this->get_single_context();
    auto policy = context->execution_policy();

    auto single_source = this->param.single_source;
    auto d_distances = thrust::device_pointer_cast(this->result.distances);
    thrust::fill(policy, d_distances + 0, d_distances + n_vertices,
                 std::numeric_limits<weight_t>::max());

    thrust::fill(policy, d_distances + single_source,
                 d_distances + single_source + 1, 0);

    thrust::fill(policy, visited.begin(), visited.end(),
                 -1);  // This does need to be reset in between runs though

    if (!sssp_delta())
      return;
    // GAP: frontier[0] = source, bin 0
    cudaMemcpyAsync(front.first, &single_source, sizeof(vertex_t), cudaMemcpyHostToDevice, stream());
    front_n = 1;
    bin = 0;
    done = false;
    put_buffers();
    for (int s = 0; s <= R; s++)
      h_tail(s) = 0;
    h_overflow() = 0;
    upload();
    if (d_work)
      cudaMemsetAsync(d_work, 0, sizeof(unsigned long long), stream());
    cudaStreamSynchronize(stream());
  }

  ~problem_t() {
    if (!d_desc)
      return;
    for (auto& s : slots)
      cudaFreeAsync(s.first, stream());
    cudaFreeAsync(front.first, stream());
    cudaFreeAsync(d_desc, stream());
    cudaFreeAsync(marked, stream());
    if (d_work)
      cudaFreeAsync(d_work, stream());
    cudaStreamSynchronize(stream());
    cudaFreeHost(h_desc);
  }

  // A step's appends overflowed: grow those slots (keeping their entries) and
  // re-add the marked vertices, until none overflows.
  void recover_overflow() {
    const vertex_t n = this->get_graph().get_number_of_vertices();
    while (h_overflow()) {
      for (int s = 0; s <= R; s++) {
        if (h_tail(s) <= slots[s].second)
          continue;
        const unsigned long long cap = std::max(2 * slots[s].second, 2 * h_tail(s));
        vertex_t* p = allocate(cap);
        cudaMemcpyAsync(p, slots[s].first, slots[s].second * sizeof(vertex_t), cudaMemcpyDeviceToDevice, stream());
        cudaFreeAsync(slots[s].first, stream());
        h_tail(s) = slots[s].second;
        slots[s] = {p, cap};
      }
      put_buffers();
      h_overflow() = 0;
      upload();
      const int blocks = (int)std::min<long long>(((long long)n + 255) / 256, 4096);
      delta::readd_kernel<<<blocks, 256, 0, stream()>>>(step(), n);
      download_tails();
    }
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
    if (!sssp_delta())
      f->push_back(P->param.single_source);
  }

  bool is_converged(gcuda::multi_context_t& context) override {
    if (sssp_delta())
      return this->get_problem()->done;
    return gunrock::enactor_t<problem_t>::is_converged(context);
  }

  // sssp_delta: the frontier's entries a step relaxes (stale ones skipped), as GAP's trace
  void traversal_frontier_work(std::size_t& fv, std::size_t& fe, std::size_t& max_deg) override {
    if (!sssp_delta()) {
      gunrock::enactor_t<problem_t>::traversal_frontier_work(fv, fe, max_deg);
      return;
    }
    auto P = this->get_problem();
    cudaStreamSynchronize(P->stream());
    auto G = P->get_graph();
    auto dist = P->result.distances;
    const weight_t floor_d = P->delta_w * (weight_t)P->bin;
    const vertex_t* f = P->front.first;
    auto first = thrust::make_counting_iterator<std::size_t>(0);
    auto last = first + P->front_n;
    auto deg = [G, dist, f, floor_d] __device__(std::size_t i) -> std::size_t {
      const vertex_t u = f[i];
      return dist[u] >= floor_d ? (std::size_t)G.get_number_of_neighbors(u) : 0;
    };
    auto live = [dist, f, floor_d] __device__(std::size_t i) -> std::size_t {
      return dist[f[i]] >= floor_d ? 1 : 0;
    };
    fe = thrust::transform_reduce(thrust::device, first, last, deg, std::size_t(0), thrust::plus<std::size_t>());
    max_deg = thrust::transform_reduce(thrust::device, first, last, deg, std::size_t(0),
                                       thrust::maximum<std::size_t>());
    fv = thrust::transform_reduce(thrust::device, first, last, live, std::size_t(0), thrust::plus<std::size_t>());
  }

  long long traversal_work_e() override {
    auto P = this->get_problem();
    if (!sssp_delta() || !P->d_work)
      return -1;
    unsigned long long w = 0;
    cudaMemcpy(&w, P->d_work, sizeof(w), cudaMemcpyDeviceToHost);
    return (long long)w;
  }

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    if (sssp_delta()) {
      const int R = P->R;
      const long long n = (long long)P->front_n;
      delta::step_items_t<decltype(G), weight_t> items{G, P->front.first, P->result.distances,
                                                       P->delta_w * (weight_t)P->bin};
      gacge::lb::dispatch(P->lb, [&](auto tag) {
        constexpr int Gs = decltype(tag)::value;
        if constexpr (Gs == 0) {
          P->merge.reserve(n);
          P->merge.prefix(items, n, P->stream());
          const gacge::lb::merge_args_t m = P->merge.plan(P->merge.offsets, n, P->stream());
          if (P->merge.tiles > 0)
            delta::step_kernel<0><<<(unsigned int)P->merge.tiles, delta::kBlock, 0, P->stream()>>>(G, P->step(), items,
                                                                                                 n, m);
        } else {
          delta::step_kernel<Gs><<<gacge::lb::grid(n), delta::kBlock, 0, P->stream()>>>(G, P->step(), items, n,
                                                                                        gacge::lb::merge_args_t{});
        }
      });
      P->download_tails();
      if (gacge::task::enabled())
        P->thread.print("SSSP", "thread-step", "step=" + std::to_string(P->thread_step++));
      P->recover_overflow();
      // the next frontier: the current bin again, else the lowest non-empty bin
      int s = -1;
      if (P->h_tail(R) > 0) {
        s = R;
      } else {
        for (long long b = P->bin + 1; b < P->bin + R; b++) {
          if (P->h_tail((int)(b % R)) > 0) {
            s = (int)(b % R);
            P->bin = b;
            break;
          }
        }
      }
      if (s < 0) {
        P->done = true;
        return;
      }
      std::swap(P->front, P->slots[s]);
      P->front_n = P->h_tail(s);
      P->h_tail(s) = 0;
      P->put_buffers();
      P->upload();
      return;
    }

    auto single_source = P->param.single_source;
    auto distances = P->result.distances;
    auto visited = P->visited.data().get();

    auto iteration = this->iteration;

    auto shortest_path = [distances, single_source] __host__ __device__(
                             vertex_t const& source,    // ... source
                             vertex_t const& neighbor,  // neighbor
                             edge_t const& edge,        // edge
                             weight_t const& weight     // weight (tuple).
                             ) -> bool {
      weight_t source_distance = thread::load(&distances[source]);
      weight_t distance_to_neighbor = source_distance + weight;

      // Check if the destination node has been claimed as someone's child
      weight_t recover_distance =
          math::atomic::min(&(distances[neighbor]), distance_to_neighbor);

      return (distance_to_neighbor < recover_distance);
    };

    auto remove_completed_paths = [G, visited, iteration] __host__ __device__(
                                      vertex_t const& vertex) -> bool {
      if (visited[vertex] == iteration)
        return false;

      visited[vertex] = iteration;
      /// @todo Confirm we do not need the following for bug
      /// https://github.com/gunrock/essentials/issues/9 anymore.
      // return G.get_number_of_neighbors(vertex) > 0;
      return true;
    };

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::load_balance_t::block_mapped>(
        G, E, shortest_path, context);

    // Execute filter operator on the provided lambda
    operators::filter::execute<operators::filter_algorithm_t::bypass>(
        G, E, remove_completed_paths, context);

    /// @brief Execute uniquify operator to deduplicate the frontier
    /// @note Not required.
    // // bool best_effort_uniquification = true;
    // // operators::uniquify::execute<operators::uniquify_algorithm_t::unique>(
    // // E, context, best_effort_uniquification);
  }

};  // struct enactor_t

template <typename weight_t>
struct sssp_reached_t {
  __host__ __device__ bool operator()(weight_t x) const { return x < std::numeric_limits<weight_t>::max(); }
};

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::weight_type* distances,      // Output
          typename graph_t::vertex_type* predecessors,   // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t, weight_t>;

  param_type param(single_source);
  result_type result(distances, predecessors, G.get_number_of_vertices());
  // </user-defined>

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  // sssp_delta: the bins are the problem's (delta::step_t), not the enactor's
  // frontiers (|E| x 1.5 per buffer up front: 13 GB on Arabic beside its 9 GB
  // weighted graph)
  enactor_properties_t props;
  props.self_manage_frontiers = sssp_delta();
  enactor_type enactor(&problem, context, props);
  const float t = enactor.enact();
  if (gacge::task::enabled()) {  // the reached vertices: a vertex taken more than once was reinserted
    auto d = thrust::device_pointer_cast(distances);
    const long long reached = thrust::count_if(thrust::device, d, d + G.get_number_of_vertices(),
                                               sssp_reached_t<typename graph_t::weight_type>{});
    problem.task.print("SSSP", "bin-vertex", "reached=" + std::to_string(reached));
  }
  return t;
  // </boiler-plate>
}

}  // namespace sssp
}  // namespace gunrock
