/**
 * @file tc.hxx
 * @author Muhammad A. Awad (mawad@ucdavis.edu)
 * @brief Triangle Counting algorithm.
 * @date 2022-08-06
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/util/timer.hxx>
#include <gunrock/gacge_lb.hxx>
#include <gunrock/gacge_task_stats.hxx>

#include <thrust/binary_search.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include <cstdlib>
#include <string>

namespace gunrock {
namespace tc {

// GACGE: how the oriented count is spread over the GPU. By default one thread
// per edge (parallel_for over the edges, the source found by a binary search
// in the row offsets): a high-degree vertex's merges are spread over the whole
// GPU, as Ligra Triangle's dense_parallel spreads them over the workers.
// Gunrock's block_mapped advance ran all of a vertex's edges in one 256-thread
// block (Arabic's 575K-degree vertices: over 20 min a round, one SM busy);
// GACGE_TC_LB=block restores it. (merge_path, the edge-balanced advance, faults
// on this graph-as-frontier advance.)
inline bool tc_block_mapped() {
  static const bool block = [] {
    const char* e = std::getenv("GACGE_TC_LB");
    return e != nullptr && std::string(e) == "block";
  }();
  return block;
}

// Default (since 2026-10-16): GAP's OrderedCount with the edge as the tile,
// thread-mapped: one thread per edge (its source by a binary search in the
// row offsets), merging N(s) and N(d) below d for d < s, the others idle (GAP
// breaks there). Each edge's merge is an independent task of its own cost, so
// the edge, not the vertex, is TC's work unit. GACGE_TC=gap: the vertex as the
// tile under GACGE_LB (tc_count_kernel), the counts summed per block.
inline bool tc_gap() {
  static const bool v = [] {
    const char* e = std::getenv("GACGE_TC");
    return e != nullptr && std::string(e) == "gap";
  }();
  return v;
}

// GACGE_TC=split: one thread per edge as GACGE_TC=edge, but an edge whose end
// points' degrees sum to more than GACGE_TC_HEAVY (default 1024) is queued and
// intersected by a whole warp (the lanes take the shorter list's entries below
// d and binary-search them in the longer one), so a hub's long intersections
// do not stall the 31 other lanes of a warp of short ones.
inline bool tc_split() {
  static const bool v = [] {
    const char* e = std::getenv("GACGE_TC");
    return e != nullptr && std::string(e) == "split";
  }();
  return v;
}
inline int tc_heavy() {
  static const int v = [] {
    const char* e = std::getenv("GACGE_TC_HEAVY");
    return e ? std::atoi(e) : 1024;
  }();
  return v;
}

// |N(s) and N(d) below d| by a merge (Ligra's countCommon)
template <typename graph_t, typename vertex_t>
__device__ __forceinline__ vertex_t tc_merge(const graph_t& G, vertex_t s, vertex_t d) {
  using edge_t = typename graph_t::edge_type;
  edge_t i = G.get_starting_edge(s), ie = G.get_starting_edge(s + 1);
  edge_t j = G.get_starting_edge(d), je = G.get_starting_edge(d + 1);
  vertex_t count = 0;
  while (i < ie && j < je) {
    const vertex_t a = G.get_destination_vertex(i), b = G.get_destination_vertex(j);
    if (a >= d || b >= d) break;
    if (a < b) i++;
    else if (b < a) j++;
    else { count++; i++; j++; }
  }
  return count;
}

// the edges (u, v), v < u, of u: N(u)'s entries below u
template <typename graph_t, typename vertex_t>
__device__ __forceinline__ long long tc_lower(const graph_t& G, vertex_t u) {
  const auto col = G.get_column_indices();
  const auto* a = col + G.get_starting_edge(u);
  return thrust::lower_bound(thrust::seq, a, a + G.get_number_of_neighbors(u), u) - a;
}

constexpr int kTcBlock = gacge::lb::kBlock;

// every vertex u with its edges below u (N(u)'s first tc_lower entries)
template <typename graph_t>
struct lower_tiles_t {
  using vertex_type = typename graph_t::vertex_type;
  graph_t G;
  __device__ void item(long long i, vertex_type& u, long long& start) const {
    u = (vertex_type)i;
    start = G.get_starting_edge(u);
  }
  __device__ long long len(long long i) const { return tc_lower(G, (vertex_type)i); }
};

// GAP's OrderedCount under the schedule Gs (gacge_lb.hxx): each edge (u, v),
// v < u, merges N(u) and N(v) below v; the counts summed per block into the
// total (counters[0], GAP's reduction)
template <int Gs, typename graph_t>
__global__ void __launch_bounds__(kTcBlock)
    tc_count_kernel(graph_t G, lower_tiles_t<graph_t> tiles, long long n, gacge::lb::merge_args_t m,
                    unsigned long long* counters) {
  using vertex_t = typename graph_t::vertex_type;
  using reduce_t = cub::BlockReduce<unsigned long long, kTcBlock>;
  __shared__ typename reduce_t::TempStorage temp;
  unsigned long long count = 0;
  gacge::lb::for_edges<Gs>(tiles, n, m, [&](long long, vertex_t u, long long e) {
    count += tc_merge(G, u, (vertex_t)G.get_destination_vertex(e));
    return false;
  });
  const unsigned long long sum = reduce_t(temp).Sum(count);
  if (threadIdx.x == 0 && sum)
    atomicAdd(counters + 0, sum);
}

template <typename graph_t, typename vertex_t>
__global__ void tc_heavy_kernel(graph_t G, vertex_t* counts, const typename graph_t::edge_type* heavy,
                                const unsigned long long* heavy_n) {
  using edge_t = typename graph_t::edge_type;
  const unsigned long long warps = (static_cast<unsigned long long>(gridDim.x) * blockDim.x) >> 5;
  const unsigned long long nheavy = *heavy_n;
  const int lane = threadIdx.x & 31;
  const auto col = G.get_column_indices();
  for (unsigned long long k = (static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
       k < nheavy; k += warps) {
    const edge_t e = heavy[k];
    const vertex_t s = G.get_source_vertex(e), d = G.get_destination_vertex(e);
    // both lists below d
    const vertex_t* a = col + G.get_starting_edge(s);
    const vertex_t* b = col + G.get_starting_edge(d);
    const edge_t la = thrust::lower_bound(thrust::seq, a, a + G.get_number_of_neighbors(s), d) - a;
    const edge_t lb = thrust::lower_bound(thrust::seq, b, b + G.get_number_of_neighbors(d), d) - b;
    const vertex_t* shorter = la <= lb ? a : b;
    const vertex_t* longer = la <= lb ? b : a;
    const edge_t ls = la <= lb ? la : lb, ll = la <= lb ? lb : la;
    vertex_t c = 0;
    for (edge_t x = lane; x < ls; x += 32)
      c += thrust::binary_search(thrust::seq, longer, longer + ll, shorter[x]) ? 1 : 0;
    for (int o = 16; o > 0; o >>= 1)
      c += __shfl_down_sync(0xffffffffu, c, o);
    if (lane == 0 && c)
      math::atomic::add(&counts[s], c);
  }
}

template <typename vertex_t>
struct param_t {
  bool reduce_all_triangles;
  // oriented (GACGE, as Ligra's Triangle): every edge (s, d) with d < s merges
  // the sorted neighbor lists of s and d up to d and adds the count to s once,
  // so each triangle w < d < s is found once, at its edge (s, d), without an
  // atomic per triangle. Otherwise (Gunrock's own): every edge (s, d) with
  // d > s intersects the whole lists and adds 1 to each common neighbor's
  // count, so each triangle is found at all three of its edges.
  bool oriented;
  param_t(bool _reduce_all_triangles, bool _oriented = true)
      : reduce_all_triangles(_reduce_all_triangles), oriented(_oriented) {}
};

template <typename vertex_t>
struct result_t {
  vertex_t* vertex_triangles_count;
  std::size_t* total_triangles_count;
  result_t(vertex_t* _vertex_triangles_count, uint64_t* _total_triangles_count)
      : vertex_triangles_count(_vertex_triangles_count),
        total_triangles_count(_total_triangles_count) {}
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

  // GACGE_TC=split: the queue of heavy edges (one slot per edge) and its length
  thrust::device_vector<edge_t> heavy;
  thrust::device_vector<unsigned long long> heavy_n;
  gacge::task::stats_t task;  // GACGE_TASK_STATS
  // tc_gap: the total, the schedule, and merge-path's prefix sum of the edges below each vertex
  thrust::device_vector<unsigned long long> gap_counters;
  gacge::lb::mode_t lb = gacge::lb::mode(gacge::lb::mode_t::merge);
  gacge::lb::merge_t merge;  // merge-path: the prefix sum of the edges below each vertex and its merge tiles (timed)

  void init() override {
    if (tc_gap() && param.oriented) {
      gap_counters.resize(1);
      if (lb == gacge::lb::mode_t::merge) {
        merge.reserve(this->get_graph().get_number_of_vertices());
        merge.reserve_work((long long)this->get_graph().get_number_of_vertices() +
                           this->get_graph().get_number_of_edges());
      }
    }
    if (tc_split()) {
      heavy.resize(this->get_graph().get_number_of_edges());
      heavy_n.resize(1);
    }
  }

  void reset() override {}
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties)
      : gunrock::enactor_t<problem_t>(_problem, _context, _properties) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto vertex_triangles_count = P->result.vertex_triangles_count;

    if (P->param.oriented && tc_gap()) {
      auto stream = this->context->get_context(0)->stream();
      unsigned long long* counters = P->gap_counters.data().get();
      const vertex_t n = G.get_number_of_vertices();
      cudaMemsetAsync(counters, 0, sizeof(unsigned long long), stream);
      lower_tiles_t<decltype(G)> tiles{G};
      gacge::lb::dispatch(P->lb, [&](auto tag) {
        constexpr int Gs = decltype(tag)::value;
        if constexpr (Gs == 0) {
          P->merge.prefix(tiles, (long long)n, stream);
          const gacge::lb::merge_args_t m = P->merge.plan(P->merge.offsets, (long long)n, stream);
          if (P->merge.tiles > 0)
            tc_count_kernel<0><<<(unsigned int)P->merge.tiles, kTcBlock, 0, stream>>>(G, tiles, (long long)n, m,
                                                                                    counters);
        } else {
          tc_count_kernel<Gs><<<gacge::lb::grid(n), kTcBlock, 0, stream>>>(G, tiles, (long long)n,
                                                                         gacge::lb::merge_args_t{}, counters);
        }
      });
      unsigned long long total = 0;  // GAP's OrderedCount returns the total
      cudaMemcpyAsync(&total, counters, sizeof(total), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      *P->result.total_triangles_count = (std::size_t)total;
      return;
    }

    if (P->param.oriented) {
      const gacge::task::view_t task = P->task.view();  // GACGE_TASK_STATS
      auto oriented = [G, vertex_triangles_count, task] __host__ __device__(
                          vertex_t const& source, vertex_t const& neighbor,
                          edge_t const& edge, weight_t const& weight) -> bool {
        if (neighbor < source) {
          // |N(source) ∩ N(neighbor) ∩ [0, neighbor)| by a merge of the two
          // sorted lists (Ligra's countCommon)
          edge_t i = G.get_starting_edge(source), ie = G.get_starting_edge(source + 1);
          edge_t j = G.get_starting_edge(neighbor), je = G.get_starting_edge(neighbor + 1);
          vertex_t count = 0;
          unsigned long long steps = 0;  // task: this edge's intersection, its merge steps as executed
          while (i < ie && j < je) {
            vertex_t a = G.get_destination_vertex(i), b = G.get_destination_vertex(j);
            steps++;
            if (a >= neighbor || b >= neighbor)
              break;
            if (a < b) {
              i++;
            } else if (b < a) {
              j++;
            } else {
              count++;
              i++;
              j++;
            }
          }
#ifdef __CUDA_ARCH__
          gacge::task::record(task, steps);
#endif
          if (count)
            math::atomic::add(&(vertex_triangles_count[source]), count);
        }
        return false;
      };
      if (tc_split()) {
        auto stream = this->context->get_context(0)->stream();
        unsigned long long* heavy_n = P->heavy_n.data().get();
        edge_t* heavy = P->heavy.data().get();
        const long long heavy_deg = tc_heavy();
        cudaMemsetAsync(heavy_n, 0, sizeof(unsigned long long), stream);
        auto light_edge = [G, vertex_triangles_count, heavy, heavy_n, heavy_deg] __device__(edge_t const& e) {
          const vertex_t s = G.get_source_vertex(e), d = G.get_destination_vertex(e);
          if (d >= s)
            return;
          if ((long long)G.get_number_of_neighbors(s) + (long long)G.get_number_of_neighbors(d) > heavy_deg) {
            heavy[atomicAdd(heavy_n, 1ULL)] = e;  // a warp intersects it
            return;
          }
          const vertex_t c = tc_merge(G, s, d);
          if (c)
            math::atomic::add(&vertex_triangles_count[s], c);
        };
        operators::parallel_for::execute<operators::parallel_for_each_t::edge>(G, light_edge, context);
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        tc_heavy_kernel<<<sms * 16, 256, 0, stream>>>(G, vertex_triangles_count, P->heavy.data().get(),
                                                       heavy_n);
        error::throw_if_exception(cudaGetLastError(), "tc heavy kernel");
      } else if (tc_block_mapped()) {
        operators::advance::execute<operators::load_balance_t::block_mapped,
                                    operators::advance_direction_t::forward,
                                    operators::advance_io_type_t::graph,
                                    operators::advance_io_type_t::none>(
            G, E, oriented, context);
      } else {
        auto per_edge = [G, oriented] __device__(edge_t const& e) {
          oriented(G.get_source_vertex(e), G.get_destination_vertex(e), e,
                   weight_t(0));
        };
        operators::parallel_for::execute<operators::parallel_for_each_t::edge>(
            G, per_edge, context);
      }
      return;
    }

    auto intersect = [G, vertex_triangles_count] __host__ __device__(
                         vertex_t const& source,    // ... source
                         vertex_t const& neighbor,  // neighbor
                         edge_t const& edge,        // edge
                         weight_t const& weight     // weight (tuple).
                         ) -> bool {
      if (neighbor > source) {
        auto src_vertex_triangles_count = G.get_intersection_count(
            source, neighbor,
            [vertex_triangles_count, source,
             neighbor](auto intersection_vertex) {
              if (source != intersection_vertex &&
                  neighbor != intersection_vertex) {
                math::atomic::add(
                    &(vertex_triangles_count[intersection_vertex]),
                    vertex_t{1});
              }
            });
      }
      return false;
    };

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::load_balance_t::block_mapped,
                                operators::advance_direction_t::forward,
                                operators::advance_io_type_t::graph,
                                operators::advance_io_type_t::none>(
        G, E, intersect, context);
  }

  virtual bool is_converged(gcuda::multi_context_t& context) {
    if (this->iteration == 1)
      return true;
    return false;
  }

  float post_process() {
    util::timer_t timer;
    timer.begin();
    auto P = this->get_problem();
    auto G = P->get_graph();

    if (P->param.reduce_all_triangles && !(tc_gap() && P->param.oriented)) {
      auto policy = this->context->get_context(0)->execution_policy();
      *P->result.total_triangles_count = thrust::transform_reduce(
          policy, P->result.vertex_triangles_count,
          P->result.vertex_triangles_count + G.get_number_of_vertices(),
          [] __host__ __device__(const vertex_t& vertex_triangles) {
            return static_cast<std::size_t>(vertex_triangles);
          },
          std::size_t{0}, thrust::plus<std::size_t>());
    }
    return timer.end();
  }
};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          bool reduce_all_triangles,
          typename graph_t::vertex_type* vertex_triangles_count,  // Output
          std::size_t* total_triangles_count,                     // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0)),  // Context
          bool oriented = true  // false: Gunrock's own (each triangle 3 times)
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t>;

  param_type param(reduce_all_triangles, oriented);
  result_type result(vertex_triangles_count, total_triangles_count);
  // </user-defined>

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  // Disable internal-frontiers:
  enactor_properties_t props;
  props.self_manage_frontiers = true;
  enactor_type enactor(&problem, context, props);
  auto time = enactor.enact();
  time += enactor.post_process();
  problem.task.print("TC", "edge-intersection");

  // </boiler-plate>
  return time;
}

}  // namespace tc
}  // namespace gunrock