/**
 * @file ksssp.hxx
 * @brief k independent single-source shortest paths run together (k <= 64).
 *
 * Unlike multi_sssp, which gives every source distance 0 in one shared
 * distance array (the distance to the nearest source), every source here keeps
 * its own distances: dist[v * k + s] is v's distance from source s. One
 * vertex frontier serves all k traversals; a 64-bit mask per vertex says which
 * sources improved the vertex in the last step, so an edge (u, w) is read once
 * and relaxed for every source active at u. The frontier is the union of the k
 * frontiers, and a step's relaxations are its edges times the active sources.
 *
 * Per step: the advance relaxes, for each frontier vertex u and each source s
 * active at u, dist[w][s] with atomic min and records the improved sources in
 * next_mask[w]; the vertices it improved form the next frontier. The old
 * frontier's curr_mask is then cleared, and the dedup filter moves next_mask
 * into curr_mask. Distances only decrease, so concurrent reads of dist[u][s]
 * during the step are safe (label correcting, as Gunrock's SSSP).
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
namespace ksssp {

using mask_t = unsigned long long;

// Device operations behind host-device wrappers: an extended __host__ __device__
// lambda must capture the same variables in both compilations, so the lambdas
// below call these instead of guarding their own bodies with __CUDA_ARCH__.
__host__ __device__ inline int lowest_bit(mask_t m) {
#ifdef __CUDA_ARCH__
  return __ffsll((long long)m) - 1;
#else
  return __builtin_ffsll((long long)m) - 1;
#endif
}
__host__ __device__ inline void mask_or(mask_t* a, mask_t v) {
#ifdef __CUDA_ARCH__
  atomicOr(a, v);
#else
  *a |= v;
#endif
}
__host__ __device__ inline mask_t mask_take(mask_t* a) {
#ifdef __CUDA_ARCH__
  return atomicExch(a, mask_t(0));
#else
  mask_t v = *a;
  *a = 0;
  return v;
#endif
}

template <typename vertex_t>
struct param_t {
  vertex_t* sources;  // host, k entries, distinct
  int k;
  param_t(vertex_t* _sources, int _k) : sources(_sources), k(_k) {}
};

template <typename weight_t>
struct result_t {
  weight_t* distances;  // device, n * k, dist[v * k + s]
  result_t(weight_t* _distances) : distances(_distances) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  problem_t(graph_t& G, param_type& _param, result_type& _result, std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context), param(_param), result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  thrust::device_vector<vertex_t> sources;
  thrust::device_vector<mask_t> curr_mask;
  thrust::device_vector<mask_t> next_mask;
  thrust::device_vector<vertex_t> visited;

  void init() override {
    auto n = this->get_graph().get_number_of_vertices();
    sources.assign(this->param.sources, this->param.sources + this->param.k);
    curr_mask.resize(n);
    next_mask.resize(n);
    visited.resize(n);
  }

  void reset() override {
    auto n = this->get_graph().get_number_of_vertices();
    auto policy = this->get_single_context()->execution_policy();
    const int k = this->param.k;
    auto d = thrust::device_pointer_cast(this->result.distances);
    thrust::fill(policy, d, d + (std::size_t)n * k, std::numeric_limits<weight_t>::max());
    thrust::fill(policy, curr_mask.begin(), curr_mask.end(), mask_t(0));
    thrust::fill(policy, next_mask.begin(), next_mask.end(), mask_t(0));
    thrust::fill(policy, visited.begin(), visited.end(), -1);
    auto dist = this->result.distances;
    auto cm = curr_mask.data().get();
    auto src = sources.data().get();
    thrust::for_each(policy, thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(k),
                     [=] __device__(int s) {
                       vertex_t v = src[s];
                       dist[(std::size_t)v * k + s] = 0;
                       mask_or(&cm[v], mask_t(1) << s);
                     });
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem, std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::enactor_t<problem_t>(_problem, _context) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;

  // the distinct source vertices
  void prepare_frontier(frontier_t* f, gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    auto policy = context.get_context(0)->execution_policy();
    f->reserve(P->param.k);
    f->set_number_of_elements(P->param.k);
    thrust::copy(policy, P->sources.begin(), P->sources.end(), f->begin());
  }

  void loop(gcuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();
    const std::size_t k = P->param.k;
    auto dist = P->result.distances;
    auto cm = P->curr_mask.data().get();
    auto nm = P->next_mask.data().get();
    auto visited = P->visited.data().get();
    auto iteration = this->iteration;

    // relax (u, w) for every source active at u; w joins the output frontier
    // when any of its distances improved
    auto relax = [=] __host__ __device__(vertex_t const& u, vertex_t const& w, edge_t const& e,
                                          weight_t const& weight) -> bool {
      mask_t m = cm[u];
      mask_t improved = 0;
      const weight_t* du = dist + (std::size_t)u * k;
      weight_t* dw = dist + (std::size_t)w * k;
      while (m) {
        const int s = lowest_bit(m);
        m &= m - 1;
        const weight_t nd = thread::load(&du[s]) + weight;
        if (nd < math::atomic::min(&dw[s], nd))
          improved |= mask_t(1) << s;
      }
      if (improved == 0)
        return false;
      mask_or(&nm[w], improved);
      return true;
    };
    operators::advance::execute<operators::load_balance_t::block_mapped>(G, E, relax, context);

    // the old frontier (now the output buffer) leaves: clear its masks
    auto old = E->get_output_frontier();
    auto policy = context.get_context(0)->execution_policy();
    auto first = old->begin();
    thrust::for_each(policy, first, first + old->get_number_of_elements(), [cm] __device__(vertex_t v) {
      if (gunrock::util::limits::is_valid(v))
        cm[v] = 0;
    });

    // dedup the new frontier and move each vertex's improved sources into curr_mask
    // (the visited check races, so a duplicate may pass: atomics keep the masks whole)
    auto keep = [=] __host__ __device__(vertex_t const& v) -> bool {
      if (visited[v] == iteration)
        return false;
      visited[v] = iteration;
      mask_or(&cm[v], mask_take(&nm[v]));
      return true;
    };
    operators::filter::execute<operators::filter_algorithm_t::bypass>(G, E, keep, context);
  }
};

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type* sources,  // host, k distinct vertices
          int k,
          typename graph_t::weight_type* distances,  // device, n * k
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(new gcuda::multi_context_t(0))) {
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;
  using param_type = param_t<vertex_t>;
  using result_type = result_t<weight_t>;
  param_type param(sources, k);
  result_type result(distances);
  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;
  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();
  enactor_type enactor(&problem, context);
  return enactor.enact();
}

}  // namespace ksssp
}  // namespace gunrock
