/**
 * @file pr.hxx
 * @author Ben Johnson (bkj.322@gmail.com)
 * @brief PageRank
 * @date 2021-04-01
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/inner_product.h>

#include <cstdlib>
#include <string>

namespace gunrock {
namespace pr {

// GACGE (2026-10-14): PageRank as Ligra's (apps/PageRank.C): pull, float, no
// redistribution of dangling (degree-0) mass, contrib[u] = alpha * p[u] / deg(u)
// once per iteration, p[v] = (1 - alpha) / n + sum of contrib over v's
// neighbors (one warp per vertex, so a high-degree vertex's sum is split over
// 32 lanes). Used with a fixed iteration count (max_iters > 0, GACGE_PR_ITERS);
// GACGE_PR=push, or no fixed count, runs Gunrock's push PageRank (atomicAdd per
// edge, dangling mass redistributed).
// GACGE_PR=vertex: the same pull with Gunrock's parallel_for over the
// vertices (one thread per vertex sums its neighbors' contributions).
inline bool pr_vertex() {
  static const bool v = [] {
    const char* e = std::getenv("GACGE_PR");
    return e != nullptr && std::string(e) == "vertex";
  }();
  return v;
}

// GACGE_PR=nr: the same pull with Gunrock's neighborreduce operator (as its
// spmv): each edge yields its neighbor's contribution and a segmented reduction
// (moderngpu, merge-path balanced) sums every vertex's edges, no atomics.
inline bool pr_nr() {
  static const bool v = [] {
    const char* e = std::getenv("GACGE_PR");
    return e != nullptr && std::string(e) == "nr";
  }();
  return v;
}

inline bool pr_pull() {
  static const bool pull = [] {
    const char* e = std::getenv("GACGE_PR");
    return e == nullptr || std::string(e) != "push";
  }();
  return pull;
}

template <typename graph_t, typename weight_t>
__global__ void pull_warp_kernel(graph_t G,
                                 const weight_t* __restrict__ contrib,
                                 weight_t* __restrict__ p,
                                 std::size_t n,
                                 weight_t base) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  const std::size_t warp =
      (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n)
    return;  // the whole warp: n is compared per warp
  const vertex_t v = static_cast<vertex_t>(warp);
  const edge_t begin = G.get_starting_edge(v), end = G.get_starting_edge(v + 1);
  weight_t sum = 0;
  for (edge_t e = begin + lane; e < end; e += 32)
    sum += contrib[G.get_destination_vertex(e)];
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  if (lane == 0)
    p[v] = base + sum;
}

template <typename weight_t>
struct param_t {
  weight_t alpha;
  weight_t tol;
  int max_iters;  // > 0: stop after this many iterations (GACGE: 20, tol 0)

  param_t(weight_t _alpha, weight_t _tol, int _max_iters = 0)
      : alpha(_alpha), tol(_tol), max_iters(_max_iters) {}
};

template <typename weight_t>
struct result_t {
  weight_t* p;
  result_t(weight_t* _p) : p(_p) {}
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

  thrust::device_vector<weight_t>
      plast;  // pagerank values from previous iteration
  thrust::device_vector<weight_t>
      iweights;  // alpha * 1 / (sum of outgoing weights) -- used to determine
                 // out of mass spread from src to dst

  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    plast.resize(n_vertices);
    iweights.resize(n_vertices);
  }

  void reset() override {
    // Execution policy for a given context (using single-gpu).
    auto policy = this->context->get_context(0)->execution_policy();

    auto g = this->get_graph();

    auto n_vertices = g.get_number_of_vertices();
    auto alpha = this->param.alpha;

    thrust::fill_n(policy, this->result.p, n_vertices, 1.0 / n_vertices);

    thrust::fill_n(policy, plast.begin(), n_vertices, 0);

    auto get_weight = [=] __host__ __device__(const int& i) -> weight_t {
      weight_t val = 0;

      edge_t start = g.get_starting_edge(i);
      edge_t end = start + g.get_number_of_neighbors(i);
      for (edge_t offset = start; offset < end; offset++) {
        val += g.get_edge_weight(offset);
      }

      return val != 0 ? alpha / val : 0;
    };
    // pull: alpha / deg (the graph's edge weights are 1; not read)
    auto get_degree = [=] __host__ __device__(const int& i) -> weight_t {
      auto d = g.get_number_of_neighbors(i);
      return d != 0 ? alpha / weight_t(d) : 0;
    };
    if (pr_pull() && this->param.max_iters > 0) {
      thrust::transform(policy, thrust::counting_iterator<vertex_t>(0),
                        thrust::counting_iterator<vertex_t>(n_vertices),
                        iweights.begin(), get_degree);
      return;
    }

    thrust::transform(policy, thrust::counting_iterator<vertex_t>(0),
                      thrust::counting_iterator<vertex_t>(n_vertices),
                      iweights.begin(), get_weight);
  }
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

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto n_vertices = G.get_number_of_vertices();
    auto p = P->result.p;
    auto plast = P->plast.data().get();
    auto iweights = P->iweights.data().get();
    auto alpha = P->param.alpha;

    auto policy = this->context->get_context(0)->execution_policy();

    if (pr_pull() && P->param.max_iters > 0) {
      // contrib (in plast) = alpha * p / deg, then p = (1 - alpha) / n + sum
      thrust::transform(policy, p, p + n_vertices, iweights, plast,
                        thrust::multiplies<weight_t>());
      if (pr_nr()) {
        auto edge_contrib = [G, plast] __host__ __device__(edge_t edge) -> weight_t {
          return plast[G.get_destination_vertex(edge)];
        };
        auto plus_t = [] __host__ __device__(weight_t a, weight_t b) { return a + b; };
        operators::neighborreduce::execute(G, E, p, edge_contrib, plus_t, weight_t(0), context);
        const weight_t base = weight_t((1 - alpha) / n_vertices);
        thrust::transform(policy, p, p + n_vertices, p,
                          [base] __host__ __device__(weight_t x) { return base + x; });
        return;
      }
      if (pr_vertex()) {
        const weight_t base = weight_t((1 - alpha) / n_vertices);
        auto pull = [G, plast, p, base] __device__(vertex_t const& v) {
          weight_t sum = 0;
          for (auto e = G.get_starting_edge(v), end = G.get_starting_edge(v + 1); e < end; ++e)
            sum += plast[G.get_destination_vertex(e)];
          p[v] = base + sum;
        };
        operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(G, pull, context);
        return;
      }
      const std::size_t threads = 256;
      const std::size_t blocks =
          (static_cast<std::size_t>(n_vertices) * 32 + threads - 1) / threads;
      pull_warp_kernel<<<blocks, threads, 0,
                         this->context->get_context(0)->stream()>>>(
          G, plast, p, static_cast<std::size_t>(n_vertices),
          weight_t((1 - alpha) / n_vertices));
      error::throw_if_exception(cudaGetLastError(), "pr pull kernel");
      return;
    }

    thrust::copy_n(policy, p, n_vertices, plast);

    // >> handle "dangling nodes" (nodes w/ zero outdegree)
    // could skip this if no nodes have sero outdegree
    auto compute_dangling = [=] __host__ __device__(const int& i) -> weight_t {
      return iweights[i] == 0 ? alpha * p[i] : 0;
    };

    float dsum = thrust::transform_reduce(
        policy, thrust::counting_iterator<vertex_t>(0),
        thrust::counting_iterator<vertex_t>(n_vertices), compute_dangling,
        (weight_t)0.0, thrust::plus<weight_t>());

    thrust::fill_n(policy, p, n_vertices, (1 - alpha + dsum) / n_vertices);
    // -- OR --
    // skip dangling nodes
    // thrust::fill_n(policy,
    //   p, n_vertices, (1 - alpha) / n_vertices);
    // <<
    auto spread_coo_op = [=] __host__ __device__(edge_t const& e) -> void {
      auto src = G.get_source_vertex(e);
      auto dst = G.get_destination_vertex(e);
      weight_t weight = G.get_edge_weight(e);
      weight_t update = plast[src] * iweights[src] * weight;
      math::atomic::add(p + dst, update);
    };

    operators::parallel_for::execute<operators::parallel_for_each_t::edge>(
        G,              // graph
        spread_coo_op,  // lambda function
        context         // context
    );

    // -- OR --
    // You can do the following with advance operator instead:
    // auto spread_op = [=] __host__ __device__(
    //                      vertex_t const& src, vertex_t const& dst,
    //                      edge_t const& edge, weight_t const& weight) -> bool
    // {
    //   weight_t update = plast[src] * iweights[src] * weight;
    //   math::atomic::add(p + dst, update);
    //   return false;
    // };
    // operators::advance::execute<operators::load_balance_t::block_mapped,
    //                             operators::advance_direction_t::forward,
    //                             operators::advance_io_type_t::graph,
    //                             operators::advance_io_type_t::none>(
    //     G, E, spread_op, context);
    // <<
  }

  virtual bool is_converged(gcuda::multi_context_t& context) {
    if (this->iteration == 0)
      return false;

    auto P = this->get_problem();
    // a fixed iteration count (checked before every iteration: exactly max_iters)
    if (P->param.max_iters > 0 && this->iteration >= P->param.max_iters)
      return true;
    auto G = P->get_graph();
    auto tol = P->param.tol;

    auto n_vertices = G.get_number_of_vertices();
    auto p = P->result.p;
    auto plast = P->plast.data().get();

    auto abs_diff = [=] __host__ __device__(const int& i) -> weight_t {
      return abs(p[i] - plast[i]);
    };

    auto policy = this->context->get_context(0)->execution_policy();
    float err = thrust::transform_reduce(
        policy, thrust::counting_iterator<vertex_t>(0),
        thrust::counting_iterator<vertex_t>(n_vertices), abs_diff,
        (weight_t)0.0, thrust::maximum<weight_t>());

    return err < tol;
  }

};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::weight_type alpha,
          typename graph_t::weight_type tol,
          typename graph_t::weight_type* p,  // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0)),  // Context
          int max_iters = 0,         // > 0: exactly this many iterations
          int* iterations = nullptr  // Output: the iterations run
) {
  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<weight_t>;
  using result_type = result_t<weight_t>;

  param_type param(alpha, tol, max_iters);
  result_type result(p);
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
  float ms = enactor.enact();
  if (iterations)
    *iterations = enactor.iteration;
  return ms;
  // </boiler-plate>
}

}  // namespace pr
}  // namespace gunrock
