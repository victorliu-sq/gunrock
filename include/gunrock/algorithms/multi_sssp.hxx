/**
 * @file multi_sssp.hxx
 * @brief Multi-source shortest path algorithm.
 */
#pragma once

#include <cstddef>
#include <limits>

#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
namespace multi_sssp {

template <typename vertex_t>
struct param_t {
  vertex_t* sources;
  std::size_t n_sources;

  param_t(vertex_t* _sources, std::size_t _n_sources)
      : sources(_sources), n_sources(_n_sources) {}
};

template <typename vertex_t, typename weight_t>
struct result_t {
  weight_t* distances;
  vertex_t* predecessors;
  unsigned long long* edge_counter;

  result_t(weight_t* _distances,
           vertex_t* _predecessors,
           vertex_t n_vertices,
           unsigned long long* _edge_counter = nullptr)
      : distances(_distances),
        predecessors(_predecessors),
        edge_counter(_edge_counter) {}
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

  thrust::device_vector<vertex_t> sources;
  thrust::device_vector<vertex_t> visited;

  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    sources.resize(this->param.n_sources);
    visited.resize(n_vertices);

    auto policy = this->context->get_context(0)->execution_policy();
    thrust::copy(policy, this->param.sources,
                 this->param.sources + this->param.n_sources, sources.begin());
    thrust::fill(policy, visited.begin(), visited.end(), -1);
  }

  void reset() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();

    auto context = this->get_single_context();
    auto policy = context->execution_policy();

    auto d_distances = thrust::device_pointer_cast(this->result.distances);
    thrust::fill(policy, d_distances, d_distances + n_vertices,
                 std::numeric_limits<weight_t>::max());

    auto distances = this->result.distances;
    thrust::for_each(policy, sources.begin(), sources.end(),
                     [distances] __host__ __device__(vertex_t source) {
                       distances[source] = 0;
                     });

    thrust::fill(policy, visited.begin(), visited.end(), -1);
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::enactor_t<problem_t>(_problem, _context) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;

  void prepare_frontier(frontier_t* f,
                        gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    auto context0 = context.get_context(0);
    auto policy = context0->execution_policy();

    f->reserve(P->param.n_sources);
    f->set_number_of_elements(P->param.n_sources);
    thrust::copy(policy, P->sources.begin(), P->sources.end(), f->begin());
  }

  void loop(gcuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto distances = P->result.distances;
    auto edge_counter = P->result.edge_counter;
    auto visited = P->visited.data().get();
    auto iteration = this->iteration;

    auto shortest_path = [distances, edge_counter] __host__ __device__(
                             vertex_t const& source,
                             vertex_t const& neighbor,
                             edge_t const& edge,
                             weight_t const& weight) -> bool {
      if (edge_counter != nullptr) {
        atomicAdd(edge_counter, 1ULL);
      }

      weight_t source_distance = thread::load(&distances[source]);
      weight_t distance_to_neighbor = source_distance + weight;
      weight_t recover_distance =
          math::atomic::min(&(distances[neighbor]), distance_to_neighbor);

      return (distance_to_neighbor < recover_distance);
    };

    auto remove_completed_paths = [G, visited, iteration] __host__ __device__(
                                      vertex_t const& vertex) -> bool {
      if (visited[vertex] == iteration)
        return false;

      visited[vertex] = iteration;
      return true;
    };

    operators::advance::execute<operators::load_balance_t::block_mapped>(
        G, E, shortest_path, context);

    operators::filter::execute<operators::filter_algorithm_t::bypass>(
        G, E, remove_completed_paths, context);
  }
};

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type* sources,
          std::size_t n_sources,
          typename graph_t::weight_type* distances,
          typename graph_t::vertex_type* predecessors,
          unsigned long long* edge_counter = nullptr,
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))) {
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t, weight_t>;

  param_type param(sources, n_sources);
  result_type result(distances, predecessors, G.get_number_of_vertices(),
                     edge_counter);

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  return enactor.enact();
}

}  // namespace multi_sssp
}  // namespace gunrock
