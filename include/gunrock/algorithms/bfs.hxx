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

#include <cstdlib>
#include <string>

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

  void init() override {}

  void reset() override {
    auto n_vertices = this->get_graph().get_number_of_vertices();
    auto d_distances = thrust::device_pointer_cast(this->result.distances);
    thrust::fill(thrust::device, d_distances + 0, d_distances + n_vertices,
                 std::numeric_limits<vertex_t>::max());
    thrust::fill(thrust::device, d_distances + this->param.single_source,
                 d_distances + this->param.single_source + 1, 0);
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
    f->push_back(P->param.single_source);
    gap_pulling = false;
    gap_started = false;
  }

  // GAP rule state (rule_gap): edges not yet examined, the edges of the last
  // push's new vertices, the vertices the last pull woke, pulling or pushing
  long long gap_edges_to_check = 0, gap_scout = 0, gap_old_awake = 0;
  bool gap_pulling = false, gap_started = false;

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

  // Direction-optimized BFS: decides the step's direction (rule_gap: GAP's
  // DOBFS; otherwise Ligra's rule) and, for a pull step, runs it and makes the
  // next frontier; false: push.
  bool pull_step(gcuda::multi_context_t& context) {
    auto G = this->get_problem()->get_graph();
    const std::size_t m = G.get_number_of_edges();
    std::size_t n_f = 0, m_f = 0;
    if (!rule_gap()) {
      frontier_size(n_f, m_f);
      return n_f + m_f > m / 20 ? do_pull(context) >= 0 : false;
    }
    if (!gap_started) {  // the source's degree
      gap_started = true;
      frontier_size(n_f, m_f);
      gap_edges_to_check = (long long)m;
      gap_scout = (long long)m_f;
    }
    if (!gap_pulling && gap_scout > gap_edges_to_check / 15) {
      frontier_size(n_f, m_f);
      gap_pulling = true;
      gap_old_awake = (long long)n_f;
    }
    if (gap_pulling) {
      const long long awake = do_pull(context);
      const long long n = G.get_number_of_vertices();
      if (!(awake >= gap_old_awake || awake > n / 18)) {
        gap_pulling = false;
        gap_scout = 1;
      }
      gap_old_awake = awake;
      return true;
    }
    gap_edges_to_check -= gap_scout;
    return false;
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
    // GAP rule: the next step's scout count, the degrees of the vertices this
    // push found (the new input frontier; GAP adds them as it claims them)
    if (P->param.direction_optimized && rule_gap()) {
      std::size_t n_f = 0, m_f = 0;
      frontier_size(n_f, m_f);
      gap_scout = (long long)m_f;
    }

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

  enactor_type enactor(&problem, context);
  return enactor.enact();
}

}  // namespace bfs
}  // namespace gunrock