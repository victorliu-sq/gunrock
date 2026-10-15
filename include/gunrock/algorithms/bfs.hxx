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
#include <gunrock/algorithms/gacge_lb.hxx>

#include <cstdlib>
#include <string>

namespace gunrock {
namespace bfs {

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
  }

  // GACGE_BFS_PULL=advance: pull steps as an advance (see pull_step); default:
  // one thread per vertex (parallel_for), which stops at the first parent.
  static bool pull_with_advance() {
    static const bool v = [] {
      const char* e = std::getenv("GACGE_BFS_PULL");
      return e != nullptr && std::string(e) == "advance";
    }();
    return v;
  }

  // Direction-optimized BFS: decides the step's direction by Ligra's rule and,
  // for a pull step, runs it and makes the next frontier; false: push.
  bool pull_step(gcuda::multi_context_t& context) {
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto distances = P->result.distances;
    auto policy = this->context->get_context(0)->execution_policy();
    auto* in = this->get_input_frontier();
    const std::size_t len = in->get_number_of_elements();
    const vertex_t* f = in->data();
    // the frontier's vertices and their edges (push frontiers carry invalids)
    const std::size_t n_f = thrust::count_if(policy, f, f + len, [] __host__ __device__(vertex_t v) {
      return gunrock::util::limits::is_valid(v);
    });
    const std::size_t m_f = thrust::transform_reduce(
        policy, f, f + len,
        [G] __host__ __device__(vertex_t v) -> std::size_t {
          return gunrock::util::limits::is_valid(v) ? (std::size_t)G.get_number_of_neighbors(v) : 0;
        },
        std::size_t{0}, thrust::plus<std::size_t>());
    const std::size_t m = G.get_number_of_edges();
    if (n_f + m_f <= m / 20)
      return false;
    const vertex_t level = this->iteration;
    const vertex_t unvisited = std::numeric_limits<vertex_t>::max();
    const vertex_t n = G.get_number_of_vertices();
    if (pull_with_advance()) {
      // GACGE_BFS_PULL=advance: the unvisited vertices are the input of an
      // advance (GACGE_LB) over their edges; an edge (v, u) with u in the level
      // sets v. A vertex's remaining edges are still read once v is set (the
      // operator returns at once), unlike the scan below, which stops.
      auto* scratch = this->get_output_frontier();
      scratch->reserve(n);
      vertex_t* s = scratch->data();
      auto send = thrust::copy_if(policy, thrust::counting_iterator<vertex_t>(0),
                                  thrust::counting_iterator<vertex_t>(n), s,
                                  [distances, unvisited] __host__ __device__(vertex_t v) {
                                    return distances[v] == unvisited;
                                  });
      scratch->set_number_of_elements(send - s);
      auto pull_edge = [distances, level, unvisited] __host__ __device__(
                           vertex_t const& v, vertex_t const& u, edge_t const& e,
                           weight_t const& w) -> bool {
        if (distances[v] == unvisited && distances[u] == level)
          distances[v] = level + 1;  // every writer stores the same value
        return false;
      };
      gacge::advance<operators::advance_direction_t::forward,
                     operators::advance_io_type_t::vertices,
                     operators::advance_io_type_t::none>(
          G, pull_edge, scratch, in, this->get_enactor()->scanned_work_domain, context);
      // the next frontier, in the input buffer (no swap)
      in->reserve(n);
      vertex_t* o = in->data();
      auto end = thrust::copy_if(policy, thrust::counting_iterator<vertex_t>(0),
                                 thrust::counting_iterator<vertex_t>(n), o,
                                 [distances, level] __host__ __device__(vertex_t v) {
                                   return distances[v] == level + 1;
                                 });
      in->set_number_of_elements(end - o);
      return true;
    }
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
    return true;
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
    gacge::advance(G, E, search, context);  // GACGE_LB

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