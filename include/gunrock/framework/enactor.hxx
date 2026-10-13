/**
 * @file enactor.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#include <vector>
#include <cstdio>
#include <cstdlib>

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <gunrock/cuda/cuda.hxx>

#include <gunrock/framework/frontier/frontier.hxx>
#include <gunrock/framework/problem.hxx>
#include <gunrock/framework/benchmark.hxx>

#pragma once

namespace gunrock {

/**
 * @brief A simple struct to store enactor properties.
 *
 * @par Overview
 * Modify this per-algorithm if the defaults need to be changed. A
 * gunrock-specific parameter can also control the frontier sizing factor used
 * to resize and scale the frontier larger than the needed size.
 */
struct enactor_properties_t {
  /*!
   * Resizes the frontier by this factor * the size required.
   * @code frontier.resize(frontier_sizing_factor * new_size);
   */
  float frontier_sizing_factor{1.5f};

  /*!
   * Number of frontier buffers to manage.
   * @note Not used in the implementation yet.
   */
  std::size_t number_of_frontier_buffers{2};

  /*!
   * When enabled, enactor's internal frontiers are not allocated ahead of time.
   * Instead, the user is encouraged to manage the frontiers.
   */
  bool self_manage_frontiers{false};

  /**
   * @brief Construct a new enactor properties t object with default values.
   */
  enactor_properties_t() = default;
};

/**
 * @brief Building block of the algorithm within gunrock. The methods within the
 * enactor structure are overriden by an application writer (methods such as
 * loop) to extend the enactor to implement a graph algorithm.
 *
 * @par Overview
 * Note that the enactor's `enact()` member can be extended to support
 * multi-gpu contexts (using execution policies like model). Enactor also has
 * pure virtual functions, which **must** be implemented within the graph
 * algorithm by the developers. These functions prepare the initial frontier of
 * the algorithm (which could be one node, the entire graph or any subset),
 * and defines the main loop of iteration that iterates until the algorithm
 * converges.
 * @see prepare_frontier()
 * @see loop()
 * @see is_converged()
 *
 * @tparam algorithm_problem_t algorithm specific problem type.
 * @tparam frontier_kind `gunrock::frontier_kind_t` enum (vertex or
 * edge-based frontier). Currently, only vertex frontier is supported.
 *
 */
template <typename algorithm_problem_t,
          frontier::frontier_kind_t frontier_kind =
              frontier::frontier_kind_t::vertex_frontier,
          frontier::frontier_view_t frontier_view =
              frontier::frontier_view_t::vector>
struct enactor_t {
  using vertex_t = typename algorithm_problem_t::vertex_t;
  using edge_t = typename algorithm_problem_t::edge_t;

  using frontier_t = frontier::frontier_t<vertex_t, edge_t, frontier_kind>;

  /*!
   * Enactor properties (frontier resizing factor, buffers, etc.)
   */
  enactor_properties_t properties;

  /*!
   * A shared_ptr to a multi-gpu context.
   * @see `gunrock::gcuda::multi_context_t`
   */
  std::shared_ptr<gcuda::multi_context_t> context;

  /*!
   * Algorithm's problem structure.
   */
  algorithm_problem_t* problem;

  /*!
   * Vector of frontier buffers (default number of buffers = 2).
   */
  thrust::host_vector<frontier_t> frontiers;

  /*!
   * Extra space to store scan of the work domain (for advance).
   * @note This is not used for all advance algorithms.
   * @todo Find a way to only allocate this if the advance algorithm that
   * actually needs it is being run. Otherwise, it maybe a waste of memory space
   * to allocate this.
   */
  thrust::device_vector<edge_t> scanned_work_domain;

  /*!
   * Active frontier buffer, this pointer can be obtained by
   * `get_input_frontier()` method. This buffer is used as an input frontier.
   */
  frontier_t* active_frontier;

  /*!
   * Inactive frontier buffer, this pointer can be obtained by
   * `get_output_frontier()` method. This buffer is often but not always used as
   * an output frontier.
   */
  frontier_t* inactive_frontier;

  /*!
   * An internal selector used to swap frontier buffers.
   */
  int buffer_selector;

  /*!
   * Number of iterations for the `loop` method. Increments per loop-iteration.
   */
  int iteration;

  /*!
   * Disable copy ctor and assignment operator. We don't want to let the user
   * copy only a slice.
   * @todo Look at enabling copy constructors.
   */
  enactor_t(const enactor_t& rhs) = delete;
  enactor_t& operator=(const enactor_t& rhs) = delete;

  /**
   * @brief Construct a new enactor t object.
   *
   * @param _problem algorithm's problem data structure.
   * @param _context shared pointer to the gcuda::multi_context_t context that
   * stores information about multiple GPUs (such as streams, device ids,
   * events, etc.)
   * @param _properties `gunrock::enactor_properties_t`, includes
   * properties such as frontier resizing factor (default = 2) and number of
   * frontier buffers to create for the enactor.
   */
  enactor_t(algorithm_problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t())
      : properties(_properties),
        context(_context),
        problem(_problem),
        frontiers(properties.number_of_frontier_buffers),
        scanned_work_domain(problem->get_graph().get_number_of_vertices() + 1),
        active_frontier(reinterpret_cast<frontier_t*>(&frontiers[0])),
        inactive_frontier(reinterpret_cast<frontier_t*>(&frontiers[1])),
        buffer_selector(0),
        iteration(0) {
    /*!
     * If the self manage frontiers property is false, the enactor interface
     * will resize the frontier buffers ahead of time to avoid the first
     * allocation during the algorithm execution face.
     * @note We also use a resizing factor to allocate more than what we
     * actually may need.
     *
     */
    if (!(properties.self_manage_frontiers)) {
      auto g = problem->get_graph();
      std::size_t initial_size =
          (g.get_number_of_edges() > g.get_number_of_vertices())
              ? g.get_number_of_edges()
              : g.get_number_of_vertices();

      for (auto& buffer : frontiers) {
        buffer.set_resizing_factor(properties.frontier_sizing_factor);
        buffer.reserve((std::size_t)(initial_size));
      }
    }
  }

  /**
   * @brief Get the problem pointer object
   * @return algorithm_problem_t*
   */
  algorithm_problem_t* get_problem() {
    return reinterpret_cast<algorithm_problem_t*>(problem);
  }

  /**
   * @brief Get the frontier pointer object
   * @return frontier_t*
   */
  frontier_t* get_input_frontier() {
    return reinterpret_cast<frontier_t*>(active_frontier);
  }

  /**
   * @brief Get the frontier pointer object
   * @return frontier_t*
   */
  frontier_t* get_output_frontier() {
    return reinterpret_cast<frontier_t*>(inactive_frontier);
  }

  /**
   * @brief Get the pointer to the enactor object.
   * @return enactor_t*
   */
  enactor_t* get_enactor() { return this; }

  /**
   * @brief The active frontier's valid vertices and the edges they will
   * examine (the sum of their degrees), for GACGE_TRAVERSAL_STATS.
   */
  void traversal_frontier_work(std::size_t& fv, std::size_t& fe, std::size_t& max_deg) {
    auto f = get_input_frontier();
    auto n = f->get_number_of_elements();
    fv = fe = max_deg = 0;
    if (n == 0)
      return;
    auto g = problem->get_graph();
    auto first = f->begin();
    auto deg = [g] __device__(vertex_t const& v) -> std::size_t {
      return gunrock::util::limits::is_valid(v) ? (std::size_t)g.get_number_of_neighbors(v) : 0;
    };
    fe = thrust::transform_reduce(thrust::device, first, first + n, deg, std::size_t(0), thrust::plus<std::size_t>());
    max_deg = thrust::transform_reduce(thrust::device, first, first + n, deg, std::size_t(0),
                                       thrust::maximum<std::size_t>());
    fv = thrust::count_if(thrust::device, first, first + n,
                          [] __device__(vertex_t const& v) { return gunrock::util::limits::is_valid(v); });
  }

  /**
   * @brief Swap the inactive and active frontier buffers such that the inactive
   * buffer becomes the input buffer to the next operator and vice-versa.
   */
  void swap_frontier_buffers() {
    buffer_selector ^= 1;
    active_frontier =
        reinterpret_cast<frontier_t*>(&frontiers[buffer_selector]);
    inactive_frontier =
        reinterpret_cast<frontier_t*>(&frontiers[buffer_selector ^ 1]);
  }

  /**
   * @brief Run the enactor with the given problem and the loop.
   * @todo We can work on evolving this into a multi-gpu implementation.
   * @return float time took for enactor to complete (this is often used as
   * **the** time for performance measurements).
   */
  float enact() {
    auto single_context = context->get_context(0);
    prepare_frontier(get_input_frontier(), *context);
    // GACGE traversal statistics (measure/traversal-char), off unless
    // GACGE_TRAVERSAL_STATS is set: one XBTraversal line per enact(), and one
    // XBStep line per iteration for the enact() numbered GACGE_TRAVERSAL_TRACE
    // (0-based, default 1: the first after one warm-up). The counting adds work
    // between iterations, so runtimes from such runs are not used.
    static int enact_index = -1;
    ++enact_index;
    const bool stats = std::getenv("GACGE_TRAVERSAL_STATS") != nullptr;
    const char* trace_env = std::getenv("GACGE_TRAVERSAL_TRACE");
    const bool trace = stats && enact_index == (trace_env ? std::atoi(trace_env) : 1);
    std::size_t t_steps = 0, t_v = 0, t_e = 0, t_max_e = 0, t_small = 0, t_imb_n = 0;
    double t_imb = 0;  // sum over steps of max degree / mean degree of the frontier
    auto timer = single_context->timer();
    timer.begin();
    while (!is_converged(*context)) {
      if (stats) {
        std::size_t fv = 0, fe = 0, md = 0;
        traversal_frontier_work(fv, fe, md);
        t_steps++; t_v += fv; t_e += fe;
        t_max_e = fe > t_max_e ? fe : t_max_e;
        t_small += fe < 65536;
        if (fe > 0) { t_imb += (double)md * (double)fv / (double)fe; t_imb_n++; }
        if (trace)
          std::printf("XBStep: run=%d step=%zu phase=advance frontier_v=%zu frontier_e=%zu max_deg=%zu\n",
                      enact_index, t_steps - 1, fv, fe, md);
      }
      loop(*context);
      ++iteration;
    }
    finalize(*context);
    auto runtime = timer.end();
    if (stats)
      std::printf("XBTraversal: run=%d steps=%zu frontier_v=%zu frontier_e=%zu max_frontier_e=%zu small_steps=%zu "
                  "imbalance=%.3f nodes=%zu edges=%zu\n", enact_index, t_steps, t_v, t_e, t_max_e, t_small,
                  t_imb_n ? t_imb / t_imb_n : 0.0, (std::size_t)problem->get_graph().get_number_of_vertices(),
                  (std::size_t)problem->get_graph().get_number_of_edges());
#if (ESSENTIALS_COLLECT_METRICS)
    benchmark::____.search_depth = iteration;
    benchmark::____.total_runtime = runtime;
#endif
    return runtime;
  }

  /**
   * @brief This is the core of the implementation for any algorithm. Graph
   * algorithm developers should override this virtual function to implement
   * their own `loop` function, which loops till a convergence condition is
   * met `is_converged()`.
   *
   * @par Overview
   * This function is on the host and is timed, so make sure you are writing the
   * most efficient implementation possible. Avoid performing copies in this
   * function if they are not part of the algorithm's core, or running API calls
   * that are incredibly slow (such as `printfs` or debug statements).
   *
   * @param context `gunrock::gcuda::multi_context_t`.
   */
  virtual void loop(gcuda::multi_context_t& context) = 0;

  /**
   * @brief Prepare the initial frontier.
   *
   * @param context `gunrock::gcuda::multi_context_t`.
   */
  virtual void prepare_frontier(frontier_t* f,
                                gcuda::multi_context_t& context) {};

  /**
   * @brief Algorithm is converged if true is returned, keep on iterating if
   * false is returned. This function is checked at the end of every iteration
   * of the `enact()`. Graph algorithm developers may override this function
   * with their own custom convergence condition.
   *
   * @par Overview
   * Default behavior of `is_converged()` method is to claim that the algorithm
   * is converged if the frontier size is set to be empty (`frontier->is_empty()
   * == true`).
   *
   * @return true converged!
   * @return false not converged, keep looping!
   */
  virtual bool is_converged(gcuda::multi_context_t& context) {
    return active_frontier->is_empty();
  }

  /**
   * @brief `finalize()` runs only once, and after the `while()` has ended.
   *
   * @par Finalize runs after the convergence condition is met in
   * `is_converged()`. Users can extend the following virtual function to do a
   * one final wrap-up of the application. Users are not required to implement
   * this function.
   *
   * @param context `gunrock::gcuda::multi_context_t`.
   */
  virtual void finalize(gcuda::multi_context_t& context) {}

};  // struct enactor_t

}  // namespace gunrock