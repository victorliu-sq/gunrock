/**
 * @file gacge_lb.hxx
 * @brief GACGE: one load-balancing choice for every advance of a program.
 *
 * GACGE_LB=thread|block|merge selects Gunrock's thread_mapped, block_mapped
 * (default, Gunrock's) or merge_path advance for all advances of the run; the
 * choice is printed once as an "LBConfig: lb=<name>" line.
 */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

#include <gunrock/framework/operators/advance/advance.hxx>

namespace gunrock {
namespace gacge {

inline operators::load_balance_t lb() {
  static const operators::load_balance_t value = [] {
    const char* e = std::getenv("GACGE_LB");
    const std::string s = e ? e : "block";
    operators::load_balance_t v = operators::load_balance_t::block_mapped;
    if (s == "thread")
      v = operators::load_balance_t::thread_mapped;
    else if (s == "merge")
      v = operators::load_balance_t::merge_path;
    else if (s != "block") {
      std::fprintf(stderr, "GACGE_LB=%s: expected thread, block or merge\n", s.c_str());
      std::exit(2);
    }
    std::printf("LBConfig: lb=%s\n", s.c_str());
    return v;
  }();
  return value;
}

// advance on explicit frontiers (no buffer swap)
template <operators::advance_direction_t direction,
          operators::advance_io_type_t input_type,
          operators::advance_io_type_t output_type,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
void advance(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             work_tiles_t& segments,
             gcuda::multi_context_t& context) {
  using operators::load_balance_t;
  switch (lb()) {
    case load_balance_t::thread_mapped:
      operators::advance::execute<load_balance_t::thread_mapped, direction, input_type, output_type>(
          G, op, input, output, segments, context);
      break;
    case load_balance_t::merge_path:
      operators::advance::execute<load_balance_t::merge_path, direction, input_type, output_type>(
          G, op, input, output, segments, context);
      break;
    default:
      operators::advance::execute<load_balance_t::block_mapped, direction, input_type, output_type>(
          G, op, input, output, segments, context);
  }
}

// advance on the enactor's frontiers (swaps them when there is an output)
template <operators::advance_direction_t direction = operators::advance_direction_t::forward,
          operators::advance_io_type_t input_type = operators::advance_io_type_t::vertices,
          operators::advance_io_type_t output_type = operators::advance_io_type_t::vertices,
          typename graph_t,
          typename enactor_type,
          typename operator_t>
void advance(graph_t& G, enactor_type* E, operator_t op, gcuda::multi_context_t& context) {
  using operators::load_balance_t;
  switch (lb()) {
    case load_balance_t::thread_mapped:
      operators::advance::execute<load_balance_t::thread_mapped, direction, input_type, output_type>(
          G, E, op, context);
      break;
    case load_balance_t::merge_path:
      operators::advance::execute<load_balance_t::merge_path, direction, input_type, output_type>(
          G, E, op, context);
      break;
    default:
      operators::advance::execute<load_balance_t::block_mapped, direction, input_type, output_type>(
          G, E, op, context);
  }
}

}  // namespace gacge
}  // namespace gunrock
