#include <gunrock/algorithms/bc.hxx>
#include <gunrock/algorithms/bfs.hxx>
#include <gunrock/util/performance.hxx>
#include <gunrock/io/parameters.hxx>
#include <limits>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

using namespace gunrock;
using namespace memory;

template <typename csr_t, typename vertex_t, typename edge_t>
size_t count_reached_vertex_edges(csr_t& csr,
                                  thrust::device_vector<vertex_t>& distances,
                                  bool skip_source) {
  auto row_offsets = thrust::raw_pointer_cast(csr.row_offsets.data());
  auto distance_values = thrust::raw_pointer_cast(distances.data());
  auto begin = thrust::make_counting_iterator<vertex_t>(0);
  auto end = thrust::make_counting_iterator<vertex_t>(
      static_cast<vertex_t>(distances.size()));

  return thrust::transform_reduce(
      thrust::device,
      begin,
      end,
      [row_offsets, distance_values, skip_source] __device__(vertex_t src) -> size_t {
        auto src_distance = distance_values[src];
        if (src_distance == std::numeric_limits<vertex_t>::max()) {
          return 0;
        }
        if (skip_source && src_distance == 0) {
          return 0;
        }
        return static_cast<size_t>(row_offsets[src + 1] - row_offsets[src]);
      },
      size_t{0},
      thrust::plus<size_t>());
}

void test_bc(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;
  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  gunrock::io::cli::parameters_t params(num_arguments, argument_array,
                                        "Betweenness Centrality");

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(params.filename);

  csr_t csr;

  if (params.binary) {
    csr.read_binary(params.filename);
  } else {
    csr.from_coo(coo);
  }

  // --
  // Build graph

  auto G = graph::build<memory_space_t::device>(properties, csr);

  // --
  // Params and memory allocation

  size_t n_vertices = G.get_number_of_vertices();
  size_t n_edges = G.get_number_of_edges();
  thrust::device_vector<weight_t> bc_values(n_vertices);
  thrust::device_vector<vertex_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);

  // Parse sources
  std::vector<int> source_vect;
  gunrock::io::cli::parse_source_string(params.source_string, &source_vect,
                                        n_vertices, params.num_runs);
  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  // --
  // GPU Run

  size_t n_runs = source_vect.size();
  std::vector<float> run_times;
  size_t total_edges_processed = 0;

  auto benchmark_metrics = std::vector<benchmark::host_benchmark_t>(n_runs);
  for (int i = 0; i < n_runs; i++) {
    benchmark::INIT_BENCH();

    run_times.push_back(
        gunrock::bc::run(G, source_vect[i], bc_values.data().get()));

    benchmark::host_benchmark_t metrics = benchmark::EXTRACT();
    benchmark_metrics[i] = metrics;
    auto source = source_vect[i];
    gunrock::bfs::run(G, source, distances.data().get(), predecessors.data().get());
    auto forward_edges =
        count_reached_vertex_edges<csr_t, vertex_t, edge_t>(csr, distances, false);
    auto backward_edges =
        count_reached_vertex_edges<csr_t, vertex_t, edge_t>(csr, distances, true);
    total_edges_processed += forward_edges + backward_edges;

    benchmark::DESTROY_BENCH();
  }

  // Export metrics
  if (params.export_metrics) {
    gunrock::util::stats::export_performance_stats(
        benchmark_metrics, n_edges, n_vertices, run_times, "bc",
        params.filename, "market", params.json_dir, params.json_file,
        source_vect, tag_vect, num_arguments, argument_array);
  }

  // --
  // Log

  std::cout << "Single source : " << source_vect.back() << "\n";
  print::head(bc_values, 40, "GPU bc values");
  std::cout << "GPU Elapsed Time : " << run_times[params.num_runs - 1]
            << " (ms)" << std::endl;
  auto runtime_s = static_cast<double>(run_times[params.num_runs - 1]) / 1000.0;
  std::cout << "Average runtime: " << runtime_s << std::endl;
  std::cout << "Number of Processed Edges: "
            << static_cast<double>(total_edges_processed) / n_runs << std::endl;
  std::cout << "Processed Edges per Second: "
            << (static_cast<double>(total_edges_processed) / n_runs) / runtime_s << std::endl;
}

int main(int argc, char** argv) {
  test_bc(argc, argv);
}
