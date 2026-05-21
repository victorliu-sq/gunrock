#include <gunrock/algorithms/bfs.hxx>
#include <gunrock/algorithms/dawn.hxx>
#include <gunrock/util/performance.hxx>
#include <gunrock/io/parameters.hxx>
#include <gunrock/framework/benchmark.hxx>
#include <limits>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include "bfs_cpu.hxx"  // Reference implementation

using namespace gunrock;
using namespace memory;

template <typename csr_t, typename vertex_t, typename edge_t>
size_t count_processed_edges(csr_t& csr,
                             thrust::device_vector<vertex_t>& distances) {
  auto row_offsets = thrust::raw_pointer_cast(csr.row_offsets.data());
  auto distance_values = thrust::raw_pointer_cast(distances.data());
  auto begin = thrust::make_counting_iterator<vertex_t>(0);
  auto end = thrust::make_counting_iterator<vertex_t>(
      static_cast<vertex_t>(distances.size()));

  return thrust::transform_reduce(
      thrust::device,
      begin,
      end,
      [row_offsets, distance_values] __device__(vertex_t v) -> size_t {
        if (distance_values[v] == std::numeric_limits<vertex_t>::max()) {
          return 0;
        }
        return static_cast<size_t>(row_offsets[v + 1] - row_offsets[v]);
      },
      size_t{0},
      thrust::plus<size_t>());
}

void test_bfs(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  std::string DEFAULT_BFS_ALGORITHMS =
      "DAWN";  // Using 'Breadth First Search' here will call the original BFS
  // --
  // IO

  gunrock::io::cli::parameters_t params(num_arguments, argument_array,
                                        DEFAULT_BFS_ALGORITHMS);

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
  // Run problem

  size_t n_runs = source_vect.size();
  std::vector<float> run_times;

  auto benchmark_metrics = std::vector<benchmark::host_benchmark_t>(n_runs);
  for (int i = 0; i < n_runs; i++) {
    benchmark::INIT_BENCH();
    if (DEFAULT_BFS_ALGORITHMS == "DAWN")
      run_times.push_back(gunrock::dawn_bfs::run(G, source_vect[i],
                                                 distances.data().get(),
                                                 predecessors.data().get()));
    else
      run_times.push_back(gunrock::bfs::run(G, source_vect[i],
                                            distances.data().get(),
                                            predecessors.data().get()));

    benchmark::host_benchmark_t metrics = benchmark::EXTRACT();
    benchmark_metrics[i] = metrics;

    benchmark::DESTROY_BENCH();
  }

  // Export metrics
  if (params.export_metrics) {
    if (DEFAULT_BFS_ALGORITHMS == "DAWN")
      gunrock::util::stats::export_performance_stats(
          benchmark_metrics, n_edges, n_vertices, run_times, "dawn_bfs",
          params.filename, "market", params.json_dir, params.json_file,
          source_vect, tag_vect, num_arguments, argument_array);
    else
      gunrock::util::stats::export_performance_stats(
          benchmark_metrics, n_edges, n_vertices, run_times, "bfs",
          params.filename, "market", params.json_dir, params.json_file,
          source_vect, tag_vect, num_arguments, argument_array);
  }

  // Print info for last run
  std::cout << "Source : " << source_vect.back() << "\n";
  print::head(distances, 40, "GPU distances");
  std::cout << "GPU Elapsed Time : " << run_times[n_runs - 1] << " (ms)"
            << std::endl;
  auto edges_processed =
      count_processed_edges<csr_t, vertex_t, edge_t>(csr, distances);
  auto runtime_s = static_cast<double>(run_times[n_runs - 1]) / 1000.0;
  std::cout << "Average runtime: " << runtime_s << std::endl;
  std::cout << "Number of Processed Edges: " << edges_processed << std::endl;
  std::cout << "Processed Edges per Second: "
            << static_cast<double>(edges_processed) / runtime_s << std::endl;

  // --
  // CPU Run

  if (params.validate) {
    thrust::host_vector<vertex_t> h_distances(n_vertices);
    thrust::host_vector<vertex_t> h_predecessors(n_vertices);

    // Validate with last source in source vector
    float cpu_elapsed = bfs_cpu::run<csr_t, vertex_t, edge_t>(
        csr, source_vect.back(), h_distances.data(), h_predecessors.data());

    int n_errors =
        util::compare(distances.data().get(), h_distances.data(), n_vertices);
    print::head(h_distances, 40, "CPU Distances");

    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
    std::cout << "Number of errors : " << n_errors << std::endl;
  }
}

int main(int argc, char** argv) {
  test_bfs(argc, argv);
}
