#include <gunrock/algorithms/multi_sssp.hxx>
#include "multi_sssp_cpu.hxx"
#include <gunrock/io/parameters.hxx>
#include <gunrock/util/performance.hxx>

using namespace gunrock;
using namespace memory;

void test_multi_sssp(int num_arguments, char** argument_array) {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  std::string algorithm = "Single Source Shortest Path";
  gunrock::io::cli::parameters_t params(num_arguments, argument_array,
                                        algorithm);

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(params.filename);

  csr_t csr;
  if (params.binary) {
    csr.read_binary(params.filename);
  } else {
    csr.from_coo(coo);
  }

  auto G = graph::build<memory_space_t::device>(properties, csr);

  size_t n_vertices = G.get_number_of_vertices();
  size_t n_edges = G.get_number_of_edges();

  thrust::device_vector<weight_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);

  std::vector<int> source_vect;
  gunrock::io::cli::parse_source_string(params.source_string, &source_vect,
                                        n_vertices, params.num_runs);

  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  std::vector<float> run_times;
  auto benchmark_metrics = std::vector<benchmark::host_benchmark_t>(1);

  benchmark::INIT_BENCH();
  run_times.push_back(gunrock::multi_sssp::run(
      G, source_vect.data(), source_vect.size(), distances.data().get(),
      predecessors.data().get()));

  benchmark_metrics[0] = benchmark::EXTRACT();
  benchmark::DESTROY_BENCH();

  if (params.export_metrics) {
    gunrock::util::stats::export_performance_stats(
        benchmark_metrics, n_edges, n_vertices, run_times, "multi_sssp",
        params.filename, "market", params.json_dir, params.json_file,
        source_vect, tag_vect, num_arguments, argument_array);
  }

  print::head(distances, 40, "GPU distances");
  std::cout << "GPU Elapsed Time : " << run_times.back() << " (ms)"
            << std::endl;

  if (params.validate) {
    thrust::host_vector<weight_t> h_distances(n_vertices);
    thrust::host_vector<vertex_t> h_predecessors(n_vertices);

    float cpu_elapsed = multi_sssp_cpu::run<csr_t, vertex_t, edge_t, weight_t>(
        csr, source_vect, h_distances.data(), h_predecessors.data());

    int n_errors =
        util::compare(distances.data().get(), h_distances.data(), n_vertices);

    print::head(h_distances, 40, "CPU Distances");

    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
    std::cout << "Number of errors : " << n_errors << std::endl;
  }
}

int main(int argc, char** argv) {
  test_multi_sssp(argc, argv);
}
