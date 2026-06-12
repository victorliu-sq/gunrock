#include <gunrock/algorithms/multi_sssp.hxx>
#include "multi_sssp_cpu.hxx"
#include <gunrock/io/parameters.hxx>
#include <gunrock/util/performance.hxx>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace gunrock;
using namespace memory;

struct normalized_args_t {
  std::vector<std::string> storage;
  std::vector<char*> argv;
};

normalized_args_t normalize_multi_sssp_args(int argc, char** argv) {
  normalized_args_t normalized;
  normalized.storage.reserve(argc);

  for (int i = 0; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--src" || arg == "-s") {
      std::string sources = "--src=";
      bool has_source = false;
      i++;
      while (i < argc && argv[i][0] != '-') {
        if (has_source) sources += ",";
        sources += argv[i];
        has_source = true;
        i++;
      }
      if (!has_source) {
        std::cerr << "Error: --src requires at least one source id" << std::endl;
        std::exit(1);
      }
      normalized.storage.push_back(sources);
      i--;
    } else {
      normalized.storage.push_back(arg);
    }
  }

  normalized.argv.reserve(normalized.storage.size());
  for (auto& value : normalized.storage) {
    normalized.argv.push_back(const_cast<char*>(value.c_str()));
  }

  return normalized;
}

void test_multi_sssp(int num_arguments, char** argument_array) {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  std::string algorithm = "Multi Source Shortest Path";
  auto normalized_args = normalize_multi_sssp_args(num_arguments, argument_array);
  gunrock::io::cli::parameters_t params(static_cast<int>(normalized_args.argv.size()),
                                        normalized_args.argv.data(),
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
  thrust::device_vector<unsigned long long> edges_processed(1);

  std::vector<int> source_vect;
  if (!params.source_string.empty() && params.source_count >= 0) {
    std::cerr << "Error: use either --src or --src-count, not both"
              << std::endl;
    std::exit(1);
  }
  if (params.source_count >= 0) {
    gunrock::io::cli::generate_unique_random_sources(
        params.source_count, &source_vect, n_vertices);
  } else {
    gunrock::io::cli::parse_source_string(params.source_string, &source_vect,
                                          n_vertices, params.num_runs);
  }

  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  std::vector<float> run_times;
  auto benchmark_metrics = std::vector<benchmark::host_benchmark_t>(1);

  thrust::fill(edges_processed.begin(), edges_processed.end(), 0ULL);
  benchmark::INIT_BENCH();
  run_times.push_back(gunrock::multi_sssp::run(
      G, source_vect.data(), source_vect.size(), distances.data().get(),
      predecessors.data().get(), edges_processed.data().get()));

  benchmark_metrics[0] = benchmark::EXTRACT();
  benchmark::DESTROY_BENCH();

  thrust::host_vector<unsigned long long> h_edges_processed = edges_processed;

  if (params.export_metrics) {
    gunrock::util::stats::export_performance_stats(
        benchmark_metrics, n_edges, n_vertices, run_times, "multi_sssp",
        params.filename, "market", params.json_dir, params.json_file,
        source_vect, tag_vect, num_arguments, argument_array);
  }

  print::head(distances, 40, "GPU distances");
  std::cout << "GPU Elapsed Time : " << run_times.back() << " (ms)"
            << std::endl;
  auto runtime_s = static_cast<double>(run_times.back()) / 1000.0;
  std::cout << "Average runtime: " << runtime_s << std::endl;
  std::cout << "Number of Processed Edges: " << h_edges_processed[0]
            << std::endl;
  std::cout << "Processed Edges per Second: "
            << static_cast<double>(h_edges_processed[0]) / runtime_s
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
