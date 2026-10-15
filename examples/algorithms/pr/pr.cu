#include <gunrock/algorithms/pr.hxx>
#include <gunrock/util/performance.hxx>
#include <gunrock/io/parameters.hxx>
#include <gunrock/io/gacge_bin.hxx>
#include <thrust/reduce.h>
#include <cstdio>
#include <cstdlib>

using namespace gunrock;
using namespace memory;

void test_pr(int num_arguments, char** argument_array) {
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
                                        "Page Rank");

  // GACGE binary CSR (<name>.bin, gunrock/io/gacge_bin.hxx): one parallel read
  // of the canonical graph Ligra reads too; .mtx / .csr as before.
  csr_t csr;
  graph::graph_properties_t properties;
  if (params.gacge_binary) {
    gacge_bin::load_device_csr(params.filename, csr, gacge_bin::values_t::ones);
  } else {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    auto [props, coo] = mm.load(params.filename);
    properties = props;
    if (params.binary) {
      csr.read_binary(params.filename);
    } else {
      csr.from_coo(coo);
    }
  }

  auto G = graph::build<memory_space_t::device>(properties, csr);

  // --
  // Params and memory allocation

  srand(time(NULL));

  weight_t alpha = 0.85;
  weight_t tol = 1e-6;
  // GACGE_PR_ITERS=k: exactly k iterations, no convergence test (as Ligra's
  // PageRank with -maxiters k -epsilon 0)
  int max_iters = 0;
  if (const char* it = std::getenv("GACGE_PR_ITERS")) {
    max_iters = std::atoi(it);
    tol = 0;
  }

  size_t n_vertices = G.get_number_of_vertices();
  size_t n_edges = G.get_number_of_edges();
  thrust::device_vector<weight_t> p(n_vertices);

  // Parse tags
  std::vector<std::string> tag_vect;
  gunrock::io::cli::parse_tag_string(params.tag_string, &tag_vect);

  // --
  // GPU Run

  std::vector<float> run_times;

  auto benchmark_metrics =
      std::vector<benchmark::host_benchmark_t>(params.num_runs);
  for (int i = 0; i < params.num_runs; i++) {
    gacge::profile::SetTrial(i);
    benchmark::INIT_BENCH();

    int iterations = 0;
    run_times.push_back(gunrock::pr::run(
        G, alpha, tol, p.data().get(),
        std::shared_ptr<gcuda::multi_context_t>(new gcuda::multi_context_t(0)),
        max_iters, &iterations));
    weight_t rank0 = p[0];
    double rank_sum = thrust::reduce(p.begin(), p.end(), 0.0);
    std::printf("PRResult: iterations=%d rank_sum=%.9f rank0=%.9e\n", iterations, rank_sum, (double)rank0);

    benchmark::host_benchmark_t metrics = benchmark::EXTRACT();
    benchmark_metrics[i] = metrics;

    benchmark::DESTROY_BENCH();
  }

  // Placeholder since PR does not use sources
  std::vector<int> src_placeholder;

  // Export metrics
  if (params.export_metrics) {
    gunrock::util::stats::export_performance_stats(
        benchmark_metrics, n_edges, n_vertices, run_times, "pr",
        params.filename, "market", params.json_dir, params.json_file,
        src_placeholder, tag_vect, num_arguments, argument_array);
  }

  // Log

  print::head(p, 40, "GPU rank");

  for (size_t i = 0; i < run_times.size(); i++)
    std::cout << "GPU Elapsed Time : " << run_times[i] << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_pr(argc, argv);
}
