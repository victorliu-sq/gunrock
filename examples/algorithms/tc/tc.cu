#include <vector>

#include <gunrock/algorithms/tc.hxx>
#include "tc_cpu.hxx"

#include <cxxopts.hpp>

using namespace gunrock;
using namespace memory;

struct parameters_t {
  std::string filename;
  cxxopts::Options options;
  bool validate;
  bool reduce_all_triangles;
  int num_runs;
  bool all_edges;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv)
      : options(argv[0], "Traingle Counting example") {
    // Add command line options
    options.add_options()("help", "Print help")(
        "validate", "CPU validation",
        cxxopts::value<bool>()->default_value("false"))(
        "m,market", "Matrix file", cxxopts::value<std::string>())(
        "r,reduce",
        "Compute a single triangle count for the entire graph (default = "
        "false)",
        cxxopts::value<bool>()->default_value("false"))(
        "n,num_runs", "Number of runs",
        cxxopts::value<int>()->default_value("1"))(
        "all-edges",
        "Gunrock's own counting (each triangle at its three edges, an atomic "
        "per triangle) instead of the oriented one (as Ligra's Triangle)",
        cxxopts::value<bool>()->default_value("false"));

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help") || (result.count("market") == 0)) {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }
    filename = result["market"].as<std::string>();
    validate = result["validate"].as<bool>();
    reduce_all_triangles = result["reduce"].as<bool>();
    num_runs = result["num_runs"].as<int>();
    all_edges = result["all-edges"].as<bool>();
  }
};

void test_tc(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = uint32_t;
  using edge_t = uint32_t;
  using weight_t = float;
  using count_t = vertex_t;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr;

  // --
  // IO
  parameters_t params(num_arguments, argument_array);
  gunrock::graph::graph_properties_t properties =
      gunrock::graph::graph_properties_t();

  if (util::is_market(params.filename)) {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    auto [properties, coo] = mm.load(params.filename);
    if (!properties.symmetric) {
      std::cerr << "Error: input matrix must be symmetric" << std::endl;
      exit(1);
    }
    csr.from_coo(coo);
  } else if (util::is_binary_csr(params.filename)) {
    csr.read_binary(params.filename);
  } else {
    std::cerr << "Unknown file format: " << params.filename << std::endl;
    exit(1);
  }

  // --
  // Build graph

  auto G = graph::build<memory_space_t::device>(properties, csr);

  // --
  // Params and memory allocation

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<count_t> triangles_count(n_vertices, 0);

  // --
  // GPU Run

  // -n runs, each from zeroed per-vertex counts (tc::run adds into them)
  std::size_t total_triangles = 0;
  std::vector<float> run_times;
  for (int i = 0; i < params.num_runs; i++) {
    thrust::fill(triangles_count.begin(), triangles_count.end(), 0);
    total_triangles = 0;
    run_times.push_back(tc::run(
        G, params.reduce_all_triangles, triangles_count.data().get(),
        &total_triangles,
        std::shared_ptr<gcuda::multi_context_t>(new gcuda::multi_context_t(0)),
        !params.all_edges));
    std::cout << "TCResult: total=" << total_triangles
              << " oriented=" << !params.all_edges << std::endl;
  }

  // --
  // Log

  print::head(triangles_count, 40, "Per-vertex triangle count");
  if (params.reduce_all_triangles) {
    std::cout << "Total Graph Traingles : " << total_triangles << std::endl;
  }
  for (float t : run_times)
    std::cout << "GPU Elapsed Time : " << t << " (ms)" << std::endl;

  // --
  // CPU validation
  if (params.validate) {
    std::vector<count_t> reference_triangles_count(n_vertices, 0);
    std::size_t reference_total_triangles = 0;

    float cpu_elapsed =
        tc_cpu::run(csr, reference_triangles_count, reference_total_triangles);
    uint32_t n_errors = 0;
    if (total_triangles != reference_total_triangles) {
      std::cout << "Error: Total TC mismatch: " << total_triangles
                << "! = " << reference_total_triangles << std::endl;
      n_errors++;
    }
    n_errors += util::compare(
        triangles_count.data().get(), reference_triangles_count.data(),
        n_vertices, [](const auto x, const auto y) { return x != y; }, true);
    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
    std::cout << "Number of errors : " << n_errors << std::endl;
  }
}

int main(int argc, char** argv) {
  test_tc(argc, argv);
}
