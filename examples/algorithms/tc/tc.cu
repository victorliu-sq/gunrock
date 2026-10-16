#include <vector>

#include <gunrock/algorithms/tc.hxx>
#include "tc_cpu.hxx"
#include <gunrock/io/gacge_bin.hxx>

#include <cxxopts.hpp>

#include <cub/block/block_scan.cuh>
#include <gunrock/gacge_lb.hxx>
#include <cub/device/device_segmented_sort.cuh>
#include <thrust/binary_search.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <type_traits>

using namespace gunrock;
using namespace memory;

// GACGE_TC_RELABEL: gap (default since 2026-10-15) counts as the GAP benchmark's
// tc (Hybrid): when WorthRelabelling says the degrees are skewed, the graph is
// first relabelled by decreasing degree (RelabelByDegree, here on the device),
// inside the timed run as in GAP; off counts on the graph as loaded.

// GAP's UniDist (gapbs src/generator.h, BSD), verbatim, so that the degree
// sample of WorthRelabelling picks the same vertices as GAP's.
template <typename NodeID_, typename rng_t_,
          typename uNodeID_ = typename std::make_unsigned<NodeID_>::type>
class UniDist {
 public:
  UniDist(NodeID_ max_value, rng_t_& rng) : rng_(rng) {
    no_mod_ = rng_.max() == static_cast<uNodeID_>(max_value);
    mod_ = max_value + 1;
    uNodeID_ remainder_sub_1 = rng_.max() % mod_;
    if (remainder_sub_1 == mod_ - 1)
      cutoff_ = 0;
    else
      cutoff_ = rng_.max() - remainder_sub_1;
  }
  NodeID_ operator()() {
    uNodeID_ rand_num = rng_();
    if (no_mod_)
      return rand_num;
    if (cutoff_ != 0) {
      while (rand_num >= cutoff_)
        rand_num = rng_();
    }
    return rand_num % mod_;
  }

 private:
  rng_t_& rng_;
  bool no_mod_;
  uNodeID_ mod_;
  uNodeID_ cutoff_;
};

// GAP's WorthRelabelling (gapbs src/tc.cc) on the row offsets of a symmetric
// graph with m directed entries (GAP's num_edges() is m / 2 for it)
template <typename edge_t>
bool gap_worth_relabelling(const std::vector<edge_t>& off) {
  const int64_t n = (int64_t)off.size() - 1;
  const int64_t num_edges = ((int64_t)off[n] - (int64_t)off[0]) / 2;
  const int64_t average_degree = num_edges / n;
  if (average_degree < 10)
    return false;
  std::mt19937_64 rng(27491095);  // GAP's kRandSeed
  UniDist<int32_t, std::mt19937_64> udist((int32_t)(n - 1), rng);
  auto degree = [&](int64_t v) { return (int64_t)off[v + 1] - (int64_t)off[v]; };
  const int64_t num_samples = std::min<int64_t>(1000, n);
  int64_t sample_total = 0;
  std::vector<int64_t> samples(num_samples);
  for (int64_t trial = 0; trial < num_samples; trial++) {
    int32_t source;
    do {
      source = udist();
    } while (degree(source) == 0);  // SourcePicker::PickNext
    samples[trial] = degree(source);
    sample_total += samples[trial];
  }
  std::sort(samples.begin(), samples.end());
  const double sample_average = static_cast<double>(sample_total) / num_samples;
  const double sample_median = samples[num_samples / 2];
  return sample_average / 1.3 > sample_median;
}

// every vertex of the input with all its edges (RelabelByDegree's mapping)
template <typename vertex_t, typename edge_t>
struct csr_tiles_t {
  using vertex_type = vertex_t;
  const edge_t* off;
  __device__ void item(long long i, vertex_t& u, long long& start) const {
    u = (vertex_t)i;
    start = off[i];
  }
  __device__ long long len(long long i) const { return (long long)(off[i + 1] - off[i]); }
};

// RelabelByDegree's list mapping (GAP: each vertex writes its neighbors' new
// ids into its new list), the edges spread by the schedule GACGE_LB
// (gacge_lb.hxx); merge-path's prefix sum is the row offsets
template <int Gs, typename vertex_t, typename edge_t>
__global__ void __launch_bounds__(gacge::lb::kBlock) relabel_map_kernel(const edge_t* off, const vertex_t* col, const edge_t* noff,
                                                          const vertex_t* nid, vertex_t* mc, vertex_t n,
                                                          gacge::lb::merge_args_t m) {
  csr_tiles_t<vertex_t, edge_t> tiles{off};
  gacge::lb::for_edges<Gs>(tiles, (long long)n, m, [&](long long, vertex_t u, long long e) {
    mc[noff[nid[u]] + (e - off[u])] = nid[col[e]];
    return false;
  });
}

// GAP's RelabelByDegree on the device: vertices sorted by (degree, id),
// largest first, take ids 0, 1, ...; every list is mapped and sorted again.
template <typename csr_t>
void relabel_by_degree(const csr_t& in, csr_t& out) {
  using vertex_t = typename csr_t::index_type;
  using edge_t = typename csr_t::offset_type;
  const vertex_t n = in.number_of_rows;
  const edge_t m = in.number_of_nonzeros;
  const edge_t* off = thrust::raw_pointer_cast(in.row_offsets.data());
  const vertex_t* col = thrust::raw_pointer_cast(in.column_indices.data());
  thrust::device_vector<unsigned long long> key(n);
  thrust::device_vector<vertex_t> order(n), new_ids(n);
  thrust::transform(thrust::counting_iterator<vertex_t>(0), thrust::counting_iterator<vertex_t>(n),
                    key.begin(), [off] __device__(vertex_t v) {
                      return ((unsigned long long)(off[v + 1] - off[v]) << 32) | (unsigned long long)v;
                    });
  thrust::sequence(order.begin(), order.end());
  thrust::sort_by_key(key.begin(), key.end(), order.begin(), thrust::greater<unsigned long long>());
  thrust::scatter(thrust::counting_iterator<vertex_t>(0), thrust::counting_iterator<vertex_t>(n),
                  order.begin(), new_ids.begin());
  out.number_of_rows = n;
  out.number_of_columns = n;
  out.number_of_nonzeros = m;
  out.row_offsets.resize(n + 1);
  const vertex_t* ord = thrust::raw_pointer_cast(order.data());
  thrust::transform(thrust::counting_iterator<vertex_t>(0), thrust::counting_iterator<vertex_t>(n + 1),
                    out.row_offsets.begin(), [off, ord, n] __device__(vertex_t k) -> edge_t {
                      return k < n ? off[ord[k] + 1] - off[ord[k]] : 0;
                    });
  thrust::exclusive_scan(out.row_offsets.begin(), out.row_offsets.end(), out.row_offsets.begin());
  thrust::device_vector<vertex_t> mapped(m);
  out.column_indices.resize(m);
  const edge_t* noff = thrust::raw_pointer_cast(out.row_offsets.data());
  const vertex_t* nid = thrust::raw_pointer_cast(new_ids.data());
  vertex_t* mc = thrust::raw_pointer_cast(mapped.data());
  gacge::lb::dispatch(gacge::lb::mode(gacge::lb::mode_t::merge), [&](auto tag) {
    constexpr int Gs = decltype(tag)::value;
    if constexpr (Gs == 0) {  // merge-path over the row offsets (int: m < 2^31)
      static gacge::lb::merge_t merge;  // its buffers (allocated in the first, warm-up, run)
      const gacge::lb::merge_args_t args = merge.plan(reinterpret_cast<const int*>(off), (long long)n, 0);
      if (merge.tiles > 0)
        relabel_map_kernel<0><<<(unsigned int)merge.tiles, gacge::lb::kBlock>>>(off, col, noff, nid, mc, n, args);
    } else {
      relabel_map_kernel<Gs><<<gacge::lb::grid(n), gacge::lb::kBlock>>>(off, col, noff, nid, mc, n,
                                                                       gacge::lb::merge_args_t{});
    }
  });
  std::size_t temp_bytes = 0;
  vertex_t* oc = thrust::raw_pointer_cast(out.column_indices.data());
  cub::DeviceSegmentedSort::SortKeys(nullptr, temp_bytes, mc, oc, (int64_t)m, (int64_t)n, noff, noff + 1);
  thrust::device_vector<unsigned char> temp(temp_bytes);
  cub::DeviceSegmentedSort::SortKeys(thrust::raw_pointer_cast(temp.data()), temp_bytes, mc, oc, (int64_t)m,
                                     (int64_t)n, noff, noff + 1);
}

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

  if (gacge_bin::is_bin(params.filename.c_str())) {
    // GACGE binary CSR (gunrock/io/gacge_bin.hxx): sorted lists, the canonical
    // graph Ligra reads too; no edge weights (they alias the column indices)
    gacge_bin::load_device_csr(params.filename, csr, gacge_bin::values_t::none);
  } else if (util::is_market(params.filename)) {
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
  if (gacge_bin::is_bin(params.filename.c_str()))
    G.alias_values(reinterpret_cast<weight_t*>(thrust::raw_pointer_cast(csr.column_indices.data())));

  // --
  // Params and memory allocation

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<count_t> triangles_count(n_vertices, 0);

  // --
  // GPU Run

  // GACGE_TC_RELABEL (default gap): relabel by degree when GAP would
  const char* rl = std::getenv("GACGE_TC_RELABEL");
  bool relabel = !params.all_edges && (rl == nullptr || std::string(rl) == "gap");
  if (relabel) {
    std::vector<edge_t> h_off(n_vertices + 1);
    thrust::copy(csr.row_offsets.begin(), csr.row_offsets.end(), h_off.begin());
    relabel = gap_worth_relabelling(h_off);
  }
  std::cout << "TCRelabel: " << (relabel ? 1 : 0) << std::endl;

  // -n runs, each from zeroed per-vertex counts (tc::run adds into them); a
  // relabelled run times the relabelling too (GAP's Hybrid does)
  std::size_t total_triangles = 0;
  std::vector<float> run_times;
  for (int i = 0; i < params.num_runs; i++) {
    gacge::profile::SetTrial(i);
    thrust::fill(triangles_count.begin(), triangles_count.end(), 0);
    total_triangles = 0;
    auto ctx = std::shared_ptr<gcuda::multi_context_t>(new gcuda::multi_context_t(0));
    if (relabel) {
      cudaEvent_t t0, t1;
      cudaEventCreate(&t0);
      cudaEventCreate(&t1);
      gacge::profile::Begin();  // the relabelling is timed (and profiled) with the count
      cudaEventRecord(t0);
      csr_t rcsr;
      relabel_by_degree(csr, rcsr);
      auto RG = graph::build<memory_space_t::device>(properties, rcsr);
      RG.alias_values(reinterpret_cast<weight_t*>(thrust::raw_pointer_cast(rcsr.column_indices.data())));
      cudaEventRecord(t1);
      cudaEventSynchronize(t1);
      float relabel_ms = 0;
      cudaEventElapsedTime(&relabel_ms, t0, t1);
      const float count_ms = tc::run(RG, params.reduce_all_triangles, triangles_count.data().get(),
                                     &total_triangles, ctx, !params.all_edges);
      run_times.push_back(relabel_ms + count_ms);
      std::cout << "TCRelabelTime: relabel_ms=" << relabel_ms << " count_ms=" << count_ms << std::endl;
      cudaEventDestroy(t0);
      cudaEventDestroy(t1);
    } else {
      run_times.push_back(tc::run(G, params.reduce_all_triangles, triangles_count.data().get(),
                                  &total_triangles, ctx, !params.all_edges));
    }
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
