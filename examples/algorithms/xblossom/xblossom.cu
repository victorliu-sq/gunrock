// X-Blossom++ maximum matching on Gunrock (include/gunrock/algorithms/xblossom.hxx).
//
//   xblossom -m <graph.mtx> [-d dataset] [-l block|thread] [-p path_buffer_ratio]
//            [-M mat|nomat] [-T compact|segment] [-B onthefly|rootpath] [-q segment_buffer_ratio]
//            [-o odd_node_ratio] [-w warmup] [-r rounds]
//
// -B (segment table, mat only) picks the blossom-base identification: onthefly
// (default; the two walks of Sec. 4.B.1) or rootpath (the original X-Blossom's:
// both endpoint-to-root paths materialised and scanned from the root).
//
// -T selects the path table: compact (default; three vertex ids per node) or
// segment (the explicit segments of the original X-Blossom++). -p sizes the
// buffers a run materialises: the cycle buffer under mat, and under -T segment
// also the segment buffer (ratio x n nodes each) unless -q sizes that one apart.
//
// Runs warm-up + timed rounds, each a full maximum matching from the empty
// matching on a fresh engine (only FindMaximumMatch is timed, as XB++), checks
// every matching (mutual, along graph edges) and prints the GACGE lines
//   XBConfig: ...                                       once
//   XBRound: index runtime_s matching_size valid        per timed round
//   XBPhases: index aug_path_s expand_s blossom_s steps   per timed round
//   XBMemory: index paths_used blossom_buffer_peak blossom_list_peak buffer_capacity list_capacity
//             transform_tasks_peak table_bytes flip_stack_peak
//   XBResult: status matching_size valid rounds mean_runtime_s
#include <gunrock/algorithms/xblossom.hxx>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace gunrock;
using namespace memory;

namespace {

struct options_t {
  std::string graph, dataset = "unknown", lb = "block", blossom = "mat", table = "compact", base = "onthefly";
  size_t path_buffer_ratio = 1, segment_buffer_ratio = 0; // 0: as -p
  double odd_node_ratio = 30; // blossom-list entries per node; may be fractional
  int warmup = 1, rounds = 10;
};

options_t parse(int argc, char** argv) {
  options_t o;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "-m") o.graph = v;
    else if (k == "-d") o.dataset = v;
    else if (k == "-l") o.lb = v;
    else if (k == "-p") o.path_buffer_ratio = std::stoul(v);
    else if (k == "-M") o.blossom = v;
    else if (k == "-T") o.table = v;
    else if (k == "-B") o.base = v;
    else if (k == "-q") o.segment_buffer_ratio = std::stoul(v);
    else if (k == "-o") o.odd_node_ratio = std::stod(v);
    else if (k == "-w") o.warmup = std::stoi(v);
    else if (k == "-r") o.rounds = std::stoi(v);
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); std::exit(2); }
  }
  if (o.graph.empty()) { std::fprintf(stderr, "usage: xblossom -m <graph.mtx> ...\n"); std::exit(2); }
  if (o.blossom == "tuple") { o.blossom = "nomat"; o.table = "compact"; } // the earlier spelling of nomat-compact
  if (o.blossom != "mat" && o.blossom != "nomat") {
    std::fprintf(stderr, "-M must be mat or nomat\n");
    std::exit(2);
  }
  if (o.table != "compact" && o.table != "segment") {
    std::fprintf(stderr, "-T must be compact or segment\n");
    std::exit(2);
  }
  if (o.base != "onthefly" && o.base != "rootpath") {
    std::fprintf(stderr, "-B must be onthefly or rootpath\n");
    std::exit(2);
  }
  if (o.base == "rootpath" && (o.table != "segment" || o.blossom != "mat")) {
    std::fprintf(stderr, "-B rootpath needs -T segment -M mat\n");
    std::exit(2);
  }
  return o;
}

// Matching size if mate (exposed = n) is a matching of the CSR graph, else -1.
long check_matching(const std::vector<int>& row, const std::vector<int>& col, const std::vector<index_t>& mate) {
  const long n = static_cast<long>(row.size()) - 1;
  long matched = 0;
  for (long v = 0; v < n; v++) {
    long u = mate[v];
    if (u == n) continue;
    if (u < 0 || u > n || mate[u] != v) return -1;
    bool edge = false;
    for (int j = row[v]; j < row[v + 1]; j++)
      if (col[j] == u) { edge = true; break; }
    if (!edge) return -1;
    matched++;
  }
  return matched / 2;
}

// path_table_t: the compact path table (-T compact, the default) or the segment path table (-T segment)
template <operators::load_balance_t LB, typename path_table_t, typename graph_t>
int run(graph_t& G, const options_t& o, const std::vector<int>& row, const std::vector<int>& col) {
  auto context = std::make_shared<gcuda::multi_context_t>(0);
  double total = 0;
  long first = -2;
  bool all_valid = true, all_same = true;
  for (int i = 0; i < o.warmup + o.rounds; i++) {
    xblossom::engine_t<LB, graph_t, path_table_t> engine(G, o.path_buffer_ratio,
                                                         o.segment_buffer_ratio ? o.segment_buffer_ratio : o.path_buffer_ratio,
                                                         o.odd_node_ratio,
                                                         o.blossom != "mat", context, o.base == "rootpath");
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<index_t> mate = engine.FindMaximumMatch();
    double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    if (i < o.warmup) continue;
    long size = check_matching(row, col, mate);
    bool valid = size >= 0;
    all_valid = all_valid && valid;
    if (first == -2) first = size;
    else if (size != first) all_same = false;
    total += secs;
    std::printf("XBRound: index=%d runtime_s=%.9f matching_size=%ld valid=%d\n", i - o.warmup, secs, size,
                valid ? 1 : 0);
    double aug_s, expand_s, blossom_s;
    engine.PhaseSeconds(aug_s, expand_s, blossom_s);
    std::printf("XBPhases: index=%d aug_path_s=%.9f expand_s=%.9f blossom_s=%.9f steps=%zu\n", i - o.warmup, aug_s,
                expand_s, blossom_s, engine.Steps());
    std::printf("XBMemory: index=%d paths_used=%zu blossom_buffer_peak=%zu blossom_list_peak=%zu "
                "buffer_capacity=%zu list_capacity=%zu transform_tasks_peak=%zu table_bytes=%zu flip_stack_peak=%d\n",
                i - o.warmup, engine.PathsUsed(), engine.PeakBlossomBufferNodes(), engine.PeakBlossomListEntries(),
                engine.BufferCapacity(), engine.BlossomListCapacity(), engine.PeakTransformTasks(),
                engine.PathTableFixedBytes(), engine.FlipStackPeak());
    std::fflush(stdout);
  }
  const char* status = !all_valid ? "invalid_matching" : (!all_same ? "matching_size_varies" : "ok");
  std::printf("XBResult: status=%s matching_size=%ld valid=%d rounds=%d mean_runtime_s=%.9f\n", status, first,
              all_valid ? 1 : 0, o.rounds, o.rounds ? total / o.rounds : 0.0);
  return all_valid ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
  options_t o = parse(argc, argv);

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;
  using csr_t = format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // The CSR is built on the host and only its offsets and column indices go to
  // the device: X-Blossom never reads an edge weight, and the weights Gunrock's
  // advances load per edge are aliased to the column indices (on Webbase, 1.7
  // billion arcs, that is 6.8 GB of device memory the weights would take).
  using host_csr_t = format::csr_t<memory_space_t::host, vertex_t, edge_t, weight_t>;
  graph::graph_properties_t properties;
  host_csr_t hcsr;
  {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    auto [props, coo] = mm.load(o.graph);
    properties = props;
    hcsr.from_coo(coo);
  }  // the COO is freed here
  csr_t csr;
  csr.number_of_rows = hcsr.number_of_rows;
  csr.number_of_columns = hcsr.number_of_columns;
  csr.number_of_nonzeros = hcsr.number_of_nonzeros;
  csr.row_offsets = hcsr.row_offsets;
  csr.column_indices = hcsr.column_indices;
  auto G = graph::build<memory_space_t::device>(properties, csr);
  G.alias_values(reinterpret_cast<weight_t*>(thrust::raw_pointer_cast(csr.column_indices.data())));
  static_assert(sizeof(weight_t) == sizeof(vertex_t), "the weights alias the column indices");

  std::vector<int> row(hcsr.row_offsets.begin(), hcsr.row_offsets.end());
  std::vector<int> col(hcsr.column_indices.begin(), hcsr.column_indices.end());
  hcsr = host_csr_t();  // the host copies above are what the matching check reads
  long max_degree = 0;
  for (size_t v = 0; v + 1 < row.size(); v++) max_degree = std::max<long>(max_degree, row[v + 1] - row[v]);

  std::printf("XBConfig: arm=gunrock variant=xb-pp-gunrock reuse=1 lb=%s blossom=%s path_table=%s base=%s dataset=%s nodes=%d edges=%ld "
              "max_degree=%ld warmup=%d rounds=%d path_buffer_ratio=%zu segment_buffer_ratio=%zu odd_node_ratio=%g\n",
              o.lb.c_str(), o.blossom.c_str(), o.table.c_str(), o.base.c_str(), o.dataset.c_str(), G.get_number_of_vertices(),
              static_cast<long>(G.get_number_of_edges()) / 2, max_degree, o.warmup, o.rounds,
              o.path_buffer_ratio, o.segment_buffer_ratio ? o.segment_buffer_ratio : o.path_buffer_ratio, o.odd_node_ratio);
  std::fflush(stdout);

  using table_t = zblossom::dev::PathTable;
  using tuple_t = zblossom::dev::TuplePathTable;
  const bool tuple = o.table == "compact";
  if (o.lb == "block")
    return tuple ? run<operators::load_balance_t::block_mapped, tuple_t>(G, o, row, col)
                 : run<operators::load_balance_t::block_mapped, table_t>(G, o, row, col);
  if (o.lb == "thread")
    return tuple ? run<operators::load_balance_t::thread_mapped, tuple_t>(G, o, row, col)
                 : run<operators::load_balance_t::thread_mapped, table_t>(G, o, row, col);
  // merge_path is left out: on this Gunrock build it issues no work for these
  // advances (no error, the operator is never called), and none of Gunrock's
  // own algorithms here use it.
  std::fprintf(stderr, "-l must be block or thread\n");
  return 2;
}
