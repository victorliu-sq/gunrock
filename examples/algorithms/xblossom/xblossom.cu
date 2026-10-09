// X-Blossom++ maximum matching on Gunrock (include/gunrock/algorithms/xblossom.hxx).
//
//   xblossom -m <graph.mtx> [-d dataset] [-l block|thread] [-p path_buffer_ratio]
//            [-w warmup] [-r rounds]
//
// Runs warm-up + timed rounds, each a full maximum matching from the empty
// matching on a fresh engine (only FindMaximumMatch is timed, as XB++), checks
// every matching (mutual, along graph edges) and prints the GACGE lines
//   XBConfig: ...                                       once
//   XBRound: index runtime_s matching_size valid        per timed round
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
  std::string graph, dataset = "unknown", lb = "block";
  size_t path_buffer_ratio = 1;
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
    else if (k == "-w") o.warmup = std::stoi(v);
    else if (k == "-r") o.rounds = std::stoi(v);
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); std::exit(2); }
  }
  if (o.graph.empty()) { std::fprintf(stderr, "usage: xblossom -m <graph.mtx> ...\n"); std::exit(2); }
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

template <operators::load_balance_t LB, typename graph_t>
int run(graph_t& G, const options_t& o, const std::vector<int>& row, const std::vector<int>& col) {
  auto context = std::make_shared<gcuda::multi_context_t>(0);
  double total = 0;
  long first = -2;
  bool all_valid = true, all_same = true;
  for (int i = 0; i < o.warmup + o.rounds; i++) {
    xblossom::engine_t<LB, graph_t> engine(G, o.path_buffer_ratio, context);
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

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(o.graph);
  csr_t csr;
  csr.from_coo(coo);
  auto G = graph::build<memory_space_t::device>(properties, csr);

  std::vector<int> row(csr.row_offsets.size()), col(csr.column_indices.size());
  thrust::copy(csr.row_offsets.begin(), csr.row_offsets.end(), row.begin());
  thrust::copy(csr.column_indices.begin(), csr.column_indices.end(), col.begin());
  long max_degree = 0;
  for (size_t v = 0; v + 1 < row.size(); v++) max_degree = std::max<long>(max_degree, row[v + 1] - row[v]);

  std::printf("XBConfig: arm=gunrock variant=xb-pp-gunrock reuse=1 lb=%s dataset=%s nodes=%d edges=%ld "
              "max_degree=%ld warmup=%d rounds=%d path_buffer_ratio=%zu\n",
              o.lb.c_str(), o.dataset.c_str(), G.get_number_of_vertices(),
              static_cast<long>(G.get_number_of_edges()) / 2, max_degree, o.warmup, o.rounds,
              o.path_buffer_ratio);
  std::fflush(stdout);

  if (o.lb == "block") return run<operators::load_balance_t::block_mapped>(G, o, row, col);
  if (o.lb == "thread") return run<operators::load_balance_t::thread_mapped>(G, o, row, col);
  // merge_path is left out: on this Gunrock build it issues no work for these
  // advances (no error, the operator is never called), and none of Gunrock's
  // own algorithms here use it.
  std::fprintf(stderr, "-l must be block or thread\n");
  return 2;
}
