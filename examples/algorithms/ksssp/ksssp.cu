/**
 * ksssp: k independent SSSPs from k distinct sources run together (ksssp.hxx),
 * against the same k sources run one after another with Gunrock's SSSP, and
 * against Gunrock's shared-distance multi_sssp from the same sources.
 *
 *   ksssp -m <weighted.mtx> --src-count <k <= 64> [-n runs] [--validate] [--compare]
 *     --validate   CPU Dijkstra from every source; every distance compared
 *     --compare    also time the k sequential SSSPs and the shared multi_sssp
 *
 * Prints per run
 *   KSSSP: run k ms
 *   KSSSP_SEQ: run k ms            (--compare: the k SSSPs, summed)
 *   KSSSP_SHARED: run k ms         (--compare: one multi_sssp)
 * and KSSSP_VALID: sources errors  (--validate).
 */
#include <gunrock/algorithms/ksssp.hxx>
#include <gunrock/algorithms/sssp.hxx>
#include <gunrock/algorithms/multi_sssp.hxx>
#include "../sssp/sssp_cpu.hxx"
#include <gunrock/io/parameters.hxx>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace gunrock;
using namespace memory;

int main(int argc, char** argv) {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;
  using csr_t = format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --compare is ours; the rest goes to Gunrock's parser
  bool compare = false;
  std::vector<char*> args;
  for (int i = 0; i < argc; i++) {
    if (std::string(argv[i]) == "--compare") compare = true;
    else args.push_back(argv[i]);
  }
  gunrock::io::cli::parameters_t params((int)args.size(), args.data(), "Multi Source Shortest Path");

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto [properties, coo] = mm.load(params.filename);
  csr_t csr;
  csr.from_coo(coo);
  auto G = graph::build<memory_space_t::device>(properties, csr);
  const std::size_t n = G.get_number_of_vertices();

  const int k = params.source_count;
  if (k < 1 || k > 64) {
    std::fprintf(stderr, "--src-count must be 1..64\n");
    return 2;
  }
  std::vector<vertex_t> sources;
  gunrock::io::cli::generate_unique_random_sources(k, &sources, (int)n);

  thrust::device_vector<weight_t> dist(n * k);
  thrust::device_vector<weight_t> one(n);
  thrust::device_vector<vertex_t> pred(n);
  const int runs = params.num_runs > 0 ? params.num_runs : 1;
  std::printf("KSSSPConfig: nodes=%zu edges=%zu k=%d runs=%d\n", n, (std::size_t)G.get_number_of_edges(), k, runs);

  for (int r = 0; r < runs; r++) {
    float ms = ksssp::run(G, sources.data(), k, dist.data().get());
    std::printf("KSSSP: run=%d k=%d ms=%.4f\n", r, k, ms);
    if (compare) {
      float seq = 0;
      for (int s = 0; s < k; s++) {
        vertex_t src = sources[s];
        seq += sssp::run(G, src, one.data().get(), pred.data().get());
      }
      std::printf("KSSSP_SEQ: run=%d k=%d ms=%.4f\n", r, k, seq);
      float shared = multi_sssp::run(G, sources.data(), sources.size(), one.data().get(), pred.data().get());
      std::printf("KSSSP_SHARED: run=%d k=%d ms=%.4f\n", r, k, shared);
    }
    std::fflush(stdout);
  }

  if (params.validate) {
    thrust::host_vector<weight_t> h(dist);
    std::vector<weight_t> ref(n);
    std::vector<vertex_t> rp(n);
    long errors = 0;
    for (int s = 0; s < k; s++) {
      vertex_t src = sources[s];
      sssp_cpu::run<csr_t, vertex_t, edge_t, weight_t>(csr, src, ref.data(), rp.data());
      for (std::size_t v = 0; v < n; v++)
        if (h[v * k + s] != ref[v]) errors++;
    }
    std::printf("KSSSP_VALID: sources=%d errors=%ld\n", k, errors);
    return errors ? 3 : 0;
  }
  return 0;
}
