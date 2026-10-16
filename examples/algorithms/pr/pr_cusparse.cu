// pr_cusparse: PageRank's pull as cuSPARSE's SpMV, the GPU reference for the
// GACGE PageRank (GAP's pr_spmv step for step: contrib = alpha * p / deg, one
// SpMV over the CSR (float, ones as values), then base + sum and the error;
// a fixed iteration count, GACGE_PR_ITERS, default 20). The timed region is
// the iterations, as pr.cu's: cusparseSpMV with the buffer allocated once and
// the preprocess done outside it. Output as pr.cu: PRResult and GPU Elapsed
// Time per run.
//
//   pr_cusparse -m <graph.bin> [-n runs]
#include <gunrock/io/gacge_bin.hxx>
#include <gunrock/formats/formats.hxx>
#include <gunrock/memory.hxx>

#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#define CK(x)                                                                                   \
  do {                                                                                          \
    auto s_ = (x);                                                                              \
    if (s_ != 0) {                                                                              \
      std::fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #x, (int)s_);         \
      std::exit(1);                                                                             \
    }                                                                                           \
  } while (0)

__global__ void contrib_kernel(const float* p, const int* off, float* c, int n, float alpha) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < n) {
    const int d = off[v + 1] - off[v];
    c[v] = d ? alpha * p[v] / d : 0.f;
  }
}

__global__ void finish_kernel(const float* sums, float* p, int n, float base, float* error) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < n) {
    const float s = base + sums[v];
    atomicAdd(error, fabsf(s - p[v]));
    p[v] = s;
  }
}

int main(int argc, char** argv) {
  std::string graph;
  int runs = 1;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "-m") graph = argv[i + 1];
    else if (k == "-n") runs = std::atoi(argv[i + 1]);
  }
  if (graph.empty()) {
    std::fprintf(stderr, "usage: pr_cusparse -m <graph.bin> [-n runs]\n");
    return 2;
  }
  const char* it = std::getenv("GACGE_PR_ITERS");
  const int iters = it ? std::atoi(it) : 20;
  // GACGE_CUSPARSE_ALG: default (cuSPARSE's choice), alg1 or alg2 (CSR merge-based)
  const char* ae = std::getenv("GACGE_CUSPARSE_ALG");
  const std::string alg_name = ae ? ae : "default";
  const cusparseSpMVAlg_t alg = alg_name == "alg1"   ? CUSPARSE_SPMV_CSR_ALG1
                                : alg_name == "alg2" ? CUSPARSE_SPMV_CSR_ALG2
                                                     : CUSPARSE_SPMV_ALG_DEFAULT;
  const float alpha = 0.85f;

  using namespace gunrock;
  using csr_t = format::csr_t<memory::memory_space_t::host, int, int, float>;
  csr_t h;
  gacge_bin::load_host_csr(graph, h, gacge_bin::values_t::ones);
  const int n = h.number_of_rows, m = h.number_of_nonzeros;
  thrust::device_vector<int> off(h.row_offsets.begin(), h.row_offsets.end());
  thrust::device_vector<int> col(h.column_indices.begin(), h.column_indices.end());
  thrust::device_vector<float> val(h.nonzero_values.begin(), h.nonzero_values.end());
  thrust::device_vector<float> p(n), contrib(n), sums(n), error(1);

  cusparseHandle_t handle;
  CK(cusparseCreate(&handle));
  cusparseSpMatDescr_t A;
  cusparseDnVecDescr_t x, y;
  CK(cusparseCreateCsr(&A, n, n, m, off.data().get(), col.data().get(), val.data().get(), CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  CK(cusparseCreateDnVec(&x, n, contrib.data().get(), CUDA_R_32F));
  CK(cusparseCreateDnVec(&y, n, sums.data().get(), CUDA_R_32F));
  const float one = 1.f, zero = 0.f;
  size_t bytes = 0;
  CK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, x, &zero, y, CUDA_R_32F,
                             alg, &bytes));
  void* buffer = nullptr;
  cudaMalloc(&buffer, bytes ? bytes : 1);
#if CUSPARSE_VERSION >= 12300
  CK(cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, x, &zero, y, CUDA_R_32F,
                             alg, buffer));
#endif

  const int blocks = (n + 255) / 256;
  const float base = (1.f - alpha) / n;
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  for (int r = 0; r < runs; r++) {
    thrust::fill(p.begin(), p.end(), 1.f / n);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int k = 0; k < iters; k++) {
      contrib_kernel<<<blocks, 256>>>(p.data().get(), off.data().get(), contrib.data().get(), n, alpha);
      CK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, x, &zero, y, CUDA_R_32F,
                      alg, buffer));
      cudaMemsetAsync(error.data().get(), 0, sizeof(float));
      finish_kernel<<<blocks, 256>>>(sums.data().get(), p.data().get(), n, base, error.data().get());
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    const double rank_sum = thrust::reduce(p.begin(), p.end(), 0.0);
    std::printf("PRResult: iterations=%d rank_sum=%.9f rank0=%.9e alg=%s\n", iters, rank_sum, (double)p[0], alg_name.c_str());
    std::printf("GPU Elapsed Time : %g (ms)\n", ms);
  }
  return 0;
}
