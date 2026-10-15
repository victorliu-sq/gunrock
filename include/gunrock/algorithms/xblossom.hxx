/**
 * @file xblossom.hxx
 * @brief X-Blossom++ (maximum matching in general graphs) on Gunrock.
 *
 * The algorithm is XB++'s (X-Blossom repo, src/xblsm_gpu/xb_gpu_engine.cu,
 * XB_REUSE=1): search phases grow alternating trees from the exposed nodes;
 * each iteration of a phase runs
 *   1. augment  -- an edge between even nodes of two trees is an augmenting
 *                  path; lock both trees and flip it in the matching
 *   2. expand   -- a neighbour w in no tree joins as an odd node and its mate
 *                  as an even node
 *   3. blossom  -- an edge between even nodes of one tree closes a blossom
 *                  whose odd nodes then become even
 * until a path is found (next phase) or no even node is left (maximum).
 * Trees whose root is still exposed are kept across phases.
 *
 * What Gunrock does here: each step's frontier (the even nodes of the
 * X-Blossom node queue) goes into a gunrock frontier_t and the three steps are
 * advance operators with load balancing LB (block_mapped = Gunrock's default,
 * or thread_mapped); the two per-vertex initialisation passes are
 * parallel_for. What stays X-Blossom's own (vendored, ./xblossom/): the
 * device path table, the even-node queue, the per-tree / per-match /
 * per-odd-node locks and the blossom transform. One semantic difference from
 * the hand-written XB++ node-level mode: the advance lambda sees one edge at a
 * time, so there is no "skip this node's remaining edges".
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>

#include <cub/cub.cuh>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

// X-Blossom's device structures (vendored; the example adds
// include/gunrock/algorithms/xblossom to the include path).
#include "utils/utils_gpu.h"
#include "utils/dev_array.h"
#include "utils/dev_value.h"
#include "utils/dev_list.h"
#include "utils/stream.h"
#include "utils/launcher.h"
#include "utils/cuda_algo.h"
#include "enode_queue/enode_queue.h"
#include "path_table/path_table.h"
#include "path_table/path_table_tuple.h"
#include <type_traits>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

namespace gunrock {
namespace xblossom {

// Size of the blossom lists, per node (XB++'s ODD_NODE_RATIO).
constexpr size_t kOddNodeRatio = 30; // default capacity of the blossom lists, per node

// nomat: walk even node s's chain down to the blossom base and claim every
// odd node still odd (per-node lock); each claim is one transform task
// (node, s, o, base), the node becoming even with the path around the cycle
// through o. Same as X-Blossom++'s ClaimOddNodesOnChain (xb_gpu_engine.cu).
template <typename path_table_t, typename flags_t, typename list_t, typename array_t>
__device__ __forceinline__ void ClaimOddNodesOnChain(path_table_t& path_table, flags_t& is_even,
                                                     flags_t& atomic_locks_odd_nodes, list_t& task_node,
                                                     array_t& task_s, array_t& task_o, array_t& task_base,
                                                     index_t s, index_t o, index_t base) {
  path_table.ForEachRecordedOddOnChain(s, base, [&](index_t n) {
    if (!is_even[n] && atomicExch(&atomic_locks_odd_nodes[n], DFLAG_TRUE) == DFLAG_FALSE) {
      const offset_t t = task_node.Append(n);
      if (static_cast<size_t>(t) >= task_node.capacity()) return; // the list is full; the host stops the run
      task_s[t] = s;
      task_o[t] = o;
      task_base[t] = base;
    }
  });
}

// path_table_t: zblossom::dev::TuplePathTable (the compact path table, three
// vertex ids per node; the default) or zblossom::dev::PathTable (the segment
// path table). Both run mat and nomat; mat on the compact table materialises
// the cycles from the records (TuplePathTableView::MaterializeCycle).
template <operators::load_balance_t LB, typename graph_t, typename path_table_t = zblossom::dev::PathTable>
class engine_t {
  static constexpr bool kTuple = std::is_same<path_table_t, zblossom::dev::TuplePathTable>::value;
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
  using frontier_t = frontier::frontier_t<vertex_t, edge_t, frontier::frontier_kind_t::vertex_frontier>;

 public:
  // nomat: no blossom buffer, odd nodes claimed while the closing edge walks
  // its chains and made even from the path table (X-Blossom++'s XB_BLOSSOM_MAT 1);
  // odd_node_ratio sizes the blossom lists (records under mat, tasks under nomat),
  // in entries per node; it may be fractional (nomat needs well under one per node).
  // path_table_buffer_ratio sizes the cycle buffer (mat), segment_buffer_ratio
  // the segment table's buffer (-T segment); both x n nodes.
  // base_rootpath (segment table, mat only): find each blossom's base as the
  // original X-Blossom does, by materialising both endpoint-to-root paths and
  // scanning them from the root; otherwise on the fly, by the two anchor walks.
  engine_t(graph_t& G, size_t path_table_buffer_ratio, size_t segment_buffer_ratio, double odd_node_ratio, bool nomat,
           std::shared_ptr<gcuda::multi_context_t> context, bool base_rootpath = false)
      : G_(G),
        context_(context),
        nomat_(nomat),
        base_rootpath_(base_rootpath),
        nnodes_(G.get_number_of_vertices()),
        dev_matching_(nnodes_, nnodes_),
        path_table_(nnodes_, path_table_buffer_ratio, segment_buffer_ratio, !nomat), // the compact table keeps a cycle buffer under mat only
        is_even_(nnodes_),
        tree_roots_(nnodes_),
        atomic_locks_tree_(nnodes_),
        atomic_locks_match_(nnodes_),
        atomic_locks_odd_nodes_(nnodes_),
        enode_queue_(nnodes_),
        blossom_offset_list_(ListSize(odd_node_ratio, nnodes_)),
        num_nodes_in_blossom_list_(ListSize(odd_node_ratio, nnodes_)),
        num_odd_nodes_vside_in_blossom_list_(ListSize(odd_node_ratio, nnodes_)),
        num_odd_nodes_in_blossom_list_(ListSize(odd_node_ratio, nnodes_)),
        num_odd_nodes_in_blossom_psum_(nomat ? 1 : ListSize(odd_node_ratio, nnodes_)), // nomat scans nothing,
        input_frontier_(nnodes_),
        output_frontier_(1) {
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, num_odd_nodes_in_blossom_list_.begin(),
                                             num_odd_nodes_in_blossom_psum_.begin(),
                                             static_cast<int>(num_odd_nodes_in_blossom_psum_.size()),
                                             cuda_stream_.cuda_stream()));
    scan_temp_.resize(bytes);
    scan_temp_bytes_ = bytes;
  }

  // entries of a blossom list: ratio x n, rounded up, plus the one a prefix sum needs
  static size_t ListSize(double ratio, size_t n) { return static_cast<size_t>(std::ceil(ratio * static_cast<double>(n))) + 1; }

  // Maximum matching from the empty matching; mate[v] == nnodes for exposed v.
  // Maximum matching from the empty matching: PrepareMatch, Match, Mate. The
  // driver times Match only (GACGE): the empty matching is set before, and the
  // mates copied back after, its timer, as Gunrock's enactors time their loop
  // after the problem's reset.
  std::vector<index_t> FindMaximumMatch() {
    PrepareMatch();
    Match();
    return Mate();
  }

  // Untimed: the statistics' and counters' reset and the empty matching.
  void PrepareMatch() {
    // GACGE traversal statistics (measure/traversal-char), off unless
    // GACGE_TRAVERSAL_STATS is set: one XBTraversal line per matching, and one
    // XBStep line per advance for the matching numbered GACGE_TRAVERSAL_TRACE
    // (0-based, default 1: the first after one warm-up).
    static int run_index = -1;
    ++run_index;
    ts_on_ = std::getenv("GACGE_TRAVERSAL_STATS") != nullptr;
    const char* trace_env = std::getenv("GACGE_TRAVERSAL_TRACE");
    ts_trace_ = ts_on_ && run_index == (trace_env ? std::atoi(trace_env) : 1);
    ts_run_ = run_index;
    for (int k = 0; k < 3; k++) ts_steps_[k] = ts_v_[k] = ts_e_[k] = 0;
    ts_max_e_ = ts_small_ = 0;
    ts_imb_sum_ = 0;
    ts_imb_n_ = 0;
    if constexpr (kTuple) {
      path_table_.EnableCounters(ts_on_);
      path_table_.ResetCounters();
    }
    dev_matching_.Fill(static_cast<index_t>(nnodes_));
    cudaDeviceSynchronize();
  }

  // Timed: the searches, from the empty matching to the maximum one.
  void Match() {
    found_ = DFLAG_FALSE;
    size_t searches = 1;
    FindAndFlipAugmentingPath();
    while (found_ == DFLAG_TRUE) {
      found_ = DFLAG_FALSE;
      FindAndFlipAugmentingPath();
      searches++;
    }
    if (ts_on_) {
      size_t steps = 0, fv = 0, fe = 0;
      for (int k = 0; k < 3; k++) { steps += ts_steps_[k]; fv += ts_v_[k]; fe += ts_e_[k]; }
      unsigned long long walks = 0, walk_steps = 0, paths = 0, flips = 0, direct = 0, max_flips = 0;
      if constexpr (kTuple) {
        walks = path_table_.BaseWalks();
        walk_steps = path_table_.BaseWalkSteps();
        paths = path_table_.AugmentingPaths();
        flips = path_table_.AugmentingPathFlips();
        direct = path_table_.DirectAugmentingPaths();
        max_flips = path_table_.MaxAugmentingPathFlips();
      }
      std::printf("XBTraversal: run=%d steps=%zu frontier_v=%zu frontier_e=%zu max_frontier_e=%zu small_steps=%zu "
                  "imbalance=%.3f nodes=%zu edges=%zu searches=%zu aug_steps=%zu aug_e=%zu expand_steps=%zu "
                  "expand_e=%zu blossom_steps=%zu blossom_e=%zu base_walks=%llu base_walk_steps=%llu "
                  "aug_paths=%llu aug_path_flips=%llu aug_paths_direct=%llu aug_path_max_flips=%llu\n",
                  ts_run_, steps, fv, fe, ts_max_e_, ts_small_, ts_imb_n_ ? ts_imb_sum_ / ts_imb_n_ : 0.0, nnodes_,
                  (size_t)G_.get_number_of_edges(), searches, ts_steps_[0], ts_e_[0], ts_steps_[1], ts_e_[1],
                  ts_steps_[2], ts_e_[2], walks, walk_steps, paths, flips, direct, max_flips);
    }
    cudaDeviceSynchronize();
  }

  // Untimed: the mates copied back to the host.
  std::vector<index_t> Mate() {
    std::vector<index_t> mate(nnodes_);
    thrust::copy(dev_matching_.begin(), dev_matching_.end(), mate.begin());
    return mate;
  }

  // Accumulated phase times (s) and step count of the matching so far
  void PhaseSeconds(double& aug_path_s, double& expand_s, double& blossom_s) const {
    aug_path_s = t_aug_;
    expand_s = t_expand_;
    blossom_s = t_blossom_;
  }
  double InitSeconds() const { return t_init_; }
  size_t Steps() const { return steps_; }
  // Memory footprint of the blossom phase: the most nodes the blossom buffer
  // held in a step (mat only), the most entries the blossom lists held in a
  // step (closing-edge records under mat, odd-node tasks under nomat), the
  // path segments written, and the capacities
  size_t PeakBlossomBufferNodes() const { return peak_buffer_; }
  size_t PeakBlossomListEntries() const { return peak_list_; }
  size_t PeakTransformTasks() const { return peak_tasks_; }
  size_t PathsUsed() const { return path_table_.PathsUsed(); }
  size_t BufferCapacity() const { return path_table_.BufferCapacity(); }
  size_t PathTableFixedBytes() const { return path_table_.FixedBytes(); }
  int FlipStackPeak() const { return path_table_.FlipStackPeak(); }
  size_t BlossomListCapacity() const { return num_odd_nodes_in_blossom_list_.size(); }
  // Counters (EnableCounters, driver -c 1): blossoms, counted as the closing edges whose
  // base was identified (one record per closing edge under mat, one base walk under nomat)
  void EnableCounters() { count_ = true; }
  size_t Blossoms() const { return blossoms_; }

  // Public: nvcc requires the enclosing function of an extended __device__
  // lambda to be public.
  graph_t& G_;
  std::shared_ptr<gcuda::multi_context_t> context_;
  bool nomat_;
  bool base_rootpath_;
  size_t nnodes_;
  double t_aug_{0}, t_expand_{0}, t_blossom_{0}, t_init_{0};
  size_t steps_{0}, peak_buffer_{0}, peak_list_{0}, peak_tasks_{0};
  bool count_{false};
  size_t blossoms_{0};
  int found_{DFLAG_FALSE};
  int exhausted_{DFLAG_FALSE};

  DArray<index_t> dev_matching_;
  DValue<DFlag> dev_found_;
  DValue<unsigned long long> dev_blossoms_; // counter builds: closing edges of this step (nomat)
  path_table_t path_table_;
  DArray<DFlag> is_even_;
  DArray<index_t> tree_roots_;
  DArray<DAtomicFlag> atomic_locks_tree_;
  DArray<DAtomicFlag> atomic_locks_match_;
  DArray<DAtomicFlag> atomic_locks_odd_nodes_;
  zblossom::ENodeQueue enode_queue_;

  DList<offset_t> blossom_offset_list_; // cycle offsets (mat) or odd-node tasks (nomat)
  DArray<index_t> num_nodes_in_blossom_list_;
  DArray<index_t> num_odd_nodes_vside_in_blossom_list_;
  DArray<index_t> num_odd_nodes_in_blossom_list_;
  DArray<offset_t> num_odd_nodes_in_blossom_psum_; // 64-bit: a step's odd-node tasks can exceed 2^31

  CudaStream cuda_stream_;
  DVector<char> scan_temp_;
  size_t scan_temp_bytes_{0};

  frontier_t input_frontier_;
  frontier_t output_frontier_;  // unused: the advances write no output frontier
  thrust::device_vector<edge_t> segments_;

  // GACGE_XB_PHASE_SYNC (default 1): a device synchronization at the end of
  // init, expand and blossom, so each phase's host-clock time holds its own
  // kernels (augment already ends with the found flag copied back). 0: no
  // extra synchronization; expand's kernels then run into blossom's time.
  static bool PhaseSync() {
    static const bool on = [] {
      const char* e = std::getenv("GACGE_XB_PHASE_SYNC");
      return e == nullptr || e[0] != '0';
    }();
    return on;
  }

  void FindAndFlipAugmentingPath() {
    using clock = std::chrono::high_resolution_clock;
    auto secs = [](clock::time_point a, clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    const bool sync = PhaseSync();
    auto ti = clock::now();
    InitAlternatingForest();
    if (sync) cudaDeviceSynchronize();
    t_init_ += secs(ti, clock::now());
    while (true) {
      if (IsExhausted()) break;
      auto t0 = clock::now();
      bool found = FindAndFlipAugmentingPathInAlternatingForest();
      auto t1 = clock::now();
      t_aug_ += secs(t0, t1);
      if (found) break;
      ExpandAlternatingForest();
      if (sync) cudaDeviceSynchronize();
      auto t2 = clock::now();
      t_expand_ += secs(t1, t2);
      TransformOddNodesInBlossom();
      if (sync) cudaDeviceSynchronize();
      auto t3 = clock::now();
      t_blossom_ += secs(t2, t3);
      steps_++;
    }
  }

  bool IsExhausted() {
    if (enode_queue_.Size() == 0) exhausted_ = DFLAG_TRUE;
    return exhausted_;
  }

  // The even nodes of the queue (the current step's frontier) as a gunrock
  // frontier; returns its size.
  size_t LoadFrontier() {
    const index_t begin = enode_queue_.Begin();
    const size_t size = enode_queue_.End() - begin;
    if (input_frontier_.get_capacity() < size) input_frontier_.reserve(size);
    input_frontier_.set_number_of_elements(size);
    thrust::copy(thrust::device, enode_queue_.BeginIterator() + begin,
                 enode_queue_.BeginIterator() + begin + size, input_frontier_.begin());
    return size;
  }

  // GACGE traversal statistics: per phase (0 AugPath, 1 Expand, 2 Blossom)
  bool ts_on_{false}, ts_trace_{false};
  int ts_run_{0}, ts_phase_{0};
  size_t ts_steps_[3]{0, 0, 0}, ts_v_[3]{0, 0, 0}, ts_e_[3]{0, 0, 0}, ts_max_e_{0}, ts_small_{0};
  double ts_imb_sum_{0};  // sum over steps of max degree / mean degree of the frontier
  size_t ts_imb_n_{0};

  // The loaded frontier's vertices and the edges the advance will examine (their degrees)
  void TraversalStep() {
    const size_t n = input_frontier_.get_number_of_elements();
    size_t fe = 0, max_deg = 0;
    if (n > 0) {
      auto g = G_;
      auto first = input_frontier_.begin();
      auto deg = [g] __device__(vertex_t const& v) -> size_t { return (size_t)g.get_number_of_neighbors(v); };
      fe = thrust::transform_reduce(thrust::device, first, first + n, deg, size_t(0), thrust::plus<size_t>());
      max_deg = thrust::transform_reduce(thrust::device, first, first + n, deg, size_t(0), thrust::maximum<size_t>());
    }
    if (fe > 0) {
      ts_imb_sum_ += (double)max_deg * (double)n / (double)fe;
      ts_imb_n_++;
    }
    const int k = ts_phase_;
    ts_steps_[k]++;
    ts_v_[k] += n;
    ts_e_[k] += fe;
    ts_max_e_ = fe > ts_max_e_ ? fe : ts_max_e_;
    ts_small_ += fe < 65536;
    if (ts_trace_) {
      static const char* names[3] = {"augpath", "expand", "blossom"};
      std::printf("XBStep: run=%d step=%zu phase=%s frontier_v=%zu frontier_e=%zu max_deg=%zu\n", ts_run_,
                  ts_steps_[0] + ts_steps_[1] + ts_steps_[2] - 1, names[k], n, fe, max_deg);
    }
  }

  // One advance over the loaded frontier with LB; op returns false (no output).
  template <typename op_t>
  void Advance(op_t op) {
    if (ts_on_) TraversalStep();
    operators::advance::execute<LB, operators::advance_direction_t::forward,
                                operators::advance_io_type_t::vertices, operators::advance_io_type_t::none>(
        G_, op, &input_frontier_, &output_frontier_, segments_, *context_);
    // merge_path launches without synchronizing (block_mapped does) and a
    // failed launch would otherwise go unnoticed; the next step reads results
    // on the host.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void InitAlternatingForest() {
    atomic_locks_tree_.Fill(DFLAG_FALSE);
    atomic_locks_match_.Fill(DFLAG_FALSE);
    atomic_locks_odd_nodes_.Fill(DFLAG_FALSE);
    dev_found_.SetH2D(DFLAG_FALSE);

    const index_t nnodes = nnodes_;
    auto matching_v = dev_matching_.DeviceView();
    auto is_even_v = is_even_.DeviceView();
    auto tree_roots_v = tree_roots_.DeviceView();
    auto queue_v = enode_queue_.DeviceView();
    auto path_table_v = path_table_.DeviceView();

    enode_queue_.Clear();
    enode_queue_.PrepareForAppendingENode1();
    // Exposed nodes start trees; matched nodes keep their tree only if its root
    // is still exposed.
    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        G_,
        [=] __device__(vertex_t const& x) {
      auto matching = matching_v;
      auto is_even = is_even_v;
      auto tree_roots = tree_roots_v;
      auto queue = queue_v;
      auto path_table = path_table_v;
          if (matching[x] == nnodes) {
            is_even[x] = 1;
            tree_roots[x] = x;
            queue.AppendENode1(x);
            path_table.ResetNode(x);
          } else {
            index_t root = tree_roots[x];
            if (root == nnodes || matching[root] != nnodes) {
              is_even[x] = 0;
              tree_roots[x] = nnodes;
              path_table.ResetNode(x);
            }
          }
        },
        *context_);
    context_->get_context(0)->synchronize();

    enode_queue_.PrepareForAppendingENode2();
    // Even nodes of the kept trees join the frontier.
    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        G_,
        [=] __device__(vertex_t const& x) {
      auto matching = matching_v;
      auto is_even = is_even_v;
      auto tree_roots = tree_roots_v;
      auto queue = queue_v;
      auto path_table = path_table_v;
          if (matching[x] != nnodes) {
            index_t root = tree_roots[x];
            if (root != nnodes && matching[root] == nnodes && is_even[x]) queue.AppendENode2(x);
          }
        },
        *context_);
    context_->get_context(0)->synchronize();
  }

  bool FindAndFlipAugmentingPathInAlternatingForest() {
    ts_phase_ = 0;
    if (LoadFrontier() == 0) return false;
    const index_t nnodes = nnodes_;
    auto matching_v = dev_matching_.DeviceView();
    auto path_table_v = path_table_.DeviceView();
    auto is_even_v = is_even_.DeviceView();
    auto tree_roots_v = tree_roots_.DeviceView();
    auto locks_v = atomic_locks_tree_.DeviceView();
    auto found_v = dev_found_.DeviceView();

    Advance([=] __device__(vertex_t const& v, vertex_t const& w, edge_t const&, weight_t const&) -> bool {
      auto matching = matching_v;
      auto path_table = path_table_v;
      auto is_even = is_even_v;
      auto tree_roots = tree_roots_v;
      auto locks = locks_v;
      auto found = found_v;
      index_t root_v = tree_roots[v];
      index_t root_w = tree_roots[w];
      if (is_even[w] && root_v != root_w && root_v != nnodes && root_w != nnodes) {
        index_t first_lock = MIN(root_v, root_w);
        index_t second_lock = MAX(root_v, root_w);
        if (atomicExch(&locks[first_lock], DFLAG_TRUE) != DFLAG_FALSE) return false;
        if (atomicExch(&locks[second_lock], DFLAG_TRUE) != DFLAG_FALSE) {
          locks[first_lock] = DFLAG_FALSE;
          return false;
        }
        path_table.AddAugmentingPath(v, w, matching);
        atomicExch(&found, DFLAG_TRUE);
      }
      return false;
    });
    found_ = dev_found_.GetD2H();
    return found_;
  }

  void ExpandAlternatingForest() {
    ts_phase_ = 1;
    if (enode_queue_.Size() == 0) return;
    LoadFrontier();
    const index_t nnodes = nnodes_;
    auto matching_v = dev_matching_.DeviceView();
    auto path_table_v = path_table_.DeviceView();
    auto is_even_v = is_even_.DeviceView();
    auto tree_roots_v = tree_roots_.DeviceView();
    auto queue_v = enode_queue_.DeviceView();
    auto locks_v = atomic_locks_match_.DeviceView();

    enode_queue_.PrepareForAppendingENode1();
    Advance([=] __device__(vertex_t const& v, vertex_t const& w, edge_t const&, weight_t const&) -> bool {
      auto matching = matching_v;
      auto path_table = path_table_v;
      auto is_even = is_even_v;
      auto tree_roots = tree_roots_v;
      auto queue = queue_v;
      auto locks = locks_v;
      index_t root_v = tree_roots[v];
      index_t root_w = tree_roots[w];
      if (root_w == nnodes) {
        index_t x = matching[w];
        index_t match_idx = MIN(w, x);
        if (atomicExch(&locks[match_idx], DFLAG_TRUE) == DFLAG_FALSE) {
          is_even[w] = 0;
          is_even[x] = 1;
          tree_roots[w] = root_v;
          tree_roots[x] = root_v;
          __threadfence();
          path_table.ExpandTwoNodes(v, w, x);
          __threadfence();
          queue.AppendENode1Warp(x);
        }
      }
      __threadfence();
      return false;
    });
  }

  // The blossom lists hold odd_node_ratio x n entries; the prefix sum over them
  // needs one more. Past that the run is invalid: stop with an error.
  void CheckListFits(size_t used, const char* what) const {
    if (used + 1 > num_odd_nodes_in_blossom_list_.size()) {
      std::fprintf(stderr, "XBError: blossom list full (%zu %ss, capacity %zu); raise -o\n", used, what,
                   num_odd_nodes_in_blossom_list_.size() - 1);
      std::exit(4);
    }
  }

  void TransformOddNodesInBlossom() {
    ts_phase_ = 2;
    if (enode_queue_.Size() == 0) return;
    LoadFrontier();
    const index_t nnodes = nnodes_;
    auto matching_v = dev_matching_.DeviceView();
    auto path_table_v = path_table_.DeviceView();
    auto is_even_v = is_even_.DeviceView();
    auto tree_roots_v = tree_roots_.DeviceView();
    auto queue_v = enode_queue_.DeviceView();
    auto locks_v = atomic_locks_odd_nodes_.DeviceView();

    enode_queue_.PrepareForAppendingENode2();
    path_table_.ResetBlossomBuffer();

    blossom_offset_list_.Clear();
    auto blossom_offsets_v = blossom_offset_list_.DeviceView();
    auto nodes_in_blossom_v = num_nodes_in_blossom_list_.DeviceView();
    auto odd_in_blossom_v = num_odd_nodes_in_blossom_list_.DeviceView();
    auto odd_vside_in_blossom_v = num_odd_nodes_vside_in_blossom_list_.DeviceView();

    if (nomat_) {
      // nomat: find the base, claim the odd nodes still odd on both chains
      // (the lists hold: node, s, o, base), no cycle written
      const bool count = count_;
      if (count) dev_blossoms_.SetH2D(0);
      auto blossoms_v = dev_blossoms_.DeviceView();
      Advance([=] __device__(vertex_t const& v, vertex_t const& w, edge_t const&, weight_t const&) -> bool {
        auto matching = matching_v;
        auto path_table = path_table_v;
        auto is_even = is_even_v;
        auto tree_roots = tree_roots_v;
        auto locks = locks_v;
        auto task_node = blossom_offsets_v;
        auto task_s = nodes_in_blossom_v;
        auto task_o = odd_vside_in_blossom_v;
        auto task_base = odd_in_blossom_v;
        index_t root_v = tree_roots[v];
        index_t root_w = tree_roots[w];
        if (is_even[w] && root_v == root_w && matching[v] != w && root_w != nnodes) {
          index_t base;
          path_table.FindBlossomBaseByAnchorNode(v, w, base);
          if (count) atomicAdd(&blossoms_v, 1ULL);
          ClaimOddNodesOnChain(path_table, is_even, locks, task_node, task_s, task_o, task_base, v, w, base);
          ClaimOddNodesOnChain(path_table, is_even, locks, task_node, task_s, task_o, task_base, w, v, base);
        }
        return false;
      });
      if (count) blossoms_ += dev_blossoms_.GetD2H();
      const size_t num_tasks = blossom_offset_list_.size();
      CheckListFits(num_tasks, "odd-node task");
      peak_list_ = std::max(peak_list_, num_tasks);
      if (num_tasks == 0) return;
      // one task per claimed odd node: its path around the cycle, from the path table
      LaunchKernelForEachMax(cuda_stream_, num_tasks, [=] __device__(index_t t) {
        auto path_table = path_table_v;
        auto is_even = is_even_v;
        auto queue = queue_v;
        auto task_node = blossom_offsets_v;
        auto task_s = nodes_in_blossom_v;
        auto task_o = odd_vside_in_blossom_v;
        auto task_base = odd_in_blossom_v;
        const index_t u = task_node[t];
        const index_t s = task_s[t];
        const index_t o = task_o[t];
        const index_t base = task_base[t];
        path_table.ExpandOddNodeAroundBlossom(u, s, o, base);
        atomicExch(&is_even[u], 1);
        queue.AppendENode2(u);
      });
      return;
    }

    // mat (as is): find the blossoms closed by frontier edges, each materialised in the blossom buffer
    const bool base_rootpath = base_rootpath_;
    Advance([=] __device__(vertex_t const& v, vertex_t const& w, edge_t const&, weight_t const&) -> bool {
      auto matching = matching_v;
      auto path_table = path_table_v;
      auto is_even = is_even_v;
      auto tree_roots = tree_roots_v;
      auto queue = queue_v;
      auto locks = locks_v;
      auto blossom_offsets = blossom_offsets_v;
      auto nodes_in_blossom = nodes_in_blossom_v;
      auto odd_in_blossom = odd_in_blossom_v;
      auto odd_vside_in_blossom = odd_vside_in_blossom_v;
      const bool rootpath = base_rootpath; // captured here: nvcc rejects a first capture inside if constexpr
      index_t root_v = tree_roots[v];
      index_t root_w = tree_roots[w];
      if (is_even[w] && root_v == root_w && matching[v] != w && root_w != nnodes) {
        offset_t blossom_offset;
        size_t nnodes_in_blossom;
        index_t v_idx_in_blossom;
        if constexpr (kTuple) {
          if (!path_table.MaterializeCycle(v, w, matching, blossom_offset, nnodes_in_blossom, v_idx_in_blossom))
            return false; // the cycle buffer is full; the host aborts the run after this kernel
        } else if (rootpath) {
          index_t w_idx_in_blossom;
          if (!path_table.FindBlossomInBlossomBuffer(v, w, blossom_offset, nnodes_in_blossom, v_idx_in_blossom,
                                                     w_idx_in_blossom))
            return false; // the buffer is full (flagged) or the forest changed under the worker
        } else {
          if (!path_table.FindAndAddBlossomImplicitByAnchorNodeInBlossomBuffer(v, w, blossom_offset, nnodes_in_blossom,
                                                                              v_idx_in_blossom))
            return false;
        }
        size_t num_odd_vside = v_idx_in_blossom >> 1;
        size_t num_odd = (nnodes_in_blossom - 2) >> 1;
        if (num_odd == 0) return false;
        offset_t slot = blossom_offsets.Append(blossom_offset);
        if (static_cast<size_t>(slot) >= blossom_offsets.capacity()) return false; // full; the host stops the run
        nodes_in_blossom[slot] = nnodes_in_blossom;
        odd_in_blossom[slot] = num_odd;
        odd_vside_in_blossom[slot] = num_odd_vside;
      }
      return false;
    });

    // Make their odd nodes even (one tasklet per odd node).
    CheckListFits(blossom_offset_list_.size(), "closing-edge record");
    if (path_table_.CycleOverflow()) {
      // the reserve counter kept counting past the capacity: what this step alone asked for
      const size_t asked = path_table_.BlossomBufferUsed();
      std::fprintf(stderr, "XBError: cycle buffer full: this step asked for %zu nodes (%.1f GB), capacity %zu nodes (%.1f GB); raise -p\n",
                   asked, asked * 4e-9, path_table_.BufferCapacity(), path_table_.BufferCapacity() * 4e-9);
      std::exit(4);
    }
    const size_t num_blossom = blossom_offset_list_.size();
    if (count_) blossoms_ += num_blossom;
    peak_list_ = std::max(peak_list_, num_blossom);
    peak_buffer_ = std::max(peak_buffer_, path_table_.BlossomBufferUsed());
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(thrust::raw_pointer_cast(scan_temp_.data()), scan_temp_bytes_,
                                             num_odd_nodes_in_blossom_list_.begin(),
                                             num_odd_nodes_in_blossom_psum_.begin(),
                                             static_cast<int>(num_blossom + 1), cuda_stream_.cuda_stream()));
    cuda_stream_.Sync();
    const size_t total_transforms = num_odd_nodes_in_blossom_psum_[num_blossom];
    peak_tasks_ = std::max(peak_tasks_, total_transforms);
    if (total_transforms == 0) return;
    auto psum_v = num_odd_nodes_in_blossom_psum_.DeviceView();
    LaunchKernelForEachMax(cuda_stream_, total_transforms, [=] __device__(size_t tid) {
      auto matching = matching_v;
      auto path_table = path_table_v;
      auto is_even = is_even_v;
      auto tree_roots = tree_roots_v;
      auto queue = queue_v;
      auto locks = locks_v;
      auto blossom_offsets = blossom_offsets_v;
      auto nodes_in_blossom = nodes_in_blossom_v;
      auto odd_in_blossom = odd_in_blossom_v;
      auto odd_vside_in_blossom = odd_vside_in_blossom_v;
      auto psum = psum_v;
      index_t b = FindRightmostLTEQ64(psum, num_blossom + 1, static_cast<offset_t>(tid));
      offset_t blossom_offset = blossom_offsets[b];
      index_t nnodes_in_blossom = nodes_in_blossom[b];
      index_t num_odd_vside = odd_vside_in_blossom[b];
      index_t delta = static_cast<index_t>(static_cast<offset_t>(tid) - psum[b]);
      int is_vside = delta <= (num_odd_vside - 1);
      index_t odd_idx = 1 + 2 * delta + (1 - is_vside);
      index_t odd_node;
      if constexpr (kTuple) {
        odd_node = path_table.CycleNode(blossom_offset, odd_idx);
      } else {
        odd_node = path_table.GetNodeFromBlossomInBlossomBuffer(blossom_offset, odd_idx);
      }
      if (!is_even[odd_node] && path_table.IsPathEmpty(odd_node) &&
          atomicExch(&locks[odd_node], DFLAG_TRUE) == DFLAG_FALSE) {
        int step = is_vside ? +1 : -1;
        index_t start_idx = odd_idx + step;
        size_t path_len = is_vside ? (nnodes_in_blossom - odd_idx - 1) : static_cast<size_t>(odd_idx);
        if constexpr (kTuple) {
          if (path_table.CycleSpanHasDuplicate(blossom_offset, start_idx, path_len, step)) return;
          path_table.ConvertFromCycle(blossom_offset, 2 * num_odd_vside, odd_idx, step);
        } else {
          if (path_table.CheckPathDuplicationInBlossomBufferDirectional(blossom_offset, start_idx, path_len, step))
            return;
          path_table.ExpandNodesFromBlossomDirectional(odd_node, blossom_offset, nnodes_in_blossom, start_idx,
                                                       path_len, step);
        }
        atomicExch(&is_even[odd_node], 1);
        queue.AppendENode2(odd_node);
      }
    });
  }
};

}  // namespace xblossom
}  // namespace gunrock
