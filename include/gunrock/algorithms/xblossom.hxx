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
 * merge_path, thread_mapped); the two per-vertex initialisation passes are
 * parallel_for. What stays X-Blossom's own (vendored, ./xblossom/): the
 * device path table, the even-node queue, the per-tree / per-match /
 * per-odd-node locks and the blossom transform. One semantic difference from
 * the hand-written XB++ node-level mode: the advance lambda sees one edge at a
 * time, so there is no "skip this node's remaining edges".
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>

#include <cub/cub.cuh>

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

namespace gunrock {
namespace xblossom {

// Size of the blossom lists, per node (XB++'s ODD_NODE_RATIO).
constexpr size_t kOddNodeRatio = 30;

template <operators::load_balance_t LB, typename graph_t>
class engine_t {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
  using frontier_t = frontier::frontier_t<vertex_t, edge_t, frontier::frontier_kind_t::vertex_frontier>;

 public:
  engine_t(graph_t& G, size_t path_table_buffer_ratio, std::shared_ptr<gcuda::multi_context_t> context)
      : G_(G),
        context_(context),
        nnodes_(G.get_number_of_vertices()),
        dev_matching_(nnodes_, nnodes_),
        path_table_(nnodes_, path_table_buffer_ratio),
        is_even_(nnodes_),
        tree_roots_(nnodes_),
        blossom_to_base_(nnodes_),
        atomic_locks_tree_(nnodes_),
        atomic_locks_match_(nnodes_),
        atomic_locks_odd_nodes_(nnodes_),
        enode_queue_(nnodes_),
        blossom_offset_list_(kOddNodeRatio * nnodes_ + 1),
        num_nodes_in_blossom_list_(kOddNodeRatio * nnodes_ + 1),
        num_odd_nodes_vside_in_blossom_list_(kOddNodeRatio * nnodes_ + 1),
        num_odd_nodes_in_blossom_list_(kOddNodeRatio * nnodes_ + 1),
        num_odd_nodes_in_blossom_psum_(kOddNodeRatio * nnodes_ + 1),
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

  // Maximum matching from the empty matching; mate[v] == nnodes for exposed v.
  std::vector<index_t> FindMaximumMatch() {
    dev_matching_.Fill(static_cast<index_t>(nnodes_));
    found_ = DFLAG_FALSE;
    FindAndFlipAugmentingPath();
    while (found_ == DFLAG_TRUE) {
      found_ = DFLAG_FALSE;
      FindAndFlipAugmentingPath();
    }
    std::vector<index_t> mate(nnodes_);
    thrust::copy(dev_matching_.begin(), dev_matching_.end(), mate.begin());
    return mate;
  }

  // Public: nvcc requires the enclosing function of an extended __device__
  // lambda to be public.
  graph_t& G_;
  std::shared_ptr<gcuda::multi_context_t> context_;
  size_t nnodes_;
  int found_{DFLAG_FALSE};
  int exhausted_{DFLAG_FALSE};

  DArray<index_t> dev_matching_;
  DValue<DFlag> dev_found_;
  zblossom::dev::PathTable path_table_;
  DArray<DFlag> is_even_;
  DArray<index_t> tree_roots_;
  DArray<index_t> blossom_to_base_;
  DArray<DAtomicFlag> atomic_locks_tree_;
  DArray<DAtomicFlag> atomic_locks_match_;
  DArray<DAtomicFlag> atomic_locks_odd_nodes_;
  zblossom::ENodeQueue enode_queue_;

  DList<index_t> blossom_offset_list_;
  DArray<index_t> num_nodes_in_blossom_list_;
  DArray<index_t> num_odd_nodes_vside_in_blossom_list_;
  DArray<index_t> num_odd_nodes_in_blossom_list_;
  DArray<index_t> num_odd_nodes_in_blossom_psum_;

  CudaStream cuda_stream_;
  DVector<char> scan_temp_;
  size_t scan_temp_bytes_{0};

  frontier_t input_frontier_;
  frontier_t output_frontier_;  // unused: the advances write no output frontier
  thrust::device_vector<edge_t> segments_;

  void FindAndFlipAugmentingPath() {
    InitAlternatingForest();
    while (!IsExhausted() && !FindAndFlipAugmentingPathInAlternatingForest()) {
      ExpandAlternatingForest();
      TransformOddNodesInBlossom();
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

  // One advance over the loaded frontier with LB; op returns false (no output).
  template <typename op_t>
  void Advance(op_t op) {
    operators::advance::execute<LB, operators::advance_direction_t::forward,
                                operators::advance_io_type_t::vertices, operators::advance_io_type_t::none>(
        G_, op, &input_frontier_, &output_frontier_, segments_, *context_);
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

  void TransformOddNodesInBlossom() {
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

    // Find the blossoms closed by frontier edges.
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
      index_t root_v = tree_roots[v];
      index_t root_w = tree_roots[w];
      if (is_even[w] && root_v == root_w && matching[v] != w && root_w != nnodes) {
        index_t blossom_offset;
        size_t nnodes_in_blossom;
        index_t v_idx_in_blossom;
        path_table.FindAndAddBlossomImplicitByAnchorNodeInBlossomBuffer(v, w, blossom_offset, nnodes_in_blossom,
                                                                       v_idx_in_blossom);
        size_t num_odd_vside = v_idx_in_blossom >> 1;
        size_t num_odd = (nnodes_in_blossom - 2) >> 1;
        if (num_odd == 0) return false;
        index_t slot = blossom_offsets.Append(blossom_offset);
        nodes_in_blossom[slot] = nnodes_in_blossom;
        odd_in_blossom[slot] = num_odd;
        odd_vside_in_blossom[slot] = num_odd_vside;
      }
      return false;
    });

    // Make their odd nodes even (one tasklet per odd node).
    const size_t num_blossom = blossom_offset_list_.size();
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(thrust::raw_pointer_cast(scan_temp_.data()), scan_temp_bytes_,
                                             num_odd_nodes_in_blossom_list_.begin(),
                                             num_odd_nodes_in_blossom_psum_.begin(),
                                             static_cast<int>(num_blossom + 1), cuda_stream_.cuda_stream()));
    cuda_stream_.Sync();
    const size_t total_transforms = num_odd_nodes_in_blossom_psum_[num_blossom];
    if (total_transforms == 0) return;
    auto psum_v = num_odd_nodes_in_blossom_psum_.DeviceView();
    LaunchKernelForEachMax(cuda_stream_, total_transforms, [=] __device__(index_t tid) {
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
      index_t b = FindRightmostLTEQ(psum, num_blossom + 1, tid);
      index_t blossom_offset = blossom_offsets[b];
      index_t nnodes_in_blossom = nodes_in_blossom[b];
      index_t num_odd_vside = odd_vside_in_blossom[b];
      index_t delta = tid - psum[b];
      int is_vside = delta <= (num_odd_vside - 1);
      index_t odd_idx = 1 + 2 * delta + (1 - is_vside);
      index_t odd_node = path_table.GetNodeFromBlossomInBlossomBuffer(blossom_offset, odd_idx);
      if (!is_even[odd_node] && path_table.IsPathEmpty(odd_node) &&
          atomicExch(&locks[odd_node], DFLAG_TRUE) == DFLAG_FALSE) {
        int step = is_vside ? +1 : -1;
        index_t start_idx = odd_idx + step;
        size_t path_len = is_vside ? (nnodes_in_blossom - odd_idx - 1) : static_cast<size_t>(odd_idx);
        if (path_table.CheckPathDuplicationInBlossomBufferDirectional(blossom_offset, start_idx, path_len, step))
          return;
        path_table.ExpandNodesFromBlossomDirectional(odd_node, blossom_offset, nnodes_in_blossom, start_idx,
                                                     path_len, step);
        atomicExch(&is_even[odd_node], 1);
        queue.AppendENode2(odd_node);
      }
    });
  }
};

}  // namespace xblossom
}  // namespace gunrock
