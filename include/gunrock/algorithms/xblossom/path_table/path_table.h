// Vendored from X-Blossom (src/include/path_table/path_table.h) for the Gunrock X-Blossom app.
#ifndef PATH_TABLE_H
#define PATH_TABLE_H
#include <utils/dev_array.h>
#include <utils/types_gpu.h>
#include <utils/dev_value.h>

#include "path_table.h"
#include "utils/cpp_types.h"
#include "utils/dev_list.h"

#define TWO_NODES 2
// GPlus
// #define PATH_TABLE_BUFFER_RATIO 6000

// PATENT + Hyperlink
// #define PATH_TABLE_BUFFER_RATIO 300

// LiveJournal
// #define PATH_TABLE_BUFFER_RATIO 200

#define PATH_TABLE_BUFFER_RATIO 80

#define PATH_TABLE_ODD_NODE_RATIO 80
// #define PATH_TABLE_ODD_NODE_RATIO 30
// #define PATH_TABLE_ODD_NODE_RATIO 10
// #define PATH_TABLE_ODD_NODE_RATIO 0

// Stackoverflow
// #define PATH_TABLE_BUFFER_RATIO 1

class TestPathTableFixture;

namespace zblossom {
  // class XBlossomGpuEngineParEdgeTester;
  // class XBlossomGpuEngineTester;
  class PathTableTester;

  template<typename XBlossomEngine>
  class XbsmGpuEngineTester;

  namespace host {
    class PathTable {
    public:
      explicit PathTable(size_t nnnodes)
        : paths_(nnnodes) {
      }

      void Reset() {
        for (auto &path: paths_) {
          path.clear();
        }
      }

      void FindAugmentingPath(index_t v, index_t w, Vector<index_t> &augmenting_path) const {
        Vector<index_t> path_v, path_w;
        FindPath(v, path_v);
        FindPath(w, path_w);

        augmenting_path.clear();
        // insert in the reverse order
        augmenting_path.insert(augmenting_path.end(), path_v.rbegin(), path_v.rend());
        // insert in the regular order
        augmenting_path.insert(augmenting_path.end(), path_w.begin(), path_w.end());
      }

      // It is guaranteed that v and w can form a blossom
      void FindBlossom(index_t v, index_t w, Vector<index_t> &blossom) const {
        Vector<index_t> path_v, path_w;
        FindPath(v, path_v);
        FindPath(w, path_w);

        blossom.clear();
        blossom.reserve(path_v.size() + path_w.size());
        int cur_v_idx = path_v.size() - 1;
        int cur_w_idx = path_w.size() - 1;

        // Find the base
        while (cur_v_idx >= 0 && cur_w_idx >= 0 && path_v[cur_v_idx] == path_w[cur_w_idx]) {
          cur_v_idx--;
          cur_w_idx--;
        }

        const size_t base_idx_v = cur_v_idx + 1;
        const size_t base_idx_w = cur_w_idx + 1;

        // Add the path from base to v
        for (int i = base_idx_v; i >= 0; i--) {
          blossom.push_back(path_v[i]);
        }

        // Add the path from w to v
        for (int j = 0; j <= base_idx_w; j++) {
          blossom.push_back(path_w[j]);
        }
      }

      bool IsPathEmpty(index_t v) const {
        return paths_[v].empty();
      }

      // v --- w --- x
      // => x: <w, v>
      void ExpandTwoNodes(index_t v, index_t w, index_t x) {
        paths_[x].push_back(w);
        paths_[x].push_back(v);
      }

      // Add nodes (clockwise)
      void AddNodesFromBlossomToRight(index_t node, const Vector<index_t> &blossom, size_t start_idx) {
        for (size_t idx = start_idx; idx < blossom.size(); idx++) {
          paths_[node].push_back(blossom[idx]);
        }
      }

      // Add nodes (anti-clockwise)
      void AddNodesFromBlossomToLeft(index_t node, const Vector<index_t> &blossom, size_t start_idx) {
        for (int idx = static_cast<int>(start_idx); idx >= 0; idx--) {
          paths_[node].push_back(blossom[idx]);
        }
      }

    private:
      Vector<Vector<index_t> > paths_;

      // Find a path from node v to the root, output to 'path'
      // v will be at path[0], and the root will be at path[path.size() - 1]
      void FindPath(index_t v, Vector<index_t> &path) const {
        path.clear();
        path.push_back(v);
        index_t cur_v = v;
        while (!paths_[cur_v].empty()) {
          path.insert(path.end(), paths_[cur_v].begin(), paths_[cur_v].end());
          cur_v = path.back();
        }
      }
    };
  }

  namespace dev {
    class PathTableView;

    class PathTable {
      friend class ::zblossom::PathTableTester;
      friend class ::TestPathTableFixture;

      template<typename XBlossomEngine>
      friend class zblossom::XbsmGpuEngineTester;

    public:
      explicit PathTable(size_t nnnodes)
        : nnnodes_(nnnodes),
          path_buffer_size_(PATH_TABLE_BUFFER_RATIO * nnnodes),
          paths_(path_buffer_size_),
          offset_length_(nnnodes),
          root_distance_(nnnodes),
          blossom_buffer_(path_buffer_size_) {
        try {
          // offset_length_.resize(nnnodes);
          // root_distance_.resize(nnnodes);

          // paths_.resize(PATH_TABLE_BUFFER_RATIO * nnnodes);
          // blossom_buffer_.resize(path_buffer_size_);

          // paths_offset_.resize(1);
          // blossom_buffer_offset_.resize(1);
        } catch (const thrust::system_error &e) {
          std::cerr << "PathTable Initialization fails due to CUDA Thrust allocation failed: " << e.what() << std::endl;
          std::abort();
        }
      }

      explicit PathTable(size_t nnnodes, size_t buffer_ratio)
        : nnnodes_(nnnodes),
          path_buffer_size_(buffer_ratio * nnnodes),
          paths_(path_buffer_size_),
          offset_length_(nnnodes),
          root_distance_(nnnodes),
          blossom_buffer_(buffer_ratio * nnnodes) {
        try {
          // offset_length_.resize(nnnodes);
          // root_distance_.resize(nnnodes);

          // paths_.resize(PATH_TABLE_BUFFER_RATIO * nnnodes);
          // blossom_buffer_.resize(path_buffer_size_);

          // paths_offset_.resize(1);
          // blossom_buffer_offset_.resize(1);
        } catch (const thrust::system_error &e) {
          std::cerr << "PathTable Initialization fails due to CUDA Thrust allocation failed: " << e.what() << std::endl;
          std::abort();
        }
      }

      // Invoked when Resetting Alternating Forest.
      void ResetPathTable() {
        // thrust::fill(offset_length_.begin(), offset_length_.end(), DPair<index_t, size_t>(0, 0));
        offset_length_.Fill(DPair<index_t, size_t>(0, 0));
        // thrust::fill(root_distance_.begin(), root_distance_.end(), 0);
        root_distance_.Fill(0);

        paths_.Clear();
        blossom_buffer_.Clear();
      }

      // Invoked when Transforming Odd nodes into even nodes.
      void ResetBlossomBuffer() {
        blossom_buffer_.Clear();
      }

      void ResetPathTableBuffer() {
        paths_.Clear();
      }

      // defined after the PathTable class
      PathTableView DeviceView();

    private:
      size_t nnnodes_;

      size_t path_buffer_size_;
      // Store (offset, length) pair for paths either to the last even node or blossom base
      DArray<DPair<index_t, size_t> > offset_length_;
      // length from a node to the root of its alternating tree: helpful when reconstructing a path
      DArray<index_t> root_distance_;

      DList<index_t> paths_;
      DList<index_t> blossom_buffer_;
    };

    class PathTableView {
    public:
      DEV_HOST_INLINE
      explicit PathTableView(size_t nnnodes,
                             size_t path_buffer_size,
                             DArrayView<DPair<index_t, size_t> > paths_offset_length,
                             DArrayView<index_t> paths_root_distance,
                             DListView<index_t> paths,
                             DListView<index_t> path_buffer
                             // DArrayView<index_t> paths,
                             // DArrayView<index_t> path_buffer,
                             // DValueView<index_t> paths_offsets,
                             // DValueView<index_t> path_buffer_offset
      )
        : nnnodes_(nnnodes),
          path_buffer_size_(path_buffer_size),
          offset_length_(paths_offset_length),
          root_distance_(paths_root_distance),
          paths_(paths),
          blossom_buffer_(path_buffer) {
      }

      DEV_INLINE bool IsPathEmpty(index_t v) const {
        DPair<index_t, size_t> offset_length_pair = offset_length_[v];
        return offset_length_pair.second == 0;
      }


      DEV_INLINE void ExpandTwoNodes(index_t v, index_t w, index_t x) {
        // Worse Performance
        // index_t offset = paths_.ReserveWarp(TWO_NODES);

        index_t offset = paths_.Reserve(TWO_NODES);
        paths_[offset] = w;
        paths_[offset + 1] = v;

        // root_distance_[x] = TWO_NODES + root_distance_[v];
        atomicExch(&root_distance_[x], TWO_NODES + root_distance_[v]);

        __threadfence(); // <- make the list visible before meta‑data

        offset_length_[x] = {offset, TWO_NODES};
        __threadfence(); // <- make the list visible before meta‑data
      }

      DEV_INLINE void AddAugmentingPath(index_t v,
                                        index_t w,
                                        DArrayView<index_t> matching) {
        // Update maching for the bridge (v, w)
        // matching[v] = w;
        // matching[w] = v;
        atomicExch(&matching[v], w);
        atomicExch(&matching[w], v);

        // Traverse the augmenting path starting from v
        index_t cur_v = v;
        DPair<index_t, size_t> offset_length = offset_length_[cur_v];

        while (offset_length.second != 0) {
          index_t offset = offset_length.first;
          size_t length = offset_length.second;

          // Iterate over pairs of nodes along the augmenting path and update their matchings
          for (index_t i = 0; i < length; i += 2) {
            index_t node1 = paths_[offset + i];
            index_t node2 = paths_[offset + i + 1];
            // matching[node1] = node2;
            // matching[node2] = node1;
            atomicExch(&matching[node1], node2);
            atomicExch(&matching[node2], node1);
          }

          // update the cur_v
          cur_v = paths_[offset + length - 1];
          offset_length = offset_length_[cur_v];
        }

        // Do similar traversal for w
        index_t cur_w = w;
        offset_length = offset_length_[cur_w];

        while (offset_length.second != 0) {
          index_t offset = offset_length.first;
          size_t length = offset_length.second;

          // Iterate ower pairs of nodes along the augmenting path and update their matchings
          for (index_t i = 0; i < length; i += 2) {
            index_t node1 = paths_[offset + i];
            index_t node2 = paths_[offset + i + 1];
            // matching[node1] = node2;
            // matching[node2] = node1;
            atomicExch(&matching[node1], node2);
            atomicExch(&matching[node2], node1);
          }

          // update the cur_w
          cur_w = paths_[offset + length - 1];
          offset_length = offset_length_[cur_w];
        }
        __threadfence();
      }

      DEV_INLINE void FindBlossom(index_t v,
                                  index_t w,
                                  index_t &blossom_offset,
                                  size_t &nnodes_in_blossom,
                                  index_t &v_idx_in_blossom) {
        // 1-1 Get length of path v and path w using paths_root_length
        size_t nnodes_v_root = this->root_distance_[v] + 1;
        size_t nndoes_w_root = this->root_distance_[w] + 1;

        // 1-2 Resevere space for both path v and path w
        index_t path_v_offset = paths_.Reserve(nnodes_v_root + nndoes_w_root);
        index_t path_w_offset = path_v_offset + nnodes_v_root;

        // 1-3 Traverse path v to root and store it in paths_buffer
        index_t buffer_idx_v = path_v_offset;
        index_t cur_v = v;
        // add v to path buffer

        // TODO: save the space for blossom buffer
        if (buffer_idx_v > path_buffer_size_) {
          printf("[ERROR]: buffer_idx_v is %d, whereas path_buffer_size is %lu\n", buffer_idx_v, path_buffer_size_);
          assert(false);
        }

        paths_[buffer_idx_v++] = cur_v;

        DPair<index_t, size_t> offset_length = offset_length_[cur_v];
        while (offset_length.second != 0) {
          index_t cur_offset = offset_length.first;
          size_t cur_length = offset_length.second;

          // Iterate each node in path
          for (index_t i = 0; i < cur_length; i++) {
            paths_[buffer_idx_v++] = paths_[cur_offset + i];
          }

          // update the cur_v
          cur_v = paths_[cur_offset + cur_length - 1];
          offset_length = offset_length_[cur_v];
        }

        index_t buffer_idx_v_end = buffer_idx_v;

        // 1-4 Traverse path w to root and store it in paths_buffer
        index_t buffer_idx_w = path_w_offset;
        index_t cur_w = w;
        // add w to path buffer
        paths_[buffer_idx_w++] = cur_w;
        offset_length = offset_length_[cur_w];
        while (offset_length.second > 0) {
          index_t cur_offset = offset_length.first;
          size_t cur_length = offset_length.second;

          // Iterate each node in path
          for (index_t i = 0; i < cur_length; i++) {
            paths_[buffer_idx_w++] = paths_[cur_offset + i];
          }

          // update the cur_w
          cur_w = paths_[cur_offset + cur_length - 1];
          offset_length = offset_length_[cur_w];
        }

        index_t buffer_idx_w_end = buffer_idx_w;

        // *********************************************************
        // 2. Find the blossom base
        // Find the base for path v and path w
        index_t cur_buffer_v_idx = path_v_offset + nnodes_v_root - 1;
        index_t cur_buffer_w_idx = path_w_offset + nndoes_w_root - 1;

        while (cur_buffer_v_idx >= path_v_offset &&
               cur_buffer_w_idx >= path_w_offset &&
               paths_[cur_buffer_v_idx] == paths_[cur_buffer_w_idx]) {
          cur_buffer_v_idx--;
          cur_buffer_w_idx--;
        }

        // base_idx_* points to the blossom base clearly
        const index_t base_idx_v = cur_buffer_v_idx + 1;
        const index_t base_idx_w = cur_buffer_w_idx + 1;

        const size_t nnodes_v_base = base_idx_v - path_v_offset + 1;
        const size_t nnodes_w_base = base_idx_w - path_w_offset + 1;

        // 3. Create the blossom
        // 3-1 Add the path from base to v by reversing the path from v to base
        index_t left = path_v_offset;
        index_t right = base_idx_v;
        while (left < right) {
          index_t tmp = paths_[left];
          paths_[left++] = paths_[right];
          paths_[right--] = tmp;
        }

        // 3-2 Add the path from w to base by shifting the path from w to base, preserving the original order
        v_idx_in_blossom = nnodes_v_base - 1;

        for (index_t i = 0; i < nnodes_w_base; i++) {
          paths_[path_v_offset + nnodes_v_base + i] = paths_[path_w_offset + i];
        }

        // 3-3 Update Blossom offset and length
        blossom_offset = path_v_offset;
        nnodes_in_blossom = nnodes_v_base + nnodes_w_base;
      }

      DEV_INLINE void FindBlossomInBlossomBuffer(index_t v,

                                                 index_t w,
                                                 index_t &blossom_offset,
                                                 size_t &nnodes_in_blossom,
                                                 index_t &v_idx_in_blossom,
                                                 index_t &w_idx_in_blossom) {
        // 1-1 Get length of path v and path w using paths_root_length
        size_t nnnodes_v_root = this->root_distance_[v] + 1;
        size_t nnode_w_root = this->root_distance_[w] + 1;

        // 1-2 Resevere space for both path v and path w
        // index_t path_v_offset = atomicAdd(blossom_buffer_offset_.Addr(), path_v_len + path_w_len);
        index_t path_v_offset = blossom_buffer_.Reserve(nnnodes_v_root + nnode_w_root);
        index_t path_w_offset = path_v_offset + nnnodes_v_root;

        // 1-3 Traverse path v to root and store it in paths_buffer
        index_t buffer_idx_v = path_v_offset;
        index_t cur_v = v;
        // add v to path buffer

        // TODO: save the space for blossom buffer
        if (buffer_idx_v > path_buffer_size_) {
          printf("[ERROR]: buffer_idx_v is %d, whereas path_buffer_size is %lu\n", buffer_idx_v, path_buffer_size_);
          assert(false);
        }

        blossom_buffer_[buffer_idx_v++] = cur_v;

        DPair<index_t, size_t> offset_length = offset_length_[cur_v];
        int it = 0;
        while (offset_length.second != 0) {
          index_t cur_offset = offset_length.first;
          size_t cur_length = offset_length.second;

          // Iterate each node in path
          for (index_t i = 0; i < cur_length; i++) {
            blossom_buffer_[buffer_idx_v++] = paths_[cur_offset + i];
          }

          // update the cur_v
          cur_v = paths_[cur_offset + cur_length - 1];
          offset_length = offset_length_[cur_v];
          it++;
          if (it > 10000) {
            printf("Dead Loop\n");
          }
        }

        // 1-4 Traverse path w to root and store it in paths_buffer
        index_t buffer_idx_w = path_w_offset;
        index_t cur_w = w;
        // add w to path buffer
        blossom_buffer_[buffer_idx_w++] = cur_w;
        offset_length = offset_length_[cur_w];

        it = 0;
        while (offset_length.second > 0) {
          index_t cur_offset = offset_length.first;
          size_t cur_length = offset_length.second;

          // Iterate each node in path
          for (index_t i = 0; i < cur_length; i++) {
            blossom_buffer_[buffer_idx_w++] = paths_[cur_offset + i];
          }

          // update the cur_w
          cur_w = paths_[cur_offset + cur_length - 1];
          offset_length = offset_length_[cur_w];

          it++;
          if (it > 10000) {
            printf("Dead Loop\n");
          }
        }

        // 2. Find the blossom base
        // Find the base for path v and path w
        index_t cur_buffer_v_idx = path_v_offset + nnnodes_v_root - 1;
        index_t cur_buffer_w_idx = path_w_offset + nnode_w_root - 1;

        while (cur_buffer_v_idx >= path_v_offset &&
               cur_buffer_w_idx >= path_w_offset &&
               blossom_buffer_[cur_buffer_v_idx] == blossom_buffer_[cur_buffer_w_idx]) {
          cur_buffer_v_idx--;
          cur_buffer_w_idx--;
        }

        // base_idx_* points to the blossom base clearly
        const index_t base_idx_v = cur_buffer_v_idx + 1;
        const index_t base_idx_w = cur_buffer_w_idx + 1;

        // 3. Create the blossom
        // 3-1 Add the path from base to v by reversing the path from v to base
        index_t left = path_v_offset;
        index_t right = base_idx_v;
        while (left < right) {
          index_t tmp = blossom_buffer_[left];
          blossom_buffer_[left++] = blossom_buffer_[right];
          blossom_buffer_[right--] = tmp;
        }

        // 3-2 Add the path from w to base by shifting the path from w to base, preserving the original order
        const size_t nnnodes_v_base = base_idx_v - path_v_offset + 1;
        const size_t nnodes_w_base = base_idx_w - path_w_offset + 1;
        for (index_t i = 0; i < nnodes_w_base; i++) {
          blossom_buffer_[path_v_offset + nnnodes_v_base + i] = blossom_buffer_[path_w_offset + i];
        }

        // 3-3 Update Blossom offset and length
        blossom_offset = path_v_offset;
        nnodes_in_blossom = nnnodes_v_base + nnodes_w_base;

        v_idx_in_blossom = nnnodes_v_base - 1;
        w_idx_in_blossom = nnnodes_v_base;
      }

      DEV_INLINE void FindAndAddBlossomImplicit(index_t v,
                                                index_t w,
                                                index_t &blossom_offset,
                                                size_t &num_nodes_in_blossom,
                                                index_t &v_idx_in_blossom) {
        // find the length of blossom
        index_t cur_v = v;
        index_t cur_w = w;

        size_t len_traversal_v = 0;
        size_t len_traversal_w = 0;

        int reach_root_v = 0;
        int reach_root_w = 0;

        size_t len_v_to_root = 0;
        size_t len_w_to_root = 0;

        DPair<index_t, size_t> offset_length_cur_v = this->offset_length_[cur_v];
        DPair<index_t, size_t> offset_length_cur_w = this->offset_length_[cur_w];

        index_t idx_in_path_v = 1;
        index_t idx_in_path_w = 1;
        // printf("v: %d, w: %d starts to traversal\n", v, w);

        int it = 0;
        while (cur_v != cur_w) {
          if (offset_length_cur_v.second == 0) {
            // cur_v reach the root node, switch to w
            reach_root_v = 1;
            cur_v = w;
            offset_length_cur_v = this->offset_length_[cur_v];
            idx_in_path_v = 1;
          } else {
            // move to the next node in the path
            cur_v = paths_[offset_length_cur_v.first + idx_in_path_v];

            len_traversal_v += 2;
            len_v_to_root += 2 * (1 - reach_root_v);

            if (idx_in_path_v == offset_length_cur_v.second - 1) {
              offset_length_cur_v = this->offset_length_[cur_v];
              idx_in_path_v = 1;
            } else {
              idx_in_path_v += 2;
            }
          }

          if (offset_length_cur_w.second == 0) {
            // cur_v reach the root node, switch to w
            reach_root_w = 1;
            cur_w = v;
            offset_length_cur_w = this->offset_length_[cur_w];
            idx_in_path_w = 1;
          } else {
            // move to the next node in the path
            cur_w = paths_[offset_length_cur_w.first + idx_in_path_w];

            len_traversal_w += 2;
            len_w_to_root += 2 * (1 - reach_root_w);

            // if the cur_v is already the last one in the path
            if (idx_in_path_w == offset_length_cur_w.second - 1) {
              offset_length_cur_w = this->offset_length_[cur_w];
              idx_in_path_w = 1;
            } else {
              idx_in_path_w += 2;
            }
          }
        }

        index_t blossom_base = cur_v;
        // reserve space in paths to store the blossom from base to v to w to base

        // Bug fixed: len_v_to_base should exclude path_w_len, not path_v_len
        size_t len_v_to_base = reach_root_v ? len_traversal_v - len_w_to_root : len_traversal_v;
        size_t len_w_to_base = reach_root_v ? len_traversal_w - len_v_to_root : len_traversal_w;

        size_t nnodes_v_to_base = len_v_to_base + 1;
        size_t nnodes_w_to_base = len_w_to_base + 1;

        // reserve blossom offset
        blossom_offset = paths_.Reserve(nnodes_v_to_base + nnodes_w_to_base);
        num_nodes_in_blossom = nnodes_v_to_base + nnodes_w_to_base;

        // read v to base in reverse order
        // add v to path buffer
        index_t buffer_idx_v = blossom_offset + nnodes_v_to_base - 1;
        if (buffer_idx_v - 1 > this->path_buffer_size_) {
          // printf("[ERROR] v is at idx: %d\n", buffer_idx_v - 1);
        }

        index_t last_v = v;
        this->paths_[buffer_idx_v--] = v;
        offset_length_cur_v = this->offset_length_[v];
        idx_in_path_v = 0;
        while (last_v != blossom_base) {
          index_t cur_v_to_add = paths_[offset_length_cur_v.first + idx_in_path_v];
          this->paths_[buffer_idx_v--] = cur_v_to_add;
          if (idx_in_path_v == offset_length_cur_v.second - 1) {
            offset_length_cur_v = this->offset_length_[cur_v_to_add];
            idx_in_path_v = 0;
          } else {
            idx_in_path_v += 1;
          }
          last_v = cur_v_to_add;
        }

        // read w to base in regular order in paths_
        index_t buffer_idx_w = blossom_offset + nnodes_v_to_base;
        // add w to path buffer
        this->paths_[buffer_idx_w++] = w;
        offset_length_cur_w = this->offset_length_[w];
        index_t last_w = w;
        idx_in_path_w = 0;
        while (last_w != blossom_base) {
          index_t cur_w_to_add = paths_[offset_length_cur_w.first + idx_in_path_w];
          this->paths_[buffer_idx_w++] = cur_w_to_add;
          if (idx_in_path_w == offset_length_cur_w.second - 1) {
            offset_length_cur_w = this->offset_length_[cur_w_to_add];
            idx_in_path_w = 0;
          } else {
            idx_in_path_w += 1;
          }
          last_w = cur_w_to_add;
        }

        v_idx_in_blossom = nnodes_v_to_base - 1;
      }

      DEV_INLINE void FindAndAddBlossomImplicitByAnchorNode(index_t v,
                                                            index_t w,
                                                            index_t &blossom_offset,
                                                            size_t &num_nodes_in_blossom,
                                                            index_t &v_idx_in_blossom) {
        // find the blossom base
        index_t cur_v = v;
        index_t cur_w = w;

        DPair<index_t, size_t> offset_length_cur_v = this->offset_length_[cur_v];
        DPair<index_t, size_t> offset_length_cur_w = this->offset_length_[cur_w];

        int it = 0;
        while (cur_v != cur_w) {
          if (offset_length_cur_v.second == 0) {
            // cur_v reach the root node, switch to w
            cur_v = w;
            offset_length_cur_v = this->offset_length_[cur_v];
          } else {
            // move to the next node in the path
            cur_v = paths_[offset_length_cur_v.first + offset_length_cur_v.second - 1];
            offset_length_cur_v = this->offset_length_[cur_v];
          }

          if (offset_length_cur_w.second == 0) {
            // cur_v reach the root node, switch to w
            cur_w = v;
            offset_length_cur_w = this->offset_length_[cur_w];
          } else {
            // move to the next node in the path
            cur_w = paths_[offset_length_cur_w.first + offset_length_cur_w.second - 1];
            offset_length_cur_w = this->offset_length_[cur_w];
          }
        }

        index_t blossom_base = cur_v;
        // reserve space in paths to store the blossom from base to v to w to base

        size_t len_v_to_base = root_distance_[v] - root_distance_[blossom_base];
        size_t len_w_to_base = root_distance_[w] - root_distance_[blossom_base];

        size_t nnodes_v_to_base = len_v_to_base + 1;
        size_t nnodes_w_to_base = len_w_to_base + 1;

        // reserve blossom offset
        blossom_offset = paths_.Reserve(nnodes_v_to_base + nnodes_w_to_base);
        num_nodes_in_blossom = nnodes_v_to_base + nnodes_w_to_base;

        // read v to base in reverse order
        // add v to path buffer
        index_t buffer_idx_v = blossom_offset + nnodes_v_to_base - 1;
        if (buffer_idx_v - 1 > this->path_buffer_size_) {
          // printf("[ERROR] v is at idx: %d\n", buffer_idx_v - 1);
        }

        index_t last_v = v;
        this->paths_[buffer_idx_v--] = v;
        offset_length_cur_v = this->offset_length_[v];
        index_t idx_in_path_v = 0;
        while (last_v != blossom_base) {
          index_t cur_v_to_add = paths_[offset_length_cur_v.first + idx_in_path_v];
          this->paths_[buffer_idx_v--] = cur_v_to_add;
          if (idx_in_path_v == offset_length_cur_v.second - 1) {
            offset_length_cur_v = this->offset_length_[cur_v_to_add];
            idx_in_path_v = 0;
          } else {
            idx_in_path_v += 1;
          }
          last_v = cur_v_to_add;
        }

        // read w to base in regular order in paths_
        index_t buffer_idx_w = blossom_offset + nnodes_v_to_base;
        // add w to path buffer
        this->paths_[buffer_idx_w++] = w;
        offset_length_cur_w = this->offset_length_[w];
        index_t last_w = w;
        index_t idx_in_path_w = 0;
        while (last_w != blossom_base) {
          index_t cur_w_to_add = paths_[offset_length_cur_w.first + idx_in_path_w];
          this->paths_[buffer_idx_w++] = cur_w_to_add;
          if (idx_in_path_w == offset_length_cur_w.second - 1) {
            offset_length_cur_w = this->offset_length_[cur_w_to_add];
            idx_in_path_w = 0;
          } else {
            idx_in_path_w += 1;
          }
          last_w = cur_w_to_add;
        }

        v_idx_in_blossom = nnodes_v_to_base - 1;
      }

      DEV_INLINE void FindAndAddBlossomImplicitByAnchorNodeInBlossomBuffer(index_t v,
                                                            index_t w,
                                                            index_t &blossom_offset,
                                                            size_t &num_nodes_in_blossom,
                                                            index_t &v_idx_in_blossom) {
        // find the blossom base
        index_t cur_v = v;
        index_t cur_w = w;

        DPair<index_t, size_t> offset_length_cur_v = this->offset_length_[cur_v];
        DPair<index_t, size_t> offset_length_cur_w = this->offset_length_[cur_w];

        int it = 0;
        while (cur_v != cur_w) {
          if (offset_length_cur_v.second == 0) {
            // cur_v reach the root node, switch to w
            cur_v = w;
            offset_length_cur_v = this->offset_length_[cur_v];
          } else {
            // move to the next node in the path
            cur_v = paths_[offset_length_cur_v.first + offset_length_cur_v.second - 1];
            offset_length_cur_v = this->offset_length_[cur_v];
          }

          if (offset_length_cur_w.second == 0) {
            // cur_v reach the root node, switch to w
            cur_w = v;
            offset_length_cur_w = this->offset_length_[cur_w];
          } else {
            // move to the next node in the path
            cur_w = paths_[offset_length_cur_w.first + offset_length_cur_w.second - 1];
            offset_length_cur_w = this->offset_length_[cur_w];
          }
        }

        index_t blossom_base = cur_v;
        // reserve space in paths to store the blossom from base to v to w to base

        size_t len_v_to_base = root_distance_[v] - root_distance_[blossom_base];
        size_t len_w_to_base = root_distance_[w] - root_distance_[blossom_base];

        size_t nnodes_v_to_base = len_v_to_base + 1;
        size_t nnodes_w_to_base = len_w_to_base + 1;

        // reserve blossom offset
        blossom_offset = blossom_buffer_.Reserve(nnodes_v_to_base + nnodes_w_to_base);
        num_nodes_in_blossom = nnodes_v_to_base + nnodes_w_to_base;

        // read v to base in reverse order
        // add v to path buffer
        index_t buffer_idx_v = blossom_offset + nnodes_v_to_base - 1;
        if (buffer_idx_v - 1 > this->path_buffer_size_) {
          // printf("[ERROR] v is at idx: %d\n", buffer_idx_v - 1);
        }

        index_t last_v = v;
        this->blossom_buffer_[buffer_idx_v--] = v;
        offset_length_cur_v = this->offset_length_[v];
        index_t idx_in_path_v = 0;
        while (last_v != blossom_base) {
          index_t cur_v_to_add = paths_[offset_length_cur_v.first + idx_in_path_v];
          this->blossom_buffer_[buffer_idx_v--] = cur_v_to_add;
          if (idx_in_path_v == offset_length_cur_v.second - 1) {
            offset_length_cur_v = this->offset_length_[cur_v_to_add];
            idx_in_path_v = 0;
          } else {
            idx_in_path_v += 1;
          }
          last_v = cur_v_to_add;
        }

        // read w to base in regular order blossom_buffer_
        index_t buffer_idx_w = blossom_offset + nnodes_v_to_base;
        // add w to path buffer
        this->blossom_buffer_[buffer_idx_w++] = w;
        offset_length_cur_w = this->offset_length_[w];
        index_t last_w = w;
        index_t idx_in_path_w = 0;
        while (last_w != blossom_base) {
          index_t cur_w_to_add = paths_[offset_length_cur_w.first + idx_in_path_w];
          this->blossom_buffer_[buffer_idx_w++] = cur_w_to_add;
          if (idx_in_path_w == offset_length_cur_w.second - 1) {
            offset_length_cur_w = this->offset_length_[cur_w_to_add];
            idx_in_path_w = 0;
          } else {
            idx_in_path_w += 1;
          }
          last_w = cur_w_to_add;
        }

        v_idx_in_blossom = nnodes_v_to_base - 1;
      }

      DEV_INLINE void FindBlossomBaseByAnchorNode(index_t v,
                                                  index_t w,
                                                  index_t &blossom_base) {
        // find the length of blossom
        index_t cur_v = v;
        index_t cur_w = w;

        DPair<index_t, size_t> offset_length_cur_v = this->offset_length_[cur_v];
        DPair<index_t, size_t> offset_length_cur_w = this->offset_length_[cur_w];

        int it = 0;
        while (cur_v != cur_w) {
          if (offset_length_cur_v.second == 0) {
            // cur_v reach the root node, switch to w
            cur_v = w;
            offset_length_cur_v = this->offset_length_[cur_v];
          } else {
            // move to the next node in the path
            cur_v = paths_[offset_length_cur_v.first + offset_length_cur_v.second - 1];
            offset_length_cur_v = this->offset_length_[cur_v];
          }

          if (offset_length_cur_w.second == 0) {
            // cur_v reach the root node, switch to w
            cur_w = v;
            offset_length_cur_w = this->offset_length_[cur_w];
          } else {
            // move to the next node in the path
            cur_w = paths_[offset_length_cur_w.first + offset_length_cur_w.second - 1];
            offset_length_cur_w = this->offset_length_[cur_w];
          }
        }

        blossom_base = cur_v;
      }

      DEV_INLINE void ConstructBlossom(index_t v,
                                       index_t w,
                                       index_t blossom_base,
                                       index_t &blossom_offset,
                                       size_t &num_nodes_in_blossom) {
        // reserve space in paths to store the blossom from base to v to w to base

        size_t len_v_to_base = root_distance_[v] - root_distance_[blossom_base];
        size_t len_w_to_base = root_distance_[w] - root_distance_[blossom_base];

        size_t nnodes_v_to_base = len_v_to_base + 1;
        size_t nnodes_w_to_base = len_w_to_base + 1;

        // reserve blossom offset
        blossom_offset = paths_.Reserve(nnodes_v_to_base + nnodes_w_to_base);
        num_nodes_in_blossom = nnodes_v_to_base + nnodes_w_to_base;

        // read v to base in reverse order
        // add v to path buffer
        index_t buffer_idx_v = blossom_offset + nnodes_v_to_base - 1;
        if (buffer_idx_v - 1 > this->path_buffer_size_) {
          // printf("[ERROR] v is at idx: %d\n", buffer_idx_v - 1);
        }

        index_t last_v = v;
        this->paths_[buffer_idx_v--] = v;
        DPair<index_t, size_t> offset_length_cur_v = this->offset_length_[v];
        index_t idx_in_path_v = 0;
        while (last_v != blossom_base) {
          index_t cur_v_to_add = paths_[offset_length_cur_v.first + idx_in_path_v];
          this->paths_[buffer_idx_v--] = cur_v_to_add;
          if (idx_in_path_v == offset_length_cur_v.second - 1) {
            offset_length_cur_v = this->offset_length_[cur_v_to_add];
            idx_in_path_v = 0;
          } else {
            idx_in_path_v += 1;
          }
          last_v = cur_v_to_add;
        }

        // read w to base in regular order in paths_
        index_t buffer_idx_w = blossom_offset + nnodes_v_to_base;
        // add w to path buffer
        this->paths_[buffer_idx_w++] = w;
        DPair<index_t, size_t> offset_length_cur_w = this->offset_length_[w];
        index_t last_w = w;
        index_t idx_in_path_w = 0;
        while (last_w != blossom_base) {
          index_t cur_w_to_add = paths_[offset_length_cur_w.first + idx_in_path_w];
          this->paths_[buffer_idx_w++] = cur_w_to_add;
          if (idx_in_path_w == offset_length_cur_w.second - 1) {
            offset_length_cur_w = this->offset_length_[cur_w_to_add];
            idx_in_path_w = 0;
          } else {
            idx_in_path_w += 1;
          }
          last_w = cur_w_to_add;
        }

        __threadfence();
        // size_t nnodes_in_buffer_w = 1;
        // while (nnodes_in_buffer_w < nnodes_w_to_base) {
        //   DPair<index_t, size_t> offset_length_w = offset_length_[cur_w];
        //   index_t offset = offset_length_w.first;
        //   size_t length = offset_length_w.second;
        //
        //   for (index_t i = 0; i < length; i++) {
        //     this->paths_[buffer_idx_w++] = paths_[offset + i];
        //   }
        //
        //   nnodes_in_buffer_w += length;
        //   cur_w = paths_[offset + length - 1];
        // }

        // v_idx_in_blossom = nnodes_v_to_base - 1;
      }


      DEV_INLINE void ExpandNodesFromBlossomReusePath(index_t node,
                                                      index_t blossom_base,
                                                      index_t path_offset,
                                                      index_t path_len) {
        // only root_distance and offset_length are updated
        // this->root_distance_[node] = path_len + this->root_distance_[blossom_base];
        atomicExch(&this->root_distance_[node], path_len + this->root_distance_[blossom_base]);

        __threadfence(); // <- make the list visible before meta‑data

        offset_length_[node] = {path_offset, path_len};

        __threadfence(); // <- make the list visible before meta‑data
      }

      DEV_INLINE void ExpandNodesFromBlossomRightwards(index_t node,
                                                       index_t blossom_offset,
                                                       size_t blossom_length,
                                                       index_t start_idx) {
        // 1. update metadata - offset and length
        // start idx is in [0, blossom_length - 1]
        // size_t path_len = blossom_length - 1 - start_idx + 1;
        size_t path_len = blossom_length - start_idx;
        // printf("Node %d has path_len %lu\n", node, path_len);
        // index_t path_offset = atomicAdd(paths_offsets_.Addr(), path_len);
        index_t path_offset = paths_.Reserve(path_len);

        // 3. expands nodes (current node to base) to paths
        for (index_t i = 0; i < path_len; i++) {
          this->paths_[path_offset + i] = this->blossom_buffer_[blossom_offset + start_idx + i];
        }

        __threadfence(); // <- make the list visible before meta‑data

        // check duplication
        // if (this->CheckDuplicationInPath(path_offset, path_len)) {
        //   // printf("Duplication Detected in Path!\n");
        //   return;
        // }

        // 2. update root length
        index_t base = this->blossom_buffer_[blossom_offset + blossom_length - 1];
        this->root_distance_[node] = path_len + this->root_distance_[base];

        __threadfence();

        offset_length_[node] = {path_offset, path_len};
        __threadfence();
      }

      DEV_INLINE void ExpandNodesFromBlossomLeftwards(index_t node,
                                                      index_t blossom_offset,
                                                      index_t start_idx) {
        // 1. update metadata - offset and length
        // start idx is in [0, blossom_length - 1]
        // size_t path_len = start_idx - 0 + 1;
        size_t path_len = start_idx + 1;
        // printf("Node %d has path_len %lu\n", node, path_len);
        // index_t path_offset = atomicAdd(paths_offsets_.Addr(), path_len);
        index_t path_offset = paths_.Reserve(path_len);

        // 2. update root length
        index_t base = this->blossom_buffer_[blossom_offset + 0];
        this->root_distance_[node] = path_len + this->root_distance_[base];

        // 3. expands nodes (current node to base) to paths
        for (index_t i = 0; i < path_len; i++) {
          this->paths_[path_offset + i] = this->blossom_buffer_[blossom_offset + start_idx - i];
        }

        __threadfence(); // <- make the list visible before meta‑data

        offset_length_[node] = {path_offset, path_len};

        __threadfence(); // <- make the list visible before meta‑data
      }

      DEV_INLINE void ExpandNodesFromBlossomDirectional(index_t node,
                                                  index_t blossom_offset,
                                                  size_t  blossom_len,
                                                  index_t start_idx,  // [0, blossom_len)
                                                  size_t  path_len,
                                                  int     step) {     // +1 or -1
        // Reserve path storage
        index_t path_offset = paths_.Reserve(path_len);

        // Choose base and set root distance
        index_t base = this->blossom_buffer_[blossom_offset + (step > 0 ? (blossom_len - 1) : 0)];
        this->root_distance_[node] = path_len + this->root_distance_[base];

        // Materialize nodes into paths_
        index_t start = blossom_offset + start_idx;
        for (index_t k = 0; k < static_cast<index_t>(path_len); ++k) {
          this->paths_[path_offset + k] = this->blossom_buffer_[start + k * step];
        }

        __threadfence();
        offset_length_[node] = {path_offset, path_len};
        __threadfence();
      }

      // ********************************************************************************************
      // Just For Testing
      DEV_INLINE void ExpandNodesFromBlossomRightwardsTillV(index_t blossom_offset,
                                                            size_t nnodes_in_blossom,
                                                            index_t v_idx) {
        for (int i = 1; i < v_idx; i += 2) {
          if (this->IsPathEmpty(i)) {
            index_t node = this->paths_[blossom_offset + i];

            // bug fixed: offset = blossom_offset + i + 1, the node right after it, not itself
            index_t offset = blossom_offset + i + 1;
            // (blossom length - 1) - i + 1 (- 1) (exclude node itself)
            size_t length = nnodes_in_blossom - static_cast<size_t>(i) - 1;

            index_t blossom_base = this->paths_[blossom_offset];

            this->ExpandNodesFromBlossomReusePath(node, blossom_base, offset, length);

            __threadfence();
          }
        }
      }

      // ********************************************************************************************
      DEV_INLINE index_t GetNodeFromBlossomInBlossomBuffer(index_t blossom_offset, index_t idx) {
        return this->blossom_buffer_[blossom_offset + idx];
      }

      DEV_INLINE index_t GetNodeFromBlossom(index_t blossom_offset, index_t idx) {
        return this->paths_[blossom_offset + idx];
      }

      DEV_INLINE bool CheckDuplication(index_t path_offset, index_t path_len) {
        // Check for duplicates in blossom
        for (index_t i = path_offset; i < path_offset + path_len; i++) {
          index_t node_i = paths_[i];
          for (index_t j = i + 1; j < path_offset + path_len; j++) {
            if (node_i == paths_[j]) {
              return true;
            }
          }
        }
        return false;
      }

      DEV_INLINE bool CheckDuplicationInBlossomBuffer(index_t blossom_offset, index_t blossom_len) {
        // Check for duplicates in blossom
        for (index_t i = blossom_offset + 1; i < blossom_offset + blossom_len - 1; i++) {
          index_t node_i = this->blossom_buffer_[i];
          for (index_t j = i + 1; j < blossom_offset + blossom_len; j++) {
            if (node_i == this->blossom_buffer_[j]) {
              return true;
            }
          }
        }
        return false;
      }

      DEV_INLINE bool CheckPathDuplicationInBlossomBufferRightwards(index_t path_offset_in_blossom_buffer,
                                                                    index_t nnodes_in_path) {
        // Check for duplicates inpath
        for (index_t i = path_offset_in_blossom_buffer; i < path_offset_in_blossom_buffer + nnodes_in_path; i++) {
          index_t node_i = this->blossom_buffer_[i];
          for (index_t j = i + 1; j < path_offset_in_blossom_buffer + nnodes_in_path; j++) {
            if (node_i == this->blossom_buffer_[j]) {
              return true;
            }
          }
        }
        return false;
      }

      DEV_INLINE bool CheckPathDuplicationInBlossomBufferLeftwards(index_t path_offset_in_blossom_buffer,
                                                                   index_t nnodes_in_path) {
        // Check for duplicates in path
        for (index_t i = path_offset_in_blossom_buffer; i >= path_offset_in_blossom_buffer + 1 - nnodes_in_path; i--) {
          index_t node_i = this->blossom_buffer_[i];
          for (index_t j = i - 1; j >= path_offset_in_blossom_buffer + 1 - nnodes_in_path; j--) {
            if (node_i == this->blossom_buffer_[j]) {
              return true;
            }
          }
        }
        return false;
      }

      DEV_INLINE bool CheckPathDuplicationInBlossomBufferDirectional(index_t blossom_offset,
                                                                     index_t start_idx, // index in [0, nnodes)
                                                                     size_t len,
                                                                     int step) {
        // +1 or -1
        // Compare all pairs within the span: buffer[ start, start+step, ..., start+(len-1)*step ]
        index_t start = blossom_offset + start_idx;
        for (index_t a = 0; a < static_cast<index_t>(len); ++a) {
          index_t ia = start + a * step;
          index_t node_a = this->blossom_buffer_[ia];
          for (index_t b = a + 1; b < static_cast<index_t>(len); ++b) {
            index_t ib = start + b * step;
            if (node_a == this->blossom_buffer_[ib]) return true;
          }
        }
        return false;
      }

      // ===================== For ParAugPath() --- Transform Odd Nodes =============================
      // Same implementation as IsPathEmpty
      DEV_INLINE bool IsRoot(index_t v) const {
        DPair<index_t, size_t> offset_length_pair = offset_length_[v];
        return offset_length_pair.second == 0;
      }

      DEV_INLINE index_t GetRootDistance(index_t v) const {
        return this->root_distance_[v];
      }

      DEV_INLINE DPair<index_t, size_t> GetOffsetLength(index_t v) const {
        return this->offset_length_[v];
      }

      DEV_INLINE index_t GetAnchorNode(DPair<index_t, size_t> offset_length) const {
        return this->paths_[offset_length.first + offset_length.second - 1];
      }

      DEV_INLINE index_t GetNode(index_t offset, index_t idx_in_path) const {
        return this->paths_[offset + idx_in_path];
      }

      DEV_INLINE void AddAugmentingPath(index_t v, DArrayView<index_t> matching) {
        // Traverse the augmenting path starting from v
        index_t cur_v = v;
        DPair<index_t, size_t> offset_length = offset_length_[cur_v];

        while (offset_length.second != 0) {
          index_t offset = offset_length.first;
          size_t length = offset_length.second;

          // Iterate over pairs of nodes along the augmenting path and update their matchings
          for (index_t i = 0; i < length; i += 2) {
            index_t node1 = paths_[offset + i];
            index_t node2 = paths_[offset + i + 1];
            // matching[node1] = node2;
            // matching[node2] = node1;
            atomicExch(&matching[node1], node2);
            atomicExch(&matching[node2], node1);
          }

          // update the cur_v
          cur_v = paths_[offset + length - 1];
          offset_length = offset_length_[cur_v];
        }
      }

      // ===================== For ParAugPath() --- Construction of Blossom  =============================
      // Same implementation as IsPathEmpty
      DEV_INLINE index_t ReserveBlossom(size_t nnnodes_in_blossom) {
        // reserve blossom offset
        return paths_.Reserve(nnnodes_in_blossom);
      }

      DEV_INLINE void SetNode(index_t set_node, index_t set_node_idx_in_path) {
        // reserve blossom offset
        paths_[set_node_idx_in_path] = set_node;
      }

      // For Lazy Init
      DEV_INLINE void ResetNode(index_t node) {
        // reserve blossom offset
        root_distance_[node] = 0;
        offset_length_[node] = {0, 0};
      }

    private
    :
      size_t nnnodes_;
      size_t path_buffer_size_;

      // Store (offset, length) pair for paths either to the last even node or blossom base
      DArrayView<DPair<index_t, size_t> > offset_length_;
      // length from a node to the root of its alternating tree: helpful when reconstructing a path
      DArrayView<index_t> root_distance_;
      // Paths either to the last even node or blossom base

      // Global offsets for atomic reservation for paths
      DListView<index_t> paths_;

      DListView<index_t> blossom_buffer_;
    };

    inline PathTableView PathTable::DeviceView() {
      return PathTableView(nnnodes_,
                           path_buffer_size_,
                           this->offset_length_.DeviceView(),
                           this->root_distance_.DeviceView(),
                           this->paths_.DeviceView(),
                           this->blossom_buffer_.DeviceView()
      );
    }
  }
}

#endif //PATH_TABLE_H
