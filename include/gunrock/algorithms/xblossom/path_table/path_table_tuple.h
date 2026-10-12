#ifndef PATH_TABLE_TUPLE_H
#define PATH_TABLE_TUPLE_H
// Vendored from X-Blossom (src/include/path_table/path_table_tuple.h) for the Gunrock X-Blossom app.
// The tuple path table (XB_PATH_TABLE_TUPLE 1, flat nomat only): a node's
// path to the root is not written out; every node keeps a fixed record of
// three vertex ids (3 x 4 B, 3|V| in all, no path buffer, no blossom buffer,
// no root distances):
//
//   empty (root or odd node)      (NONE, -, -)
//   even node x reached by expand (w, v, NONE)   w = x's mate, v = w's tree
//                                                parent: the segment [w, v]
//                                                of the path table as is
//   odd node u made even by a     (s, o, base)   the route u ~> s -> o ~> base,
//   blossom with closing edge                    then base's own route
//   (s, o) and base
//
// A record is read only from the node it belongs to, so u is implicit; the
// route of u is reverse(route(s) before u), o, route(o) up to base, route(base).
// The engine never needs it in order: flipping an augmenting path needs its
// matched pairs (an unordered set, the same pairs route(s) holds before u),
// the base walk needs only each record's anchor (v or base), and claiming a
// chain's odd nodes needs only the w of every TWO record on the anchor chain
// (every node inside a BLOSSOM record's own parts was made even when that
// blossom was processed, so the chains' odd nodes are exactly those w).
// Flipping recurses into BLOSSOM records with an explicit per-thread stack
// (one entry per blossom nesting level; its peak is reported).
#include <vector>
#include <utils/dev_array.h>
#include <utils/types_gpu.h>
#include <utils/dev_value.h>

#include "utils/cpp_types.h"

#define XB_TUPLE_FLIP_STACK 64

namespace zblossom {
  namespace dev {
    class TuplePathTableView;

    class TuplePathTable {
    public:
      explicit TuplePathTable(size_t nnnodes)
        : nnnodes_(nnnodes), rec_(3 * nnnodes), flip_stack_peak_() {
        flip_stack_peak_.SetH2D(0);
      }

      // the buffer ratio of the materialising table is accepted and ignored
      explicit TuplePathTable(size_t nnnodes, size_t /*buffer_ratio*/)
        : nnnodes_(nnnodes), rec_(3 * nnnodes), flip_stack_peak_() {
        flip_stack_peak_.SetH2D(0);
      }

      void ResetPathTable() { rec_.Fill(static_cast<index_t>(nnnodes_)); }
      void ResetBlossomBuffer() {}
      void ResetPathTableBuffer() {}

      // Occupancy: the materialising table's counters are zero here; the
      // table is the records alone
      size_t PathsUsed() const { return 0; }
      size_t BlossomBufferUsed() const { return 0; }
      size_t BufferCapacity() const { return 0; }
      size_t FixedBytes() const { return 3 * nnnodes_ * sizeof(index_t); }
      int FlipStackPeak() const { return flip_stack_peak_.GetD2H(); }

      TuplePathTableView DeviceView();

    private:
      size_t nnnodes_;
      DArray<index_t> rec_; // 3 per node
      DValue<int> flip_stack_peak_;
    };

    class TuplePathTableView {
    public:
      DEV_HOST_INLINE
      TuplePathTableView(size_t nnnodes, DArrayView<index_t> rec, DValueView<int> flip_stack_peak)
        : nnnodes_(nnnodes), none_(static_cast<index_t>(nnnodes)), rec_(rec), flip_stack_peak_(flip_stack_peak) {
      }

      struct Rec {
        index_t a, b, c;
      };

      DEV_INLINE Rec Load(index_t x) const {
        Rec r;
        r.a = rec_[3 * static_cast<size_t>(x)];
        r.b = rec_[3 * static_cast<size_t>(x) + 1];
        r.c = rec_[3 * static_cast<size_t>(x) + 2];
        return r;
      }
      DEV_INLINE void Store(index_t x, index_t a, index_t b, index_t c) {
        rec_[3 * static_cast<size_t>(x)] = a;
        rec_[3 * static_cast<size_t>(x) + 1] = b;
        rec_[3 * static_cast<size_t>(x) + 2] = c;
      }
      DEV_INLINE bool IsEmpty(const Rec &r) const { return r.a == none_; }
      DEV_INLINE bool IsTwo(const Rec &r) const { return r.c == none_; }
      // the next even node towards the root (the segment's last node)
      DEV_INLINE index_t Anchor(const Rec &r) const { return IsTwo(r) ? r.b : r.c; }

      DEV_INLINE bool IsPathEmpty(index_t v) const { return IsEmpty(Load(v)); }

      // For Lazy Init
      DEV_INLINE void ResetNode(index_t node) { Store(node, none_, none_, none_); }

      // x = w's mate becomes even through even v: record (w, v)
      DEV_INLINE void ExpandTwoNodes(index_t v, index_t w, index_t x) {
        Store(x, w, v, none_);
        __threadfence();
      }

      // Odd node u on s's chain becomes even around the blossom with closing
      // edge (s, o) and base: record (s, o, base). O(1): no chain is walked
      DEV_INLINE void ExpandOddNodeAroundBlossom(index_t u, index_t s, index_t o, index_t base) {
        Store(u, s, o, base);
        __threadfence();
      }

      // The blossom base of closing edge (v, w): the first anchor the two
      // anchor chains share (each walks to its root, then switches to the
      // other side; same as PathTableView::FindBlossomBaseByAnchorNode)
      DEV_INLINE void FindBlossomBaseByAnchorNode(index_t v, index_t w, index_t &blossom_base) const {
        index_t cur_v = v, cur_w = w;
        Rec rv = Load(cur_v), rw = Load(cur_w);
        while (cur_v != cur_w) {
          if (IsEmpty(rv)) { cur_v = w; } else { cur_v = Anchor(rv); }
          rv = Load(cur_v);
          if (IsEmpty(rw)) { cur_w = v; } else { cur_w = Anchor(rw); }
          rw = Load(cur_w);
        }
        blossom_base = cur_v;
      }

      // f(n) for every node n that was odd when it was recorded, on s's
      // anchor chain down to base (the w of each TWO record; a BLOSSOM
      // record's own parts hold only even nodes and are skipped)
      template<typename F>
      DEV_INLINE void ForEachRecordedOddOnChain(index_t s, index_t base, F f) const {
        index_t cur = s;
        while (cur != base) {
          const Rec r = Load(cur);
          if (IsEmpty(r)) return; // the root without meeting the base: cannot happen in a static forest
          if (IsTwo(r)) f(r.a);
          cur = Anchor(r);
        }
      }

      // Flip the matching along the augmenting path v ~> root_v, (v, w), w ~> root_w
      DEV_INLINE void AddAugmentingPath(index_t v, index_t w, DArrayView<index_t> matching) {
        atomicExch(&matching[v], w);
        atomicExch(&matching[w], v);
        FlipRoute(v, matching);
        FlipRoute(w, matching);
        __threadfence();
      }

    private:
      // A pending walk: from x up to stop, which is an anchor (stop >= 0) or,
      // encoded as ~stop, the w of a TWO record (an odd node); none_: the root
      struct Walk {
        index_t x, stop;
      };

      DEV_INLINE void Flip(index_t a, index_t b, DArrayView<index_t> &matching) const {
        atomicExch(&matching[a], b);
        atomicExch(&matching[b], a);
      }

      // The matched pairs of route(x) up to the root, set as the new matching.
      // A BLOSSOM record (s, o, base) met at u contributes (s, o), the pairs of
      // route(s) before u, and the pairs of route(o): route(o) reaches base as
      // an anchor and continues as route(base), so "route(o) up to this walk's
      // stop" is the rest of the walk. The walk of route(s) before u is done
      // first, the walk of route(o) kept on a stack (one entry per nesting level)
      DEV_INLINE void FlipRoute(index_t x, DArrayView<index_t> &matching) {
        Walk stack[XB_TUPLE_FLIP_STACK];
        int depth = 0, peak = 0;
        Walk cur{x, none_};
        while (true) {
          while (true) {
            if (cur.stop >= 0 && cur.x == cur.stop) break; // the anchor stop
            const Rec r = Load(cur.x);
            if (IsEmpty(r)) break; // the root
            if (IsTwo(r)) {
              if (cur.stop < 0 && r.a == ~cur.stop) break; // the odd stop
              Flip(r.a, r.b, matching);
              cur.x = r.b;
            } else {
              assert(depth < XB_TUPLE_FLIP_STACK);
              stack[depth++] = Walk{r.b, cur.stop};
              if (depth > peak) peak = depth;
              Flip(r.a, r.b, matching);
              cur = Walk{r.a, ~cur.x}; // route(s) before u
            }
          }
          if (depth == 0) break;
          cur = stack[--depth];
        }
        if (peak > 0) atomicMax(&flip_stack_peak_, peak);
      }

      size_t nnnodes_;
      index_t none_;
      DArrayView<index_t> rec_;
      DValueView<int> flip_stack_peak_;
    };

    inline TuplePathTableView TuplePathTable::DeviceView() {
      return TuplePathTableView(nnnodes_, rec_.DeviceView(), flip_stack_peak_.DeviceView());
    }
  }
}

#endif //PATH_TABLE_TUPLE_H
