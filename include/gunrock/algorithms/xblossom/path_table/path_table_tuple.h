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
//
// mat on the compact table (constructed with mat = true): the blossom phase
// still materialises every closing edge's cycle, as X-Blossom++'s mat does,
// into a cycle buffer, expanding the records in order; to place each part of
// a nested record at its offset it keeps a root distance per node (the number
// of nodes on the route after it), 4 B per node, mat only. A converted odd
// node then gets its record (m, n, b) from the cycle (the closing edge's
// endpoint on its side, the other one, the cycle's first node) instead of a
// copied segment.
#include <vector>
#include <utils/dev_array.h>
#include <utils/types_gpu.h>
#include <utils/dev_value.h>

#include "utils/cpp_types.h"
#include "utils/dev_list.h"

#define XB_TUPLE_FLIP_STACK 256  // XB-GAP's depth
#define XB_TUPLE_CYCLE_STACK 128

namespace zblossom {
  namespace dev {
    class TuplePathTableView;

    class TuplePathTable {
    public:
      explicit TuplePathTable(size_t nnnodes)
        : TuplePathTable(nnnodes, 0, false) {}

      // buffer_ratio sizes the cycle buffer (ratio x n nodes) under mat; nomat
      // keeps the records alone
      // the engine's table-agnostic constructor (no segment buffer here)
      explicit TuplePathTable(size_t nnnodes, size_t cycle_ratio, size_t /*segment_ratio*/, bool mat)
        : TuplePathTable(nnnodes, cycle_ratio, mat) {}

      explicit TuplePathTable(size_t nnnodes, size_t buffer_ratio, bool mat = false)
        : nnnodes_(nnnodes), mat_(mat), rec_(3 * nnnodes), rd_(mat ? nnnodes : 1),
          cycles_(mat ? buffer_ratio * nnnodes : 1), flip_stack_peak_(), cycle_overflow_() {
        flip_stack_peak_.SetH2D(0);
        cycle_overflow_.SetH2D(0);
        ResetCounters();
      }

      // Traversal counters (GACGE_TRAVERSAL_STATS; an atomic per base walk and
      // per augmenting path, so off by default): base walks and their lockstep
      // steps, augmenting paths and the matched pairs they flip
      void EnableCounters(bool on = true) { count_ = on; }
      void ResetCounters() {
        for (auto *c : {&walks_, &walk_steps_, &paths_, &path_flips_, &direct_paths_, &max_flips_}) c->SetH2D(0);
      }
      unsigned long long DirectAugmentingPaths() const { return direct_paths_.GetD2H(); } // a single edge, nothing flipped
      unsigned long long MaxAugmentingPathFlips() const { return max_flips_.GetD2H(); }
      unsigned long long BaseWalks() const { return walks_.GetD2H(); }
      unsigned long long BaseWalkSteps() const { return walk_steps_.GetD2H(); }
      unsigned long long AugmentingPaths() const { return paths_.GetD2H(); }
      unsigned long long AugmentingPathFlips() const { return path_flips_.GetD2H(); }

      void ResetPathTable() { rec_.Fill(static_cast<index_t>(nnnodes_)); }
      void ResetBlossomBuffer() { if (mat_) cycles_.Clear(); }
      void ResetPathTableBuffer() {}

      // Occupancy: the materialising table's counters are zero here; the
      // table is the records alone
      size_t PathsUsed() const { return 0; }
      size_t BlossomBufferUsed() const { return mat_ ? cycles_.size() : 0; }
      size_t BufferCapacity() const { return mat_ ? cycles_.capacity() : 0; }
      // the records, plus the root distances under mat
      size_t FixedBytes() const { return (mat_ ? 4 : 3) * nnnodes_ * sizeof(index_t); }
      int FlipStackPeak() const { return flip_stack_peak_.GetD2H(); }
      // mat: a closing edge's cycle did not fit the cycle buffer (the run is then invalid)
      bool CycleOverflow() const { return mat_ && cycle_overflow_.GetD2H() != 0; }

      TuplePathTableView DeviceView();

    private:
      size_t nnnodes_;
      bool mat_;
      DArray<index_t> rec_; // 3 per node
      DArray<index_t> rd_;  // mat: nodes on the route after each node
      DList<index_t> cycles_; // mat: the materialised cycles of one step
      DValue<int> flip_stack_peak_;
      DValue<int> cycle_overflow_;
      bool count_{false};
      DValue<unsigned long long> walks_, walk_steps_, paths_, path_flips_, direct_paths_, max_flips_;
    };

    class TuplePathTableView {
    public:
      DEV_HOST_INLINE
      TuplePathTableView(size_t nnnodes, bool mat, DArrayView<index_t> rec, DArrayView<index_t> rd,
                         DListView<index_t> cycles, DValueView<int> flip_stack_peak, DValueView<int> cycle_overflow,
                         bool count, DValueView<unsigned long long> walks, DValueView<unsigned long long> walk_steps,
                         DValueView<unsigned long long> paths, DValueView<unsigned long long> path_flips,
                         DValueView<unsigned long long> direct_paths, DValueView<unsigned long long> max_flips)
        : nnnodes_(nnnodes), none_(static_cast<index_t>(nnnodes)), mat_(mat), rec_(rec), rd_(rd),
          cycles_(cycles), flip_stack_peak_(flip_stack_peak), cycle_overflow_(cycle_overflow), count_(count),
          walks_(walks), walk_steps_(walk_steps), paths_(paths), path_flips_(path_flips), direct_paths_(direct_paths),
          max_flips_(max_flips) {
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
      DEV_INLINE void ResetNode(index_t node) {
        Store(node, none_, none_, none_);
        if (mat_) rd_[node] = 0;
      }

      // x = w's mate becomes even through even v: record (w, v)
      DEV_INLINE void ExpandTwoNodes(index_t v, index_t w, index_t x) {
        Store(x, w, v, none_);
        if (mat_) rd_[x] = 2 + rd_[v];
        __threadfence();
      }

      // ---- mat on the compact table ------------------------------------------------
      // The cycle of closing edge (v, w) into the cycle buffer, laid out as the
      // materialising table lays it out: base ... v | w ... base (v at v_idx,
      // w at v_idx + 1, base first and last). Each side is the route of its
      // endpoint down to the base, expanded from the records.
      // Returns false, and flags the table, when the cycle does not fit the buffer
      // (the release build has no bounds check, and a cycle written past the end
      // corrupts what the walks read)
      DEV_INLINE bool MaterializeCycle(index_t v, index_t w, const DArrayView<index_t> &matching,
                                       offset_t &offset, size_t &len, index_t &v_idx) {
        index_t base;
        FindBlossomBaseByAnchorNode(v, w, base);
        const size_t nv = static_cast<size_t>(rd_[v] - rd_[base]) + 1;
        const size_t nw = static_cast<size_t>(rd_[w] - rd_[base]) + 1;
        offset = cycles_.Reserve(nv + nw);
        if (static_cast<size_t>(offset) + nv + nw > cycles_.capacity()) {
          atomicExch(&cycle_overflow_, 1);
          return false;
        }
        WriteRoute(v, base, offset + static_cast<offset_t>(nv) - 1, -1, matching);
        WriteRoute(w, base, offset + static_cast<offset_t>(nv), +1, matching);
        len = nv + nw;
        v_idx = static_cast<index_t>(nv - 1);
        return true;
      }
      DEV_INLINE index_t CycleNode(offset_t offset, index_t idx) const { return cycles_[offset + idx]; }
      // a repeated node on the span of the cycle a conversion would use (as mat checks)
      DEV_INLINE bool CycleSpanHasDuplicate(offset_t offset, index_t start_idx, size_t len, int step) const {
        const offset_t start = offset + start_idx;
        for (index_t a = 0; a < static_cast<index_t>(len); ++a) {
          const index_t node_a = cycles_[start + a * step];
          for (index_t b = a + 1; b < static_cast<index_t>(len); ++b) {
            if (node_a == cycles_[start + b * step]) return true;
          }
        }
        return false;
      }
      // odd node u at odd_idx of the cycle (step +1: v's side, -1: w's side)
      // becomes even: record (m, n, b) and its root distance
      DEV_INLINE void ConvertFromCycle(offset_t offset, index_t v_idx, index_t odd_idx, int step) {
        const index_t u = cycles_[offset + odd_idx];
        const index_t v = cycles_[offset + v_idx], w = cycles_[offset + v_idx + 1];
        const index_t m = step > 0 ? v : w, n = step > 0 ? w : v, b = cycles_[offset];
        const index_t mate = cycles_[offset + odd_idx + step]; // u's tree mate, the next node towards m
        rd_[u] = rd_[m] - rd_[mate] + 2 + rd_[n];
        Store(u, m, n, b);
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
        unsigned long long steps = 0;
        while (cur_v != cur_w) {
          if (IsEmpty(rv)) { cur_v = w; } else { cur_v = Anchor(rv); }
          rv = Load(cur_v);
          if (IsEmpty(rw)) { cur_w = v; } else { cur_w = Anchor(rw); }
          rw = Load(cur_w);
          steps++;
        }
        blossom_base = cur_v;
        last_steps_ = steps;  // GACGE_TASK_STATS: this walk's dependent steps
        if (count_) { atomicAdd(&walks_, 1ULL); atomicAdd(&walk_steps_, steps); }
      }

      // f(n) for every node n that was odd when it was recorded, on s's
      // anchor chain down to base (the w of each TWO record; a BLOSSOM
      // record's own parts hold only even nodes and are skipped)
      template<typename F>
      DEV_INLINE void ForEachRecordedOddOnChain(index_t s, index_t base, F f) const {
        index_t cur = s;
        last_steps_ = 0;  // GACGE_TASK_STATS: the records this chain visits
        while (cur != base) {
          const Rec r = Load(cur);
          last_steps_++;
          if (IsEmpty(r)) return; // the root without meeting the base: cannot happen in a static forest
          if (IsTwo(r)) f(r.a);
          cur = Anchor(r);
        }
      }

      // GACGE_TASK_STATS: the steps of the last base walk, chain scan or path flip of this view
      DEV_INLINE unsigned long long LastSteps() const { return last_steps_; }

      // Flip the matching along the augmenting path v ~> root_v, (v, w), w ~> root_w
      DEV_INLINE void AddAugmentingPath(index_t v, index_t w, DArrayView<index_t> matching) {
        atomicExch(&matching[v], w);
        atomicExch(&matching[w], v);
        unsigned long long flips = 0;
        FlipRoute(v, matching, flips);
        FlipRoute(w, matching, flips);
        last_steps_ = flips;  // GACGE_TASK_STATS: the pairs this path flips
        __threadfence();
        if (count_) {
          atomicAdd(&paths_, 1ULL);
          atomicAdd(&path_flips_, flips);
          if (flips == 0) atomicAdd(&direct_paths_, 1ULL);
          atomicMax(&max_flips_, flips);
        }
      }

    private:
      // Writes route(x) up to the anchor stop (both included) into the cycle
      // buffer, x at pos and each next node at pos + dir. A BLOSSOM record
      // (m, n, b) at u stands for reverse(route(m) up to u's mate), n, route(n)
      // up to b; the root distances give each part's offset, so the parts are
      // independent frames: the reversed one and route(n) are pushed, and the
      // walk goes on from b (written by the route(n) frame). Depth: two frames
      // per nesting level.
      struct Frame {
        index_t x, stop;
        offset_t pos;
        int dir;
      };
      DEV_INLINE void WriteRoute(index_t x, index_t stop, offset_t pos, int dir, const DArrayView<index_t> &matching) {
        Frame stack[XB_TUPLE_CYCLE_STACK];
        int depth = 0;
        stack[depth++] = Frame{x, stop, pos, dir};
        while (depth > 0) {
          Frame f = stack[--depth];
          index_t cur = f.x;
          offset_t p = f.pos;
          cycles_[p] = cur;
          while (cur != f.stop) {
            const Rec r = Load(cur);
            if (IsEmpty(r)) break; // the root (cannot happen before the stop in a static forest)
            if (IsTwo(r)) {
              cycles_[p + f.dir] = r.a;
              cycles_[p + 2 * f.dir] = r.b;
              p += 2 * f.dir;
              cur = r.b;
            } else {
              const index_t mate = matching[cur];
              const offset_t la = static_cast<offset_t>(rd_[r.a] - rd_[mate]) + 1;
              assert(depth + 2 <= XB_TUPLE_CYCLE_STACK);
              stack[depth++] = Frame{r.a, mate, p + f.dir * la, -f.dir};          // reverse(route(m) .. mate)
              stack[depth++] = Frame{r.b, r.c, p + f.dir * (la + 1), f.dir};       // n .. b
              p += f.dir * (la + 1 + static_cast<offset_t>(rd_[r.b] - rd_[r.c]));
              cur = r.c;
            }
          }
        }
      }

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
      DEV_INLINE void FlipRoute(index_t x, DArrayView<index_t> &matching, unsigned long long &flips) {
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
              flips++;
              cur.x = r.b;
            } else {
              if (depth >= XB_TUPLE_FLIP_STACK) {  // XB-GAP aborts: mark the run (peak past the stack)
                atomicMax(&flip_stack_peak_, XB_TUPLE_FLIP_STACK + 1);
                return;
              }
              stack[depth++] = Walk{r.b, cur.stop};
              if (depth > peak) peak = depth;
              Flip(r.a, r.b, matching);
              flips++;
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
      bool mat_;
      DArrayView<index_t> rec_;
      DArrayView<index_t> rd_;
      DListView<index_t> cycles_;
      DValueView<int> flip_stack_peak_;
      DValueView<int> cycle_overflow_;
      bool count_;
      DValueView<unsigned long long> walks_, walk_steps_, paths_, path_flips_, direct_paths_, max_flips_;
      mutable unsigned long long last_steps_ = 0;  // GACGE_TASK_STATS: the last walk's, chain's or flip's steps
    };

    inline TuplePathTableView TuplePathTable::DeviceView() {
      return TuplePathTableView(nnnodes_, mat_, rec_.DeviceView(), rd_.DeviceView(), cycles_.DeviceView(),
                                flip_stack_peak_.DeviceView(), cycle_overflow_.DeviceView(), count_,
                                walks_.DeviceView(), walk_steps_.DeviceView(), paths_.DeviceView(),
                                path_flips_.DeviceView(), direct_paths_.DeviceView(), max_flips_.DeviceView());
    }
  }
}

#endif //PATH_TABLE_TUPLE_H
