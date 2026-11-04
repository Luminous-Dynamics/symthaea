# D4v2 dynamic-BCE exhaustive state/oracle corpus v1

**Claim ceiling:** independent local model of the source-pinned D4v2 manager transition subset plus exhaustive exact projected-support/PWMC oracle. Not a D4 binary differential, full solver, formal proof, or performance benchmark.

## Why this corpus exists

The D4 manager has two distinct watch-lifecycle cases:

1. Constructor cleanup removes watch-list entries for targets already inactive after the initial formula scan. Their protected records can remain in the protected-record arena without a live watch-list entry.
2. A target deleted dynamically by inprocessing leaves the watch parked on the triggering inactive clause. Backtracking restores clause activity and assignments but does not move that watch back.

Thus the valid invariant is conditional: every active target has exactly one valid sentinel; an inactive target may have no watch after constructor cleanup or a parked watch after dynamic deletion. The current sentinel must match the head of its protected record's candidate list; when the target is active and its blocker is unassigned, the sentinel must be active.

## Corpus

- Exhaustive: every one of the 256 CNFs formed from all subsets of the eight nonempty, non-tautological clauses over two variables, crossed with all three nonempty selected-variable partitions. Total: 768 cases.
- Seeded: 250 random three-variable CNFs, each crossed with all seven nonempty selected-variable partitions. Total: 1,750 cases.
- Branch traversal assigns every selected variable in a depth-first binary tree, invokes preUpdate/postUpdate, verifies clause and assignment restoration, and deliberately allows persistent watcher relocation.
- Every case uses nontrivial positive rational literal weights. Exact PWMC comparisons are done with rational arithmetic, never float approximations.

## Expected local reference-model run

| Measure | Exhaustive 2-variable | Seeded 3-variable |
|---|---:|---:|
| Cases | 768 | 1,750 |
| Decision nodes | 3,328 | 11,250 |
| Leaf branches | 2,048 | 6,500 |
| Backtracks | 2,560 | 9,500 |
| Initial deletions | 320 | 960 |
| Watch relocations | 128 | 6,631 |
| Dynamic deletions | 128 | 584 |

All root and leaf checks preserved exact projected support and exact rational PWMC. Full-model counts are not invariant and may change. The reproducible script imports the sibling source-derived manager model and writes a machine-readable result manifest; exact uploaded-script execution is a separate gate from this prior in-memory run.

## Reproduction and next gates

Run python3 d4v2_dynamic_bce_exhaustive_model.py beside verify_d4v2_dynamic_bce_trace.py.

Next: validate the exact uploaded runner, expand to four variables with a bounded corpus, then build a pinned unchanged D4v2 binary harness and compare active-clause/watch/queue state traces. A successful reference test is not a D4 differential or HOL4 theorem.