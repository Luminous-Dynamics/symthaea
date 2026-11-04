# ETH-028 — context-indexed projected BCE trace v1

**Claim ceiling:** two-branch reference trace checker plus exhaustive finite oracle. This is not a D4 trace replay, a complete dynamic-preprocessing verifier, or a formally verified checker.

## Trace example

Root visible variables are x and a; h is hidden. The root CNF is:

- c1 = (h OR x OR a)
- c2 = (NOT h OR x OR NOT a)

The search branches on visible x.

### Branch x = 0

Conditioning removes the false x literals and yields:

- c1 = (h OR a)
- c2 = (NOT h OR NOT a)

The trace deletes c1 using hidden blocker h. Its only opposing clause is c2; the resolvent is (a OR NOT a), with tautology witness a.

Before deletion, this conditioned branch has 2 satisfying full assignments. After deletion it has 3. Its projected support over a remains {0,1}.

### Backtrack and sibling x = 1

The x=0 branch must restore the exact root formula digest when backtracking. The sibling branch x=1 is then conditioned independently; both root clauses are satisfied and its residual formula is True. The x=0 deletion is not allowed to leak into this sibling.

### Root accounting

| Quantity | Original root | Branch-processed |
|---|---:|---:|
| Full satisfying assignments | 6 | 7 |
| Projected support | {00, 01, 10, 11} | {00, 01, 10, 11} |
| PMC | 4 | 4 |
| PWMC | 1 | 1 |

Visible weights are x=false/true = 1/2 and a=false/true = 1/3, 2/3. This fixture demonstrates preservation of projected support and PWMC despite a changed full-model count.

## Receipt checks

The reference checker recomputes:
- root formula digest;
- branch context digest from parent digest, branch ID and partial assignment;
- conditioned formula and residual variable partition;
- per-operation context and source digest;
- hidden blocker eligibility and complete opposing-clause set;
- tautological resolvent and explicit witness;
- exact target formula digest;
- branch-local projected support before/after;
- backtrack restoration to the original root digest;
- union of both branch supports against the original root support.

Ten negative controls target missing/duplicate branches, stale context, conditioned-formula tampering, blocker visibility, missing opposing clauses, cross-branch context reuse, wrong backtrack restoration/ancestry and stale root digests.

## Limitations

- Only one Boolean decision variable and two branches.
- Only one BCE deletion in one branch.
- Exhaustive enumeration is a teaching oracle, not a scalable algorithm.
- Does not replay D4's actual mutable watched-clause data structure.
- Does not yet cover nested decisions, multiple deletions, clause activation/inactivation, implication propagation, restarts, or asynchronous workers.
- Not a HOL4/CakeML formal proof.

## Next gate

Import a real D4 dynamic-BCE event trace or instrument an unchanged D4 build. Extend the receipt with clause active/inactive state, watched-literal changes, propagation reasons, stack frames, nested contexts and restoration events. Differentially compare every event and leaf projected support with D4 on a small exhaustive corpus before scaling.
