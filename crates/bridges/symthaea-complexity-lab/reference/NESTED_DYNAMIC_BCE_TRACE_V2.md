# ETH-028 — nested context-indexed BCE trace v2

**Claim ceiling:** finite nested branch trace checker and exhaustive oracle. Not a D4 dynamic BCE trace replay, not a complete dynamic preprocessing verifier, and not a HOL4/CakeML formal proof.

## Decision tree

The root formula is:

- c1 = (h OR x OR y OR a)
- c2 = (NOT h OR x OR y OR NOT a)

Visible variables are x, y and a; h is hidden. The trace branches on x. Under x=0 it branches again on y. Only the leaf x=0,y=0 performs a BCE deletion, removing c1 using hidden blocker h. Its resolvent against c2 is (a OR NOT a).

The x=0,y=1 sibling is a no-op because y satisfies both root clauses. The x=1 sibling is also a no-op because x satisfies both root clauses. Each child is conditioned from its own parent formula digest and each backtrack receipt restores the exact parent context and formula digest.

## Expected exact accounting

| Quantity | Root source | Leaf-processed |
|---|---:|---:|
| Full satisfying assignments | 14 | 15 |
| Visible support size | 8 | 8 |
| PMC | 8 | 8 |
| PWMC | 1 | 1 |

Visible weights: x and y each have (1/2, 1/2); a has (1/3, 2/3). The single branch-local deletion changes full-model count but preserves root projected support and exact PWMC.

Leaf model counts:
- x=0,y=0: 2 -> 3.
- x=0,y=1: 4 -> 4.
- x=1: 8 -> 8.

The source leaf counts sum to 14, the root count. The processed leaf counts sum to 15. The union of all processed leaf supports equals the root support.

## Mutations

Eleven negative controls cover missing/duplicate siblings, wrong nested parent, cross-branch context reuse, conditioned formula tampering, visible blocker, missing opposing clause coverage, invalid backtrack restoration, sibling contamination and stale root/context digests.

## Reproduction

Run python3 verify_nested_dynamic_bce_trace.py. It writes nested_dynamic_bce_trace_results.v2.json with exact checker/certificate hashes.

## Limitations and next gate

Only two decision levels and three leaves are exercised. The trace does not model D4's actual watched-clause activation/deactivation stack, propagation reasons, restarts or parallel workers. Next, instrument an unchanged D4 BCEi build and compare its true branch-state event stream with this context contract; only then generalize the verifier.


## Local reference execution provenance

A local reference implementation was executed against the public v2 certificate after its Git blob SHA-1 matched the fetched repository file.

- Certificate Git blob SHA-1: edeaa6bd2b27b48cd89c070a8755dee1fa9833f8
- Certificate SHA-256: 2b57e824b30df5452f7d3824a268f98715544adc7c586c93e669cd347c49a5e4
- Local reference checker SHA-256: 55beb7db67de6e91f4822b203e95ba7b00ee95cc4ab7d0ad1c7951f93880ce97
- Local reference checker Git blob SHA-1: cb4443853ca5bded57c6917f4e2e505386ca9f45
- Runtime: Python 3.13.5, Linux.
- Result: 14 root/source leaf models, 15 processed leaf models; eight visible assignments; PMC 8; PWMC 1/1; 11/11 negative controls rejected.

**Important:** the public uploaded checker currently has a different Git blob (61c7acc4dd54f10904e0a287b6c0ee97e182806f) and is marked not_run in the result manifest. The pass above belongs to the local reference variant, not that uploaded checker. Re-run the uploaded checker before promoting its status.
