# ETH-032 — rollback-aware candidate successor index

**Status:** dependency-free Rust module and tests committed; Python mirror exercised; Rust compilation and GitHub CI remain unverified. No D4 binary differential, formal proof, or runtime speedup is claimed.

## Design

The module in src/candidate_successor.rs preserves D4's exact candidate permutation and swap-to-front promotion. It adds:
- a hierarchical bitmap over active candidate positions;
- upper levels summarizing non-empty 64-bit words;
- reverse incidence from clause IDs to protected records;
- inverse clause-to-position maps for swap updates.

Backtracking reactivates bitmap positions through reverse incidence but does not undo candidate permutations or sentinel relocation.

Public API: CandidateSuccessorIndex::new, set_clause_active, next_active_after_head, promote_to_head, validate and stats.

## Complexity hypothesis

Let k be a protected record's candidate count, deg(c) the number of records containing clause c, and W=64.

- Build: O(E) for E total candidate entries.
- Clause-state update: O(deg(c) log_W k).
- Successor query: O(log_W k).
- Promotion: expected O(1) inverse-map updates plus O(log_W k) bitmap updates.
- Storage: O(E) candidate, inverse-map and reverse-incidence entries plus O(E/W) summary bits.

These are hypotheses for the current implementation, not yet formally proved or measured in Rust. HashMap and per-record allocation overhead may dominate small records; an adaptive list/bitmap representation is likely worthwhile.

## Python mirror operation counts

A Python mirror of the same bitmap and reverse-index scheme passed 10,000 randomized bit set/query operations at each of 12 boundary sizes from 0 to 65,537 bits. It also passed 4,000 randomized active-state updates over 23 records / 193 clauses, with 92,000 successor comparisons against a linear oracle and state validation after every update.

Adversarial fan-in family:

| k | Linear candidate inspections | Indexed successor queries | Bitmap words read, including promotion lookups |
|---:|---:|---:|---:|
| 2 | 2 | 2 | 6 |
| 4 | 9 | 4 | 14 |
| 8 | 35 | 8 | 30 |
| 16 | 135 | 16 | 62 |
| 32 | 527 | 32 | 126 |
| 64 | 2,079 | 64 | 254 |
| 128 | 8,255 | 128 | 639 |
| 256 | 32,895 | 256 | 1,407 |
| 512 | 131,327 | 512 | 2,943 |
| 1,024 | 524,799 | 1,024 | 6,015 |

The word-access column includes reads for swap-bit operations, not only successor queries. Both policies return the same first active candidate and perform the same swap-to-front operation. This is a Python mirror operation-count experiment, not a Rust measurement or D4 executable benchmark.

## Rust tests authored

- Hierarchical bitmap vs ordered-set oracle at 0, 1, 2, 63, 64, 65, 127, 128, 129, 4,096, 8,193 and 65,537 bits.
- Persistent permutation across deactivation and backtrack reactivation.
- Reverse-incidence fanout and malformed-input rejection.
- A 65,537-entry successor query whose only active candidate is the final position.
- 4,000 randomized active-state updates compared to a linear oracle.

**Execution limitation:** this environment has no rustc or cargo, and shell network access is unavailable. The Rust module and tests are committed, but compilation and execution are not claimed.

## Next gates

1. Exact-head Cargo test and clippy.
2. End-to-end scan-vs-indexed state differential after every assignment, queue event, promotion and backtrack.
3. Differential against an unchanged pinned D4v2 build.
4. HOL4 refinement theorem for the protected-triple, sentinel and backtrack invariants.
5. Benchmark on the pinned D4 build and SAT 2024 BCEi cohort; include initialization, update fanout, memory and runtime.
