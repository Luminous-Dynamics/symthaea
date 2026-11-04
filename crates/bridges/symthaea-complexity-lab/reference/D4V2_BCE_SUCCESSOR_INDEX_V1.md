# ETH-032 — rollback-aware candidate successor index

**Status:** dependency-free Rust module, source-semantic transition guards and tests authored; Python mirror previously exercised; Rust compilation and GitHub CI remain unverified. No D4 binary differential, formal proof, or runtime speedup is claimed.

## Design

The module in src/candidate_successor.rs preserves D4's exact candidate permutation and swap-to-front promotion. It adds:
- a hierarchical bitmap over active candidate positions;
- upper levels summarizing non-empty 64-bit words;
- reverse incidence from clause IDs to protected records;
- inverse clause-to-position maps for swap updates.

Backtracking reactivates bitmap positions through reverse incidence but does not undo candidate permutations or sentinel relocation.

Public API: CandidateSuccessorIndex::new, set_clause_active, advance_after_deactivation, next_active_after_head, promote_to_head, validate and stats. `advance_after_deactivation(record, clause)` is the canonical D4-shaped operation: it requires that `clause` is the record's current sentinel and is already inactive, then returns either `Relocated` (the first active successor was promoted) or `Exhausted` (no active successor remains). The lower-level query and promotion methods independently reject an active sentinel; promotion also rejects an inactive replacement. This prevents accidental successor queries on a non-sentinel candidate and prevents promoting an inactive candidate.

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


## Sentinel-precondition hardening — 2026-11-04

The Rust API now makes D4's `inprocessing()` call preconditions explicit:

- A successor transition must name the record's current sentinel clause.
- The sentinel must already have transitioned to inactive in the authoritative clause-activity table.
- The returned successor is the first active candidate in the current permutation after position zero.
- Promotion is accepted only when the old sentinel is inactive and the replacement candidate is active.
- A no-successor result is represented as `SuccessorTransition::Exhausted`; the caller remains responsible for the theorem-backed target deletion and queue event.
- Candidate order remains persistent across later clause reactivation; only activity bits are updated on backtrack.

New Rust unit tests cover active-sentinel query rejection, mismatched-sentinel rejection, inactive-replacement rejection without mutation, exhausted records, and reactivation of an old candidate at its current non-head position. The randomized transition test now triggers successor processing only when the deactivated clause is the current sentinel, matching the source-level watch-trigger semantics rather than querying every record after every clause update.

**Execution status:** these are authored Rust tests and source changes. They have not been compiled or run because `rustc` and `cargo` are unavailable in this execution environment. No test-pass claim is made for the new Rust code. The existing Python reference model remains separate evidence and does not substitute for Rust execution or D4-binary differential.
