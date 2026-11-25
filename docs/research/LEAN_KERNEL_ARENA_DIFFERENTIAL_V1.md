# Lean Kernel Arena — live checker landscape snapshot (2026-11-25)

**Issue:** [#13643 — ARENA-DIFFERENTIAL-001](https://github.com/Luminous-Dynamics/symthaea/issues/13643)  
**Related comparator-bundle registry:** [LEAN_CHECKER_DIVERSITY_V1.md](LEAN_CHECKER_DIVERSITY_V1.md)  
**Status:** sourced transcription of the Arena homepage on 2026-11-25. This is not an immutable archived round and not a mirror of the 9.4 MB results.json.

## 1. Why two separate checker views are necessary

There are two different populations:

1. **Lake comparator paranoid bundle:** the six execution paths in the Trust Closure profile. They are the checker set relevant to a standard lake comparator --paranoid result. Our registry models their source/algorithm lineages and common-mode relationships.
2. **Lean Kernel Arena landscape:** the broader, evolving ecosystem of checker implementations submitted to the Arena. The current homepage snapshot lists 37 entrants across 15 reported languages/implementation families.

The Arena snapshot is a competitive landscape, not a claim that every one of its 37 entrants is safe to use in the production comparator. An experimental checker can be useful for differential testing without being a trusted assurance authority.

The current homepage includes five rows that overlap by name with the comparator model: official, lean4lean, nanoda, con-leche, and con-ron. The separate leanchecker-paranoid hardened comparator path is not listed as a distinct row on this homepage snapshot.

## 2. What the Arena measures

The homepage distinguishes three outcome groups:

- **Invalid proofs:** correct rejection, false acceptance, decline.
- **Valid proofs:** correct acceptance, false rejection, decline.
- **Corner cases:** accept or decline; no prescribed outcome is used for correctness scoring.

The page states that declines include unsupported inputs, crashes and timeouts. These are not pooled into false rejection or false acceptance. The ordering prioritizes false acceptance of invalid proofs, then false rejection of valid proofs, then Mathlib test performance, then decline counts.

The downloadable test tarball lists 138 good and 92 bad tests (8.0 MB), excluding individual tests larger than 10 MB. The current table reports 30 corner cases per checker. The raw results.json is approximately 9.4 MB and contains checker, test and result metadata. The current environment could not retrieve that raw file due the webpage content-size limit; the JSON artifact here is therefore a sourced homepage transcription, not the underlying full dataset.

### Mathlib scale measurement

The time column is **virtual CPU time**, not wall-clock duration. The Arena derives it from instruction counts using a fixed 6.0-Ginstr/s conversion. This improves hardware comparability for instruction count but does not model parallel throughput. The memory column is peak memory/RSS as displayed by the Arena. Some checkers have no published Mathlib measurement.

## 3. Selected observations from the captured table

The current live table shows why raw theorem checking and performance must be measured separately:

| Checker | Invalid proof correct / wrong | Valid proof correct / wrong / declined | Mathlib virtual CPU | Peak memory |
|---|---:|---:|---:|---:|
| gruszka | 92 / 0 | 148 / 0 / 0 | 76.8 s | 2.6 GiB |
| nano-lean | 92 / 0 | 147 / 0 / 1 | 65.8 s | 5.0 GiB |
| rpylean | 92 / 0 | 146 / 0 / 2 | 96.3 s | 4.4 GiB |
| official v4.35.0 | 92 / 0 | 148 / 0 / 0 | 33.7 min | 9.9 GiB |
| nanoda | 92 / 0 | 146 / 0 / 2 | 16.0 min | 5.9 GiB |
| con-leche | 88 / 4 | 141 / 0 / 7 | 17.0 min | 4.8 GiB |
| con-ron | 88 / 4 | 142 / 0 / 6 | 9.65 min | 4.4 GiB |
| lean4lean | 92 / 0 | 141 / 7 / 0 | 40.3 min | 9.9 GiB |

These are table transcriptions for this one homepage capture. They are not estimates of the checkers' general false-acceptance probabilities. The small control set does not support a general soundness ranking. For example, gruszka has a better displayed Mathlib time and memory than the official v4.35.0 row on this snapshot, while the official checker has a different implementation/trust status and a different assurance role. A single Mathlib time does not capture all resource profiles, supported constructs, proof parser behavior or metatheoretic assurance.

### Distribution across all 37 visible rows

Recomputing the counts from the captured table:
- 17/37 have zero false acceptances on the 92 invalid-proof cases; 20/37 have at least one.
- 29/37 have zero false rejections on the 148 valid-proof cases; 8/37 have at least one.
- 4/37 have at least one decline on the invalid-proof cohort; 23/37 have at least one decline on the valid-proof cohort.
- The largest displayed false-acceptance counts are `mini` (19/92), `overfull` (16/92), `evmlean` (11/92), `mathgraph` (8/92), and `nanoclo-fortran` (6/92).
- The largest displayed false-rejection counts are `canonical-min` (62/148), `lean4cobol` (20/148), and `lean4lean` (7/148).

This distribution is a reason to keep the outcomes separate, not a definitive checker ranking. A small frozen cohort can reveal specific seeded failure modes, but cannot establish a general soundness probability. The current page is also a live round, so the counts must always be paired with the capture date and source snapshot digest.

## 4. Full 37-checker snapshot

The complete transcription, including all published versions, languages, invalid/valid/corner outcome counts, and Mathlib time/memory where available, is in [LEAN_KERNEL_ARENA_LIVE_SNAPSHOT_2026_11_25.json](LEAN_KERNEL_ARENA_LIVE_SNAPSHOT_2026_11_25.json). Its strict schema is [LEAN_KERNEL_ARENA_LIVE_SNAPSHOT_2026_11_25.schema.json](LEAN_KERNEL_ARENA_LIVE_SNAPSHOT_2026_11_25.schema.json).

The data intentionally keeps null for a missing time/memory cell. It does not infer a performance number for a checker that did not complete the Mathlib-scale case.

## 5. How Symthaea should benchmark against this

A strong competitive programme has two tiers.

### Tier A — production assurance (Lake comparator bundle)

Use the six-path profile in the Trust Closure Capsule:
- exact byte-identical export SHA-256 for every executed checker;
- pinned checker source revisions and binary digests;
- versioned axiom policy;
- full comparator statement/challenge check;
- independent checker lineage relationships and common-mode risk;
- raw result, stderr, exit code and resource receipts;
- accepted, rejected, declined, error, timeout, resource_exhausted, not_run kept separate.

This tier answers: **what was checked on this exact proof artifact, and which trust failure domains did the checkers cover?**

### Tier B — research differential testing (Arena ecosystem)

Use a frozen Arena round and run as many compatible entrants as can consume the same exported artifact. The dataset must include:
- exact round archive and capture digest;
- every test file digest and expected outcome;
- checker commit, build recipe, binary digest, language/runtime;
- export format and Lean revision;
- instruction count, wall time, peak RSS and hardware;
- accept/reject/decline/timeout/crash/resource state for every test;
- per-test pairwise disagreements and a minimal reproducer where possible.

This tier answers: **where do independent checker implementations disagree, what constructs cause it, and which implementation/algorithm/resource trade-offs explain the difference?**

Do not call the current homepage capture a frozen round. Before a leaderboard claim, acquire an archived round or reproduce a frozen test/checker bundle and archive raw receipts. The Arena itself warns that results are only comparable within a round because checkers, tests and Lean versions change.

## 6. Research priorities to exceed the current baseline

1. **Differential soundness testing:** prioritize minimization of every wrong acceptance. A single false acceptance on a seeded invalid export is a release-blocking event, not a score offset by speed.
2. **Completeness without silent rejection:** measure supported positive cases, and classify decline separately from wrong rejection.
3. **Semantic coverage:** construct test matrices for universes, inductives, mutual/nested recursion, constructors, recursors, quotient types, projections, proof irrelevance, definitional equality, native computation, and imported module boundaries.
4. **Metamorphic testing:** alpha-renaming, declaration/root order changes, harmless context reorder, equivalent export rewrites and byte-preserving rebuilds must not change verdicts where the format contract says they should not.
5. **Parser differential fuzzing:** test byte-level export parsers independently, preserve malformed-input errors, and compare accepted ASTs / declaration graphs before kernel checking.
6. **Fault-domain diversity:** add new checkers only when they reduce a documented common-mode risk. A new language port is valuable implementation diversity; it is not automatically a new algorithm lineage.
7. **Resource-aware orchestration:** use per-checker time/memory budgets and classify timeout/OOM/unsupported features as inconclusive rather than rejection. Large formalizations can take radically different time across implementations.
8. **Proof-explanation integration:** connect accepted proof exports to the Trust Closure Capsule and ECPCB, then ask whether the proof plan and explanation correspond to the exact checked declaration graph.

## 7. Prospective benchmark targets

These are research targets, not current Symthaea measurements:

| Metric | Initial target |
|---|---:|
| False acceptance on frozen invalid controls | 0 observed |
| False rejection on supported valid controls | 0 observed |
| Unclassified checker outcomes | 0 |
| Exact exported artifact digest binding | 100% |
| Reproducible run receipts for each compared checker | 100% |
| Frozen-round task coverage disclosed | 100% |
| Disagreement minimization / triage | 100% |
| Runtime, instruction count and peak RSS | median and p95 by test family |
| Independent algorithm lineages | reported from reviewed registry; unknown remains unknown |

A zero observed error count is not a proof of zero error probability. Publish test counts, confidence bounds, unsupported coverage and every decline. A checker with a soundness proof but narrower feature support may be a better assurance component than a fast checker with opaque failure modes.

## 8. Source and limitations

Source page: https://arena.lean-lang.org/ (captured 2026-11-25). The page itself states that:
- invalid/valid/corner-case scoring groups have different semantics;
- crashes/timeouts count as declined;
- virtual time is instruction-derived and does not model parallel processing;
- the raw results file includes checker, test and result metadata;
- the homepage is the in-progress round and comparisons across rounds are invalid.

This snapshot is a bounded transcription. It is not an immutable round archive, a complete mirror of raw results, a proof that any checker is sound, or a Symthaea performance result.
