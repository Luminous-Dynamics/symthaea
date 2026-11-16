# EGDDM ShiftBench D6.13 — Whole-DAG admissible histories and graph-level causal identifiability v0.1

**Date:** 2026-11-16
**Status:** evaluator-only prototype; synthetic known-answer fixture; not independently adjudicated
**Related work:** [Issue #12453](https://github.com/Luminous-Dynamics/symthaea/issues/12453), [D6.11 pairwise identifiability](https://github.com/Luminous-Dynamics/symthaea/pull/12445), [D6.10 partial-order histories](https://github.com/Luminous-Dynamics/symthaea/pull/12428)

## Why this increment exists

D6.11 v0.1 stores pairwise admissible relation sets. These are useful marginals, but they do not preserve correlation between pairs. A candidate can make a collection of individually plausible claims that no complete admissible graph satisfies. D6.13 therefore stores and evaluates complete DAG worlds.

## Paradigm shift: possible-world causal semantics

Evaluator gold contains a finite non-empty set of complete DAG completions over one exact event universe. For every world, the scorer checks acyclicity and computes transitive closure. Pair relations are derived from each complete closure, never independently invented.

Across worlds:
- A unique BEFORE or AFTER relation is entailed.
- A unique CONCURRENT relation is entailed only when the scoped observation boundary is complete.
- Multiple possible relations are underdetermined.
- A compromised boundary blocks unsupported concurrency, while independently entailed positive order remains reportable.
- An empty world set is invalid, not vacuous truth.

The candidate's determinate pair claims are then checked against each whole world. This detects jointly impossible claim sets even where every individual claim appears in at least one world.

## Candidate/evaluator separation

Candidate-visible data: event IDs, event/availability timestamps, observation-boundary state, trusted positive relations, forbidden unordered pairs, per-pair decision cutoff and evidence IDs.

Evaluator-only data: the exact admissible whole-DAG worlds. Never place these completions or report-side oracle identities in the candidate workspace.

## Metrics

- Per-pair possible relations and world-by-world relation witnesses.
- Individually plausible claim count.
- Jointly consistent world count and identities; zero means the determinate claims are jointly impossible.
- False concurrency, unsupported causal claims, unnecessary abstention and critical unsupported claims.
- Future evidence IDs used before their availability cutoff.
- Observation-boundary claim accuracy.

No scalar accuracy score can hide a jointly impossible claim set, a false concurrency assertion, or future-evidence leakage.

## Synthetic known-answer fixture

Six events produce 15 unordered pairs. Two evaluator-only worlds contain opposite A/B/C chains while preserving E→F. The six A/B/C-to-E/F/Z pairs are constrained unordered; the trusted E→F relation must hold in both worlds.

Three candidate traces:
- **world-consistent-agent:** correctly abstains on the three contingent A/B/C pair relations, recovers E→F, and asserts the twelve certified concurrent pairs. One admissible world satisfies all determinate claims.
- **jointly-impossible-cycle:** claims A→B, C→A, and B→C. Each relation appears in at least one world, but their conjunction forms a cycle and matches zero complete worlds. It also uses event Z before Z's availability.
- **always-abstain:** correctly abstains on the three ambiguous pairs but unnecessarily abstains on the twelve identifiable relations.

These are synthetic fixture calculations, not agent trials, external causal adjudication or performance claims.

## Adversarial controls and tests

Tests cover known-answer equality, pairwise ambiguity from whole worlds, transitive entailment, incomplete-boundary concurrency, empty-world rejection, cyclic-world rejection, trusted and forbidden constraints, missing pair claims, unknown nested fields, duplicate worlds, wrong boundary claims and future evidence.

## Proposed scale and baselines

Development: 10 deterministic graphs with 4–8 events and exhaustive enumeration. Validity pilot: 8 observation/constraint families × 6 graph compositions × 3 independent seeds = 144 tasks. Research target: 10 × 8 × 5 = 400 graph tasks, with 2–50 admissible worlds per task and graph-size holdouts. These are targets, not current corpus/result.

Baselines: D6.11 pairwise marginals; pairwise marginals plus cycle rejection; timestamp serialization; closed-world missing-as-concurrent; always-abstain; explicit finite-DAG enumeration; validated constraint solver; independent expert adjudication.

## Claim ceiling and next gates

D6.13 v0.1 scores supplied finite whole-DAG worlds. It does not prove those worlds are complete, solve arbitrary causal discovery, authenticate telemetry, or establish production concurrency safety. Next gates: exact-head Rust formatting/compile/tests/Clippy/CI; independent JSON Schema engine; exhaustive independent oracle for small DAGs; independent world-set adjudication; complexity/cost curves; blinded agent trials and external replication.