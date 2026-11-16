# EGDDM ShiftBench D6.13 — Whole-DAG admissible histories and graph-level identifiability v0.1

**Date:** 2026-11-16
**Status:** evaluator-only prototype; finite synthetic known-answer worlds; not independently adjudicated
**Related:** [Issue #12453](https://github.com/Luminous-Dynamics/symthaea/issues/12453), [D6.11 pairwise identifiability](https://github.com/Luminous-Dynamics/symthaea/pull/12445)

## Why this exists

D6.11 v0.1 represents admissible causal relations independently for each event pair. That loses correlations across whole histories: individually plausible edges can be jointly impossible or create a cycle. D6.13 therefore enumerates complete admissible DAG worlds and computes what is entailed by all of them.

## Paradigm shift

Move from pairwise marginal reasoning to **whole-world quantification**. Every admissible world is a complete DAG over visible and evaluator-only hidden event nodes. The evaluator computes transitive closure and projects relations onto visible event pairs. A hidden intermediary may create a causal path even though it is absent from candidate-visible logs.

Possible-world semantics: a pair is BEFORE or AFTER only when every world agrees; CONCURRENT only when every world leaves it unordered; otherwise it is UNDERDETERMINED. An empty world set is invalid, never vacuous truth.

## v0.1 evaluator contract

Candidate-visible input contains visible event IDs, evidence availability, and pair tasks with decision cutoffs. Evaluator-only gold contains one or more complete admissible DAGs plus hidden event IDs. Candidate traces provide entailed-edge claims, pair classifications, and claimed complete visible-world projections.

Candidate world projections are checked in two ways:
- **Joint feasibility:** does one gold world contain every asserted edge in the candidate graph?
- **Exact projection:** is the candidate graph the visible transitive projection of an admissible world?

An edge can be individually possible across the world set while a set of such edges is jointly impossible. This v0.1 evaluator reports that discrepancy separately from candidate cycles.

## Metrics

- Entailed visible edges: intersection of projected transitive closures over all worlds.
- Unsupported asserted causal edges: candidate edges not entailed in every admissible world.
- Pairwise before/after/concurrent/underdetermined classifications derived from full worlds.
- False concurrency, unsupported causal claims, unnecessary abstention, critical unsupported claims and future-evidence leakage.
- Claimed-world exact projection precision/recall and ambiguity-set width.
- Individually possible but jointly impossible candidate graph count.
- Cyclic candidate graph count.

No single score may trade a critical false causal assertion or cyclic graph for speed.

## Synthetic known-answer controls

The fixture has 9 visible event records, 2 admissible whole DAG worlds, one evaluator-only hidden intermediary, and 4 pair tasks. Both worlds entail z→c. They disagree on the ordering of a/b/c. Only one world contains e→hidden-h→f; the other leaves e/f concurrent. Both worlds leave m/n concurrent.

Three candidate traces:
- **whole-world-calibrated:** asserts only the shared z→c edge, abstains on pair relations that vary, identifies m/n concurrency, and reports both exact visible-world projections.
- **pairwise-marginals-combined:** combines a→b and c→b, each individually present in some world but never together; claims e/f concurrent despite a possible hidden causal path; includes a cyclic candidate world; and uses a late-arriving event before availability.
- **always-abstain:** avoids false causal claims but misses the entailed z→c relation and claims no admissible projections.

These are synthetic scorer controls, not model performance or independent causal adjudication.

## Acceptance gates

- 100% exact relation entailment/refutation/contingency on finite controls.
- Zero accepted cyclic candidate worlds as admissible.
- Zero accepted jointly impossible graph claims.
- Zero false concurrency where any admissible world contains a causal path.
- Zero future-evidence leakage.
- Empty world set is invalid, never a pass.
- Candidate world projections cover every admissible visible projection on the deterministic positive control.
- Always-abstain does not pass: it must identify entailed facts.

## Research scale and claim ceiling

Development: 10 graphs with 4–8 visible events and exhaustive world enumeration. Validity pilot: 8 observation/constraint families × 6 graph-composition families × 3 independent seeds = 144 tasks. Research target: 10 × 8 × 5 = 400 graphs with 2–50 admissible worlds and graph-size/composition holdouts. These are proposed targets, not an existing corpus or results.

D6.13 v0.1 evaluates a supplied finite set of complete DAG worlds; it does not prove that the set is complete, authenticate real event capture, or solve arbitrary causal discovery. Next gates are Rust compile/test/format/Clippy, independent world adjudication, exhaustive small-graph controls, two independent evaluators, and blinded trials.