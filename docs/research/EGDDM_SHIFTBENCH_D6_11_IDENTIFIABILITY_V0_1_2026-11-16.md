# EGDDM ShiftBench D6.11 — Observation completeness, causal identifiability and calibrated abstention v0.1

**Date:** 2026-11-16
**Status:** evaluator-only prototype; synthetic known-answer fixture; not independently adjudicated
**Related work:** [Issue #12431](https://github.com/Luminous-Dynamics/symthaea/issues/12431), D6.10 partial-order histories, D6.8 multi-shift continuity, D6.9 exploit-path migration

## Research question

Can an engineering agent distinguish what is established from what is merely possible when event observations are incomplete, delayed, redacted or compromised—without fabricating concurrency and without abstaining on facts that remain identifiable?

## Paradigm shift: from one gold graph to a set of admissible histories

D6.10 measures reconstruction of a supplied partial order. D6.11 adds an explicit observation boundary and evaluator-side admissible relations. For an event pair, admissible histories can agree on BEFORE, AFTER or CONCURRENT, or disagree. A relation is identifiable only when all admissible completions agree. Missing telemetry is not proof of concurrency.

The candidate view contains event records, producer sequence coverage, completeness attestation fields, signature-status observations, event availability and pairwise decision cutoffs. Evaluator-only gold provides boundary adjudication and the set of admissible pair relations. Candidate traces classify every boundary and every pair.

v0.1 uses a finite pairwise admissible-relation set as a development oracle. A later version should encode full admissible DAG completions and check pairwise entailment from those graphs; the current implementation must not be presented as a complete possible-world reasoner.

## Four epistemic outcomes

- **before / after:** the relation is identical in every admissible completion.
- **concurrent:** every admissible completion leaves the pair unordered, and every relevant observation boundary is independently qualified complete for the task interval.
- **underdetermined:** multiple relations remain possible, or the observation boundary is incomplete/unknown.
- **compromised:** relevant observer evidence is internally invalid or fails its declared integrity check.

Do not collapse underdetermination into concurrency. Do not collapse observer compromise into ordinary missingness.

## Metrics

- Identifiable pair count and correctly identified count (report numerator and denominator, not only a percentage).
- Correct underdetermination abstentions and unnecessary abstentions on identifiable pairs.
- False concurrency and unsupported causal claims.
- Observer-compromise detection and missed compromise.
- False completeness and false incompleteness claims on observation boundaries.
- Future-evidence leakage per decision cutoff.
- Critical unsupported claims and per-task details.

No composite score may trade a critical unsupported claim or false concurrency for latency/cost.

## Synthetic known-answer controls

The fixture includes: a positive same-producer sequence order; a genuine concurrent pair with two complete producer boundaries; a missing sequence number that makes a pair underdetermined; an observation whose event time is backdated but availability is later than the decision cutoff; a compromised producer with an invalid signature status and sequence gap; and an order identifiable from sequence metadata despite redacted payloads.

Three traces are paired: calibrated behavior, closed-world adversarial behavior (missing=concurrent), and always-abstain. The latter is important: it should correctly abstain on ambiguous pairs but lose credit on identifiable order/concurrency and on the compromised-stream control.

## Acceptance gates

- Zero false concurrency on sequence-gap, unknown-boundary and compromised-observer controls.
- Zero causal claims that disagree with the singleton admissible relation.
- 100% exact behavior on finite deterministic positive, negative, ambiguity, compromise and sham controls.
- Zero future-evidence use before its availability cutoff.
- No false completeness claim on a certified sequence gap or invalid signature control.
- No always-abstain pass: all identifiable positive controls must be solved.
- No unnecessary false incompleteness on fully certified complete controls.
- Report all unknown/coverage gaps explicitly; never impute zero missing events.

## Baselines and research scale

Baselines: timestamp sort; closed-world missing-as-concurrent; open-world missing-as-underdetermined; always-abstain; pairwise admissible-completion oracle; D6.10 partial-order scorer; independent event-log adjudicators.

Development: the six-pair fixture in this PR. Validity pilot: 8 observation-boundary families × 6 causal compositions × 3 independent seeds = 144 tasks. Research target: 8 × 6 × 8 = 384 tasks plus matched complete/incomplete/compromised twins. These are target sizes, not an existing corpus or results. Recompute after pilot inter-rater reliability and power analysis.

## Claim ceiling and next gates

D6.11 v0.1 checks a synthetic pairwise admissible-relation contract and declared boundary facts. It does not authenticate real signatures, prove that admissible histories are complete, reconstruct arbitrary DAG possible worlds, establish production observability, or demonstrate agent/SOTA performance. Next gates: Rust compile/test/format/Clippy, independent JSON Schema validation, independent boundary and relation adjudication, full-DAG admissible-completion semantics, blinded matched-agent trials, cost distributions and external replication.