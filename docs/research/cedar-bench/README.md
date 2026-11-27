# CEDAR-Bench counterfactual evidence dependence v0

Status: synthetic benchmark template specification and generator. No agent has been evaluated and no measured score is claimed.

## Why

Citations can be supportive without being the evidence that generated the answer. This benchmark uses controlled counterfactual interventions to test whether claims track evidence, remain stable under placebo changes, abstain when sole evidence is removed, and correctly represent conflicts.

The benchmark is motivated by Abbas, Zia and Windridge, “Cited but Unused: Parametric Provenance in Retrieval-Augmented Generation” (12 Nov 2026): https://arxiv.org/abs/2611.17161. That work reports a gap between citation support and actual context use, while cautioning that its intervention detector is not accurate enough to act as a standalone policy signal.

## Files

- CEDAR_BENCH_V0_TEMPLATES.json — 20 synthetic task templates, source aliases and intervention definitions.
- scripts/generate_cedar_bench.py — expands templates using a per-run secret seed; writes candidate inputs and auditor key to separate files.
- scripts/validate_cedar_bench_templates.py — standard-library-only template contract checker.

## Separation of duties

The generator uses CEDAR_BENCH_SEED (at least 32 bytes; operators must supply 256 bits from a cryptographically secure random source). Do not expose it to the candidate. It emits:

- candidate JSONL: one randomized trial per condition, with only prompt, source bundle and opaque task ID;
- auditor JSONL: condition labels, generated values, source digests and expected behavior.

Store auditor output separately from the candidate environment. Do not upload generated auditor keys to public CI artifacts. The static CI test verifies template structure only; do not use a public fixed seed for a scored evaluation.

## Conditions

- baseline: original randomized source facts;
- counterfactual: replace the selected fact consistently across all declared aliases;
- placebo: change an unrelated field without changing the target fact;
- source_removed: remove the declared source and test whether the answer becomes unresolved or remains supported by independent evidence.

Additional template metadata calls out citation-swap, contradictory-source, multi-hop, alias and prompt-injection controls. The initial generator is contract scaffolding; it does not implement a model runner or an automated semantic judge.

## Metrics and guardrails

Keep evidence support, evidence dependence and world truth separate. Report counterfactual tracking, placebo stability, source-removal sensitivity, conflict representation, citation-swap detection, abstention, stability and all-in cost separately. TRACKS_COUNTERFACTUAL does not mean that the changed value is true.

## Claim boundary

This is a synthetic task-template corpus and generator specification. It does not establish hidden model causality, a production provenance detector, or a benchmark result.
