# CEDAR-Bench counterfactual evidence dependence v0

Status: synthetic benchmark template specification and generator. No agent has been evaluated and no measured score is claimed.

## Why

Citations can be supportive without being the evidence that generated the answer. This benchmark uses controlled counterfactual interventions to test whether claims track evidence, remain stable under placebo changes, abstain when sole evidence is removed, and correctly represent conflicts.

The benchmark is motivated by Abbas, Zia and Windridge, “Cited but Unused: Parametric Provenance in Retrieval-Augmented Generation” (12 Nov 2026): https://arxiv.org/abs/2611.17161. That work reports a gap between citation support and actual context use, while cautioning that its intervention detector is not accurate enough to act as a standalone policy signal.

## Files

- CEDAR_BENCH_V0_TEMPLATES.json — 20 synthetic task templates, source aliases and intervention definitions.
- scripts/generate_cedar_bench.py — expands a selected split using a per-run secret seed; writes candidate inputs and auditor key to separate files.
- scripts/validate_cedar_bench_templates.py — standard-library-only template contract checker.
- scripts/validate_cedar_bench_streams.py — checks generated candidate/auditor stream separation, source hashes, answer-oracle fields, citation eligibility and paired intervention semantics.

## Split isolation

Templates are assigned by task family to disjoint train (10), development (5) and sealed-test (5) groups. The generator defaults to `sealed_test`; pass `train`, `development` or `sealed_test` as the third argument to select a split. `all` exists only for the non-scoring CI smoke test. The candidate output does not disclose split labels; the auditor stream records them. The current v0 templates and family assignments are public in this PR, so this is a structural held-out split, not a truly secret benchmark. The HMAC-generated fact values remain hidden without the run seed. Before confirmatory scoring, move sealed-test templates to an independent private evaluator (or regenerate held-out families from private seeds) and freeze candidate/model versions first.

## Separation of duties

The generator uses CEDAR_BENCH_SEED (at least 32 bytes; operators must supply 256 bits from a cryptographically secure random source). Do not expose it to the candidate. It emits:

- candidate JSONL: one randomized trial per condition, with only prompt, source bundle and opaque task ID;
- auditor JSONL: condition labels, generated values, expected answer oracle, allowed/forbidden citation sources, source digests and expected behavior.

Store auditor output separately from the candidate environment. Do not upload generated auditor keys to public CI artifacts. The CI stream check uses a public deterministic smoke seed and is explicitly non-scoring. Never use that seed for evaluation. Current template structures are public; only generated values are hidden by the per-run seed.

## Conditions

- baseline: original randomized source facts;
- counterfactual: replace the selected fact consistently across all declared aliases;
- placebo: change an unrelated field without changing the target fact;
- source_removed: remove the declared source and test whether the answer becomes unresolved or remains supported by independent evidence.

Each template now binds an answer-oracle specification to exact fact IDs and supporting source IDs. The generator chooses a counterfactual value only when the oracle answer actually changes, and records expected answer and citation rules in the auditor stream only. Citation-swap decoys are explicitly forbidden as support; prompt-injection tasks preserve source text as untrusted data. The initial generator is contract scaffolding; it does not implement a model runner or an automated semantic judge.

## Metrics and guardrails

Keep evidence support, evidence dependence and world truth separate. Report counterfactual tracking, placebo stability, source-removal sensitivity, conflict representation, citation-swap detection, abstention, stability and all-in cost separately. TRACKS_COUNTERFACTUAL does not mean that the changed value is true.

## Claim boundary

This is a synthetic task-template corpus and generator specification. It does not establish hidden model causality, a production provenance detector, or a benchmark result.
