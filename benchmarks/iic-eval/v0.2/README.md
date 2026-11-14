# IIC-Eval v0.2 — adversarial source-fidelity development corpus

**Status:** synthetic development seed. It is not frozen, independently adjudicated, or a performance result.

Related work:
- [Symthaea #12018 — semantic fidelity and adversarial change-impact benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/12018)
- [Symthaea #12019 — institutional assurance vertical slice](https://github.com/Luminous-Dynamics/symthaea/issues/12019)
- [PR #11967 — Institutional Intent Compiler v0.1](https://github.com/Luminous-Dynamics/symthaea/pull/11967)

## What changes in v0.2

The v0.1 prototype focuses on structural validation: source references, spans, trace links, decisions, modalities, ownership, verification disposition, dependencies, and known-answer diagnostics. This separate corpus begins testing semantic fidelity: whether a compiler preserves the obligation actually expressed, keeps preferences distinct from duties, retains conditions and exceptions, identifies unresolved conflict, treats embedded hostile text as untrusted source data, and reacts correctly to amendments.

## Current seed cases

Nine synthetic cases cover modality/prohibition, conditional scope, negation and permissive language, conflicting source records, embedded instruction injection, future-effective amendment, Unicode/quantity fidelity, a missing referenced attachment, and separate quantity/threshold/measurement semantics.

The input corpus and known-answer oracle are separate files. The oracle is a provisional reference and has not been independently adjudicated. Do not expose it to systems under test.

## Evaluation protocol

1. Keep the corpus versioned. Do not silently rewrite prior known answers.
2. Freeze a release by hashing the corpus, oracle, schema, evaluator, and candidate system configuration.
3. Obtain two independent domain adjudications; preserve disagreements and record adjudication rationale.
4. Keep hidden holdout cases out of all tuning and prompt development.
5. Compare expert manual analysis, one-shot LLM, structured LLM with equal reviewer time, configured incumbent requirements tooling for supported scope, and the proposed compiler.
6. Report exact candidate/source-span matches and adjudicated semantic errors. Do not rely on an LLM judge alone.
7. Report errors by failure family, consequence, modality, language, and source format. Do not create a single weighted score.

## Metrics

- Atomic obligation precision/recall; high-consequence omission rate.
- False mandatory promotion; false prohibition; condition/exception loss.
- Source-span fidelity; source version binding; derivation-chain completeness.
- Unsupported inference and contradiction detection.
- Correct abstention and clarification-question quality.
- Change-impact precision/recall and unaffected-control stability.
- Expert review time and correction effort.
- Prompt injection and privacy leakage.

## Prospective target ladder (not results)

- Zero false mandatory promotions and zero lost explicit prohibitions on frozen hard negatives.
- 100% accepted candidates trace to a source span or explicitly approved derivation.
- ≥99% high-consequence obligation recall on adjudicated evaluation cases.
- ≥95% held-out modality/condition classification accuracy.
- ≥90% change-impact recall while limiting unaffected-control invalidations to <5%.
- ≥30% median expert review-time reduction without increasing high-consequence defects.

These thresholds are hypotheses to preregister before holdout evaluation, not claims of current performance.

## Claim ceiling

This seed does not establish semantic correctness, requirement completeness, legal meaning, institutional authority, multilingual competence, safety certification, standards conformance, or production readiness. The current files are intended to expose failure modes and guide an independently reviewed evaluator.


## Decision-time evidence coverage

Case IIC-010 asks for two views with different effective times but the same later knowledge time. The oracle maps each temporal query to the candidate versions required to reconstruct it. The evaluator reports the proportion of temporal query slices for which all required candidate references were recovered. This adapts temporal coverage from retrieval benchmarks to decision reconstruction: retrieving a current source is insufficient if an earlier decision-time rule is missing.


See [Decision-time evidence coverage research](./DECISION_TIME_COVERAGE.md) for the distinction between temporal retrieval coverage and evidence completeness for a specific decision.
