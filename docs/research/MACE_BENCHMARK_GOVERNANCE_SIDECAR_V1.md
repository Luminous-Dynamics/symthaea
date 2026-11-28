# MACE Benchmark Governance Sidecar V1

Status: standalone Rust contract and strict schemas authored; dedicated validation is pending. It extends, rather than replaces, the existing Benchmark Result Evidence Capsule V1.

## 1. Keep official scores immutable

The official capsule remains the source of truth for benchmark-native scores, denominators, resource use and claim status. This sidecar binds to the exact capsule digest and to a digest of its official metrics array. It contains only MACE-specific evidence/governance diagnostics and verifier receipts. It cannot replace or mutate official metric records.

Files:

- data/schemas/mace_benchmark_governance_sidecar_v1.schema.json
- data/schemas/mace_benchmark_comparison_receipt_v1.schema.json
- research/mace-benchmark-governance-sidecar-v1/

## 2. Evidence flow is not one recall metric

Track the nested observable sequence: retrieved, admissible, exposed, cited. Then measure influence only when a separate intervention is run: exposed, influence assessed, positive/negative/indeterminate.

These transitions are not interchangeable. A candidate citation is not proof of actual context exposure. Exposure is not proof of attention. A controlled influence result requires its own intervention, assignment and evaluator receipts. The sidecar reports counts and explicit denominators, never a composite evidence score.

Counts are at evidence-item/span level. The validator enforces cited <= exposed <= admissible <= retrieved. Influence classifications must close exactly to influence_assessed and cannot exceed exposed spans.

## 3. Governance outcomes are non-compensable

Separate counts include stale, expired, revoked, superseded, wrong-scope and unsupported-provenance use; unauthorized actions; cross-scope disclosure; prohibited proactive disclosure; erasure resurrection; and privacy violations. These can overlap and must not be summed as though they were disjoint root causes.

hard_gates_passed may not be true when any non-compensable violation count is nonzero. A better official answer score cannot offset a governance failure. Each rate must be reconstructed from its named opportunity denominator. Zero observed events is not zero population risk; publish confidence bounds and the actual number of independent clusters.

## 4. Influence claim discipline

- diagnostic: aggregate counts or synthetic/locally generated ablations only.
- instrumented: exact provider request and trace receipts are present.
- independently_verified: a separate evaluator has checked request, outcome and trace custody.
- causally_qualified: independent verifier, assignment/randomization receipt, split audit receipt and paired influence report are all present.

The code refuses to label an influence report causally qualified when the required receipts are missing. A hash is an integrity commitment, not a signature.

## 5. Matched comparison gate

The comparison gate requires exact equality of benchmark ID, track, release, dataset split, grader, metric contract, reader model, prompt, tools, resource budget, task IDs and cluster unit. Candidate commit/configuration may differ; the comparison is intended to isolate that change under otherwise matched conditions.

If benchmark or track differs, the receipt is explicitly cross_benchmark_or_track_not_comparable. Do not average AML textual, multimodal and coding scores, or LongMemEval-V2 Small and Medium, into one SOTA number.

Partial comparisons are not marked matched. The list of mismatching fields is returned so an experiment can be repaired or transparently labelled non-comparable.

## 6. Recommended benchmark arms

For each official benchmark/track, freeze:

1. official no-memory/native baseline;
2. official RAG or full-context reference;
3. one competitive external memory backend;
4. MACE with governance gates disabled (diagnostic ablation only);
5. MACE with revalidation, exposure lineage and governance gates enabled.

Use the official answerer, grader, split, task ordering and scoring denominator. Keep official metrics untouched. Add the governance sidecar and total lifecycle costs separately. Pair by task and preserve the true independent cluster (conversation family, person/profile, repository or trajectory family).

## 7. Tests

Nine synthetic Rust tests exercise capsule/official-metric binding, monotone evidence flow, influence count closure, denominator and cost validation, hard-gate contradiction, causal-custody requirements, same-track comparison, cross-track rejection and sidecar tamper detection.

## 8. Claim boundary

This implementation validates declared records and digest linkage. It does not authenticate the benchmark provider, prove the capsule's metric digest came from the official evaluator, authenticate the evidence observer, or establish independent custody. Those require externally held signing keys, raw trace retention, official evaluator artifacts and independent replay.

Run cargo test and cargo clippy against this standalone workspace. Until the exact-head workflow passes, treat the code as authored research implementation, not validated capability.