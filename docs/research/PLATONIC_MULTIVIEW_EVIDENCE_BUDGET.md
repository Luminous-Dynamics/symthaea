# Multi-View Retrieval Versus Identity Evidence

**Status:** executable experiment added; exact-head validation pending  
**Implementation:** crates/core/symthaea-core/examples/platonic_multiview_evidence_budget.rs  
**Runner:** .github/workflows/levin-pattern-experiments.yml

## Research question

Does combining independent noisy observations improve pattern retrieval, and does it solve the separate problem of establishing semantic identity? The expected answer is asymmetric: score fusion can improve recognition of registered patterns, but if known and unknown semantic identities have exactly the same observed views, every vector-only policy must make the same decision for both.

## Protocol

The experiment uses five deterministic seeds, 64 registered prototypes, 4,096-dimensional bipolar vectors, eight query pairs per identity and noise level, and noise levels of 5%, 10%, 20%, and 30%.

It compares three policies under the same fixed similarity/margin thresholds:

- **Single view:** rank candidates from the first observation.
- **Two-view fusion:** average each candidate's similarity over two independent noisy observations.
- **Adaptive second view:** obtain the second observation only when the first-view score is below 0.55 or its top-two margin is below 0.08. Those action thresholds are predeclared diagnostics, not calibrated risk guarantees.

For every pair of observations, a counterfactual unregistered semantic identity is assigned exactly the same two vectors. The experiment asserts that the single-view ranks, fused ranks, and adaptive acquisition decision remain identical under this change of hidden semantics. If a vector-only policy accepts, that same accepted claim is a false identity acceptance for the paired unknown interpretation.

A second measurement separates enrollment from observation evidence. An enrollment receipt alone must produce EvidenceRequired when vector geometry is consistent; it may never produce Verified. The synthetic ground-truth oracle then creates an observation-bound identity assertion for the registered interpretation. This emulates the contract required of an upstream trusted source but is not cryptographic authentication and must not be interpreted as real-world identity evidence.

## Reported metrics

Per seed and noise level, the executable reports top-1 accuracy, accepted coverage, accepted selective risk, paired counterfactual unknown false-accept rate, number of second-view requests, registration-only EvidenceRequired outcomes, registration conflicts, observation-evidence-backed verification, evidence conflicts, and the exact-counterfactual output-equality count.

The experiment is descriptive: multiple queries share each stored prototype, so these sample counts are not independent across all rows. The fixed policy is not calibrated, and observed outcomes do not certify a deployment risk level. Keep all noise strata separate; do not use pooled metrics to conceal a high-unknown false-accept rate.

## Falsifiable hypotheses

1. Averaging independent views improves top-1 recall or accepted-query risk at one or more noise levels, at the cost of an additional observation.
2. The adaptive policy requests fewer second observations than unconditional fusion, but high-confidence near-prototype unknowns can escape its query trigger.
3. Every vector-only method has identical outputs for the exact same observed views regardless of the hidden known/unknown label.
4. Enrollment-only evidence never yields Verified; only the observation-bound channel can do so, and only when the query digest, registry/enrollment scope, and geometric checks agree.
5. Replaying observation evidence against any altered query or different capture ID fails closed, even if vector bytes are identical.

These are testable hypotheses, not assumed successes. If multi-view fusion offers no improvement or the adaptive policy saves little observation cost, retain that result and compare different evidence-selection actions before changing thresholds.

## Field comparison

This is not a replacement for established selective-classification baselines or an ANN systems benchmark. Compare reliability with one-sided selective prediction, classwise risk profiles, and conformal/finite-sample risk controls. Compare retrieval systems on matched workloads using exact-ground-truth Recall@K, p50/p95 latency, throughput, build/update cost, and measured memory. Keep identity verification as its own outcome channel.

Relevant grounding:

- Traub et al., NeurIPS 2024, Overcoming Common Flaws in the Evaluation of Selective Classification Systems, proposes AUGRC to address weaknesses in fixed-working-point comparisons: https://papers.nips.cc/paper_files/paper/2024/hash/047c84ec50bd8ea29349b996fc64af4b-Abstract-Conference.html
- Vergés et al. (2025), Classification using hyperdimensional computing: a review with comparative analysis, uses a common framework across seven established HDC/VSA datasets and 121 UCI tasks: https://doi.org/10.1007/s10462-025-11181-2
- ANN-Benchmarks documents reproducible exact-ground-truth recall/latency trade-offs: https://github.com/erikbern/ann-benchmarks
