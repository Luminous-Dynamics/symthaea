# Broca Angular Calibration Benchmark (2026-10-22)

## Research question

Can supervised angular calibration distinguish an intended new token from a protected old-token context that shares the same compositional context?

This is a follow-up to [issue #8285](https://github.com/Luminous-Dynamics/symthaea/issues/8285) and complements [PR #8280](https://github.com/Luminous-Dynamics/symthaea/pull/8280). It is deliberately a benchmark harness and synthetic stress test, not a production initializer or a quality claim.

## Paradigm shift

A vocabulary extension is an expansion of a cosine classifier. The important quantity is the *directional margin* between the new token and the best competing old token, on both intended and protected contexts. Stored norm matching does not control these logits when cosine similarity is used.

The proposed method under test is a calibrated contrastive direction:

`candidate(λ) = normalize(context + λ · (positive_centroid − protected_centroid))`

Here the positive and protected centroids are learned only from the deterministic training split. A small grid of λ values is selected by maximizing intended-token global top-1 subject to a 1% training false-activation constraint. If no candidate satisfies the constraint, the selection falls back to the lowest false-activation rate, then the highest training accuracy. Final results are calculated on independent held-out noise samples.

## Synthetic fixture

- 64 existing token rows; 8 new concepts; 16,384-dimensional continuous HDC vectors.
- Every new concept begins with a shared context made from two old token embeddings.
- An orthogonal planted feature distinguishes positive queries from protected queries; both sides retain the shared context. This deliberately creates an ambiguity trap for context-only initialization.
- 8 training and 64 held-out queries per concept and side, with deterministic named seeds and a SHA-256 fixture receipt.
- The benchmark compares seeded random, context-only bundle, positive-context centroid, context plus a deterministic residual, and the calibrated contrastive candidate.

This is a synthetic geometry fixture: its planted distinction is known by construction. It is not linguistic evidence, does not reproduce a natural-data distribution, and must never be presented as semantic quality or an SOTA result.

## Metrics

- Intended target beats all old logits.
- Intended target is global top-1 among old and newly appended candidates.
- Protected-context false activation rate: any new candidate reaches or exceeds the best old-token logit.
- Mean positive target margin over the old vocabulary.
- Mean protected old-best margin over the new vocabulary.
- Mean and maximum absolute pairwise cosine between new candidate vectors.
- Exact seed scheme, fixture digest, selected λ, and training-grid diagnostics in JSON.

## Research hypotheses

1. Context-only bundling can activate on protected queries that share its constituent context.
2. Positive-only grounding may improve intended-query recall but can ignore protected false activation.
3. Contrastive calibration can improve held-out target margins when positive/protected queries contain a stable discriminative direction.
4. These gains should weaken under distribution shift, label noise, overlapping concepts, more tokens, or missing positive examples.

The benchmark is designed to falsify these hypotheses as well as support them. It does not demonstrate that the same method will improve natural language behavior.

## What would justify claiming a real gain?

1. Run the smoke benchmark and archive the exact-head JSON receipt.
2. Obtain independently labelled Broca query/target states with explicit provenance, split by lexical family or source to prevent leakage.
3. Compare against seeded random, legacy context bundle, subword/constituent composition where meaningful, positive-only centroid, and supervised/contrastive calibration.
4. Select hyperparameters on calibration data only; keep a sealed test split. Report paired confidence intervals over at least five seeds and multiple vocabulary-growth sizes.
5. Require both higher intended-token quality and non-inferior protected-old behavior. Then run short adaptation ablations because published work shows initialization-only signals may not predict later convergence.
6. Do not claim SOTA until a reproducible, apples-to-apples comparison against strong published baselines on an independently labelled task passes.

## False-activation uncertainty (diagnostic, not a guarantee)

The benchmark now emits a one-sided exact Clopper–Pearson 95% upper bound for the observed protected-query false-activation count, and tests its implementation at 0, 1, and 2 events out of 512. The held-out bank has 64 samples per concept and side (512 positives and 512 protected queries across eight concepts). Under an IID Bernoulli assumption, zero events gives an upper bound below 1%; one event also remains just below 1%, while two exceed 1%.

**Important limitation:** the synthetic queries share eight planted concepts, so event rates may be heterogeneous and dependent within concept. The IID bound is therefore a diagnostic only—not a distribution-free or deployment risk guarantee. The selected 1% training condition is an empirical calibration constraint; with only 64 protected training queries, it effectively requires zero observed false activations and by itself cannot certify a 1% population rate. A real risk-control claim requires a justified sampling unit, enough independent examples, source-/family-separated data, and distribution-shift evaluation.

The broader principle is risk-controlled activation: report confidence bounds and the risk/coverage trade-off, rather than only a raw rate or cosine margin. Recent selective-risk work uses held-out confidence bounds to set acceptance thresholds; this motivates the measurement design without claiming Broca inherits any guarantee: [Fixed-Sequence Confidence-Interval Calibration (2026)](https://www.mdpi.com/2571-905X/9/5/109).

## Complementary direction: coverage versus vocabulary budget

Vocabulary expansion is not free. At 16,384 dimensions with f32 components, a raw row occupies 64 KiB, and the weight-tied cosine score evaluates each row. The related [Vocab Diet (Findings ACL 2026)](https://aclanthology.org/2026.findings-acl.1618/) reports reclaiming 10–40% of vocabulary slots while reallocating capacity in its tested LLM settings. Broca needs a separate matched-budget comparison of append-only growth versus opt-in pruning/merging or sparse/hierarchical scoring, with token-ID/checkpoint compatibility and rollback invariants. This work is tracked in [issue #8290](https://github.com/Luminous-Dynamics/symthaea/issues/8290).

## Validation boundary

The benchmark source is checked into a separate branch so the experiment can be reviewed independently from the initializer implementation. Rust compilation, formatting, linting, and execution have not been verified in this environment. No benchmark results are asserted here.

## Related research

- Mundra et al., CoNLL 2024, [An Empirical Comparison of Vocabulary Expansion and Initialization Approaches for Language Models](https://aclanthology.org/2024.conll-1.8/).
- Tejaswi et al., EMNLP Findings 2024, [Exploring Design Choices for Building Language-Specific LLMs](https://aclanthology.org/2024.findings-emnlp.614/).
- Joshi et al., August 2026, [Beyond Initialization Loss](https://arxiv.org/abs/2608.03494).
- Chen et al., 2026, [Grounded Token Initialization](https://arxiv.org/abs/2604.02324).

The external studies motivate stronger methods and held-out convergence checks. They do not establish an empirical Broca advantage.
