# Optical ISRU v0.2 — propensity receipts and audit sampling reference v0.1

**Status:** standalone Rust reference; 11 tests authored, not compiled or executed  
**Parent:** Symthaea #12526 — audit-sentinel design and propensity-aware adaptive evaluation  
**Related:** Symthaea #12483 / #12496 — sealed experimental-design tournament; #12525 — likelihood-based EIG/EVSI

## Why this exists

Adaptive experiment selection creates a data distribution that depends on the policy's earlier decisions. A campaign that repeatedly samples a small region may produce a low prediction error on a random split of its own observations while remaining poorly supported elsewhere in the intended operating envelope.

The 2026 materials-science active-learning review discusses this active-learning bias, the risks of evaluating on a sample stream that does not represent the target design distribution, and inverse-acquisition-probability weighting as one possible correction. It also warns that reweighting has assumptions and should not be presented as a universal repair: https://doi.org/10.1039/d6dd00081a

## First implementation

`research/asteroid-optical-isru-v0.2/selection_receipts.rs` adds:

- an explicit candidate catalogue, including candidates rejected by the external admission gate;
- an externally supplied gate-receipt identity for each candidate;
- a frozen target-distribution probability for every candidate;
- a policy selection propensity for every candidate;
- candidate-set and evidence-snapshot digests;
- policy identity, version, configuration digest and random-stream identity;
- a deterministic, seeded uniform audit sampler over admitted candidates;
- rejection-sampled bounded integer selection (no modulo bias);
- a target/selection importance-weight calculation; and
- Kish effective sample size for a realized weight vector.

The audit sampler never randomizes over a candidate marked as not admitted. It assigns blocked candidates zero target mass and zero selection mass. The full catalogue is retained in the receipt so the evaluation can distinguish "not eligible" from "eligible but not selected."

## Statistical contract

For candidate i, the reference computes the basic importance weight:

[
w_i = \frac{p_{target}(i)}{p_{select}(i)}.
]

If the target distribution assigns positive mass to a candidate with zero selection probability, the function returns a positivity-support error. It does not clip the zero, invent a probability, or claim the target metric is identifiable.

For a realized set of non-negative weights, Kish effective sample size is:

[
ESS = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
]

The helper deliberately does not claim independent samples. In a physical materials campaign, repeated measurements must first be grouped at the correct level: batch, specimen, physical acquisition, laboratory run, calibration episode, or common-cause component. The ESS calculation alone does not discover those dependencies.

## Reproducibility and trust boundaries

The sampler uses a deterministic xorshift64* stream for reproducible research fixtures, not cryptographic unpredictability. It is not appropriate for security-critical randomization or hidden-evaluation secrecy. A sealed tournament must use a custodian-controlled seed stream and independently retain the seed/receipt.

Digests and gate receipt IDs are opaque caller-supplied identifiers. This code checks that they are present, not that they are cryptographically correct or authorized. Mycelix or another independent verifier must resolve the underlying gate and authority records.

The target distribution is also caller-supplied and must be frozen before the campaign. A target distribution reconstructed from the adaptively sampled observations would reintroduce the bias this mechanism is intended to audit.

## Authored tests

Eleven tests cover gate-rejected candidate exclusion, deterministic replay, seed variation, retention of all candidate and receipt metadata, zero-propensity positivity rejection, importance-weight arithmetic, Kish ESS, blocked-action probability rejection, selected-action propensity consistency, invalid weights, zero-seed refusal and missing gate receipts.

### Reproduce locally

From the repository root:

```sh
rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/selection_receipts.rs -o /tmp/isru-selection-tests
/tmp/isru-selection-tests
```

**Validation honesty:** all tests are authored but have not been compiled or executed in the authoring environment because rustc/cargo are unavailable there. The implementation is a receipt and sampling reference, not evidence that a real policy's probabilities, target distribution or external gate declarations are correct.

## Required next integration

1. Have every benchmark policy emit the entire eligible candidate distribution, not just its selected candidate and score.
2. Recompute deterministic-policy propensities from frozen policy code/configuration; require stochastic policies to supply their actual conditional distribution.
3. Pin the target evaluation distribution and hidden audit worlds before policy execution.
4. Keep random audit actions within the same externally approved operating envelope as all other candidates.
5. Record rejected candidates and refusal reasons without including them in the audit lottery.
6. Apply propensity weighting only with adequate positivity/overlap; publish raw support, weight distribution, ESS and unsupported strata.
7. Use whole-batch and whole-family holdouts and independent sealed audit observations.
8. Never feed audit outcomes into the evaluation policy until the evaluation snapshot is closed.

This is the first implementation for #12526, not a complete bias-correction system or proof of generalization.
