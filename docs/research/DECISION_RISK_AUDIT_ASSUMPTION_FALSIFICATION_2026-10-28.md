# Decision-risk audit assumptions and support completeness: falsification benchmark

**Date:** 2026-10-28  
**Status:** Synthetic contract benchmark; not a deployment certificate  
**Scope:** Four policy/world cells, 4,096 observations per cell in the repeated-observation stress cases.

## Why this lane exists

The previous decision-risk gate uses a simultaneous empirical-Bernstein envelope over a declared finite set of policy/world risk cells. Such an envelope is conditional on its assumptions: observations need the required bounded sampling properties, the audited support needs to be complete, and the audited policy/world identities must refer to the exact decision-risk matrix being certified. Confidence calculations alone cannot establish those facts.

This benchmark tests two failure modes that can create false confidence: (1) a shared systematic audit bias that masks an equal-sized model error and (2) omission of the very cell that contains the model error. It also includes a low-noise iid positive control. The preflight is a contract-level prototype; its boolean assumption-receipt input must come from a separately trusted verifier and is not itself cryptographic verification.

## Fixed-seed protocol

- Seed: 20261028.
- Toy risk matrix: 2 policies × 2 worlds, with losses in a declared [0, 4] range.
- Per-cell observations: n = 4,096.
- Inherited decision-risk radius: r = 0.0666424883 bits.
- Regret budget: 0.20 bits, hence the remaining model-error allowance is epsilon <= 0.0333575117 bits.
- The numeric certificate uses 2(r + epsilon_upper); the synthetic simulator truth is used only to score whether the admission was unsafe.

## Results

| Scenario | Numeric gate | Ground-truth error used for scoring | Preflight | Interpretation |
|---|---|---:|---|---|
| Independent low-noise audits, correct matrix | Bound 0.16490 bits; admit | 0 bits | Pass | Positive control under the simulated iid sampling model |
| Shared systematic bias masks a +0.04 matrix error | Estimated epsilon 0.01315; regret bound 0.15958 bits; admit | 0.04000 bits, above 0.03336 allowance | Reject: assumptions unverified | **Unsafe numeric admission** if dependent/biased data are incorrectly treated as iid |
| Missing support cell | Incomplete-support bound 0.15827 bits; naive method admits | 0.04000 bits | Reject: AUDIT_SUPPORT_INCOMPLETE | Numeric bound is meaningless when the full declared cell family is not observed |

In the shared-bias scenario, the repeated audit readings exactly match the corrupted planning matrix, so the empirical variance is zero and the confidence radius shrinks with the nominal sample count. The observations are deliberately dependent and biased; they violate the iid premise. This does not refute the statistical theorem under its assumptions. It demonstrates that a correct formula cannot rescue an invalid evidence pipeline.

In the missing-support scenario, the failed cell is omitted from the numeric calculation and the partial matrix appears to fit the budget. The support preflight detects the omitted p0::w0 cell and rejects the certificate.

## Changes and local validation

Added:

- validation/decision_risk_audit_assumption_falsification.py
- validation/test_decision_risk_audit_assumption_falsification.py
- validation/DECISION_RISK_AUDIT_ASSUMPTION_FALSIFICATION_2026-10-28.json

The five regression tests pass locally. The executed local source Git blob matches the source blob fetched from GitHub (18146f43e895b3b4c6c284ff19e1bd2bf5d3a8b3); the test blob matches exactly (a10c6e5c22e565018cb899fa907e97102de430fa). Source SHA-256: 35986bf06755f3c00d628f1c78eda30e05f1d158c31c42fdc75a8cdbb3945641. Raw result SHA-256: d2268b11be0d0b6cc547bd93a434fd02262c48d361d41b0e9d08cc724cec08e7.

## Next engineering gate

A production preflight should require a signed, independently verified receipt binding at least: the policy-risk manifest digest; complete, ordered policy and world identifiers; loss bounds; observation protocol; independent sampling unit/cluster design; calibration and audit splits; and the code/data digests. Any mismatch, missing cell, unverifiable dependence assumption, or unsupported world must prevent certification. The same policy mapping must be rechecked immediately before dispatch. Statistical intervals should never be used to paper over identity failures.

The current Python helper does not verify signatures, infer independence from observations, or integrate into the Rust optimizer/dispatch path. Its assumption_receipt_verified argument is a seam to a future trusted verifier, not an independent proof. A production system must supply that verifier and an explicitly qualified fallback.

## Benchmark and SOTA boundary

No published method or real-world dataset was run, so this study makes no novelty or SOTA claim. Relevant matched comparators include [Conformal Policy Control (ICML 2026)](https://proceedings.mlr.press/v306/prinster26a.html), which regulates optimized policies using a calibrated safe reference policy; [CREDO (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/11e0fdc1f9ec9303e5de60670e884e53-Abstract-Conference.html), which bounds decision suboptimality for a candidate decision; and [Robust Decision-Focused Learning via Worst-Case Regret Minimization (UAI 2026)](https://proceedings.mlr.press/v337/yamao26a.html), which addresses measurement error and distribution shift. These should be compared on the same fixed policy family, audit budgets, misspecification shifts, regret budgets, numerical tolerances, and false-admission metric.

The field-level improvement target is not just lower mean regret: **zero unsafe admissions in a predeclared adversarial suite, calibrated coverage under defensible assumptions, explicit rejection of unsupported evidence, and lower audit cost at equal risk tolerance**. Zero failures in a finite synthetic suite is a regression target, not proof of zero real-world risk.
