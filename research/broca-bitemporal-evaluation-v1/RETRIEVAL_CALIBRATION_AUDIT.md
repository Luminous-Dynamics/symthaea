# Broca retrieval-utility calibration audit

**Policy version:** BROCA-RETRIEVAL-CALIBRATION-V1  
**Status:** deterministic audit implementation; Rust tests authored, not compiled/executed here.

## Why calibration is a prerequisite for the VOI gate

The retrieval value gate consumes conservative estimates: a lower bound on correctness gain, a lower bound on lifecycle-risk reduction, an upper bound on retrieval harm and measured retrieval cost. The decision rule is only as safe as those bounds. A threshold applied to poorly calibrated estimates merely gives false precision to an untrusted score.

This module audits whether the estimator's predicted net-utility lower bound actually covers the realized paired outcome on a distinct audit split. For each query:

`covered = observed_net_utility >= predicted_net_utility_lower_bound`

The observed coverage fraction is a Bernoulli sample. Under a fixed estimator and i.i.d./exchangeable audit observations, the one-sided Hoeffding lower confidence bound is:

`coverage_lcb = max(0, observed_coverage - sqrt(ln(1 / alpha_per_test) / (2 * n)))`

`alpha_per_test = family_alpha / family_size`

The Bonferroni correction controls family-wise error over a pre-registered family of slices/settings. The audit only mints an opaque calibration capability when this finite-sample lower confidence bound meets the pre-registered coverage target. Integer-ppm quantization is also checked conservatively before capability issuance.

## Data and split hygiene

Every audit request binds:

- estimator ID/version and exact estimator artifact SHA-256;
- feature-schema and utility-conversion-schema SHA-256 values;
- three pairwise-distinct dataset fingerprints: training, calibration and audit;
- a family-wise alpha budget, fixed family size and target lower-bound coverage;
- a minimum audit sample count (at least 30);
- one unique paired observation per audit query, with predicted positive-utility probability, predicted net-utility lower bound and realized net utility.

The estimator must be frozen before the audit split is opened. If the estimator, threshold, feature schema, utility conversion or calibration settings are changed after examining audit outcomes, that audit split is no longer an untouched hold-out and must not be reused for a pass claim. Repeated adaptive tuning requires a sequential risk-control method (for example, an e-process / adaptive Learn-then-Test design) rather than repeatedly applying a fixed-sample bound.

The code checks that the three split fingerprints differ and that audit query IDs are unique. It cannot prove that the caller correctly computed those dataset fingerprints or that the supplied per-query outcomes are authentic. The caller must preserve signed/immutable raw receipts and compute canonical dataset digests.

## What the report means

The report includes:

- observed lower-bound coverage and covered/total counts;
- one-sided Hoeffding radius and coverage lower confidence bound;
- target coverage, family-wise alpha, family size and per-test alpha;
- Brier score for the positive-marginal-utility probability;
- equal-width-bin expected calibration error (ECE) and per-bin diagnostics;
- estimator, schema, training/calibration/audit split and report digests.

Brier and ECE are diagnostics only. They are not used as a pass condition and do not prove lower-bound coverage. The calibration capability is minted solely on the finite-sample coverage lower confidence bound. A high ECE can still signal a poor probability estimate even when the lower-bound coverage test passes; report both.

## Opaque capability and VOI integration

`audit_retrieval_value_estimator` returns a serializable report and an optional opaque `ValidatedRetrievalCalibration` capability. The capability has private fields and is not serializable. It binds estimator artifact, feature schema, utility schema, calibration split and the exact report digest.

`evaluate_calibrated_retrieval_value` requires this capability before turning a `RetrievalUtilityEstimate` into a VOI decision. The canonical `decide_lifecycle_action_with_voi` adapter requires it again, then also verifies the decision/principal/scope/time, unresolved claim set and exact `EvidenceIndexSnapshotReceipt`. Thus a changed model/schema/calibration report or evidence index invalidates the path.

The capability is an in-process trust object, not a digital signature or remote attestation. It protects the API path from accidental uncalibrated use inside one process. Cross-process or persistent use requires a signed calibration receipt, key management, provenance for audit observations and anti-replay semantics.

## Benchmark targets and test protocol

The default synthetic test uses 300 perfectly covered audit observations, family alpha 0.05, one pre-registered test and target coverage 0.90. The one-sided Hoeffding radius is about 0.0707, giving a lower confidence bound about 0.9293. This is a mechanics fixture, not empirical estimator validation.

Required empirical reporting:

| Metric | Required report | Promotion interpretation |
|---|---|---|
| Lower-bound coverage | covered / n and Hoeffding LCB | Must meet the pre-registered target after family correction |
| Slice transfer | per-domain / lifecycle-slice coverage LCB, fixed family size | No pooled pass can hide a failing stale/conflict/multi-hop slice |
| Brier score | positive-utility probability Brier | Diagnostic probability quality, not a lower-bound guarantee |
| ECE | equal-width ECE + bin counts | Diagnostic only; report binning and sample size |
| Calibration drift | periodic untouched audit split and estimator/schema digests | A changed model/schema must be re-audited |
| Realized policy value | paired actual net utility, LifeEM, retrieval harm and call/token/cost use | Confirms that a passing estimate actually supports good retrieval decisions |

Prospective gates (not measured Broca results):

- The lower confidence bound on lower-bound coverage meets at least 90% on the pre-registered audit distribution and every safety-critical slice.
- Family-wise error is controlled across all reported slices via a pre-registered family size and alpha.
- No estimator or threshold tuning occurs on the audit split.
- Any model artifact, feature schema, utility schema or calibration split change invalidates the opaque capability and requires a new audit.
- Calibration must be reported together with retrieval harm and end-to-end LifeEM / answer-action quality; do not promote a calibrated estimator that improves its own score while worsening the downstream decision.

## Implementation limitations

The module uses a one-sided Hoeffding bound with Bonferroni correction for fixed, pre-registered families. This is intentionally simple and conservative. The guarantee assumes the estimator is fixed and audit observations are i.i.d. or exchangeable samples from the intended deployment distribution. Correlated queries, domain shift or adaptive threshold selection can invalidate that interpretation.

The implementation does not train the estimator, compute real token/latency cost, sign the observation receipts or establish data independence. It audits supplied paired outcomes. The default synthetic test demonstrates arithmetic and capability issuance only; it does not constitute empirical calibration.

No calibration pass, retrieval-call reduction, LifeEM gain, deployment qualification or SOTA result is claimed by this reference implementation.

## Research references

- Learn-then-Test, *A New Framework for Machine Learning Risk Control* (2025): https://arxiv.org/abs/2410.02855
- Adaptive Learn-then-Test / e-process risk control: https://arxiv.org/abs/2504.07862
- Active RAG, *When Should Active RAG Retrieve? A Budget-Aware Evaluation of Utility, Calibration, and Cost* (2026): https://arxiv.org/abs/2607.24010
- HALT, *Verification-Aware Stopping for Retrieval-Augmented Search Agents*, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.343/
