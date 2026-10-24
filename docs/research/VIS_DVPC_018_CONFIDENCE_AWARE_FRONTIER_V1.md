# VIS-DVPC-018 — Uncertainty-aware Pareto-frontier reclassification

**Status:** draft reference analyzer; its output is conditional on externally supplied simultaneous utility bands. It is not a risk certificate or performance result.  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8619  
**Upstream point-estimate frontier:** https://github.com/Luminous-Dynamics/symthaea/pull/8599  
**Cluster-level paired inference reference:** https://github.com/Luminous-Dynamics/symthaea/pull/8579  
**Research anchor:** Ek, Zachariah & Stoica, [Learning Pareto-Efficient Decisions with Confidence (AISTATS 2022)](https://proceedings.mlr.press/v151/ek22a.html). The paper motivates incorporating uncertainty into multi-objective decisions; it does not validate this tool's input bands.

## Why this exists

VIS-DVPC-016 computes a safety-gated Pareto frontier from point estimates. VIS-DVPC-014 adds cluster-level paired inference, but a separate point-estimate frontier can still promote a noisy candidate or discard a point because of sampling variation. This stage makes uncertainty visible in the dominance relation itself.

It reads the exact receipt emitted by VIS-DVPC-016 and a separate, frozen confidence-band TSV. It does **not** re-run the upstream safety gates, recompute cluster inference, verify risk certificates, or establish interval coverage. It checks that the declared inference receipt digest and all experiment/data/evidence identities match. The supplied bands remain an upstream statistical assertion.

## Build and run

~~~sh
rustc --edition=2024 --test tools/vision-dvpc-confidence-frontier.rs -o /tmp/vis-dvpc-018-tests
/tmp/vis-dvpc-018-tests
rustc --edition=2024 -O tools/vision-dvpc-confidence-frontier.rs -o /tmp/vis-dvpc-018

sha256sum pareto-receipt.tsv confidence-bands.tsv
/tmp/vis-dvpc-018 \
  pareto-receipt.tsv EXPECTED_PARETO_SHA256 \
  confidence-bands.tsv EXPECTED_BANDS_SHA256 \
  EXPECTED_INFERENCE_RECEIPT_SHA256 > confidence-frontier-receipt.tsv
~~~

All three expected digests must be frozen outside the analysis command and recorded in the experiment manifest. The program hashes the exact input bytes for the PARETO receipt and bands file. The inference-receipt digest is a caller-pinned identifier: the program does not receive or independently verify that upstream receipt's bytes.

## Confidence-band contract

The bands file uses the following metadata keys, each represented as `META<TAB>key<TAB>value`:

| Key | Meaning |
| --- | --- |
| `format` | Exactly `VIS-DVPC-CONFIDENCE-BANDS-001-v1` |
| `experiment_id`, `dataset_id`, `dataset_revision`, `split` | Must match the PARETO receipt |
| `utility_spec_sha256`, `bundle_lock_sha256`, `paired_unit_set_sha256` | Must match the PARETO receipt exactly |
| `source_sweep_sha256` | Must match the exact sweep digest inside that receipt |
| `source_pareto_receipt_sha256` | SHA-256 of the exact PARETO receipt bytes |
| `inference_receipt_sha256` | Must equal the separately caller-pinned expected digest |
| `confidence_level` | At least 0.95 and below 1.0 |
| `familywise_simultaneous` | Exactly `true` |
| `coverage_scope` | Exactly `ALL_POLICY_BUDGET_POINTS` |
| `point_count` | Exact count of PARETO result rows |

The exact data header is:

~~~text
BAND_HEADER<TAB>budget_point_id<TAB>policy_id<TAB>utility_lcb<TAB>utility_ucb
~~~

Every upstream budget-point/policy pair must appear exactly once; no missing, extra, duplicate or unknown rows are accepted. Bounds must be finite and ordered. These intervals must be **simultaneous across the entire declared point family**, not separate marginal 95% intervals. The analyzer cannot determine whether the statistical procedure truly provides that coverage; it checks the declaration, digest bindings, and complete grid only.

## Dominance under bands

The upstream receipt still decides eligibility. An `INFEASIBLE` point remains infeasible regardless of its utility interval. Eligible points are compared on the same separate objectives as before: higher downstream utility, lower all-in latency, and lower all-in energy. Latency and energy are never scalarized.

Let A and B be eligible points, and let `[L_A, U_A]` and `[L_B, U_B]` be their supplied simultaneous utility bands.

- **BAND_DOMINATED:** A certainly dominates B under the supplied bands when A is no worse in both measured resource dimensions, `L_A >= U_B`, and at least one objective is strictly better.
- **BAND_NONDOMINATED:** no eligible point can dominate B under the supplied bands and resource ordering.
- **UNRESOLVED:** at least one point might dominate B, but the intervals do not establish dominance or rule it out.
- **DUPLICATE_EQUIVALENT:** a lexicographically earlier point has identical resource values and interval bounds. This is deterministic deduplication, not statistical evidence that one point is better.
- **INFEASIBLE:** the upstream safety/resource gate failed; uncertainty cannot rehabilitate it.

“Certified” here is conditional on the validity of the supplied simultaneous bands and the measured resource values. The analyzer explicitly records `interval_procedure_verified_here=false` and `upstream_safety_certificate_verified_here=false`; it does not issue a formal statistical or safety certificate.

## Required adversarial tests and research target

The focused tests include separated bands, overlapping bands despite a favorable point estimate, latency/energy incomparability, an infeasible high-utility point, exact duplicate handling, an incomplete grid, non-simultaneous bands, mismatched input digests, and the SHA-256 known vector.

**Research target, not a current result:** compare point-estimate and band-aware frontier classifications on preregistered held-out traces; report how often apparent dominations become unresolved, how many points need additional data, and whether a budget-matched decision winner survives multiplicity control. Use cluster-level inference, sufficient independent clusters, subgroup harm gates and a second held-out domain. Do not infer SOTA from synthetic tests or a narrower frontier.

## Non-claims

- No model or router training occurs here.
- It does not create confidence intervals or validate their finite-sample coverage.
- It does not re-check risk-certificate validity, data provenance, licensing, sampling independence, or hardware meter calibration.
- It does not claim an empirical gain or SOTA.
