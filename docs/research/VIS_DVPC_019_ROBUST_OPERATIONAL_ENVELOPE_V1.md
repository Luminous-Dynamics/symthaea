# VIS-DVPC-019 — Robust operational envelope and feasibility frontier

**Status:** draft evaluator; no real-data result or SOTA claim.  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8630  
**Utility uncertainty input:** https://github.com/Luminous-Dynamics/symthaea/pull/8621  
**Budget frontier:** https://github.com/Luminous-Dynamics/symthaea/pull/8599

## Why another stage?

VIS-DVPC-016's point-estimate frontier uses measured latency/energy summaries and upstream risk/harm fields. VIS-DVPC-018 makes utility dominance uncertainty-aware, but still compares resource point estimates and inherits the upstream eligibility decision. Measurement noise, load variation, tail latency, energy-meter uncertainty and harm-rate uncertainty can turn an apparently feasible operating point into an unproven or unsafe choice.

Robust multi-objective optimization under uncertainty is an established research area. Useful references include Veyna et al. (2025), [Strategy for obtaining robust solutions in multi-objective design with uncertainties](https://doi.org/10.1016/j.apm.2024.115767), and Rivier et al. (2024), [Non-parametric measure approximations for constrained multi-objective optimisation under uncertainty](https://doi.org/10.1002/nme.7403). These motivate the direction but do not validate this analyzer.

## Build and run

~~~sh
rustc --edition=2024 --test tools/vision-dvpc-robust-envelope.rs -o /tmp/vis-dvpc-019-tests
/tmp/vis-dvpc-019-tests
rustc --edition=2024 -O tools/vision-dvpc-robust-envelope.rs -o /tmp/vis-dvpc-019

sha256sum confidence-frontier.tsv operational-envelope.tsv
/tmp/vis-dvpc-019 \
  confidence-frontier.tsv EXPECTED_CONFIDENCE_SHA256 \
  operational-envelope.tsv EXPECTED_ENVELOPE_SHA256 \
  EXPECTED_INFERENCE_RECEIPT_SHA256 > robust-envelope-receipt.tsv
~~~

The caller must pin all expected digests outside the analysis command. The input confidence receipt must be the exact output from VIS-DVPC-018. The operational-envelope file is frozen separately.

## Operational-envelope input

The exact metadata keys are one each as `META<TAB>key<TAB>value`:

| Key | Contract |
| --- | --- |
| `format` | `VIS-DVPC-OP-ENVELOPE-001-v1` |
| `experiment_id`, `dataset_id`, `dataset_revision`, `split` | Must exactly match the confidence receipt |
| `bundle_lock_sha256`, `paired_unit_set_sha256` | Exact evidence/paired-unit identity |
| `source_sweep_sha256` | Must match the frozen sweep digest in the confidence receipt |
| `source_confidence_receipt_sha256` | SHA-256 of the exact confidence-receipt bytes |
| `inference_receipt_sha256` | Must match both the confidence receipt and caller-pinned expected digest |
| `confidence_level` | Simultaneous envelope confidence level in (0,1) |
| `familywise_simultaneous` | Exactly `true` |
| `coverage_scope` | Exactly `ALL_POLICY_BUDGET_SAFETY_METRICS` |
| `point_count` | Exact count of the full point grid |
| `pooled_harm_limit`, `worst_group_harm_limit` | Frozen values in [0,1] |
| `cvar95_loss_limit` | Frozen finite non-negative value |
| `safety_violations_max` | Frozen non-negative integer |

The exact header is:

~~~text
ENVELOPE_HEADER<TAB>budget_point_id<TAB>policy_id<TAB>latency_lcb_us<TAB>latency_ucb_us<TAB>energy_lcb_uj<TAB>energy_ucb_uj<TAB>pooled_harm_lcb<TAB>pooled_harm_ucb<TAB>worst_group_harm_lcb<TAB>worst_group_harm_ucb<TAB>cvar95_loss_lcb<TAB>cvar95_loss_ucb<TAB>safety_violations_lower<TAB>safety_violations_upper<TAB>cost_measurement_complete<TAB>risk_certificate_state
~~~

Risk states are exactly `CERTIFIED`, `FAILED`, or `UNKNOWN`. All numeric intervals must be finite and ordered; harm intervals must lie in [0,1]. The full policy × budget grid must exactly match the source receipt.

## Joint coverage—not two unlinked 95% claims

Utility and operational bands are separate input families. Two separately produced 95% family-wise bands do **not** automatically imply 95% joint coverage. This tool applies the conservative union bound

`P(both families cover) >= confidence_utility + confidence_envelope - 1`.

It rejects inputs whose declared lower bound is below 95%. For example, 97.5% simultaneous coverage for each family gives a conservative 95% joint lower bound. A jointly calibrated family-wise procedure can be used instead in future versions, but its proof and receipt must be explicit. The analyzer checks the declarations/digests; it does not prove the interval procedures' coverage.

## Robust-feasibility rules

For a point to be robust-feasible, every operational upper bound must satisfy its frozen limit: latency, energy, pooled harm, worst-group harm, CVaR95 loss and safety count. Cost measurement must be complete and risk state must be `CERTIFIED`.

- If a lower bound already exceeds a hard ceiling, or the upstream state is explicitly failed/infeasible, emit `ROBUST_INFEASIBLE`.
- If an interval crosses a ceiling, a meter/measurement is incomplete, or the risk state is unknown, emit `ROBUST_UNRESOLVED_FEASIBILITY`; do not include the point in the deployable frontier.
- Upstream `INFEASIBLE` cannot be rehabilitated by any high utility interval.

Among robust-feasible points, declare A certainly dominant over B only if A's utility lower bound is at least B's utility upper bound and A's latency and energy upper bounds are both no worse, with at least one strict improvement. If a potential dominator cannot be confirmed or ruled out, emit `ROBUST_FRONTIER_UNRESOLVED`. Latency and energy remain separate objectives; no weighted score permits one to compensate for the other.

## Acceptance and non-claims

The source includes adversarial tests for latency ceiling crossings and exceedances, harm intervals crossing limits, explicit failed/unknown risk states, incomplete measurement, upstream-infeasible preservation, insufficient joint coverage, digest binding, and the SHA-256 known vector.

This is an evaluator over supplied intervals. It does not validate licensing, evidence provenance, hardware meters, risk certificates, sampling independence, or statistical coverage; it does not create confidence intervals or establish empirical superiority. Test-data mechanics are not performance evidence. A credible milestone still requires preregistered paired held-out traces, target-hardware latency/energy measurements, independent-cluster inference, worst-group safety gates, and a second-domain replication.
