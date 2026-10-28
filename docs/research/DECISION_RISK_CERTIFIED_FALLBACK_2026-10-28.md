# Error-Budgeted Decision Regret and Fail-Closed Admission Gate

**Status:** synthetic finite-game diagnostic. The error envelope is computed from simulator truth (oracle-known) and is not yet an estimator or certificate for a deployed risk model.

## Hypothesis

If the candidate-policy expected-risk vector is inside a split-conformal region of radius r in the planner model, and the maximum absolute per-cell difference between the planner risk matrix and true risk matrix is bounded by epsilon, then true regret is at most 2(r + epsilon) on the calibrated inclusion event. The extra 2 epsilon term accounts for model error in the chosen and comparator policy losses.

This motivates a fail-closed admission rule: deploy the optimized policy only when its certified regret bound fits a declared budget; otherwise use a separately qualified safe reference policy or abstain. The rule is conditional on a valid epsilon envelope, calibrated inclusion, and a fixed policy identity mapping.

## Protocol

- Six synthetic regimes, 600 calibration episodes, and 120 independent known-regime test cases; seed 20261032.
- Sixteen latent hypotheses, eight fixed candidate policies; observation budget 120, nominal split-conformal target 90%.
- Declared regret budget: 0.20 bits.
- Four model conditions: exact matrix, additive 5%-scale perturbation, additive 20%-scale perturbation, and a policy-row identity permutation.
- Pooled and maximum-known-regime calibration use the same episodes. Epsilon is the exact maximum cell error against simulator truth—an oracle in this experiment.
- The fallback is an oracle-computed full-simplex minimax mixture using the true simulation risk matrix. It is only a proxy for a separately qualified reference controller, not an implementable baseline.

## Results

| Model condition | Geometry | Radius r | Error bound epsilon | Certified bound 2(r+epsilon) | Admitted at 0.20 bits? | Planner mean regret | Gated mean regret |
|---|---|---:|---:|---:|---|---:|---:|
| Correct model | Pooled | 0.06664 | 0.00000 | 0.13328 | Yes | 0.00152 | 0.00152 |
| Correct model | Group-max | 0.07538 | 0.00000 | 0.15077 | Yes | 0.00152 | 0.00152 |
| 5%-scale perturbation | Pooled | 0.06798 | 0.04779 | 0.23155 | No — fallback | 0.00451 | 0.10992 |
| 5%-scale perturbation | Group-max | 0.07777 | 0.04779 | 0.25111 | No — fallback | 0.00451 | 0.10992 |
| 20%-scale perturbation | Pooled | 0.06717 | 0.17887 | 0.49209 | No — fallback | 0.01274 | 0.10992 |
| 20%-scale perturbation | Group-max | 0.07739 | 0.17887 | 0.51252 | No — fallback | 0.01274 | 0.10992 |
| Policy-row mismatch | Pooled | 0.06664 | 0.63699 | 1.40726 | No — fallback | 0.09781 | 0.10992 |
| Policy-row mismatch | Group-max | 0.07538 | 0.63699 | 1.42474 | No — fallback | 0.09781 | 0.10992 |

Both pooled and group-max geometry admitted the exact-model planner and rejected every corrupted model under the declared 0.20-bit budget. There were zero observed violations of the 2(r+epsilon) bound among test cases covered by the corresponding model-risk calibration region across all four conditions. Maximum independently checked LP certificate gaps were approximately 2.0e-9 bits.

## Critical caveat: an admission gate is not free performance

The gate is a certification mechanism, not a claim that average regret automatically improves. In this test, the oracle reference fallback had mean regret 0.10992 bits, higher than the planner's observed mean regret under the 5% and 20% corruption conditions. It also had a larger p95 regret than the planner in those conditions.

The gate still rejects those cases because the stated certified bound exceeds budget. A deployed design therefore needs an independently qualified fallback with measured risk/cost properties; a valid epsilon bound; and an abstention/fallback cost model. If epsilon is merely estimated by the same misspecified planner, the certificate is invalid.

The policy-row mismatch should also be rejected by the identity binding—not rescued by a large epsilon. The epsilon envelope is for bounded model uncertainty when policy identities are already aligned.

## Next benchmark gates

1. Replace oracle epsilon with an independently justified model-error envelope: held-out residual calibration, model ensembles with a valid coverage argument, or certified simulator/plant uncertainty bounds.
2. Add a held-out seventh regime and shift detection; this bound is conditional on model-risk coverage and does not grant arbitrary OOD coverage.
3. Measure fallback/abstention cost, false rejection, high-tail regret, safety violations, runtime, memory, and calibration overhead.
4. Verify signed, content-addressed policy/risk receipts before optimization and again before action dispatch. Reject action-order/schema mismatch outright.
5. Compare against MDCP/max-p, ROCP, Utility-Directed Conformal Prediction, Conformal Policy Control, loss-aware DRO and ERT on identical held-out SCM splits. These methods have not yet been run in the same protocol; no SOTA claim is made.

## Reproducibility

- Source path: validation/decision_risk_certified_fallback.py
- Source Git blob: 093762836d0b2a238aa9cc8d46c35d57aa70a62d
- Source SHA-256: a00411b9e68df4746613557b595939fd2f6839e6aaab64201f97faa240cdc881
- Test path: validation/test_decision_risk_certified_fallback.py
- Test Git blob: d18d5fd0da75c16c253adafba948b266517675f5
- Test SHA-256: d1167b41b3c41bcabc957ed557318924668a93a146fc1f14c14469509fdb555d
- Raw result JSON SHA-256: 363b3f3e98a077d09caf0b1b3b62e98f6115067b84ad18cb92c4ca7a6eeaf977
- Compressed result SHA-256: 452f365ff861490fda3fec442a21683486faf33f8a652ed28e2f492000bb0709
- Combined Python validation: 19/19 tests passed in 16.816 seconds; full log SHA-256 21e0abde673150f15d48a87f16ce53013b73793f146b629518c76148c2e7d656.
- The Rust policy-risk binding module and its eight inline Rust tests are committed separately but unexecuted: no cargo/rustc is installed in this environment, and the draft PR CI was skipped. No Rust build or remote CI pass is claimed.
- Downloadable bundle SHA-256: c7ce1154d64263a49d1a63c618c6657ea2a9373aaf316d3d1abc59ed404df4a0.
