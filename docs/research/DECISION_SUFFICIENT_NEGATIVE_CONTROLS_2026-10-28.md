# Decision-Sufficient Ambiguity: Falsification and Shift Stress Tests

**Status:** synthetic finite-game diagnostic; not a real SCM, production result, or external SOTA benchmark.

## Why this experiment exists

The positive pilot showed substantially lower regret when the ambiguity region was calibrated in candidate-policy expected-risk space rather than in ten prior-feature coordinates. This run tries to break that result. It injects small and large loss-matrix errors, corrupts action identity through a row permutation, and tests an unseen seventh prior regime deliberately concentrated on the pair of latent hypotheses with the greatest Euclidean separation in the true eight-policy risk vectors.

## Protocol

- True loss matrix: 8 fixed candidate policies by 16 latent hypotheses, joint cross-entropy loss in bits.
- Calibration: 100 episodes for each of six known regimes (600 total).
- In-distribution test: 20 new episodes for each known regime (120 total).
- Out-of-distribution test: 800 new priors from a seventh, uncalibrated Dirichlet regime; alpha 0.5 at latent hypotheses 9 and 15 and 0.002 elsewhere. These two hypotheses have the largest Euclidean distance between their columns in the true risk matrix.
- Observation budget 120; nominal split-conformal target 90%.
- The same calibration set and known-regime test cases are reused for every loss-model condition; all OOD cases are generated from a fixed seed.
- 4 local regression tests pass. Maximum independently checked LP certificate gap is about 2.01e-9 bits, using floating-point LPs and explicit margins rather than formal interval arithmetic.

## Known-regime stress results

| Condition | Risk-matrix RMSE vs. truth | Pooled coverage | Pooled mean regret | p95 regret | Bound violations among pooled-covered cases |
|---|---:|---:|---:|---:|---:|
| Correct model | 0.00000 | 88.3% | 0.00152 | 0.01267 | 0 |
| 5%-scale perturbation | 0.01933 | 88.3% | 0.00451 | 0.01957 | 0 |
| 20%-scale perturbation | 0.07110 | 88.3% | 0.01274 | 0.05075 | 0 |
| Policy-row identity mismatch | 0.26847 | 88.3% | 0.09781 | 0.20009 | 31 |

Regret is evaluated against the true loss matrix; **5%** / **20%** mean the Gaussian additive noise scale as a fraction of the global per-cell standard deviation of the matrix. The row-identity condition permutes the planner's loss rows but preserves deployed action identities, deliberately modeling a policy/receipt/schema mismatch.

The perturbation comparisons are paired against the same 120 in-distribution cases. For the 5% perturbation, the mean increase in regret was 0.002988 bits (paired bootstrap 95% CI [0.001751, 0.004465]); at 20%, 0.011214 bits ([0.007946, 0.014883]). Under row-identity mismatch the mean increase was 0.096287 bits ([0.084759, 0.108085]). Group-max coverage on these in-distribution test cases was 91.7% under the correct matrix, 94.2% under 5% perturbation, 92.5% under 20% perturbation, and 91.7% under row mismatch, with the same action-regret pattern as pooled in this sample.

The exact-risk **2r** conditional regret inequality is observed to hold among covered cases under the correct risk matrix. When action rows are permuted, the claimed inequality fails against the real deployed action losses on 31 pooled-covered and 26 group-max-covered cases; maximum regret / (2r) was 1.836 (pooled) and 1.623 (group-max). This demonstrates that geometric calibration does not fix an action identity or risk-model integrity failure. The 5% and 20% perturbation cases happened not to violate **2r** in this particular finite sample; that is not a guarantee under general misspecification.

## Unseen seventh-regime results

| Planner model | Calibration radius | Empirical coverage | Wilson 95% interval | Mean regret | p95 regret |
|---|---|---:|---:|---:|---:|
| Correct risk matrix, pooled radius | Known six regimes only | 84.6% | [82.0%, 87.0%] | 0.00235 | 0.03948 |
| Correct risk matrix, max-known-regime radius | Known six regimes only | 90.0% | [87.7%, 91.9%] | 0.00380 | 0.03990 |
| 20%-perturbed matrix, pooled radius | Known six regimes only | 84.5% | [81.8%, 86.8%] | 0.00784 | 0.03989 |
| 20%-perturbed matrix, max-known-regime radius | Known six regimes only | 89.8% | [87.5%, 91.7%] | 0.00784 | 0.03989 |

Pooled calibration undercovered the 90% target on this deliberately shifted prior family: the 95% Wilson upper bound was 87.0%, below 90%. Taking the maximum radius across known regimes recovered 90.0% empirical coverage on the 800 OOD cases, at a higher mean regret in the correct-model setting. This is a single constructed seventh regime, not evidence of protection against arbitrary unknown shifts. With 20%-scale matrix perturbation, even max-known-regime coverage was 89.75% empirically; the interval includes 90% but the result shows why shift detection and model-integrity checks are still required.

## Research implications

1. **Treat action identity as a security boundary.** Risk rows, policy IDs, simulation receipts and executed action IDs need a single canonical digest / typed binding. Fail closed on reorder, missing rows, stale risk tensors or policy hashes that do not match.
2. **Separate two kinds of uncertainty.** Statistical uncertainty in expected policy loss is not the same as model uncertainty in the loss matrix or policy semantics. The former is what the conformal radius addresses; the latter needs a model ensemble, independent verifier, or fallback.
3. **Detect support shift.** Max-over-known-regime calibration can help for a new mixture that is still near known score behavior, but it cannot create a universal conditional guarantee for arbitrary unseen regimes. Track coverage diagnostics and OOD risk features and abstain / choose a safe baseline when out of support.
4. **Benchmark risk against real outcomes.** The next evaluation must include loss-matrix perturbation levels, action-ID mismatch, held-out mechanisms, previously unseen actions, and explicit safe-fallback cost. Do not optimize only average regret while ignoring high-tail errors.

## External literature: direct baselines still required

The results are not an SOTA comparison. Direct, same-split baselines should include Multi-Distribution Robust Conformal Prediction (MDCP, max-p), Risk-Optimal Conformal Prediction (ROCP), Utility-Directed Conformal Prediction, loss-aware DRO, ERT conditional-coverage diagnostics, and Conformal Policy Control (CPC):

- MDCP (ICML 2026): https://proceedings.mlr.press/v306/yang26ad.html ; implementation https://github.com/AragornBFRer/MDCP
- ROCP (ICML 2026): https://proceedings.mlr.press/v306/wang26ar.html ; implementation https://github.com/TaoWangPenn/Risk-Optimal-Conformal-Prediction
- Utility-Directed Conformal Prediction (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/0c6b452f1bbfb6905f6bac957d73b321-Abstract-Conference.html
- Loss-Aware Distributionally Robust Optimization (ICML 2026): https://proceedings.mlr.press/v306/ohnemus26a.html
- Conditional Coverage Diagnostics / ERT (ICML 2026): https://proceedings.mlr.press/v306/braun26c.html ; implementation https://github.com/EISacho/covmetrics
- Conformal Policy Control (ICML 2026) uses conformal calibration from a safe reference policy to regulate an optimized but untested policy under declared risk tolerance: https://proceedings.mlr.press/v306/prinster26a.html ; implementation https://github.com/samuelstanton/conformal-policy-control

## Provenance

- Source: **decision_sufficient_negative_controls.py**
- Tests: **test_decision_sufficient_negative_controls.py**
- Raw JSON: **DECISION_SUFFICIENT_NEGATIVE_CONTROLS_2026-10-28.json**
- Source SHA-256: **79679c17b6746321912ddba963d161c11c60ab5000057df325b834c9c85c1111**
- Test SHA-256: **aef228cd788a83cf5d3a55a9685d786800f8298c9dea832e95e9d22cf3ee96b7**
- Raw JSON SHA-256: **519da9c0f6b1f05dc8c058c0205e9321616d961b1341c5417bddeb347ce4e9e5**
- Local regression tests: 4/4 passed. The test suite re-runs the fixed-seed experiment; runtime about 11.6 seconds.
- Latest remote check: CI was skipped and PR Governance remained in progress. No remote CI pass is claimed.
