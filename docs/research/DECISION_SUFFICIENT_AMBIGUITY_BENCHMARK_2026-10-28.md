# Decision-Sufficient Ambiguity Sets: A Paired Pilot (2026-10-28)

## Hypothesis

Robust decision systems may not need to identify an entire latent prior when the downstream decision layer only uses a finite family of candidate actions. Instead, calibrate an uncertainty region over the vector of candidate-action expected losses and optimize the policy against that region. This is a decision-targeted representation—not a claim that prior estimation is generally unnecessary or that the method is novel.

## Experiment

- Synthetic finite game: 16 latent hypotheses, 8 candidate policies, horizon 2, joint cross-entropy regret in bits.
- Six synthetic prior regimes; 50 calibration episodes and 12 independent test episodes per regime; 300 calibration and 72 test episodes total.
- Observation budgets 60, 120, 240; split-conformal targets 90% and 95%; fixed seed 20261031.
- Paired comparisons use the same calibration episodes, test episodes and nested observation prefixes.
- Compared 10-feature prior-space boxes (pooled and max-over-regime radii), 8-dimensional candidate-policy expected-loss boxes (pooled and max-over-regime radii), and unrestricted simplex minimax.
- Four local unit/smoke tests passed. Maximum independently checked LP certificate gap: 2.0000042e-9 bits. This is a floating-point numerical check with tolerance margins, not formal interval arithmetic.

## Results: mean decision regret

These regret values are directly comparable because they use the same test cases and regret objective. Coverage events are not interchangeable: feature methods cover 10 prior features, while risk-space methods cover the vector of 8 candidate-policy expected losses.

| Target | Budget | Feature pooled | Feature group-max | Decision-risk pooled | Decision-risk group-max | Unrestricted prior |
|---:|---:|---:|---:|---:|---:|---:|
| 90% | 60 | 0.03040 | 0.03744 | **0.00253** | 0.00444 | 0.10661 |
| 90% | 120 | 0.01229 | 0.01427 | **0.00159** | 0.00159 | 0.10661 |
| 90% | 240 | 0.00868 | 0.00903 | **0.00064** | 0.00064 | 0.10661 |
| 95% | 60 | 0.03646 | 0.05345 | **0.00253** | 0.00461 | 0.10661 |
| 95% | 120 | 0.01449 | 0.03137 | **0.00159** | 0.00239 | 0.10661 |
| 95% | 240 | 0.00940 | 0.01054 | **0.00064** | 0.00064 | 0.10661 |

At target 90%, pooled decision-risk calibration reduced observed mean regret against pooled feature-space calibration by 91.7%, 87.1% and 92.6% at budgets 60, 120 and 240. At target 95%, the reductions were 93.1%, 89.0% and 93.2%. Paired bootstrap 95% intervals for decision-risk pooled minus feature pooled regret excluded zero in all six settings:

| Target | Budget | Mean paired regret difference | Paired bootstrap 95% interval |
|---:|---:|---:|---:|
| 90% | 60 | -0.02787 | [-0.03539, -0.02078] |
| 90% | 120 | -0.01071 | [-0.01436, -0.00712] |
| 90% | 240 | -0.00804 | [-0.01100, -0.00509] |
| 95% | 60 | -0.03393 | [-0.04195, -0.02646] |
| 95% | 120 | -0.01290 | [-0.01690, -0.00897] |
| 95% | 240 | -0.00877 | [-0.01192, -0.00588] |

The unrestricted-prior minimax baseline admitted every prior but had mean regret 0.10661 bits and p95 regret 0.21243 bits in this toy game.

## Coverage diagnostic

At target 90%, pooled decision-risk-region coverage / worst-regime coverage was 90.3% / 75.0% at budget 60, 95.8% / 75.0% at 120, and 91.7% / 83.3% at 240. The max-over-regime risk radius recorded 100% / 100% at budgets 60 and 120, and 95.8% / 83.3% at 240. At target 95%, pooled risk-vector coverage was 94.4%, 98.6%, and 95.8%; max-over-regime was 100%, 100%, and 97.2% at budgets 60, 120, 240. With only 12 test cases per regime, each test case shifts subgroup coverage by 8.3 percentage points. These are pilot frequencies, not future-coverage guarantees.

## Literature and novelty boundary

Related methods already incorporate downstream decisions into uncertainty sets. Utility-Directed Conformal Prediction (ICLR 2025) uses downstream utility and conformal sets: https://proceedings.iclr.cc/paper_files/paper/2025/hash/0c6b452f1bbfb6905f6bac957d73b321-Abstract-Conference.html. Multi-Distribution Robust Conformal Prediction (ICML 2026) develops max-p aggregation for heterogeneous source distributions: https://proceedings.mlr.press/v306/yang26ad.html. Loss-Aware Distributionally Robust Optimization via Trainable Optimal Transport Ambiguity Sets (ICML 2026) learns loss-informed ambiguity sets: https://proceedings.mlr.press/v306/ohnemus26a.html. Optimal Decision-Making Based on Prediction Sets (ICML 2026) derives risk-optimal prediction sets and robust decisions: https://proceedings.mlr.press/v306/wang26ar.html. Conditional Coverage Diagnostics for Conformal Prediction (ICML 2026) develops ERT, a classifier-risk diagnostic for conditional coverage failures: https://proceedings.mlr.press/v306/braun26c.html.

The potential contribution to investigate is narrower: calibrated candidate-policy loss-vector ambiguity coupled to causal intervention selection, explicit subgroup requirements and safe abstention. Direct implementations of MDCP/max-p, ERT, utility-directed conformal prediction and risk-optimal conformal prediction must be compared before any claim of superiority.

For external causal evaluation, ACDB reports PC plus greedy active orientation at directed F1 42.7% and SHD 4.79 under its specific protocol: https://arxiv.org/abs/2609.31675. Those structural metrics are not comparable with toy-game regret. CausalArena reports ranking shifts across synthetic, semantic-operational, formula-grounded and real-data regimes: https://arxiv.org/abs/2609.11897.

## Next gates

1. Add MDCP/max-p, ERT, utility-directed conformal and risk-optimal conformal as direct baselines on the same splits and objective.
2. Expand to at least 1,000 held-out episodes per regime and 10+ seeds with paired regret CIs and regime-wise coverage CIs.
3. Remove the oracle-data assumption by calibrating policy-loss vectors from simulator rollouts, posterior-predictive samples or logged interventional outcomes; model uncertainty in the risk matrix itself.
4. Add unseen policies/objectives, a seventh unseen regime, a misspecified loss matrix and a permuted-risk negative control.
5. Evaluate cost-adjusted intervention regret, decision quality per intervention, graph F1/SHD/SID, runtime, memory, and safe fallback frequency on held-out SCM families.

## Provenance

Status: SYNTHETIC FINITE GAME, NOT A REAL SCM OR EXTERNAL SOTA BENCHMARK.

Source: validation/decision_sufficient_ambiguity_benchmark.py  
Tests: validation/test_decision_sufficient_ambiguity_benchmark.py  
Source SHA-256: 867239a0e0c6b22b41f3b097867f6f99855430cac5f62a3fd64fc8d32438ec78  
Test SHA-256: 1aa212622ad7c9058d358addc9007b1ccc3eeb950258384cbf351ce4f801f7dd  
Raw result SHA-256: 158eed4c09bedbf9ee8368bad3a960715065ecd7386c30567df23b3c7c64c07e  
Gzip result SHA-256: a01ad8348e0f808c4fbca0233c885f104270e64a8ab2d5cc6517472ee03695b9  
Summary SHA-256: 1a602d215ac3527e767f55cff1417ec7166832809e469f19b8bbc3bb3a4e8c2d

The attempted 100-calibration / 30-test-per-regime run timed out and produced no valid result. Only the completed 50-calibration / 12-test-per-regime run is reported. Four local tests passed. No remote CI pass is claimed.
