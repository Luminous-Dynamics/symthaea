# Risk-Qualified Policy Improvement Benchmark v0.1

Status: finite-support Rust reference and synthetic fixture. No formal CPC or action-conditional conformal guarantee is claimed by this code.

Tracking: #10042. Related: #10052 (risk-controlled reasoning compute), #9882 (exact constrained controller), #9954 and #9980 (temporal evaluation).

## 1. Research foundation

### Conformal Policy Control (CPC)

Prinster et al., Conformal Policy Control, ICML 2026, PMLR 306:99772-99824: https://proceedings.mlr.press/v306/prinster26a.html

CPC constrains an optimized candidate policy relative to a safe reference policy through a likelihood-ratio threshold beta:

    pi_beta(a|x) = min(pi_candidate(a|x), beta * pi_safe(a|x)) / Z_beta(x)

It calibrates allowable aggressiveness using weighted conformal risk control and bounded losses, including non-monotonic constraint losses. Its paper and appendix include likelihood-ratio grids, normalizer estimation, calibration-mixture weights, a worst-case test-point contribution, and acceptance/rejection sampling.

Author implementation inspected: https://github.com/samuelstanton/conformal-policy-control

- Public main commit inspected: 83ba982e14480f5e986cf7776de87a9b7e3d4245 (2026-10-24).
- The source repository declares the MIT license.
- Its CBO path estimates constrained-policy normalizers and worst-case test weights from Monte Carlo proposal samples. These are implementation approximations, not automatically proof artifacts.

### Action-conditional conformal risk-averse decisions (AC-CP)

Zhu, Kiyani, Pappas and Hassani, Conformal Risk-Averse Decision Making with Action Conditional Guarantee, ICML 2026, PMLR 306:167248-167276: https://proceedings.mlr.press/v306/zhu26z.html

This work targets safety conditioned on the action actually selected and connects action-conditional conformal prediction to action-conditional CVaR. This is not the same guarantee as CPC's marginal policy risk control.

### Why compose them

A policy can satisfy an average risk budget while concentrating risk in a small action subset. Conversely, an action-conditional guarantee does not by itself regulate the distribution shift induced by replacing a safe reference policy with an optimized candidate. The benchmark therefore reports marginal policy risk, per-action evaluator risk, action-conditional CVaR, policy movement away from a reference, reward and regret to an exact feasible oracle as separate outputs.

No marginal result may be labeled action-conditional.

## 2. Finite-support exact reference implementation

The crate at research/conformal-policy-control/ implements discrete policy clipping using exact integer masses and checked u128 arithmetic.

For safe masses s_i with total S, candidate masses c_i with total C, and beta = b_n/b_d, each clipped mass uses a common denominator:

    m_i = min(c_i * S * b_d, b_n * s_i * C)

The normalized policy is m_i / sum_j m_j. The common denominator cancels, so policy probabilities and normalizers are exact rational numbers. Candidate probability outside safe-reference support becomes zero under clipping.

The evaluator computes exact expected evaluator loss and reward, total variation from the safe reference, upper-tail discrete CVaR, each action's probability/loss/reward, and an exact finite-support supremum test weight for a declared calibration-mixture policy.

The weighted calibration diagnostic uses calibration observations from a declared mixture q_mix, importance weights pi_beta(a_i)/q_mix(a_i), the exact finite-support supremum of pi_beta/q_mix as the hypothetical test weight, and the declared worst-case test loss as the test-point loss.

This is a diagnostic for one fixed beta, not a completed proof of CPC's beta-grid selection. Evaluating several beta values on one calibration sample and selecting the best one needs a simultaneous-validity or selection argument. The output explicitly marks beta-grid selection as not formally qualified.

## 3. Why finite support first

The CPC reference CBO path approximates the constrained-policy normalizer and maximum test-point weight using Monte Carlo proposals. When the action domain is finite and enumerable, Symthaea can calculate both exactly. This removes Monte Carlo error from those two subroutines and makes policy clipping, normalization, mixture weights, worst-case test weights, and reward/risk trade-offs easier to audit.

Calibration outcomes remain random. Exact finite-support arithmetic does not remove calibration sampling uncertainty or replace the CPC theorem.

## 4. Synthetic non-monotonic example

The fixture includes four actions:

| Action | Reward | Constraint loss |
|---|---:|---:|
| safe hold | 10 | 0% |
| incremental | 25 | 10% |
| high-reward hazard | 100 | 90% |
| specialist safe | 55 | 2% |

The safe reference masses are [700, 200, 90, 10], while the optimized candidate masses are [100, 200, 300, 400]. The action loss profile is deliberately non-monotonic: the specialist action has high reward and low loss, while the high-reward hazard action has high reward and high loss.

The exact reference sweep demonstrates that evaluator risk can rise and then fall as beta increases. This is a counterexample to using a monotone binary search over beta without a justified theorem. The implementation enumerates the complete preregistered beta grid and reports every point; it does not auto-select beta.

The fixture's evaluator risks are synthetic ground truth. The weighted calibration output is a separate synthetic diagnostic using its own sample and worst-case test loss. The two must not be conflated.

## 5. Pinned external reference implementations

- **CPC:** paper https://proceedings.mlr.press/v306/prinster26a.html; code https://github.com/samuelstanton/conformal-policy-control; inspected main commit `83ba982e14480f5e986cf7776de87a9b7e3d4245` (2026-10-24), MIT. Its CBO implementation estimates constrained-policy normalizers and maximum test weights from Monte Carlo proposals. Reproduce its paper protocol; the Symthaea exact finite-support calculation is a separate discrete reference.
- **AC-RAC / AC-CP:** paper https://proceedings.mlr.press/v306/zhu26z.html; code https://github.com/Telvc/AC-RAC; inspected main commit `df7194a6fa7b283108b96672833e3f85ecc25835` (2026-05-20), MIT. The repository includes action-specific lambda calibration, per-action coverage/FDR reporting, vectorized-versus-literal-loop tests, cached fast reproduction, and full COVID/MovieLens rerun instructions. Its pinned commit predates the July 2026 proceedings, so verify paper-to-code alignment before calling it definitive.

These are separate comparators: CPC's marginal policy-risk guarantee is not AC-RAC's action-conditional guarantee. Neither is currently reproduced by this Symthaea crate.

## 6. Required benchmark comparisons

1. Safe reference policy.
2. Unconstrained optimized candidate.
3. CPC-style likelihood-ratio clipped policy.
4. Paper-matched CPC implementation/configuration after reproduction.
5. Marginal conformal risk-averse decision baseline.
6. Action-conditional conformal risk/CVaR baseline.
7. Risk-neutral controller.
8. Robust constrained optimization baseline.
9. Exact constrained finite-support oracle (upper bound only).

The exact oracle knows the synthetic environment's ground truth and is never deployable.

## 7. Evaluation metrics

- Overall constraint violation probability, with uncertainty.
- Per-action and per-context/action violation probability.
- Action-conditional CVaR90, CVaR95 and CVaR99.
- Expected reward and regret to exact constrained oracle.
- Total variation and KL from the safe reference where defined.
- Candidate feasible-support coverage.
- Exact beta, normalizer, test-point supremum weight and weight concentration.
- Marginal versus per-action risk gaps.
- Resource cost and receipt-verification cost.

## 8. Acceptance targets (not results)

- Exact policy normalization for every finite-support test.
- Zero accepted overflow; overflow must fail closed.
- Zero actions outside registered support.
- Zero future-label or revision leakage.
- Marginal risk and per-action risk are always reported separately.
- At least two non-monotonic/rare-catastrophe fixtures where marginal and per-action risk differ.
- Reward improvement over the safe reference under the declared evaluator risk budget, with uncertainty and exact-oracle regret reported.
- No beta-selection guarantee without a theorem/reproduction of the paper's selection conditions.
- No temporal/delayed-feedback guarantee inherited from static CPC/AC-CP.

## 9. Execution and integration

- Symthaea's planner/controller proposes candidate policies and actions.
- The exact resource-vector controller (#9882) applies deterministic hard constraints first.
- The policy risk engine qualifies stochastic constraint exposure, not physical hard limits.
- Symtropy supplies simulation outcomes and exact finite-domain oracles.
- Sol-Atlas can later supply physical context/state.
- Mycelix records provenance, policy digests and revocation/evidence lineage; it does not make the risk decision.

Dynamic, delayed and revised feedback are a separate extension. Static CPC/AC-CP guarantees must not be transferred to nonstationary sequential deployment without new assumptions and analysis.
