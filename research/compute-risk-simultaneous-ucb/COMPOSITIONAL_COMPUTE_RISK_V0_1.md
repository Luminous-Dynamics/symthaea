# Joint Simultaneous Risk Control for Dual-Threshold Reasoning Compute v0.1

**Status:** independent Rust reference implementation and synthetic finite-grid fixture. It is not a full reproduction of Conformal Thinking, not a distribution-shift guarantee, and not a real-model result.

Tracking: [#10052](https://github.com/Luminous-Dynamics/symthaea/issues/10052).

## 1. Paper-faithful risk definitions

Wang et al., *Conformal Thinking: Risk Control for Reasoning on a Compute Budget*, ICML 2026, PMLR 306:130032-130046:
https://proceedings.mlr.press/v306/wang26hd.html

The paper separates two early-exit errors.

- **Upper-threshold false-positive loss:** at the first upper-threshold exit, loss is one if the answer emitted at that step is wrong, otherwise zero.
- **Lower-threshold farsighted false-negative loss:** at a lower-threshold exit at step t, loss is the fraction of outputs f_t,...,f_T that would have been correct. This measures correct answers forfeited by stopping at the lower threshold even though later reasoning could have solved the task.

The reference implementation evaluates those two losses separately, with separate user risk budgets. Both are bounded in [0,1]. The efficiency metrics also mirror the paper's two views:
- upper efficiency regret: normalized number of reasoning steps spent after the first correct intermediate answer;
- lower efficiency loss: normalized number of incorrect intermediate answers produced up to the lower-threshold exit.

The implementation uses cumulative token counts for its primary compute objective, allowing unequal token cost per reasoning step.

## 2. What the paper does and the limitation we target

The paper uses distribution-free risk control / an upper confidence bound over threshold parameters and assumes risk monotonicity in the upper threshold and lower schedule parameter. It then describes a two-step procedure: select the upper threshold at its risk target, then select the lower schedule conditional on that upper threshold. The paper explicitly notes that this decoupled procedure loses some theoretical rigor and discusses a theoretical possibility where lower-threshold exits can disturb the upper false-positive guarantee under extreme distribution shift.

This reference tests a different finite-grid contract:

1. Freeze the complete candidate set of joint (upper threshold, lower schedule) policies before calibration outcomes are inspected.
2. Evaluate every candidate on every complete validation trajectory.
3. Compute the empirical upper and lower risks for every candidate.
4. Construct a one-sided Hoeffding UCB for every candidate-risk pair.
5. Bonferroni allocate the total statistical failure probability delta across exactly 2M bounds, where M is candidate count.
6. Keep candidates whose upper-risk and lower-risk UCBs both meet their respective budgets.
7. Select the feasible candidate with the lowest empirical mean normalized token cost.

There is no monotonicity assumption and no binary search. Because all 2M bounds are simultaneous, selecting a candidate by a validation-set efficiency objective among the feasible set does not invalidate the risk bounds for that frozen finite family.

## 3. Bound

Let X_1,...,X_n be independent/exchangeable **complete reasoning trajectories** from the target calibration population. For each frozen candidate m and risk axis r in {+, -}, define a bounded loss L_(i,m,r) in [0,1]. Let the empirical mean be:

    Lbar_(m,r) = (1/n) sum_i L_(i,m,r)

For M candidates and two risk axes, allocate:

    delta_(m,r) = delta / (2 M)

Hoeffding's one-sided inequality gives:

    UCB_(m,r) = min(1, Lbar_(m,r)
                       + sqrt(log(2 M / delta) / (2 n)))

A union bound over all 2M pairs implies:

    P( for every m,r: E[L_(m,r)] <= UCB_(m,r) ) >= 1 - delta

No independence between the two risks or between candidate policies is required for the union bound. Independence/exchangeability is required across complete trajectory calibration units. Steps within a trajectory are dependent and are never treated as independent samples.

The implementation rounds every per-trajectory bounded loss upward to PPM, rounds the empirical mean upward, and rounds the Hoeffding radius upward. Those integer roundings are conservative. The exact Bonferroni fraction is recorded as delta_ppm/(2M), not truncated to an integer allocation.

## 4. Why this improves the selection procedure

This finite-grid simultaneous method addresses three weaknesses:

- **Joint selection:** both exit risks are constrained simultaneously for every candidate rather than selecting the upper threshold first and the lower threshold second.
- **Selection-aware validity:** the confidence correction covers the complete predeclared candidate family, so the final choice can minimize empirical compute over the feasible candidates without using an uncorrected winner's risk estimate.
- **Non-monotonicity tolerance:** the candidate grid is exhaustively evaluated. The procedure does not assume that increasing a threshold or changing the lower schedule changes risk monotonically.

It is more conservative as the candidate grid grows. The Hoeffding radius scales with log(2M/delta); this is the price of a transparent finite-family guarantee. Future work can improve efficiency with sharper bounded-loss UCBs, empirical-Bernstein bounds under their variance assumptions, or valid learn-then-test / e-value selection. Those alternatives must be benchmarked against this simple reference.

## 5. Synthetic fixture

The committed fixture has 512 eight-step trajectories and four frozen joint candidate policies. Its fixture schema is `fixture-v0.1.schema.json`. Scenarios include easy, unsolvable, late-solvable, overconfident-wrong and noisy-solved cases. It uses:
- upper risk target: 230,000 PPM (23%);
- lower risk target: 150,000 PPM (15%);
- family statistical delta: 50,000 PPM (5%);
- simultaneous bounds: eight (two risk axes × four candidate policies).

Independent replay of the fixture gives:
- simultaneous Hoeffding radius: 70,401 PPM;
- selected candidate: `aggressive_dual`;
- selected candidate empirical upper risk: 146,485 PPM; upper UCB: 216,886 PPM;
- selected candidate empirical lower risk: 56,641 PPM; lower UCB: 127,042 PPM;
- mean normalized token cost: 507,813 PPM (about 50.8% of the full reasoning budget).

All four candidates are feasible under this synthetic pair of budgets; the chosen policy is the lowest-compute one:

| Candidate | Upper risk UCB | Lower risk UCB | Mean normalized compute | Feasible |
|---|---:|---:|---:|---|
| aggressive_dual | 216,886 PPM | 127,042 PPM | 507,813 PPM | yes |
| dual_balanced | 216,886 PPM | 114,347 PPM | 735,840 PPM | yes |
| upper_conservative | 216,886 PPM | 70,401 PPM | 770,020 PPM | yes |
| late_confidence | 216,886 PPM | 70,401 PPM | 929,200 PPM | yes |

A separate 20,000-family Monte Carlo sanity experiment used eight independent Bernoulli risk streams (four candidate policies × two risk axes), each with true mean 0.5 and 512 samples per family. The simultaneous UCB covered every population mean in 99.43% of families (95% Wilson interval 99.316%–99.525%). The Hoeffding target was 95%. This tests the numerical bound implementation under a simple synthetic design; it is not evidence about LLM task distributions or the Rust execution itself.

This is a deterministic synthetic fixture, not 512 independent real observations. These figures only verify that the implementation and independent reference calculation agree; they are not model performance or a valid deployment risk certificate.

## 6. Required benchmark expansion

Compare against:
1. Fixed full-budget reasoning.
2. Fixed small-budget reasoning.
3. Upper-threshold-only conformal risk control.
4. Lower-threshold-only stopping heuristic.
5. Paper-matched two-step Conformal Thinking.
6. Simultaneous joint finite-grid Hoeffding reference (this crate).
7. Simultaneous empirical-Bernstein UCB, with variance assumptions and coverage verified.
8. Group-conditional / task-family simultaneous risk control.
9. Exact finite-horizon oracle that knows all future correctness and compute costs (upper bound only).

Stress families must separately vary:
- solvable/easy, hard-but-solvable, unsolvable and adversarially overconfident tasks;
- confidence oscillation, calibration quality, varying token cost and non-monotone risk-vs-threshold curves;
- trajectory-level dependence across reasoning steps;
- task-family imbalance and rare high-risk subgroups;
- covariate shift, model/checkpoint change, prompt/template change, evaluator change;
- missing or delayed labels and later outcome revisions.

Under shift, do not reuse the exchangeability guarantee. Add shift detection, importance weighting with valid density-ratio assumptions, or abstention/recalibration as separate methods.

## 7. Acceptance targets (prospective)

- Zero candidate omitted from the frozen manifest.
- Zero invalid lower/upper threshold ordering accepted.
- Zero calibration steps counted as independent units when they belong to the same trajectory.
- Simultaneous risk-bound coverage at the declared familywise delta under synthetic iid trajectory simulations (with a preregistered Monte Carlo uncertainty interval).
- Risk UCBs below both targets on selected held-out candidates.
- At least 20% lower mean token use than fixed full-budget reasoning at matched selected risk, without hiding subgroup degradation.
- Selected candidate and all 2M bounds replay exactly from frozen fixture, code SHA, candidate-grid digest and calibration-trajectory digest.
- No guarantee under distribution shift, model change or adaptive candidate expansion unless separately addressed.

## 8. Reproduction source audit

Author reference: https://github.com/xidulu/reasoning_risk_control

The inspected public README describes the repository as preliminary; no LICENSE file was present in the inspected tree, and its lower-threshold helper had an unfinished variable reference. The code is therefore not copied or redistributed here. The PMLR paper is used as the method specification. The fixture retains the public source commit as audit metadata only.

## 9. Claim ceiling

This is a finite candidate family risk-control reference under independent/exchangeable complete-trajectory calibration units and a frozen candidate grid. It does not establish distribution-shift robustness, subgroup-conditional guarantees, anytime-valid online stopping, theorem-level validity of arbitrary learned confidence signals, risk control after model/prompt/verifier/task-distribution changes, or model correctness/suitability for production deployment.
