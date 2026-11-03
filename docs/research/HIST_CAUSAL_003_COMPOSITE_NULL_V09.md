# HIST-CAUSAL-001 v0.9: finite-sample composite-null model adequacy

**Status:** research reference; synthetic development evaluation only.  
**Issue:** #10180.  
**Purpose:** replace v0.8's estimated least-favourable threshold with candidate-specific finite-sample Monte Carlo p-values and an intersection-union rule.

## 1. Design correction

v0.8's pilot pooled false-OOD rate looked low, but actuator-noncompliance showed excess conditional alarms. v0.9 changes the calibration itself and corrects the statistic so one candidate must explain the complete six-probe dataset.

The statistic is:

`T(D) = max_j sum_p sum_r log P_{M_j,theta_j}(y_pr | p)`

The maximization is over one single candidate mechanism, after summing its joint log likelihood across all probes. The model cannot change between probe types.

For each candidate null `M_i`, generate `B` independent calibration datasets under that candidate and score each with the same `T`. For the evaluation dataset:

`p_i = (1 + #{b : T(D_b^(i)) <= T(D_eval)}) / (B + 1)`

The lower tail signals that even the best catalogue model fits unusually poorly. Ties count against rejection.

Declare `MODEL_MISSPECIFIED` only when every candidate-specific `p_i <= alpha`.

## 2. Finite-sample argument

Assume the true process is an in-catalogue model `M_k`, with the recipe's point nuisance parameters known and correct. The `B` calibration scores from `M_k` and the independent evaluation score are exchangeable. Its Monte Carlo rank p-value is super-uniform (exactly uniform without ties; conservative with ties).

The event that v0.9 rejects the catalogue requires *all* candidate p-values to be at most `alpha`; this event is therefore a subset of `{p_k <= alpha}`. Hence:

`P(false MODEL_MISSPECIFIED | true M_k) <= P(p_k <= alpha) <= alpha`.

This is an intersection-union test of the composite null “at least one candidate model is true.” No Bonferroni correction is needed because all candidates must be rejected.

## 3. Run design and results

- 24 public development recipes.
- Six candidate models and six probe contexts.
- 5,000 calibration replicates per candidate/recipe/sample-size.
- 2,000 independent evaluation replicates per candidate truth and OOD truth/recipe/sample-size.
- `alpha=0.05`.
- `n=1,2,5,10,20` independent observations per probe (6n observations per case).
- The detector knows candidate models and recipe point parameters, but not the OOD mixture likelihood or latent component.

| n per probe | OOD detection | Pooled in-family false OOD | Worst candidate false OOD |
|---:|---:|---:|---:|
| 1 | 6.98% | 0.55% | 3.05% |
| 2 | 24.29% | 1.59% | 4.47% |
| 5 | 54.06% | 1.16% | 4.97% |
| 10 | 85.09% | 1.00% | 4.96% |
| 20 | 98.48% | 0.93% | 5.08% |

The worst conditional false-OOD model is actuator non-compliance. At n=20 its observed rate is 5.08%, slightly above the nominal level due to finite evaluation Monte Carlo variation. Its between-recipe standard deviation is included in the results package. Do not hide it in a pooled score.

Detection is heterogeneous across recipes: between-recipe SD is 14.58 percentage points at n=5 and 9.32 points at n=10. Report per-recipe outcomes and lower-quantile performance in future comparisons.

## 4. Exact limitations

The argument holds only if:
1. the true candidate is in the declared catalogue;
2. candidate point parameters used for calibration are correct;
3. calibration and evaluation units are independent/exchangeable;
4. the same joint statistic is used for calibration and evaluation;
5. random streams are independent.

Unknown nuisance parameters, persistent latent regimes, within-campaign dependence, model misspecification beyond the declared alternative, and physical lab calibration are not covered. A finite nuisance grid is valid only if the true parameter is covered; continuous models need a conservative supremum/confidence-set approach. Correlated measurements must be calibrated at the actual independent cluster.

## 5. External benchmark landscape

These are complements, not directly comparable score scales:

- SciAgentArena: about 200 stepwise-verified tasks across five scientific domains and 18 agents; results are uneven by task/domain and identify weaknesses in validity checking and self-directed exploration. https://sciagentarena.github.io/ ; https://arxiv.org/abs/2606.12736
- SciConBench: 9K+ live scientific conclusion-synthesis questions; best clean-room factual F1 in its preprint evaluation was 0.337. https://arxiv.org/abs/2606.11337
- Autonomous Science Bench 0.1: six real-lab pilot campaigns; reports separate discovery on measured designs and learning on unmeasured designs, while warning that three trials/task make small leaderboard gaps non-meaningful. https://www.autonomous-science-bench.ai/
- FIRE-Bench: full-cycle rediscovery; strongest tested agents remained below 50 F1, with recurring experiment-design, execution and evidence reasoning failures. https://proceedings.mlr.press/v306/wang26d.html
- SafeLab: 64 manipulation tasks, 63 calibrated laboratory assets; exposes unsafe-but-successful trajectories. https://proceedings.mlr.press/v306/bai26e.html
- 2026 symbolic-regression survey: more than 300 methods; highlights robustness, uncertainty, high-dimensional scaling and benchmark quality. https://doi.org/10.1007/s11831-026-10681-w

HIST-CAUSAL's distinctive proposed contribution is per-candidate false-alarm control for model-catalogue rejection, explicit non-identifiability/abstention, and adversarial mechanism/sensor/actuator controls. No SOTA win is claimed.

## 6. Next research steps

1. Unknown parameters: nuisance confidence sets and supremum p-values (or a valid Berger–Boos adjustment).
2. Persistent confounding: campaign-level latent regimes and cluster-level calibration.
3. Open world: held-out mixtures, nonlinear mechanism compositions and model-catalogue corruption.
4. Rust: exact tiny-domain verifier, Monte Carlo rank logic, deterministic receipts.
5. Paired baselines: random, fixed-order, disagreement, EIG and particle-filter experiment planners under matched budgets.
6. Embodied evidence: bind requested action to realized action, specimen state, sensor witnesses, calibration and trajectory safety.

## 7. Claim ceiling

v0.9 is a finite synthetic test with known point parameters and independent exchangeable units. It is not general unknown-parameter model adequacy, physical discovery, evidence for morphic resonance, a derivation of Chain Theory, or SOTA superiority.