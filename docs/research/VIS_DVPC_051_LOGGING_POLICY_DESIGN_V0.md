# VIS-DVPC-051 — Constrained logging-policy design for portfolio OPE v0

**Status:** analytic known-answer benchmark in implementation; no SOTA, external-data, or deployment claim.

## Why this benchmark exists

Most off-policy-evaluation experiments treat the logging policy as fixed and compare estimators afterward. A second lever is to design the logging distribution before collecting data, so a family of candidate policies is easier to evaluate without sacrificing action support or acceptable immediate reward.

Motivating paper: Douglas, Persson & Provost, *Logging Policy Design for Off-Policy Evaluation* (arXiv:2605.15108, May 2026): https://arxiv.org/abs/2605.15108. It studies the reward–coverage tradeoff and logger design under different information regimes. This benchmark does not reproduce that paper's full framework. It tests transparent baseline policies and a finite-grid portfolio minimax objective.

Related work:
- [VIS-DVPC-050 / #9041 — sequential logging and reward drift](https://github.com/Luminous-Dynamics/symthaea/issues/9041), [PR #9043](https://github.com/Luminous-Dynamics/symthaea/pull/9043)
- [VIS-DVPC-046 / #8909 — multiple-policy selection](https://github.com/Luminous-Dynamics/symthaea/issues/8909), [PR #9011](https://github.com/Luminous-Dynamics/symthaea/pull/9011)
- SCOPE-RL reference evaluator and policy-selection metrics: https://scope-rl.readthedocs.io/en/latest/documentation/index.html

## Implementation

Source: `tools/vision-dvpc-051-logging-policy-design.rs`. The harness uses the frozen two-context, two-action Bernoulli reward table and exact target-policy values. It has three target-portfolio regimes (balanced, action-1-heavy, action-0-heavy), portfolio sizes M={4,8,32}, and an action-propensity floor of 0.02.

Six logger designs are compared at the same floor:
1. **Uniform**.
2. **Incumbent mixture** with explicit uniform exploration.
3. **Portfolio average** of target policies' action probabilities.
4. **Minimax weight second moment**: deterministic two-dimensional grid search minimizing the worst candidate's (E_mu[w_pi^2]), without using reward means.
5. **Oracle reward-aware minimax IPS variance**: grid search minimizing the worst candidate's exact IPS reward variance using known synthetic reward means. This is an oracle-information diagnostic, not deployable policy guidance when reward means are unknown.
6. **Reward-greedy with floor**: maximizes immediate reward in the known synthetic world while preserving the floor. Also an oracle-only diagnostic.

The grid search has a preregistered resolution: smoke uses 20 intervals per propensity axis; full mode uses 200 intervals per axis. The reported minimax result is grid-optimal only, not a general continuous optimum theorem.

## Exact objective

For action-1 target probability (q_x=pi(1mid x)), logger probability (p_x=mu(1mid x)), and two equally likely contexts, the expected squared importance weight is

[
E_mu[w_pi^2]=rac12sum_{xin{0,1}}left(rac{q_x^2}{p_x}+rac{(1-q_x)^2}{1-p_x}ight).
]

Since rewards are Bernoulli, (R^2=R). With known synthetic reward mean (r(x,a)), the exact IPS reward-score variance is

[
operatorname{Var}_mu(w_pi R)=rac12sum_xleft(rac{q_x^2r(x,1)}{p_x}+rac{(1-q_x)^2r(x,0)}{1-p_x}ight)-V(pi)^2.
]

Every deployable candidate must satisfy (p_xin[epsilon,1-epsilon]) for each context with (epsilon=0.02). The floor protects support by design; it does not guarantee sufficient finite-sample precision.

## Metrics

The program emits a stable tabular report with 54 rows (3 portfolio regimes × 3 portfolio sizes × 6 loggers):
- the action-1 propensity per context and minimum/maximum probability of any action;
- expected immediate reward under the logger and gap to the reward-greedy floor-constrained logger;
- worst-policy weight second moment and the implied asymptotic weight-only ESS fraction (1/max_pi E[w_pi^2]);
- worst-policy exact IPS reward variance;
- identity of the worst policy, grid resolution, and an explicit oracle-only flag.

These are complementary metrics. Immediate utility, weight second moment, estimator variance, and ESS should not be reduced to one unexamined score.

## Run and qualification

```sh
rustc --edition=2024 -O --test tools/vision-dvpc-051-logging-policy-design.rs -o /tmp/vis-dvpc-051-tests
/tmp/vis-dvpc-051-tests
rustc --edition=2024 -O tools/vision-dvpc-051-logging-policy-design.rs -o /tmp/vis-dvpc-051
/tmp/vis-dvpc-051 --smoke
/tmp/vis-dvpc-051
```

The focused workflow runs tests at the exact PR head, verifies deterministic output and row shape, checks all action propensities against the registered floor, bounds finite metrics, and records hashes for code, protocol, smoke/full outputs, and compiler version.

## Preregistered acceptance criteria

- Independently enumerated formulas agree with code fixtures.
- The minimax grid search matches exhaustive enumeration on a smaller test grid and is never worse than any point on its own registered grid.
- A skewed candidate portfolio can produce a non-uniform minimax logger; the code must not hard-code uniformity as the answer.
- **Proxy-mismatch known-answer control:** on the frozen balanced M=8 portfolio, the weight-second-moment minimizer must improve its optimized weight-moment objective versus uniform while *worsening* the exact reward-weighted worst-policy IPS variance. This is a deliberate positive control showing that weight-only ESS can be an insufficient design objective on reward-heterogeneous worlds.
- Every logger retains the floor; invalid floor and empty target portfolios fail closed.
- Oracle-only arms are always clearly identified and are not compared as deployable methods.
- Report reward–variance–support tradeoffs across all target regimes and M values. No universal winner claim from a finite synthetic panel.
- To establish real-world value, next validate on pinned public or licensed logged data with matched estimands and external methodology baselines. Open Bandit Dataset / Open Bandit Pipeline is a candidate, subject to suitability and license/eligibility review: https://research.zozo.com/data.html.

## Hypotheses (to test)

- Portfolio-aware minimax logging lowers the worst candidate's importance-weight second moment on skewed portfolios compared with uniform or incumbent-only logging.
- Uniform logging remains competitive for balanced portfolios and can win on fairness/symmetry; the grid method should not be presumed better everywhere.
- Reward-greedy logging can improve immediate reward while worsening the evaluation-risk frontier.
- Reward-aware oracle logging can set a lower bound on achievable worst-policy variance under this exact known-reward world, but its extra information must remain explicit.

The first stage is analytic and reproducible; it is not a field experiment, an SOTA result, or production authorization.
