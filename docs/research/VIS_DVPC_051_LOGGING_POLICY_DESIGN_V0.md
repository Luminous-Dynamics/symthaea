# VIS-DVPC-051 — Constrained logging-policy design for portfolio OPE v0

**Status:** analytic known-answer benchmark in implementation. No SOTA, external-data, or deployment claim.

## Why this benchmark exists

Most off-policy-evaluation studies treat the logging policy as fixed and compare estimators afterward. A complementary lever is to design the logging distribution before collecting data, so an entire candidate-policy portfolio is easier to evaluate without sacrificing action support or acceptable immediate reward.

Motivating reference: Douglas, Persson & Provost, *Logging Policy Design for Off-Policy Evaluation* (arXiv:2605.15108, May 2026): https://arxiv.org/abs/2605.15108. It studies the reward–coverage tradeoff and logger design under different information regimes. This benchmark does not reproduce that paper's full framework; it tests transparent baselines and finite-grid portfolio objectives.

Related work:
- [VIS-DVPC-050 / #9041 — sequential logging and reward drift](https://github.com/Luminous-Dynamics/symthaea/issues/9041), [PR #9043](https://github.com/Luminous-Dynamics/symthaea/pull/9043).
- [VIS-DVPC-046 / #8909 — multiple-policy selection](https://github.com/Luminous-Dynamics/symthaea/issues/8909), [PR #9011](https://github.com/Luminous-Dynamics/symthaea/pull/9011).
- SCOPE-RL reference evaluator and policy-selection metrics: https://scope-rl.readthedocs.io/en/latest/documentation/index.html.

## Experimental design

Source: `tools/vision-dvpc-051-logging-policy-design.rs`. The harness uses the frozen two-context, two-action Bernoulli reward table, exact target-policy values, three target-portfolio regimes (balanced, action-1-heavy, action-0-heavy), portfolio sizes M={4,8,32}, and an action-propensity floor of 0.02.

Seven logger designs are compared at the same support floor:

1. **Uniform.**
2. **Incumbent mixture** with explicit uniform exploration.
3. **Portfolio average** of candidate policies' action probabilities.
4. **Minimax weight second moment:** finite two-dimensional grid search minimizing the worst candidate's E[w²], without reward information.
5. **Reward-envelope minimax IPS second moment:** finite grid search minimizing the worst candidate's upper bound on E[(wR)²], using a preregistered reward-mean envelope.
6. **Oracle reward-aware minimax IPS variance:** finite grid search minimizing the worst candidate's exact IPS reward variance using known synthetic reward means. This is an oracle-information diagnostic, not a deployable method when reward means are unknown.
7. **Reward-greedy with floor:** maximizes immediate reward in the known synthetic world while preserving the floor. Also an oracle-only diagnostic.

The grid is preregistered: smoke uses 20 intervals per propensity axis; full mode uses 200. Each minimax result is optimal only on its registered finite grid, not a general continuous-optimality theorem.

## Exact objectives and information assumptions

Let q_x be the target policy's probability of action 1 in context x, p_x the logger's probability of action 1, and P(X=x)=1/2.

The expected squared importance weight is:

`E_mu[w_pi^2] = 1/2 * sum_x (q_x^2 / p_x + (1-q_x)^2 / (1-p_x))`.

For Bernoulli rewards, R²=R. With true reward mean r(x,a), the exact IPS score variance is:

`Var_mu(w_pi R) = 1/2 * sum_x (q_x^2*r(x,1)/p_x + (1-q_x)^2*r(x,0)/(1-p_x)) - V(pi)^2`.

The reward-envelope arm uses a registered upper bound u(x,a) on each Bernoulli reward mean and minimizes the worst candidate's upper bound:

`E_mu[(w_pi R)^2] <= 1/2 * sum_x (q_x^2*u(x,1)/p_x + (1-q_x)^2*u(x,0)/(1-p_x))`.

The envelope is fixed at u(0,0)=0.20, u(0,1)=0.95, u(1,0)=0.98, u(1,1)=0.20, which contains all four true means in this frozen synthetic world. The benchmark validates this condition at startup and fails closed if the envelope undercovers, contains non-finite/out-of-range values, or the true reward table itself is invalid. A dedicated negative test raises one true mean above its registered envelope and requires rejection. This is an explicit information assumption, not a learned guarantee. In a real system, these values would need to come from a valid external bound or a correctly constructed upper confidence bound available at logger-design time. An uncalibrated reward model is not a safety guarantee.

Even a valid envelope bound is not the exact variance: it upper-bounds the score's second moment and omits the policy-specific V(pi)² subtraction. It can therefore be conservative. The exact reward-aware arm uses true synthetic reward means and must remain oracle-only.

Every logger satisfies p_x in [0.02,0.98] in each context. The floor protects structural support by construction; it does not guarantee adequate finite-sample precision.

## Metrics

The program emits 63 stable primary result rows (3 portfolio regimes × 3 portfolio sizes × 7 loggers), plus 45 `ENVELOPE_SENSITIVITY` rows (3 regimes × 3 portfolio sizes × 5 envelope scales):
- action-1 propensity per context and minimum/maximum probability of any action;
- expected immediate logger reward and gap versus the reward-greedy floor-constrained logger;
- worst-policy weight second moment and implied asymptotic weight-only ESS fraction 1/max_pi E[w_pi²];
- worst-policy registered-envelope upper bound on IPS reward-score second moment;
- worst-policy exact IPS reward variance;
- worst weight-moment policy ID, grid resolution, and explicit oracle-only flag.

The envelope sensitivity rows use fixed scales {1.0, 1.1, 1.25, 1.5, 2.0} relative to the registered upper envelope, clipped to the Bernoulli maximum of 1.0. They record the selected logger probabilities, immediate reward, worst weight second moment, ESS proxy, envelope score bound, exact worst IPS variance, and grid resolution. The sensitivity sweep is diagnostic: a looser envelope must never reduce its bound for a fixed logger, but the selected minimax logger can change as the objective changes.

These metrics are complementary. Immediate utility, weight second moment, envelope bound, exact estimator variance, and ESS should not be collapsed into one unexamined score.

## Known-answer controls

The code includes independent enumeration of the weight second moment and exact Bernoulli IPS variance, verification that the registered reward envelope contains all true means, exhaustive comparison of the minimax grid against a smaller independent grid enumeration, and a support-floor check for every logger.

Two deliberately important controls guard against proxy overclaiming:

- **Weight-proxy mismatch:** on the frozen balanced M=8 portfolio, the weight-second-moment minimizer improves worst E[w²] relative to uniform but worsens exact reward-weighted worst-policy IPS variance. Thus weight-only ESS is not a sufficient logger-design objective on reward-heterogeneous worlds.
- **Reward-envelope design:** on the same balanced M=8 fixture, the envelope-aware logger improves worst true IPS variance over uniform. This is a property of this registered DGP and envelope, not a universal theorem.

A skewed portfolio should also produce a non-uniform weight-minimax logger; uniformity is not hard-coded as the answer.

## Run and qualification

```sh
rustc --edition=2024 -O --test tools/vision-dvpc-051-logging-policy-design.rs -o /tmp/vis-dvpc-051-tests
/tmp/vis-dvpc-051-tests
rustc --edition=2024 -O tools/vision-dvpc-051-logging-policy-design.rs -o /tmp/vis-dvpc-051
/tmp/vis-dvpc-051 --smoke
/tmp/vis-dvpc-051
```

The focused exact-head workflow verifies test results, deterministic TSV output, 63-row shape, all support floors, bounded metrics, oracle-only labels, full grid-resolution execution, and SHA-256 receipts for source, protocol, smoke/full outputs, and compiler version.

## Preregistered acceptance criteria

- Independent formulas agree with code fixtures.
- The minimax grid search matches exhaustive enumeration on a smaller test grid and is no worse than any point on its own registered grid.
- The weight-proxy mismatch and reward-envelope known-answer controls pass as registered.
- The reward envelope is explicitly validated against the four synthetic true means; a negative control with undercoverage is rejected before emitting design results. No claim that this synthetic envelope is statistically estimated.
- Every logger retains the action-propensity floor; invalid floor and empty portfolio fail closed.
- Oracle-only arms remain clearly identified and are not compared as deployable methods.
- Report reward–variance–support tradeoffs across all regimes and portfolio sizes. No universal winner claim from a finite synthetic panel.
- Real-world evaluation requires pinned data, matched estimands, appropriate industry baselines, and license/eligibility review. Open Bandit Dataset / Open Bandit Pipeline is a candidate, subject to suitability and commercial-use review: https://research.zozo.com/data.html.

## Research hypotheses

- Portfolio-aware minimax logging lowers worst candidate importance-weight second moment on skewed portfolios compared with uniform or incumbent-only logging.
- Uniform logging remains competitive for balanced portfolios and can win on fairness/symmetry.
- Reward-greedy logging can improve immediate reward while worsening OPE risk.
- A valid reward envelope can produce a better reward-weighted variance design than weight-only optimization in some regimes, but can be conservative when bounds are loose.
- Oracle reward-aware logging gives a diagnostic lower bound under exact synthetic reward information, but the extra information must remain explicit.

The first stage is analytic and reproducible, not a field experiment, SOTA result, faithful reproduction of Douglas et al., or production authorization.
