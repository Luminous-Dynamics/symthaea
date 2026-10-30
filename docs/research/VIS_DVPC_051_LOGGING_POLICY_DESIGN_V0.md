# VIS-DVPC-051 — Constrained logging-policy design for portfolio OPE v0

**Status:** analytic known-answer benchmark in implementation. No SOTA, external-data, or deployment claim.

## Why this benchmark exists

Most off-policy-evaluation studies treat the logging policy as fixed and compare estimators afterward. A complementary lever is to design the logging distribution before collecting data, so an entire candidate-policy portfolio is easier to evaluate without sacrificing action support or acceptable immediate reward.

Motivating reference: Douglas, Persson & Provost, *Logging Policy Design for Off-Policy Evaluation* (arXiv:2605.15108, May 2026): https://arxiv.org/abs/2605.15108. The paper studies reward–coverage tradeoffs and logger design under different information regimes. This benchmark reproduces selected analytic baselines for a tiny known-answer setting; it is not a full reproduction of the paper.

Related work:
- [VIS-DVPC-050 / #9041 — sequential logging and reward drift](https://github.com/Luminous-Dynamics/symthaea/issues/9041), [PR #9043](https://github.com/Luminous-Dynamics/symthaea/pull/9043).
- [VIS-DVPC-046 / #8909 — multiple-policy selection](https://github.com/Luminous-Dynamics/symthaea/issues/8909), [PR #9011](https://github.com/Luminous-Dynamics/symthaea/pull/9011).
- SCOPE-RL reference evaluator and policy-selection metrics: https://scope-rl.readthedocs.io/en/latest/documentation/index.html.

## Experimental design

Source: \`tools/vision-dvpc-051-logging-policy-design.rs\`. The harness uses the frozen two-context, two-action Bernoulli reward table, exact target-policy values, three target-portfolio regimes (balanced, action-1-heavy, action-0-heavy), portfolio sizes M={4,8,32}, and an action-propensity floor of 0.02.

Nine logger designs are compared at the same support floor:

1. **Uniform.**
2. **Incumbent mixture** with explicit uniform exploration.
3. **Portfolio average** of candidate policies' action probabilities. This matches the first moment of the policy distribution.
4. **Proposition 6 second-moment pseudo-target:** for a known distribution over target policies and known reward means, uses \(\tilde\pi_t(a|x)=\sqrt{\mathbb E[\pi_t(a|x)^2]}\) and sets the logger proportional to \(\tilde\pi_t(a|x)\sqrt{\mu(a,x)}\). This minimizes *expected* IPW MSE averaged over the target-policy prior under overlap; it is not a worst-case/minimax objective. The benchmark uses a uniform prior over the finite candidate portfolio.
5. **Proposition 3 unconstrained-target worst-case logger:** with known reward means but no informative restriction on the target policy, the full-support logger is \(\pi_l(a|x)=\mu(a,x)/(c_x+\mu(a,x)^2)\), where c_x is chosen so action probabilities sum to one. This is the paper's per-context minimax solution over the unconstrained target-policy class.
6. **Minimax weight second moment:** finite two-dimensional grid search minimizing the worst candidate's E[w²], without reward information.
7. **Reward-envelope minimax IPS second moment:** finite grid search minimizing the worst candidate's upper bound on E[(wR)²], using a preregistered reward-mean envelope.
8. **Oracle reward-aware minimax IPS variance:** finite grid search minimizing the worst candidate's exact IPS reward variance using known synthetic reward means.
9. **Reward-greedy with floor:** maximizes immediate reward in the known synthetic world while preserving the floor.

The two analytic proposition loggers and the reward-aware / reward-greedy controls use the known synthetic reward means and are explicitly **oracle-information-only**. They are not deployable estimates of an unknown reward function. Proposition 3 and Proposition 6 solve different information problems: one is worst-case over an unconstrained target class; the other is average-case under a known target-policy prior.

The finite-grid minimax results are optimal only on their registered grids, not a general continuous-optimality theorem. The Proposition 3 and Proposition 6 formulas are analytic, with the registered action-propensity floor applied as a feasibility constraint. For this frozen DGP, their analytic solutions lie inside the registered floor.

## Exact objectives and information assumptions

Let q_x be the target policy's probability of action 1 in context x, p_x the logger's probability of action 1, and P(X=x)=1/2.

The expected squared importance weight is:

\`E_mu[w_pi^2] = 1/2 * sum_x (q_x^2 / p_x + (1-q_x)^2 / (1-p_x))\`.

For Bernoulli rewards, R²=R. With true reward mean r(x,a), the exact IPS score variance is:

\`Var_mu(w_pi R) = 1/2 * sum_x (q_x^2*r(x,1)/p_x + (1-q_x)^2*r(x,0)/(1-p_x)) - V(pi)^2\`.

### Proposition 3: no informative target-policy prior

For a known reward mean \(\mu(a,x)\) and an unconstrained class of full-support target policies, the paper derives:

\`pi_star(a|x) = mu(a,x) / (c_x + mu(a,x)^2)\`

where \(c_x\) is selected such that \(\sum_a \pi_\star(a|x)=1\). For two actions, the implementation solves this monotone scalar normalization by bisection. The logger is calculated separately for each context. It is an all-policy-class worst-case comparator, not a minimax guarantee restricted to the finite portfolio used by the other grid-search methods.

### Proposition 6: known target-policy distribution

For a known distribution over target policies, Proposition 6 defines the per-action pseudo-target:

\`tilde_pi(a|x) = sqrt(E_pi[pi(a|x)^2])\`.

With known reward means, the expected-MSE-optimal logger under overlap is:

\`mu_star(a|x) = tilde_pi(a|x)*sqrt(r(x,a)) / sum_b(tilde_pi(b|x)*sqrt(r(x,b)))\`.

The benchmark's portfolio prior assigns equal probability to each registered policy. This logger minimizes the *average* per-policy IPW variance (and thus average IPW MSE divided by N) for that prior in this frozen known-reward setting. The mean variance metric in the output is the equal-weight mean over target policies; at sample size N, its corresponding mean IPW MSE is mean_variance/N.

### Reward-envelope objective

The reward-envelope arm uses a registered upper bound u(x,a) on each Bernoulli reward mean and minimizes the worst candidate's upper bound:

\`E_mu[(w_pi R)^2] <= 1/2 * sum_x (q_x^2*u(x,1)/p_x + (1-q_x)^2*u(x,0)/(1-p_x))\`.

The envelope is fixed at u(0,0)=0.20, u(0,1)=0.95, u(1,0)=0.98, u(1,1)=0.20, which contains all four true means in this frozen synthetic world. The benchmark validates this condition at startup and fails closed if the envelope undercovers, contains non-finite/out-of-range values, or the true reward table itself is invalid. A dedicated negative test raises one true mean above its registered envelope and requires rejection. This is an explicit information assumption, not a learned guarantee. In a real system, these values would need to come from a valid external bound or a correctly constructed upper confidence bound available at logger-design time. An uncalibrated reward model is not a safety guarantee.

Even a valid envelope bound is not the exact variance: it upper-bounds the score's second moment and omits the policy-specific V(pi)² subtraction. It can therefore be conservative.

Every logger satisfies p_x in [0.02,0.98] in each context. The floor protects structural support by construction; it does not guarantee adequate finite-sample precision.

## Metrics

The program emits 81 stable primary result rows (3 portfolio regimes × 3 portfolio sizes × 9 loggers), plus 45 \`ENVELOPE_SENSITIVITY\` rows (3 regimes × 3 portfolio sizes × 5 envelope scales):
- action-1 propensity per context and minimum/maximum probability of any action;
- expected immediate logger reward and gap versus the reward-greedy floor-constrained logger;
- worst-policy weight second moment and implied asymptotic weight-only ESS fraction 1/max_pi E[w_pi²];
- worst-policy registered-envelope upper bound on IPS reward-score second moment;
- worst-policy exact IPS reward variance and equal-weight mean IPS reward variance over the target-policy prior;
- worst weight-moment policy ID, grid resolution, and explicit oracle-only flag.

The envelope sensitivity rows use fixed scales {1.0, 1.1, 1.25, 1.5, 2.0} relative to the registered upper envelope, clipped to the Bernoulli maximum of 1.0. They record the selected logger probabilities, immediate reward, worst weight second moment, ESS proxy, envelope score bound, exact worst IPS variance, and grid resolution. The sensitivity sweep is diagnostic: a looser envelope must never reduce its bound for a fixed logger, but the selected minimax logger can change as the objective changes.

These metrics are complementary. Immediate utility, average and worst-case risk, weight second moment, envelope bound, exact estimator variance, and ESS should not be collapsed into one unexamined score. Proposition 6 optimizes average prior risk; Proposition 3 optimizes a different all-target worst-case objective; the finite-grid comparators optimize the finite candidate portfolio.

## Known-answer controls

The code includes independent enumeration of the weight second moment and exact Bernoulli IPS variance, validation that the registered reward envelope contains all true means, exhaustive comparison of the minimax grid against a smaller independent grid enumeration, and a support-floor check for every logger.

Important controls guard against proxy and theorem overclaiming:

- **Proposition 3:** the normalizer is solved by bisection; tests verify the action probabilities match the closed form, sum to one, and are within the registered floor. A dense target-action grid confirms that the resulting per-context worst-case IPS variance is within tolerance of the numerical logger-grid minimum.
- **Proposition 6:** an independent implementation of the policy second moments and reward square-root weighting must match the logger probabilities; a dense-grid search verifies that it minimizes average policy variance for the frozen balanced M=8 prior within numerical tolerance.
- **Weight-proxy mismatch:** on the frozen balanced M=8 portfolio, the weight-second-moment minimizer improves worst E[w²] relative to uniform but worsens exact reward-weighted worst-policy IPS variance. Thus weight-only ESS is not a sufficient logger-design objective on reward-heterogeneous worlds.
- **Reward-envelope design:** on the same balanced M=8 fixture, the envelope-aware logger improves worst true IPS variance over uniform. This is a property of this registered DGP and envelope, not a universal theorem.

## Run and qualification

\`\`\`sh
rustc --edition=2024 -O --test tools/vision-dvpc-051-logging-policy-design.rs -o /tmp/vis-dvpc-051-tests
/tmp/vis-dvpc-051-tests
rustc --edition=2024 -O tools/vision-dvpc-051-logging-policy-design.rs -o /tmp/vis-dvpc-051
/tmp/vis-dvpc-051 --smoke
/tmp/vis-dvpc-051
\`\`\`

The focused exact-head workflow verifies test results, deterministic TSV output, 81-row primary shape, 45-row envelope sensitivity, all support floors, bounded metrics, oracle-only labels, full grid-resolution execution, and SHA-256 receipts for source, protocol, smoke/full outputs, and compiler version.

## Preregistered acceptance criteria

- Independent formulas agree with code fixtures.
- Proposition 3's all-target formula matches the published expression for known rewards and is checked against a dense per-context worst-case target-policy grid.
- Proposition 6's pseudo-target formula matches the published expression for a known target-policy distribution and known reward means; its average-risk result is compared against a dense finite-grid objective check.
- The minimax grid search matches exhaustive enumeration on a smaller test grid and is no worse than any point on its own registered grid.
- The weight-proxy mismatch and reward-envelope known-answer controls pass as registered.
- The reward envelope is explicitly validated against the four synthetic true means; a negative control with undercoverage is rejected before emitting design results. No claim that this synthetic envelope is statistically estimated.
- Every logger retains the action-propensity floor; invalid floor and empty portfolio fail closed.
- Oracle-only arms remain clearly identified and are not compared as deployable methods.
- Report reward–variance–support tradeoffs across all regimes and portfolio sizes. No universal winner claim from a finite synthetic panel.
- Real-world evaluation requires pinned data, matched estimands, appropriate industry baselines, and license/eligibility review. Open Bandit Dataset / Open Bandit Pipeline is a candidate, subject to suitability and commercial-use review: https://research.zozo.com/data.html.

## Research hypotheses

- Portfolio-aware minimax logging lowers worst candidate importance-weight second moment on skewed portfolios compared with uniform or incumbent-only logging.
- Proposition 6's pseudo-target design lowers average portfolio variance relative to ordinary portfolio-average logging in the regimes covered by its assumptions, but may sacrifice worst-case variance versus minimax designs.
- Proposition 3's all-policy worst-case logger and Proposition 6's target-prior average-risk logger occupy different points on the reward–coverage frontier; which one is better depends on how the target portfolio is specified.
- Uniform logging remains competitive for balanced portfolios and can win on fairness/symmetry.
- Reward-greedy logging can improve immediate reward while worsening OPE risk.
- A valid reward envelope can produce a better reward-weighted variance design than weight-only optimization in some regimes, but can be conservative when bounds are loose.

The first stage is analytic and reproducible, not a field experiment, SOTA result, full reproduction of Douglas et al. across all information regimes, or production authorization.
