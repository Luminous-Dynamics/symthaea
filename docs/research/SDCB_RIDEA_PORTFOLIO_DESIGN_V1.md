# Greedy R-I / R-IDeA Design-Portfolios for SDCB v1

**Status:** bounded greedy design-only portfolio reference
**Parent:** R-I / R-IDeA robust acquisition v1
**Implementation:** [scripts/research/sdcb_ridea_portfolio.rs](../../scripts/research/sdcb_ridea_portfolio.rs)
**Schema:** [SDCB_RIDEA_PORTFOLIO_V1.schema.json](SDCB_RIDEA_PORTFOLIO_V1.schema.json)
**Example:** [SDCB_RIDEA_PORTFOLIO_V1.example.json](SDCB_RIDEA_PORTFOLIO_V1.example.json)
**Expected result:** [SDCB_RIDEA_PORTFOLIO_V1.expected.json](SDCB_RIDEA_PORTFOLIO_V1.expected.json)

## 1. Why portfolio selection is a different problem

Independent candidate ranking does not solve batch design. If two experiments cover the same region, selecting both may waste a scarce budget. If one experiment fills a missing region, the representativeness value of the next experiment changes.

This reference therefore recalculates the R-I/R-IDeA score after every selected design. The history supplied to MMD is augmented with the newly selected design, so the acquisition term is conditional on the portfolio already chosen.

## 2. Greedy portfolio rule

Let H_t be the selected-design history at round t, D the fixed target/test design sample, and X_t the remaining candidate designs. For each remaining candidate x, the source-faithful R-I/R-IDeA scoring code evaluates the current MMD² ratio:

\[ R\text{-}I_t(x)=EIG(x)\left(1-\lambda\frac{\widehat{MMD}^{2}(H_t\cup\{x\},D)}{\widehat{MMD}^{2}(H_t,D)}\right), \]

\[ R\text{-}IDeA_t(x)=R\text{-}I_t(x)\,\operatorname{sigmoid}\!\left(\frac{|\hat f(x)-g(x)|-\tau}{\kappa}\right). \]

The greedy planner:

1. Scores every affordable remaining design against H_t.
2. Selects the highest positive score under the declared R-I or R-IDeA mode, optionally normalized by cost using the explicit project extension.
3. Adds the selected design to H_t and removes it from the candidate pool.
4. Recomputes all remaining MMD² ratios and repeats until the target empirical MMD² is zero, the budget or experiment limit is reached, no positive acquisition remains, or the candidate pool is exhausted.

This is a greedy portfolio rule, not an exhaustive globally optimal subset solver. The reference does not claim greedy optimality.

## 3. Frozen two-step coverage example

Initial history is `{0,3}` and target support is `{0,1,2,3}` in one dimension, with RBF sigma 1.0. Initial biased empirical MMD² is 0.1417384713.

Four candidates are offered:

| Candidate | EIG | Design | |f-hat − g| | Interpretation |
|---|---:|---:|---:|---|
| cover-a | 0.70 | 1.0 | 1.5 | representative coverage point |
| cover-b | 0.65 | 2.0 | 1.2 | complementary coverage point |
| high-EIG extrapolation | 1.30 | 4.0 | 0.1 | high nominal information, poor target coverage |
| redundant low-EIG | 0.10 | 0.5 | 0.1 | low information and limited coverage value |

With lambda=0.5, tau=0.5, kappa=0.5 and cost one unit per design:

- Round 1 selects `cover-a`: MMD² falls from 0.1417384713 to 0.0376081268; R-IDeA score is 0.5347608619.
- Round 2 recalculates against the updated history and selects `cover-b`: MMD² falls from 0.0376081268 to zero; R-IDeA score is 0.5214195276.
- The higher-EIG extrapolation is not selected; after the first pick its representativeness ratio becomes negative.

The two-step portfolio exactly matches the four-point empirical target support in this fixture at total cost two. This is arithmetic demonstration of the MMD² update and greedy ranking, not a real-world generalization result.

## 4. Critical claim boundary: frozen EIG and proxy predictions

This reference deliberately isolates portfolio coverage from full adaptive learning:

- Each candidate's EIG is supplied and frozen for the entire portfolio.
- Current-model and proxy predictions are supplied and frozen.
- No experiment outcome is generated or observed.
- No posterior is updated.
- Joint EIG of the selected set is not computed, and individual EIG values must not be summed and reported as joint information gain.
- This is a design-only batch planner, not a complete outcome-adaptive sequential R-IDeA implementation.

The next increment should bind each accepted measurement to an EGBT receipt, update the posterior/model from that authenticated observation, retrain or revalidate the proxy, and recompute the candidate EIG/DeA terms before the next decision. It should then compare the adaptive policy to exact finite-horizon EVSI oracles.

## 5. Baselines and competitive evaluation

Compare, on identical candidate pools and budgets:

1. Static top-k by raw EIG.
2. Static top-k by R-I or R-IDeA, without updating the history.
3. Greedy R-I portfolio with MMD² recomputed after each selection.
4. Greedy R-IDeA portfolio with MMD² recomputed after each selection.
5. Random and space-filling designs.
6. Kernel herding / representative batch selection.
7. Cost-aware MFBO and qMFMES where the task has fidelity levels.
8. DSSI one-step EVSI and exact non-myopic policy on paired decision tasks.

Report separately: final target MMD², prediction loss on held-out target samples, calibration, R-I/R-IDeA marginal scores, decision regret, EVSI, number/cost of measurements, and resource vectors. Do not use final MMD² alone as a proxy for predictive accuracy or decision value.

## 6. Research targets

These are project targets, not achieved results:

- Reproduce every step of the frozen MMD² sequence exactly.
- On held-out target distributions, improve target prediction loss over static EIG top-k without degrading well-specified performance.
- At matched budget, reduce redundant sample selection compared with static top-k and report target coverage gains.
- Keep R-I/R-IDeA score, joint information gain, decision EVSI and final target loss in separate columns.
- Add paired seeds and held-out target-distribution families; use family-level confidence intervals.
- External claims require a faithful upstream comparison and independent reproduction.

## 7. Implementation limits

The Rust reference uses the upstream-source-faithful MMD² implementation from `sdcb_ridea.rs`. It has a 64-candidate pool limit and inherits the sibling R-IDeA limits of 512 design samples and 64 dimensions.

12 Rust tests are authored but not compiled or executed in this environment. The frozen fixture was independently recalculated in JavaScript; that is not a Rust run.

## 8. References

1. Tang, Sloman & Kaski (2026), AISTATS / PMLR 300: https://proceedings.mlr.press/v300/tang26d.html
2. Source-faithful R-IDeA implementation: https://github.com/TrbingWY/robust-boed
3. Multi-fidelity best practices for materials/molecular research: https://www.nature.com/articles/s43588-025-00822-9

## 9. Claim boundary

This is an exact calculation of greedy marginal scores under the supplied frozen EIG/proxy inputs and exact MMD² updates. It does not establish global optimality of the chosen portfolio, joint EIG, adaptive Bayesian learning, held-out generalization, SOTA performance or physical-system validity.