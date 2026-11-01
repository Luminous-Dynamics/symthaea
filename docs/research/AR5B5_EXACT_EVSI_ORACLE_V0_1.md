# AR-5B5: Exact Finite-State EVSI Oracle v0.1

**Status:** research implementation draft; tests authored, not executed.  
**Tracking:** [#9696](https://github.com/Luminous-Dynamics/symthaea/issues/9696).  
**Claim ceiling:** exact one-step EVSI for the explicitly supplied finite-state posterior, likelihood, loss, and cost model. No claim that those models describe a real evaluator or that this is a scalable optimizer.

## 1. Purpose

The existing noisy-Pareto analyzer answers: *given the observations so far, what can be certified about the front under the declared bounded-IID assumptions?* It does not answer: *which candidate/objective should be measured next to reduce the risk of making the wrong decision?*

Issue #9696 asks for cost-aware replication by decision value. The first step should be an exact reference implementation on small finite problems. Without an exact oracle, an approximate controller can look plausible while optimizing interval width, model entropy, or a proxy unrelated to the decision we actually care about.

This module is that reference oracle.

## 2. Decision-theoretic definition

Let the current posterior over finite latent worlds be p(w), let d be a terminal decision, and let L(d,w) be the declared non-negative loss.

Current Bayes risk:

    R(p) = min_d sum_w p(w) L(d,w)

For a candidate/objective measurement action a with possible outcomes y and likelihood P(y | w,a):

    P(y | a) = sum_w p(w) P(y | w,a)
    p(w | y,a) = p(w) P(y | w,a) / P(y | a)

Expected posterior Bayes risk:

    ER(a) = sum_y P(y | a) R(p(. | y,a))

Expected value of sample information:

    EVSI(a) = R(p) - ER(a)

The first policy ranks actions by EVSI(a) divided by frozen all-in incremental cost. This is a one-step measurement policy, not the globally optimal sequential policy under a finite campaign budget. A future controller must account for option value, stopping, and interactions between successive actions.

## 3. Model contract

Schema: **symthaea.architecture-engineering.exact-discrete-evsi-problem v0.1.0**.

The problem object freezes:

- a current posterior probability for every latent joint world;
- a complete loss for each terminal decision in each world;
- each available candidate/objective action;
- every possible outcome and its likelihood in every world;
- identities and SHA-256 digests for the posterior, observation, decision-loss, and all-in cost models;
- the campaign ID and positive integer cost units for each measurement action.

Rows are canonicalized by stable IDs before hashing. The likelihoods for all outcomes of one action must sum to one in every world. World posterior probabilities must sum to one. All tables must exactly cover the same world set. Model inputs are explicitly bounded to keep the exact oracle finite and predictable.

A latent world is a *joint* state. Shared physical measurements, cross-objective correlation, and correlation between candidates can be represented by a joint state space and likelihood table. The implementation never assumes independent objectives merely because they appear in different columns.

## 4. How to encode Pareto decisions

The oracle accepts an explicit terminal decision set and loss matrix. For a small Pareto-front problem, decisions can encode candidate-front membership or a discrete front classification. The loss matrix can distinguish:

- false declaration that a candidate is dominated;
- retaining a dominated candidate;
- missing a true Pareto candidate;
- missing a feasibility or safety constraint;
- missing a preregistered target;
- delaying a correct decision.

The weights are part of the frozen decision-loss model. Do not collapse a hard safety constraint into a soft quality score unless the application explicitly permits that trade-off.

## 5. Why this is an exact oracle

For each proposed action, the implementation enumerates every listed outcome, computes its predictive probability, forms the exact finite posterior, recomputes minimum expected decision loss, and takes the expectation across outcomes. There is no Monte Carlo approximation in this finite-state calculation.

The report stores every action's current risk, expected posterior risk, EVSI, EVSI-per-cost, and outcome-wise posterior risk. The problem digest binds the full model and action costs. Report reconstruction detects changed inputs or output values.

Zero-value actions are not selected. Ties are deterministic by action ID.

## 6. Mandatory baselines

Compare the oracle's selected action with:

1. uniform replication;
2. random candidate/objective allocation;
3. maximum interval width;
4. interval width per cost;
5. posterior entropy reduction per cost;
6. racing / successive halving / best-arm identification where applicable;
7. qNEHVI/qLogNEHVI and qNParEGO/qLogNParEGO when the predictive model and search space satisfy their assumptions.

The oracle is not a substitute for qNEHVI. BoTorch recommends qNEHVI for noisy or batched multi-objective Bayesian optimization; it is an industry baseline to beat on its suitable regime, not a strawman: https://botorch.org/docs/v0.17.2/tutorials/multi_objective_bo

## 7. Calibration programme

Start with exact small cases:

- two candidates, two objectives, and a few discrete latent worlds;
- independent and shared-latent/correlated worlds;
- symmetric and asymmetric decision losses;
- objective-specific measurement costs;
- informative, weakly informative, and uninformative outcomes;
- observations that alter a Pareto membership decision but not the current scalarized score;
- safety/feasibility decisions whose losses dominate quality preferences.

Then generate larger finite instances where exact enumeration is still feasible. For every approximation, report:

- regret relative to exact EVSI action ranking;
- reduction in Bayes decision risk after the selected observation;
- action cost and controller overhead;
- rank agreement and cases where the approximation chooses a dominated action;
- calibration under likelihood and posterior misspecification.

The scalable controller must be compared against the exact oracle on overlapping small instances before it is tested on large instances.

## 8. Model-misspecification controls

The current oracle is exact only *conditional on its finite model*. It will happily produce a precise answer to an incorrect likelihood or loss model.

Therefore add controls where:

- likelihoods are misspecified or omit a latent outcome;
- posterior mass is concentrated on the wrong worlds;
- the physical process drifts after the posterior is frozen;
- missingness depends on unobserved outcomes;
- multiple objective measurements share the same underlying sample;
- the cost model omits acquisition, setup, retries, verification, or artifact-resolution work.

The production controller must compare model checks against held-out outcomes and retreat to a conservative or non-EVSI action policy when its assumptions fail.

## 9. Proposed acceptance gates

These are proposed gates, not measured results:

- exact report reconstruction is deterministic for the frozen finite model;
- zero-information observations have exactly zero EVSI within declared floating-point tolerance;
- a fully revealing observation reduces Bayes risk to the known minimum;
- cost changes alter EVSI-per-cost ranking without changing raw EVSI;
- all malformed posterior/likelihood/loss/world-identity tables fail closed;
- approximate EVSI has a preregistered maximum regret versus the exact oracle on held-out small models;
- at equal all-in cost, the adaptive policy must outperform uniform and interval-width controls on held-out independent task families before being promoted.

## 10. Exact-head validation

    cargo fmt --all -- --check
    cargo test -p symthaea-architecture-engineering pareto_evsi_oracle -- --nocapture
    cargo test -p symthaea-architecture-engineering
    cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings

The current environment does not include cargo or rustc. The test suite is authored but uncompiled and unexecuted; no validation or performance result is claimed.
