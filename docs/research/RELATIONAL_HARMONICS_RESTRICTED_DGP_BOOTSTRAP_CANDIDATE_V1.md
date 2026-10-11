# RH-006 Restricted-DGP Nested Rolling Bootstrap — candidate v1

## Purpose

The block-loss simulation narrowed the problem enough to identify a stronger candidate: generate a complete bootstrap time series under an explicit null DGP, then rerun the exact RH-006 rolling evaluator from scratch.

This is a **candidate procedure**, not an approved inference method.

## Core idea

The current retained-loss MBB starts after the difficult part has already happened: both nested models have been fitted on the observed data. Resampling the resulting origin loss means therefore cannot reproduce the finite-sample parameter-estimation distribution of the nested ridge estimators.

The candidate instead bootstraps before estimation:

~~~text
observed data
    |
    v
explicit null DGP
    |
    v
bootstrap feature/outcome series
    |
    v
exact RH-006 rolling refits
    |
    v
target-level loss differentials
    |
    v
origin-level D statistics
    |
    v
same studentized T statistic
~~~

The entire rolling schedule is regenerated jointly. This is what allows shared training-window observations and serial dependence across adjacent held-out blocks to remain coupled.

## Null

The scientific null for this candidate is **no incremental relational predictive content**:

~~~text
beta_relational = 0
~~~

That is deliberately not equated with zero finite-sample forecast-loss difference.

The current simulations show why. Under beta_relational = 0, the complete nested fitting system produced a negative mean origin loss differential of about -0.00189. That is a finite-sample estimator effect, not evidence that the scientific null is false.

The bootstrap therefore does not re-center the observed loss differential at zero. It generates the null reference distribution by simulating the restricted DGP and refitting both nested estimators. This is the critical conceptual improvement.

## Conditional feature path

A pragmatic first implementation would keep the observed feature path fixed and bootstrap the outcome errors. That is only valid under a declared exogeneity/conditioning argument.

The current RH-006 code does not include lagged outcomes in the predictor vector, but that alone is not a proof that the relational feature processes are strictly exogenous. If the features are jointly stochastic and endogenous with the outcome, the candidate must move to a joint feature/outcome bootstrap or an explicit DGP for the feature process.

This assumption is therefore a hard approval gate, not an implementation detail.

## Bootstrap DGP

A candidate null world contains:

1. the declared NonRelationalContext structure;
2. no incremental relational contribution;
3. a dependence-preserving residual/innovation generator;
4. declared treatment of heteroskedasticity;
5. a frozen nuisance-parameter source policy.

The nuisance-parameter source is still open. Possible designs include estimating the restricted model on the first eligible training span or on a separate calibration span that does not contain the confirmatory evaluation. That choice must be fixed before inspecting the observed forecast improvement.

## Full refit

Every bootstrap replicate must reconstruct the complete RH-006 procedure.

That includes:

- the exact rolling origin schedule;
- the same train/test/gap sizes;
- the same fixed horizon;
- the same training-only means and scales;
- the same fixed ridge lambda;
- the same deterministic solver;
- the same nested model feature definitions;
- the same target-level squared-loss construction;
- the same origin-level aggregation;
- the same Bartlett studentizer.

There is no shortcut through retained fitted coefficients or retained loss vectors.

## Dependence and block length

There are at least two distinct dependence mechanisms:

~~~text
1. overlapping training windows
2. serial dependence in the outcome/error process
~~~

For the current geometry:

~~~text
L_overlap = 1 + floor((48 - 1) / 16) = 3
~~~

so L=3 is a defensible first fixed design anchor.

It is **not** yet established that 3 is sufficient for all dependence in the forecast-loss process. The current simulation suggests that blindly increasing the block to 6 worsens calibration. That is evidence against “bigger block is safer,” not evidence that 3 is universally optimal.

A later candidate may use a dependence-derived bandwidth/block rule, but that rule must be specified independently of the observed forecast advantage.

## Statistic

Keep the retained-geometry candidate statistic:

~~~text
D[o] = mean_j(loss_context[o,j] - loss_relational[o,j])

bar_D = mean_o D[o]

T = sqrt(O) * bar_D / sqrt(Bartlett_LRV(D[o]))
~~~

The major change is not the numerator. It is the **bootstrap reference world** used to obtain critical values.

## What the literature actually supports

Clark–McCracken develop nested forecast comparison methods specifically around finite-sample equal accuracy and bootstrap inference for recursively or rolling estimated models. Their framework reinforces the distinction between population zero coefficients and finite-sample equal-accuracy behavior.

Doko Tchatoka–Haque develop a hybrid bootstrap with moving-block and residual components, and their construction explicitly generates bootstrap data and re-estimates the forecasting system. They report good size behavior under serially correlated and heteroskedastic errors in their setting.

Those papers establish that this overall direction is scientifically well motivated. They do **not** establish the exact RH-006 bridge because RH-006 adds fixed ridge regularization, training-only standardization, disjoint held-out blocks, overlapping rolling training windows, and a custom solver. The bridge must therefore be tested rather than inherited.

## Validation matrix

The next simulation gate should include:

| Dimension | Required stress |
|---|---|
| Error dependence | IID, AR(0.5), AR(0.8), stronger weak dependence |
| Variance | homoskedastic, heteroskedastic |
| Ridge | declared small/medium/larger fixed values |
| Geometry | current 48/16/4 plus smaller and larger windows |
| Overlap | current 16-step overlap plus non-overlap control |
| Horizon | 1-step first; later horizons only as a new procedure scope |
| Design matrix | well-conditioned and near-singular deterministic cases |
| Feature process | fixed/exogenous first; endogenous/joint bootstrap later |
| Null | coefficient-restricted DGP with estimator-induced finite-sample risk asymmetry preserved |
| RNG | pinned and independently replayable |

The minimum acceptance standard is stable null size across the matrix, not one favorable scenario.

## Decision

This candidate should **replace the current loss-level MBB as the next research target**, but it must not replace the current applicability contract yet.

Therefore:

~~~text
applicability = not-approved-for-execution
selection     = stop-assumption-failure
formal p      = disabled
formal CI     = disabled
~~~

The present artifact is a blueprint for the next simulation/implementation loop.