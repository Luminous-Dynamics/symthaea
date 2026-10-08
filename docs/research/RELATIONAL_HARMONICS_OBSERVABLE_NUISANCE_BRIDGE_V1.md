# RH-006 Observable Nuisance Bridge — v1

Status: research boundary only.

## 1. Actual observable inputs

The held-out prediction sample exposed by RH-006 currently contains:

- feature_time;
- outcome_time;
- agent_a;
- agent_b;
- alignment;
- a_to_b;
- b_to_a;
- turn_taking;
- common_driver;
- future_outcome.

The production research estimator therefore observes feature/state summaries and future outcomes. It does not expose a latent innovation vector q or an outcome-error innovation e.

Source:
src/partnership/relational_prediction.rs
at exact research head 4806ee90eee02a6b1a431edf2d5ae03bcf702877.

## 2. Consequence for the generalized-K nuisance

The generalized joint-DGP layer uses

    E[e | q] = K q
    Xi = Cov(e,q) = K Sigma_q

but neither q nor e is directly represented in RelationalPredictionSample.

Therefore a real-data inference procedure cannot simply estimate K from the current RH-006 observable sample schema.

At least one additional bridge is required:

1. an observable innovation construction whose statistical meaning is explicitly established; or
2. a latent state/measurement model that identifies a quotient of the K class; or
3. a nuisance formulation entirely in observable moments that does not require recovering K or q separately.

## 3. Why this matters

The K-identifiability audit already established that latent-coordinate changes can alter K substantially while preserving the conditional law and observable covariance.

The current RH-006 schema makes that issue concrete: the estimator's actual inputs do not contain the latent coordinates being used to define K.

Consequently, synthetic K should remain a DGP parameterization for stress testing, not be promoted to an estimated real-data nuisance without an explicit measurement bridge.

## 4. Recommended binding hierarchy

The next inference architecture should prefer, in order:

    observed loss / feature data
        |
        +--> directly observable cross-moments, when justified
        |
        +--> explicitly identified innovation representation
        |
        +--> latent measurement model + identified equivalence class
        |
        +--> structural K only after normalization/identification

A synthetic K value may define a simulated DGP, but it must never silently become an externally estimated real-data nuisance merely because the simulator can produce it.

## 5. Evidence boundary

This is a source/schema audit, not a statistical validation.

It does not establish:

- identification of any real-data cross-covariance nuisance;
- a valid estimator of Xi;
- a valid estimator of K;
- a valid bootstrap;
- formal p-values or confidence intervals;
- empirical predictive validity;
- Rust implementation qualification.

Current decision remains:

applicability = not-approved-for-execution
selection = stop-assumption-failure
