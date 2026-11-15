# decision-sufficient-evidence

A standalone Rust prototype for EGDDM D3: choosing the smallest useful observation to resolve a consequential engineering uncertainty.

## Run

From this crate directory:

    cargo test
    cargo run -- ../../docs/development-map/fixtures/decision-sufficient-evidence-v0.1.json

Input contract:
- docs/research/decision-sufficient-evidence-v0.1.schema.json
- docs/development-map/fixtures/decision-sufficient-evidence-v0.1.json
- docs/development-map/fixtures/decision-sufficient-evidence-v0.1.expected.json
- docs/research/DECISION_SUFFICIENT_EVIDENCE_V0_1_2026-11-15.md

## Method

The planner computes expected value of sample information (EVSI) from caller-supplied:
- prior probabilities over mutually exclusive hypotheses;
- action loss for each hypothesis;
- observation likelihoods for each evidence outcome and hypothesis;
- explicit acquisition-cost utility; and
- separate resource budgets.

It returns the baseline minimum-risk action, each candidate's expected posterior loss, gross EVSI, net EVSI, hard-gate rejection reasons, and a Pareto frontier over information value and resource costs.

Authorization, source scope, execution isolation and prerequisites are explicit boolean gates. They are not confidence weights and cannot be offset by an attractive EVSI score. Resource budgets are also hard bounds.

## Limitations

This is a decision-theoretic ranking over supplied assumptions. It does not estimate or calibrate priors, likelihoods, action losses or costs. The boolean hard-gate fields are caller assertions, not authenticated permissions or security attestations. It does not execute an observation, authorize an engineering action, establish causality, or measure downstream results.

The acquisition cost utility must be expressed in the same declared utility units as action loss. The individual resource dimensions remain visible and separately budgeted; the planner does not silently convert dollars, latency, review time and compute into one score.

Eight Rust unit-test definitions are authored. They have not been compiled or executed. Do not treat them as passing.
