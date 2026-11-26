# SDCB robust multi-fidelity decision reference

Standalone Rust reference that evaluates one scientific design at multiple fidelity levels using RPSD's finite-scenario robust decision oracle.

## Build and run

```bash
rustc --edition=2024 --test scripts/research/sdcb_multifidelity.rs -o /tmp/sdcb-multifidelity
/tmp/sdcb-multifidelity
```

The source uses only the Rust standard library plus the sibling reference module `sdcb_robust_pareto.rs`.

## What it does

- Accepts fidelity-specific observation likelihoods under every plausible model scenario.
- Computes a maximin outcome-to-action policy shared across hidden scenarios.
- Reports scenario values, robust post-experiment value and robust decision gain.
- Retains money, instrument time, compute and energy as separate resource coordinates.
- Computes an exact componentwise Pareto frontier and filters it under hard resource caps.
- Selects the highest robust-value feasible option without inventing exchange rates between resources.

## Limits

- At most 8 fidelity options; inherits RPSD's limits of 8 scenarios, 5 actions, 4 outcomes.
- One-step exact reference only; deterministic policies only.
- Likelihoods are supplied scenario assumptions. The code does not train or calibrate a fidelity model.
- It does not implement a multi-fidelity GP, sequential low-to-high upgrade correlations or asynchronous batch scheduling.

## Frozen result

A low-fidelity channel is 70% accurate in the nominal scenario but reversed (30%) under a plausible shift. Its robust gain is zero. Medium and high fidelity have robust gains 0.20 and 0.35. The expensive medium duplicate is dominated. Under the frozen example resource caps, medium is the highest-value feasible fidelity.

## Validation boundary

13 Rust tests are authored but not compiled or executed in this environment. The example/expected JSON were fetched from GitHub, parsed and independently recalculated in JavaScript. No physical measurement, MFBO surrogate fit or external benchmark run is claimed.