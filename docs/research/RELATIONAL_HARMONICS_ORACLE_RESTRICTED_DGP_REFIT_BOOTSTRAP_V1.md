# RH-006 Oracle Restricted-DGP Refit Bootstrap — superseded record

## Status

**Superseded: prior numerical result was not independently executed.**

The earlier version of this artifact recorded a 40×99 oracle benchmark. Those numbers were entered as a research artifact without an independently executed run in the present verification loop. They must therefore **not** be treated as observed simulation evidence.

The record is intentionally retained rather than deleted so the evidence history remains auditable.

## Canonical replacement

Use:

\`docs/research/RELATIONAL_HARMONICS_ORACLE_RESTRICTED_DGP_REFIT_BOOTSTRAP_EXECUTED_V2.json\`

with executable harness:

\`scripts/research/rh006_oracle_refit_splitmix64.py\`

The canonical independently executed run uses:

- 400 Monte Carlo datasets per scenario;
- 199 oracle bootstrap draws per dataset;
- SplitMix64 v1 + Rademacher innovations;
- the exact RH-006 geometry used by the candidate statistic;
- fixed observed feature path in the oracle bootstrap;
- complete nested rolling refits;
- training-only standardization;
- fixed ridge;
- origin-level Bartlett studentization.

The benchmark remains an **independent estimator-mechanics reimplementation**, not execution of the Symthaea Rust implementation.

The canonical result therefore remains diagnostic only. It does not establish estimator compatibility, feature exogeneity, real-data null-DGP validity, or formal inferential validity.