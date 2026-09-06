# Symthaea Epistemic Governance

This crate implements policy for Recursive Cognitive Architecture (RCA) evidence handling.

It is deliberately **not** another evidence store and **not** an action-authority layer:

- `symthaea-types` owns canonical cognitive proposal/evidence wire types;
- `symthaea-evidence-plane` owns research-harness mechanical integrity, task validation, and seed discipline;
- `symthaea-epistemic-governance` owns lineage, independence, currentness, defeaters, and experiment qualification policy.

The crate must not grant `ActionAuthority` or `SelfImprovementPromotion`.

## RCA-001.1 invariant

Multiple evidence objects do not imply multiple independent observations. Independence is computed only from a closed, acyclic ancestry graph. Missing ancestry fails closed instead of being interpreted as independent corroboration.
