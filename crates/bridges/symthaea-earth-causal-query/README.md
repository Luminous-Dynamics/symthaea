# symthaea-earth-causal-query

Counterfactual-identification adapter for evidence-backed Planetary Perception causal workspaces.

## Purpose

Symthaea currently has two useful causal graph representations:

1. `causal_calculus::CausalDAG` for the general causal-calculus machinery;
2. `counterfactual::CausalDAG` for identification, d-separation, and counterfactual query workflows.

`EarthCausalQueryView` creates a read-only identification view from the reviewed graph in `symthaea-earth-causal` while preserving the evidence-id/node mapping.

## Critical semantic boundary

The counterfactual reasoner can identify a symbolic estimand before any numerical effect has been estimated from data. Its current `CausalEstimand.effect` field uses `0.0` as a placeholder in those paths.

This adapter intentionally **does not expose that placeholder as an Earth effect estimate**.

Earth-facing outcomes are limited to:

- `Identified` — estimand description, identification method/confidence, adjustment evidence;
- `Unidentified` — reason, missing information, suggestions;
- `AssumptionRequired` — explicit assumption, plausibility, conditional estimand description.

Numerical effect estimation must be a separate evidence-bearing stage with data, estimator identity, diagnostics, uncertainty, and validation.

## Non-goals

This crate does not:

- infer structural edges;
- estimate causal-effect magnitudes;
- turn identification confidence into policy confidence;
- turn counterfactual output into authority;
- hide unidentified or assumption-dependent results.
