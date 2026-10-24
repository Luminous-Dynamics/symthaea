# Passive Design Search v1

## Search primitives

`symthaea-passive-design-search` provides two deliberately small primitives:

- `ParetoArchive<T>` preserves the currently non-dominated candidate set instead of
  collapsing competing engineering objectives into one irreversible score.
- `CsgFingerprint` gives a deterministic structural identity for CSG trees, including
  primitive/transform/boolean counts, maximum depth, and a digest that normalizes
  commutative union/intersection operand order while preserving subtraction order.

Equal objective vectors are retained because objective equality does not imply design
equality. Two geometrically distinct candidates can have the same observed performance
while differing in tolerance, failure mode, manufacturability, or future robustness.

## Intended objective vector

The passive engineering layer can emit observations such as:

`[passivity, physical_performance, manufacturability, material_efficiency]`

All four are currently treated as maximize objectives when using the built-in passive
fitness observation. Other engineering problems can supply mixed maximize/minimize
directions.

## Why this matters

Recent inverse-design literature is moving toward physics-guided generation and explicit
diversity/realizability management. Physics-guided diffusion models published in 2026
directly incorporate differentiable solver guidance during generation, while other
recent work demonstrates generative optimization over microstructures under competing
physical objectives. Symthaea can adopt the architectural lesson without adopting a
transformer-based internal representation:

`HDC representation -> structural mutation/search -> solver evaluation -> Pareto archive -> provenance`.

## Scientific limitation

A Pareto frontier is only as good as the objectives and solver envelopes behind it. It is
not evidence of real-world superiority outside the simulated operating envelope.
The archive therefore sits after physics evaluation, not instead of it.