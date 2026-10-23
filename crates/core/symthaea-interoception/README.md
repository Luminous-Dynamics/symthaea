# Native Interoception v0.1

This crate provides deterministic artificial viability-state primitives for
experiments on self-regulation. It is intentionally independent of the root
cognitive loop, the existing virtual-body adapter, autobiographical memory,
neuromodulation, and semantic state categories.

## Native channels

The v0.1 substrate exposes eight compute-native channels:

- compute reserve
- memory headroom
- model stability
- epistemic resolution
- integrity
- action efficacy
- novelty balance
- interaction reliability

`interaction_reliability` is deliberately agent-neutral. It represents the
reliability of action/environment coupling and does not encode an innate social
or relational need.

Each channel carries a current value, preferred and viable ranges, precision,
importance, and measured velocity.

## Regulation

Homeostasis measures current deviation from preferred ranges. Allostasis is a
separate API and can be evaluated in two ways:

1. a kinematic baseline that extrapolates measured velocity;
2. a dynamics-aware rollout under an explicitly declared constant future drive.

Keeping these mechanisms separate makes prospective regulation independently
ablatable in later experiments.

## Causal experimentation

External drives and direct interventions are distinct. A drive participates in
the native transition law. A direct intervention produces an explicit receipt
and resets measured velocity, preventing an exogenous state jump from being
silently treated as an endogenous trend.

Each native transition also returns a mechanical receipt counting driven,
restorative, clamped, and changed channels.

## Snapshot provenance

Snapshots include a schema version, the dynamics configuration, the full state,
current homeostatic report, and the exact forecast basis/configuration used for
the prospective report.

## Parameter status

The numerical defaults in v0.1 are research hypotheses in normalized artificial
state space, not measurements of biological physiology. See `CALIBRATION.md`.

## Claim boundary

This tranche establishes only deterministic artificial self-regulation
primitives. Higher-level interpretation is deliberately out of scope.
