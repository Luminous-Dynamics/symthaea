# Joint Temporal Resource Control (v0.1)

This package contains the **candidate-side** deterministic baseline for jointly
allocating temporal observations, memory retention, I/O, serial critical-path
latency, compute, measured energy and verification work.

Latency is a conservative serial critical-path estimate and is charged against its own deadline ceiling; it is not a replacement for measured end-to-end p99 latency.

The controller is intentionally not an optimal solver. It:
- discounts each action's predicted temporal-error-debt reduction by its
  uncertainty radius;
- evaluates every resource dimension using an explicit dual price;
- enforces hard resource ceilings independently;
- deterministically orders candidate action bundles; and
- supports mutually exclusive choices through conflict groups.

A bundle may jointly describe a sample, memory retain/refresh operation,
compute refinement, and verification action. All eight resource units are explicit. Latency is a conservative serial estimate charged against a hard deadline budget; it is not a substitute for measured p99 end-to-end latency.
The current additive greedy baseline does not discover arbitrary complementarity;
the evaluator measures that with paired joint-action experiments and an exact
small-instance subset oracle.

## Candidate/evaluator separation

This crate contains no hidden future stream, event labels, ground-truth
exposure, evaluator randomization seed, witness secret, or exact subset oracle.
It is not linked into the separate evaluator package. The candidate receives
only its current observations, its own forecasts/state, and the public budget
contract.

## Scope and claim limit

The resource-price update is a simple bounded multiplicative baseline. It is
not a convergence theorem, optimal dual update, or production scheduler.
Explicit prices are policy inputs; hard vector ceilings remain binding even
when a price is zero. Missing resource dimensions are not representable as
unknown: adapters must abstain before constructing a CandidateAction if any
required cost cannot be measured or bounded.

No Rust build, test, Clippy, benchmark or hardware energy measurement has been
executed for this draft. See
`docs/research/JOINT_TEMPORAL_RESOURCE_CONTROL_V0_1.md` and issue #9666.
