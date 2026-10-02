# Regenerative Physics Evidence Contract v0.1

## Purpose

This contract adds a deterministic model-consistency boundary after sensor qualification and temporal corroboration.

The ordering is:

1. qualify individual sensor observations;
2. corroborate a sensor cohort spatially/temporally;
3. compare the corroborated trajectory with an independently identified physical/model prediction;
4. preserve disagreement as evidence state rather than converting it directly into a health or safety verdict.

Current digital-twin SHM work emphasizes sensor validation, model discrepancy handling, uncertainty treatment, and temporal alignment. Physics-model mismatch is not itself proof of physical damage because the model can be stale, simplified, miscalibrated, or operating outside its validated regime.

## Contract

### PhysicsPrediction

A model prediction contains:

- component identity;
- model identity and version;
- prediction timestamp;
- predicted trajectory delta;
- model uncertainty;
- evidence identity;
- configuration digest.

### PhysicsConsistencyPolicy

The deterministic policy bounds:

- prediction age;
- allowed absolute prediction residual;
- allowed model uncertainty.

### PhysicsConsistencyGate

The gate consumes:

- component identity;
- expected configuration digest;
- a TemporalFusionDecision;
- a PhysicsPrediction;
- current evaluation time.

It produces one of:

- Consistent — corroborated sensor trajectory and model trajectory agree within policy;
- Inconsistent — corroborated sensor evidence exists, but model residual or uncertainty exceeds policy;
- InsufficientEvidence — temporal evidence is not corroborated or no observed delta exists;
- Quarantined — identity, configuration, temporal, or freshness invariants fail.

## Safety semantics

Consistent does not mean healthy.

Inconsistent does not mean damaged.

Quarantined does not identify the root cause.

The contract intentionally keeps these conclusions separate. A model can disagree because the machine changed, because the model is wrong, or because the operating regime is outside the model's validated envelope.

Recovery remains a separate qualification boundary requiring fresh post-intervention evidence and independent verification.

## Adversarial coverage

The adversarial sensing crucible now contains a concrete coordinated-false-trajectory case:

- latent state changes from 0.5 to 0.8;
- all three sensors coherently report approximately 1.5;
- temporal corroboration therefore remains Corroborated;
- the fixture records the observed consensus separately from latent state.

This demonstrates the limitation of sensor-only agreement without pretending that the temporal layer can solve coordinated false sensing.

The next layer is therefore model/physics evidence, not a larger sensor quorum.

## Research alignment

Recent 2026 digital-twin SHM literature emphasizes uncertainty quantification, model updating, sensor validation, temporal synchronization, and explicit treatment of model discrepancy as prerequisites for dependable data/physics fusion. This contract intentionally starts with deterministic evidence qualification before introducing learned or adaptive model components.

## Independence semantics

A trusted-sensor quorum is not treated as independent merely because sensor IDs differ. Temporal corroboration now carries an explicit `independence_group`, representing a physical or common-mode dependency domain. The policy can require a minimum number of independent groups. This prevents multiple colocated, shared-path, or otherwise common-mode sensors from manufacturing a false quorum.

This is consistent with current SHM work emphasizing redundancy and decentralized sensor placement while also documenting the challenge of concurrent multi-sensor failures and loss of spatial correlation. citeturn0search1turn0search2
