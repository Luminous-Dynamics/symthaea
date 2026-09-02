# VART-002 Episodic Provenance Shadow Mode v1

Status: **development instrumentation only**  
Experiment: `VART-002-EPISTEMIC-MECHANISMS`  
Confirmatory execution authorized: **false**  
Claim authorized: **false**

## Purpose

Make epistemic provenance load-bearing at the episodic persistence boundary without prematurely changing cognitive behavior or laundering internal model state into grounded history.

The current production `Episode` contains both compressed input state and Symthaea's cognitive output. A physical observation of the input does **not** ground the entire episode. Therefore the runtime must not label a complete episode `PhysicalGrounded` merely because its input ultimately originated in perception.

## Compatibility rule

The historical `Episode` serialization remains unchanged.

Provenance lives in an `EpisodicProvenanceIndex` keyed by `episode_subject_sha256`. Legacy or otherwise unannotated episodes resolve to an explicit `RealityDomain::Unknown` envelope with zero confidence. They never silently enter grounded history.

The `EpisodicPersistenceManager` owns this sidecar. Wholesale replay-store replacement resets the sidecar unless a future paired restore operation explicitly restores both episode storage and its matching provenance index.

## Shadow-mode rule

Until perception-to-episode derivation provenance exists, provenance filtering is **observational only** in the production cognitive loop.

Shadow mode MAY:

- compute the effective provenance of recalled episodes;
- evaluate a `ProvenanceRetrievalMode` against those episodes;
- record counts excluded by unknown domain, counterfactual taint, or incompatible domain;
- compare raw retrieval and epistemically admissible retrieval for DEVART mechanism studies;
- emit receipts or telemetry that contain no hidden VART material.

Shadow mode MUST NOT:

- alter production prediction confidence;
- alter candidate generation;
- alter action authority;
- alter memory capacity or retrieval budget;
- promote `Unknown`, `Imported`, `Replay`, `Dream`, or `Counterfactual` objects into grounded history;
- infer grounding from confidence, coherence, Psi, retrieval frequency, semantic similarity, persistence, or replay count.

## Required transition before enforcement

Enforcement may replace raw recall with provenance-filtered recall only after all of the following are implemented and tested:

1. **Perception subject identity** — the exact perceived object has a stable digest.
2. **Perception grounding evidence** — physical/digital grounding is bound to that exact digest.
3. **Episode derivation receipt** — the episode digest is derived from explicitly identified parents, including the perceived input and internal cognitive response.
4. **No domain collapse** — the episode's domain represents a mixed/internal derived cognitive record rather than falsely inheriting physical grounding from one parent.
5. **Persistence pairing** — episode storage and provenance sidecar can be restored together with integrity checks.
6. **Replay invariance** — replay/reconsolidation counters may change without changing the immutable episode subject digest.
7. **DEVART qualification** — enforcement behavior is tested on development worlds not used as hidden VART-002 confirmatory worlds.
8. **No confirmatory tuning** — hidden VART-002 worlds, seeds, thresholds, or outcomes remain unavailable to development.

## Intended runtime order

```text
perception
  -> perception provenance / grounding evidence
  -> internal cognitive transformation
  -> episode derivation receipt
  -> episodic storage + provenance sidecar
  -> similarity retrieval
  -> provenance view
  -> epistemic readiness
  -> proposal
  -> normal authority
  -> action
  -> receipt
```

Provenance controls what information may support reasoning. It does not itself grant permission to act.

## Scientific boundary

A future reduction in provenance confusion, harmful premature revision, or false grounding is an empirical result to be tested on fresh hidden worlds. This document authorizes no efficacy claim.