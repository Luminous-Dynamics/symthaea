# VART-WORLD-CREATIVE-001 — World-State Equivalence v1

Status: confirmatory qualification contract. It strengthens paired and longitudinal causal identity; it does not authorize scientific claims by itself.

## Problem

A world-version label is not sufficient evidence that two policies started from the same world state. Paired policy comparisons require byte/digest identity of the actual pre-intervention state, not merely equality of a version string.

Likewise, longitudinal continuation requires the next revision to begin from the exact state produced by the previous revision.

## Trial state bindings

Every confirmatory trial binds:

- `world_state_before_sha256`
- `world_state_after_sha256` for complete trials

The corresponding raw evidence artifacts are named by `evidence_index.json` as:

- `world_state_before`
- `world_state_after`

The digest is over the exact exported state-snapshot artifact bytes.

A state snapshot contains at minimum:

- `schema = "symthaea.vart-world-creative-001.world-state-snapshot.v1"`
- `experiment_id`
- `world_version`
- `provenance_domain`
- `state_digest`

`state_digest` is the domain-typed world/Dream state digest supplied by the qualified World Forge/Reality Ledger lineage. The wrapper artifact hash and the internal typed state digest serve different purposes and both are retained.

## Paired-policy equivalence

For every paired block in which FULL, RANDOM_VALID, and HEURISTIC are compared:

- `world_state_before_sha256` must be identical;
- `world_version_before` must be identical;
- `decision_input_sha256` must be identical where the policy contract requires equal decision surfaces;
- `candidate_set_sha256` must be identical where the comparison requires a shared candidate surface.

This establishes that policy is the intended varying factor.

Each policy must execute from an isolated clone/fork of that frozen starting snapshot. A policy may never mutate the baseline snapshot used to instantiate another policy cell.

## Application closure

For complete trials, the applied receipt repeats:

- `world_state_before_sha256`
- `world_state_after_sha256`
- `world_version_before`
- `world_version_after`
- `execution_context_sha256`
- `selected_proposal_sha256`

The revisit observation repeats `world_state_after_sha256` and `world_version_after`.

A matching version label with a different state digest is rejected.

## Longitudinal closure

For adjacent complete revisions of the same persistent policy/world lineage:

`previous.world_state_after_sha256 == current.world_state_before_sha256`

and

`previous.world_version_after == current.world_version_before`.

If revision indices are not adjacent, the verifier reports the gap rather than inventing continuity.

## Provenance

Pre- and post-state snapshots used for confirmatory world-improvement claims must be committed/grounded state, not counterfactual or Dream-only substitutes unless the campaign explicitly declares a Dream-world claim family.

Counterfactual candidate states remain separate evidence and cannot satisfy committed pre/post snapshot requirements.

## Required rejection classes

- `PAIRED_WORLD_STATE_MISMATCH`
- `WORLD_STATE_DIGEST_MISMATCH`
- `WORLD_STATE_VERSION_MISMATCH`
- `WORLD_STATE_PROVENANCE_SUBSTITUTION`
- `WORLD_STATE_CHAIN_MISMATCH`
- `WORLD_STATE_BASELINE_MUTATION`

## Scientific interpretation

Passing this layer establishes causal state identity for comparisons. It does not establish that a revision was good, intelligent, creative, or aesthetically preferable; those remain outcome/analysis questions.
