# PARADOX A0-R M0 — Label-Blind Evidence Firewall

Authority: **DevelopmentOnly / Representational MeasurementOnly / ProtocolHardening**.

Issue: #3612.

This directory is a static protocol subject. It does **not** execute production cognition, collect features, join semantic labels, fit a decoder, score held-out data, authorize behavioral A0-D, or authorize A1.

## Qualified prerequisite

M0 descends from corrected A0 static subject:

`8cc2651576f0068a1ccac3ef21c8a0a0eb3c2afb`

That subject was qualified by #3652 / run `35151014357` / job `104979041434`.

Its ancestry remains:

- G2b: `09d83a1d1fddbbbd30e4eba7cc95946c8eab871f`
- frozen production subject: `eb73527d05a913e79d1f05135ad6b06c1da8e2ee`

## Purpose

M0 makes researcher degrees of freedom into explicit evidence objects before the first A0-R probe fit.

The protected ordering is:

```text
qualified A0 static contract
-> frozen M0 information-flow and commitment contract
-> label-blind feature acquisition
-> sealed feature dataset
-> independently frozen semantic/split manifests
-> development-only probe fitting
-> frozen probe/preprocessing
-> sealed held-out predictions
-> held-out label join
-> development scoring
```

Every arrow is a scientific transition. None may be collapsed merely because one implementation can perform several steps.

## Three-plane firewall

### F — measurement/features

F may see only opaque measurement plans, agent-visible fixture material, frozen runner configuration, and the production execution interface.

F must not receive condition names, semantic fixture-family IDs, target labels, expected responses, oracle outputs, capability-atom labels, scores, split membership, probe parameters, or probe predictions.

A feature receipt binds only opaque IDs, technical provenance, exact execution commitments, channel digests, validity, and repeatability metadata. Raw development feature bytes may be archived separately, but semantic labels are not part of the measurement process.

### S — semantics/split

S may use the prospectively frozen fixture, semantic-label, transform, and grouping manifests needed to construct grouped/stratified partitions.

S must not read recurrent vectors, perceptual-control values, channel digests, probe parameters, predictions, or scores. The split is frozen before feature geometry is available and may not be regenerated because a later model result is inconvenient.

### P — probe/evaluation

P remains inactive until the feature dataset is sealed and the semantic/split manifests are frozen.

Held-out labels remain unavailable until preprocessing, the development probe, and held-out predictions have each been frozen and digest-bound.

## Canonical commitments

Every commitment is derivation-verifiable. A declared digest is not evidence by itself when its source object is available.

Structured commitment objects use canonical UTF-8 JSON with sorted keys, compact separators, NFC strings, no JSON floating-point values, and explicit length-prefix framing.

Binary feature values remain binary scientific objects. Their exact bytes are committed by SHA-256; they are not round-tripped through JSON numbers.

Row and dataset commitments use domain-separated framed preimages defined in `m0_static_contract.json`. The dataset uses an explicit ordered manifest. Duplicate opaque measurement IDs fail closed.

A commitment derivation registry must bind the field, domain tag, source objects, canonicalization, ordering, normalization, hash, and reference implementation. Independent recomputation is required where practical.

## Technical repeatability and attrition

Each planned measurement has exactly two fresh-service technical replicates. They are reproducibility controls and never increase statistical `n`.

All attempts and invalidity reasons are retained. Label-aware retries are forbidden. A retry policy, if any, must exist before labels are available.

Silent complete-case selection is forbidden. Material class- or transform-differential attrition makes the affected atom `NotEstablished`; falling below the preregistered independent-group minimum also yields `NotEstablished`.

## Environment capsule

Each later executable lineage must bind its runner image, OS/kernel, CPU architecture and stable model identifier where available, exact rustc version/commit, Cargo.lock SHA-256 or equivalent build-input commitment, relevant deterministic/floating-point/threading environment variables, and exact executable SHA-256.

Different environment capsules are never silently pooled into one deterministic-instance lineage.

## Static mutation controls

`m0_static_audit.py` executes fourteen deterministic controls covering stale commitments, digest/source disagreement, row reordering/insertion/deletion, cross-binding swaps, split-seed drift, canonicalization alternatives, post-seal substitutions, forbidden F/S information flow, duplicate opaque IDs, and technical-replicate mismatch.

These are static/synthetic controls. Their PASS does not claim that the future measurement harness has executed.

## Failure taxonomy

Keep runner reachability, source bytes, commitment derivation, information flow, and evidence-object sealing separate:

```text
COMMITMENT_DERIVATION_MISMATCH   subject/data-plane defect
BOUND_OBJECT_UNREACHABLE         runner-plane defect
BOUND_SOURCE_BYTES_MISMATCH      subject/source-plane defect
FIREWALL_VIOLATION               information-flow defect
DATASET_SEAL_MISMATCH            evidence-object defect
```

A qualifier that audits Git ancestry must prove all bound commits are locally reachable before classifying a source-diff failure.

## Promotion ladder

```text
STATIC_CONTRACT_QUALIFIED
-> FEATURE_DATASET_SEALED
-> SPLIT_AND_LABEL_MANIFESTS_FROZEN
-> DEVELOPMENT_PROBE_FROZEN
-> HELDOUT_PREDICTIONS_SEALED
-> DEVELOPMENT_RESULT_SCORED
```

The first step is the only step this static subject can seek.

## Claim boundary

A static M0 PASS means only that the exact frozen protocol subject mechanically encodes the declared information-flow, commitment, attrition, and mutation-control rules.

A later successful M0 acquisition can establish reproducible collection of the specified frozen native representation under the sealed protocol.

Neither result establishes that a capability atom is decodable, that production cognition uses it, that it is causally necessary, that Symthaea has the corresponding behavior, or anything about metacognition, ontology repair, consciousness, sentience, or phenomenology.
