# LL-009D — content-addressed corridor study contracts

Status: **Phase-0 research provenance. Not site selection, corridor safety, launch authority, or economic validation.**

LL-009B/LL-009C already require every evidence receipt in a corridor bundle to share the same `study_id`, `frame_contract_id`, and `epoch_contract_id`. LL-009D makes those identifiers content-addressed rather than caller-chosen labels.

## Why

An opaque ID can accidentally remain unchanged while its meaning changes. For example, a study could silently move from one candidate node to another, add an elevator architecture, alter the lunar body-fixed frame, change the constants profile, or widen the target-track window while preserving the same text ID.

LL-009D prevents that class of lineage drift.

## Contracts

The input contains three independently hashed semantic objects.

### Study

Defines the research hypothesis being compared:

- source/destination node references;
- hypothesis status;
- cargo class and mass envelope;
- demand-scenario reference;
- candidate transport architectures;
- required evidence roles;
- declared approximations/model ceilings;
- non-claims.

Changing any of these fields changes `study_id`.

### Frame

Defines:

- native terrain frame/projection;
- study lunar body-fixed frame;
- inertial frame;
- frame-bridge policy;
- gravity/constants profile;
- length/velocity units;
- forbidden silent aliases.

If the terrain and study body-fixed frames differ, `bridge_policy = none` is rejected.

Changing any frame semantics changes `frame_contract_id`.

### Epoch

Defines:

- timescale;
- single epoch, bounded window, or explicit epoch set;
- maximum target-track interpolation gap;
- artifact time semantics.

Changing time semantics changes `epoch_contract_id`.

## Content addressing

Each semantic object is canonicalized independently and hashed with SHA-256. The generated ID is:

```text
study-<sha256>
frame-<sha256>
epoch-<sha256>
```

The generated contract file contains both the semantic hash and exact contract content. A separate manifest records all three IDs plus the SHA-256 of the source input.

This means a study architecture can change without invalidating an unchanged frame or epoch contract, while the changed study ID still propagates to every downstream evidence receipt.

## Usage

Dependency-free self-test:

```bash
python3 scripts/build_ll009d_study_contracts.py --self-test
```

Materialize a contract lineage:

```bash
python3 scripts/build_ll009d_study_contracts.py \
  --input configs/lunar_transport/<corridor>-contract-input.json \
  --output-dir docs/research/evidence/<corridor>/contracts
```

Like LL-009C, an existing differing output directory is immutable. Change the contract/lineage path rather than overwriting prior evidence.

## Capsule integration

The intended next integration is for LL-009C to include the exact generated study/frame/epoch contract bytes and verify that their computed IDs equal the IDs declared by the LL-009B bundle before materializing a promoted corridor capsule.

## Non-claims

Content-addressed contracts establish what was studied. They do not establish that the assumptions, site hypothesis, physics, economics, or safety case are correct.
