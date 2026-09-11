# LL-009E — contract-bound immutable corridor capsule

Status: **Phase-0 research provenance only. Not site selection, corridor qualification, launch authority, or economic validation.**

LL-009E closes the provenance join between the LL-009D content-addressed study/frame/epoch contracts and the LL-009C immutable corridor capsule.

## Problem

LL-009B and LL-009C already require every evidence receipt in a bundle to carry the same `study_id`, `frame_contract_id`, and `epoch_contract_id`. LL-009D makes those IDs content-addressed.

However, a promoted capsule should carry the exact contract bytes that define those IDs, rather than only repeating the strings.

## Architecture

LL-009E wraps LL-009C instead of modifying it.

```text
LL-009D contracts
  study_contract.json
  frame_contract.json
  epoch_contract.json
  contract_manifest.json
             |
             | exact semantic-hash / ID verification
             v
LL-009B bundle -------------------------------+
             |                                |
             | bundle IDs must match          |
             v                                |
          LL-009C                             |
      immutable capsule                       |
             |                                |
             +--------------+-----------------+
                            v
                    LL-009E envelope
```

The complete LL-009C capsule is preserved byte-for-byte under `ll009c/`. The exact canonical LL-009D outputs are copied under `contracts/`. LL-009E then emits a binding receipt, an outer canonical index, and an aggregate receipt over the combined envelope.

## Why an outer envelope

Changing `capsule_index.json` after LL-009C creates it would invalidate LL-009C's own receipt. LL-009E therefore does not rewrite the inner capsule.

The hierarchy preserves both proofs:

```text
LL-009C receipt proves immutable evidence composition
LL-009E receipt proves exact study-definition binding around it
```

## Contract verification

Before invoking LL-009C, LL-009E independently verifies each generated contract file:

- exact supported schema version;
- `contract` is a JSON object;
- SHA-256 is recomputed over the canonical semantic object;
- `semantic_sha256` equals the recomputed digest;
- `contract_id` equals `<kind>-<semantic_sha256>`;
- the whole record is in the canonical LL-009D JSON representation.

It then verifies `contract_manifest.json` against all three records, including the semantic hashes and content-derived IDs.

## Bundle binding

The bundle must carry the same computed:

- `study_id`;
- `frame_contract_id`;
- `epoch_contract_id`.

A stale or caller-invented alias therefore cannot be promoted simply because all downstream receipts copied the same string.

## Immutability

The outer envelope includes:

- the untouched LL-009C capsule;
- the four exact LL-009D contract files;
- `contract_binding_receipt.json`;
- `ll009e_index.json`;
- `ll009e_receipt.json`.

The outer index binds:

- inner-capsule aggregate digest;
- contract manifest digest;
- exact copied-contract file hashes;
- LL-009B validator hash;
- LL-009C builder hash;
- LL-009E builder hash;
- content-addressed study/frame/epoch IDs.

An existing differing output directory fails closed. There is no force-overwrite path.

## Self-test

```bash
python3 scripts/build_ll009e_contract_bound_capsule.py --self-test
```

The executed self-test covers:

- deterministic byte-identical rebuild;
- exact contract inclusion;
- bundle/contract ID mismatch rejection;
- semantic contract mutation under a stale ID rejection;
- tampered existing-envelope overwrite rejection.

## Intended real-site promotion chain

```text
NAIF + LOLA / PGDA authoritative sources
        ↓
real evidence receipts
        ↓
LL-009D content-addressed study/frame/epoch contracts
        ↓
LL-009B eight-role closure
        ↓
LL-009C immutable corridor capsule
        ↓
LL-009E contract-bound envelope
        ↓
LL-009 transport Pareto analysis
```

This means an economic result can always be traced not only to exact evidence bytes but also to the exact semantic definition of the corridor, frames, epoch/window, cargo envelope, candidate architectures, and declared approximations that produced it.

## Non-claims

A passing LL-009E envelope establishes research provenance and reproducibility semantics only. It does not establish that a site is suitable, that the corridor is safe, that the physics models are flight-qualified, that the economics are correct, or that any physical release is authorized.
