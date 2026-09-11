# LL-009C — immutable corridor evidence capsule

Status: **Phase-0 research evidence composition. Not launch, navigation, site, catcher, or safety qualification.**

LL-009C turns a validated LL-009B corridor bundle into an immutable, byte-bound capsule suitable for downstream LL-009 research comparison.

The capsule builder does not decide whether the underlying physics are correct. That remains the responsibility of the producing subsystems. Its job is narrower: prevent a trade study from silently mixing artifacts from different sites, frames, epochs, studies, or synthetic/real-data lineages.

## Required upstream closure

For `trade_study_ready_for_real_site_comparison`, LL-009B requires all of:

- terrain pack receipt;
- frame bridge receipt;
- release state receipt;
- target track receipt;
- dispersion receipt;
- bounded terminal correction receipt;
- catcher receipt;
- deterministic release-policy receipt.

All must share the same `study_id`, `frame_contract_id`, and `epoch_contract_id`. Synthetic-only evidence is rejected for real-site promotion.

## Capsule contents

A successful capsule contains:

```text
source/
  bundle_manifest.json
  bundle_schema.json
validation/
  ll009b_receipt.json
artifacts/
  <role>/<exact receipt bytes>
capsule_index.json
capsule_receipt.json
```

The index records the exact SHA-256 of every copied evidence receipt plus hashes of the LL-009B validator and LL-009C builder used for composition.

The capsule receipt records the canonical index hash and an aggregate digest over all capsule files except the receipt itself.

## Immutability rule

A capsule output path is write-once for a lineage.

Rebuilding the same inputs into the same path is allowed only if the resulting tree is byte-identical. If any file differs, the builder fails closed and requires a new capsule/lineage path.

There is intentionally no `--force` overwrite option.

## Usage

Dependency-free self-test:

```bash
python3 scripts/build_ll009c_corridor_capsule.py --self-test
```

Build a capsule after LL-009B closure:

```bash
python3 scripts/build_ll009c_corridor_capsule.py \
  --bundle docs/research/evidence/<study>/corridor_bundle.json \
  --artifact-root docs/research/evidence/<study>/artifacts \
  --output-dir docs/research/evidence/<study>/capsule-v1
```

The builder delegates bundle semantics to:

```text
scripts/validate_ll009b_corridor_bundle.py
```

and hashes that validator into the capsule index.

## Path safety

Artifact paths in the bundle are POSIX-relative paths beneath the declared artifact root. Absolute paths, `..`, ambiguous components, and backslash-based traversal are rejected before copy.

Every source artifact hash is verified before copy and again after copy.

## First real South-Pole capsule

The intended first non-synthetic lineage should use one declared South-Pole study region, one explicit epoch contract, and one DE440 study frame contract.

The execution order is:

1. execute the pinned DE421 and DE440 SPICE snapshots from LL-004F;
2. measure and record DE421↔DE440 frame reconciliation;
3. materialize/hash the selected LOLA elevation/slope/uncertainty products;
4. create the terrain-pack and frame-bridge receipts;
5. derive one site→inertial release-state receipt;
6. bind one moving target track at the same epoch/frame contract;
7. run seeded dispersion;
8. run bounded terminal correction;
9. close catcher momentum/energy assumptions;
10. run the deterministic study-release policy;
11. validate LL-009B;
12. materialize LL-009C.

Only after that should LL-009 economics compare rover/FLOAT/rail/cable/ballistic/orbital-launcher/elevator-feeder architectures for that corridor.

## Non-claims

A successful LL-009C capsule does **not** mean:

- the candidate site is selected;
- the corridor is safe;
- the trajectory is navigable;
- the catcher is qualified;
- the launcher may physically release a pod;
- any transport architecture is economically preferred;
- human or operational qualification exists.

It means only that the evidence supplied to the Phase-0 comparison is complete, internally referentially closed, and byte-reproducible for the declared research lineage.
