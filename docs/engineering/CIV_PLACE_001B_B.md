# CIV-PLACE-001B-B — typed Rust leaf kernel + Python differential gate

**Status:** stacked qualification tranche. Read-only. No production authority.

## Purpose

Port the frozen CIV-PLACE-001B reference theorem into a small Rust leaf kernel while preserving the existing Python oracle as the independent semantic reference.

```
frozen JSON fixture
       |
       +--> Python reference oracle
       |
       +--> Rust typed leaf kernel
                    |
                    v
          canonical evaluation surface
                    |
                    v
            Python/Rust differential
```

The kernel has no Holochain, network, device, control, governance, municipal, or mutable-global-state dependency.

## Implemented contract

The crate `symthaea-civ-place` contains typed dependency edges, node/projection/currentness inputs, scoped positive independence witnesses, deterministic dependency closure, common-mode closure, read-only service evaluation, canonical JSON normalization, SHA-256 identity, and hostile-case replay.

The Rust binary `civ-place-001b-oracle` emits the deterministic evaluation surface used by the differential gate.

## Independence law

```
no discovered shared dependency
    !=
independent
```

Only a complete, non-conflicted, scoped positive witness may yield `IndependentWitnessed`.

## Differential gate

`scripts/diff_civ_place_001b.py` imports the existing stdlib-only Python reference functions, computes Python service evaluations and hostile-case outputs, executes the Rust typed oracle, and compares the semantic result surfaces exactly.

## Claim ceiling

A differential PASS establishes equivalence between the Rust leaf kernel and the exact frozen synthetic reference model. It does not establish physical infrastructure resilience, safety, structural adequacy, utility reliability, public-service sufficiency, regulatory approval, municipal authority, or physical-control authority.
