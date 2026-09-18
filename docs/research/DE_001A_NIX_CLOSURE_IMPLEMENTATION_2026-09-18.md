# DE-001A — Nix Closure Implementation

**Date:** 2026-09-18  
**Status:** implemented and frozen for qualification; not yet qualified  
**Authority:** no scientific execution authority

## What changed

The four external scientific source hashes that blocked `DE-001A-EXECUTION-CLOSURE-v1` can be resolved from published PyPI source distributions:

| Component | Source distribution SHA-256 |
|---|---|
| Cobaya 3.6.2 | `8f1061d6347427f08380e1e0c0b766d695d3978b5439fb0b1cc1a7002152d9c8` |
| CAMB 1.6.6 | `9856202a5c05570256e52377b20431891c7b08b2e9c334e141fd08d2a085516f` |
| GetDist 1.7.4 | `1bd69c9748891fa1dc516e2474b30b3afa083d2975531c09c4377ae30e8636ac` |
| Py-BOBYQA 1.4.1 | `f698848e372fa0625fb9fd3a7a8b4f557804d7858b23a45d8187ffaea6341e33` |

Cobaya and CAMB's PyPI Trusted Publishing attestations explicitly identify the same Git commits already frozen by DE-001A. GetDist and Py-BOBYQA retain both their frozen Git tag commits and the published PyPI source-distribution hashes as independent provenance anchors.

## Standalone subflake

The new subflake lives at:

`crates/domains/symthaea-cosmology-research/nix`

It pins the same nixpkgs revision as Symthaea's root lock and packages:

- Cobaya 3.6.2;
- CAMB 1.6.6;
- GetDist 1.7.4;
- Py-BOBYQA 1.4.1;
- nixpkgs' pinned iminuit 2.32.0;
- the transitive numerical Python dependencies from the same pinned nixpkgs revision.

CAMB is built from its source distribution. `FORUTILSPATH` is explicitly pointed at the `forutils` tree shipped in the source distribution, preventing CAMB's setup script from falling back to a network `git clone`.

## Qualification check

The flake check imports all five scientific packages and asserts exact distribution versions. It emits a machine-readable `versions.json` artifact inside its Nix output.

The development shell is deliberately non-authoritative. It sets offline/no-user-site flags, but a shell is not a network sandbox.

The authoritative environment qualification runs as a Nix build, where sandboxing provides the network boundary.

## Dedicated workflow

`.github/workflows/de001a-cosmology-nix.yml` is path-scoped and binds to the exact PR head.

It uses commit-pinned actions for:

- checkout;
- Nix installation;
- artifact upload.

It then:

1. verifies exact checkout HEAD/TREE;
2. runs `nix flake check --no-update-lock-file`;
3. builds the environment check explicitly;
4. realizes the Python environment;
5. records its store path, NAR hash, closure size, Nix version, and flake metadata digest;
6. verifies Git postflight immutability;
7. uploads the receipt and path-info artifacts.

The receipt says `scientific_claim=NONE` and `authority=environment-qualification-only`.

## Remaining blockers

Exactly two remain:

1. the Nix environment check has not yet produced an exact-head PASS;
2. the realized environment store/NAR identity has not yet been sealed into a promoted closure artifact.

A workflow PASS will provide evidence for both, but it does not mutate the frozen v2 manifest. Promotion will be a new closure version/receipt.

## What this still does not do

This tranche does not download DESI scientific inputs and does not evaluate the BAO likelihood. It qualifies only the numerical environment.

That ordering is intentional:

`reference freeze → source closure → environment qualification → artifact integrity → fixed-point likelihood → optimizer reproduction`
