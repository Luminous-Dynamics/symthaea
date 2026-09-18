# DE-001A — Execution Closure Contract

**Date:** 2026-09-18  
**Status:** closure preregistered, incomplete, non-executable  
**Parent target:** `DE-001A-DESI-DR2-BAO-FLAT-LCDM-v1`  
**Authority:** none until the realized closure qualifies; eventual authority remains reproduction-only

## Purpose

The known-answer target is frozen, but a scientific reproduction is not reproducible merely because its input data and comparison numbers are pinned. The numerical environment must also be an immutable subject.

This tranche therefore separates **source selection** from **closure qualification**.

The source versions are now selected and cannot drift silently. Execution remains forbidden until all selected external sources have fixed-output identities and the Nix closure is actually realized and recorded.

## Independent reproduction lane

The first executable lane is intentionally not a bit-for-bit rerun of DESI's internal likelihood package.

DESI's published `bestfit.minimum.txt` identifies the likelihood path as:

`desi_y3_cosmo_bindings.cobaya_likelihoods.bao_likelihoods_v1p2.desi_bao_all`

DE-001A instead uses the public Cobaya `bao.desi_dr2` all-tracer likelihood over the public DR2 BAO mean/covariance.

That makes this an **independent public-likelihood reproduction**. Agreement is therefore cross-implementation evidence at the software/plumbing level; disagreement is a localization target, not evidence for or against ΛCDM.

## Root Nix identity

Symthaea's root `flake.lock` currently binds the root nixpkgs input to:

- flake node: `nixpkgs_2`
- revision: `9ae611a455b90cf061d8f332b977e387bda8e1ca`
- NAR hash: `sha256-md8WlXOlfnIeHeOScMTTHFyf2d6iaTwPl2apR5EQ3P4=`
- Python lane: explicit `python311`
- first execution platform: `x86_64-linux`

The first platform is intentionally singular. Cross-architecture reproducibility belongs in a later lane after one closure qualifies.

## Selected scientific sources

The following versions are now frozen as *selection decisions*:

| Component | Version | Git commit | Current closure status |
|---|---:|---|---|
| Cobaya | 3.6.2 | `899f30a49f85de610dac321e91a1af50018e56aa` | fixed-output hash missing |
| CAMB | 1.6.6 | `3ef0272d6f7ba1231128872e56e6d4c12af8267b` | fixed-output hash missing |
| GetDist | 1.7.4 | `f8d5fb7f39927c199dbfa1bf87eb4fac6fe7c206` | fixed-output hash missing |
| Py-BOBYQA | 1.4.1 | `3a3bd50732a5695a0a434cba4c1fde01c0204e08` | fixed-output hash missing |

Why these versions:

- Cobaya 3.6.2 is the already-frozen public likelihood implementation.
- CAMB 1.6.6 is explicitly supported by Cobaya 3.6.2 and is frozen by its upstream tag.
- GetDist 1.7.4 predates Cobaya 3.6.2 and satisfies Cobaya's `GetDist>=1.3.1` requirement without selecting a later post-Cobaya release.
- Py-BOBYQA 1.4.1 satisfies Cobaya's `py-bobyqa>=1.4` requirement and has a modern `pyproject.toml` packaging path.

At the pinned nixpkgs revision, `iminuit` is already packaged as **2.32.0** with its source hash recorded by nixpkgs, so it does not need a new external fixed-output source in this tranche.

## Network policy

Two phases have different authority:

1. **Nix realization** may retrieve only declared fixed-output sources.
2. **Scientific execution** is network-disabled and may not install packages.

A command such as `pip install`, `uv sync`, or an implicit Cobaya component download during the scientific run is a qualification failure, not a convenience fallback.

## Machine-visible blockers

The closure manifest currently contains six blockers:

1. Cobaya fixed-output source hash;
2. CAMB fixed-output source hash;
3. GetDist fixed-output source hash;
4. Py-BOBYQA fixed-output source hash;
5. Nix derivation not yet implemented;
6. realized closure identity not yet recorded.

The manifest is required to keep:

`execution_authority = false`

until the blocker set is empty.

## Promotion rule

The closure may become executable only when all of the following are true:

- every external source has a fixed-output hash;
- the Nix derivation is implemented;
- the closure realizes successfully;
- exact package-version assertions pass inside that closure;
- scientific execution can run without network access or runtime installation;
- the realized closure identity is recorded;
- the machine-readable blocker set is empty.

The promoted state will be a new frozen artifact/version. This incomplete manifest is not edited after observing a scientific result.

## Next work unit

`DE-001A3a` should package the four external sources as fixed-output Nix derivations and expose a dedicated `.#cosmology-verify` shell plus a non-interactive environment check.

`DE-001A3b` should realize that closure, record its store/NAR identity, run exact version/import assertions, and freeze the resulting receipt.

Only then may `DE-001A0` consume scientific bytes.
