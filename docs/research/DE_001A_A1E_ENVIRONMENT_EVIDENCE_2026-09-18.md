# DE-001A1E — Reusable Environment Evidence Verification

Date: 2026-09-18

Status: preregistered verifier implementation; not yet qualified

Scientific claim: NONE

Authority: environment-evidence-reuse-only

## Purpose

A1E verifies whether a previously qualified DE-001A numerical environment remains valid for reuse by a later stacked checkout without requiring unrelated repository HEAD identity.

The verifier performs no cosmology and does not authorize A1C by itself. A1C still requires both a qualified A1Q fixed-point bundle and a qualified A1E environment-reuse receipt.

## Inputs

The verifier accepts:

1. a `DE-001A-ENVIRONMENT-QUALIFICATION-v3` receipt;
2. the matching canonical package-version JSON retained by that qualification run.

It derives the current repository root from Git and reopens the current checkout's:

- `nix/flake.nix`;
- `nix/flake.lock`;
- `references/de001a_execution_closure_v2.json`.

Symlinks are forbidden for these definition files.

## Definition identity

A1E recomputes the same canonical environment-definition document used by the v3 qualification workflow:

- protocol `DE-001A-ENVIRONMENT-DEFINITION-v1`;
- platform `x86_64-linux`;
- exact runtime Nix identity `nix (Nix) 2.34.7`;
- SHA-256 of `flake.nix`;
- SHA-256 of `flake.lock`;
- SHA-256 of the closure manifest.

The recomputed definition SHA-256 must exactly equal the qualified receipt.

This permits safe reuse across unrelated Rust/documentation children while forcing requalification after any environment-defining byte changes.

## Receipt validation

A1E rejects:

- unknown receipt keys;
- duplicate receipt keys;
- missing receipt keys;
- any verdict other than PASS;
- any job status other than success;
- authority or scientific-claim drift;
- reuse-policy drift;
- malformed source Git identities;
- malformed SHA-256 identities;
- malformed Nix store/NAR identities;
- non-positive run, attempt, or closure-size fields.

## Package identity

The supplied package-version JSON must hash to the digest recorded by the qualification receipt and must contain exactly:

- Cobaya 3.6.2
- CAMB 1.6.6
- GetDist 1.7.4
- iminuit 2.32.0
- Py-BOBYQA 1.4.1

The environment must be Python 3.11 and retain `scientific_claim=NONE`.

## Cross-run realization check

A1E does not assume the original runner's `/nix/store` survives.

Instead it independently realizes the current checkout's `#environment` using:

`nix build <flake>#environment --no-update-lock-file --print-out-paths --no-link`

and then re-derives:

- store path;
- `nix hash path` NAR identity;
- `nix path-info -S` closure size.

All three must exactly match the qualified v3 receipt.

This converts environment reuse from "trust an old runner path" into a cross-run reproducibility check.

## Output semantics

A1E emits only:

- `PASS`: the qualified environment definition and realized closure reproduce exactly for the current checkout;
- `INVALID`: evidence is missing, malformed, inconsistent, or cannot be reproduced.

A1E never emits a scientific NEGATIVE result.

A1E PASS sets:

- `environment_reuse_authorized=true`;
- `a1c_execution_authorized=false`.

The second field is intentionally false: environment validity is only one prerequisite for A1C.

## Scientific boundary

A1E does not evaluate DESI data, a cosmological background, a likelihood, a sampler, or an optimizer. It cannot establish LambdaCDM, dynamic dark energy, or an observational anomaly.
