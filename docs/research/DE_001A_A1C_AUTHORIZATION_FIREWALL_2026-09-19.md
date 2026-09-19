# DE-001A1C execution authorization firewall

Date: 2026-09-19
Status: preregistered authorization contract; not executed or qualified
Scientific claim: NONE

## Purpose

A1C is the released-Cobaya fixed-point reproduction lane. Its Python executor must not be allowed to interpret raw CI artifacts or decide for itself whether prerequisite evidence is sufficient.

This tranche adds `de001a-a1c-authorize`, a Rust firewall that performs no cosmology and executes no Python. It combines three already-scoped receipts:

1. A1Q evidence-bundle integrity PASS for the current exact HEAD/TREE;
2. A1E reusable-environment evidence PASS for the current environment definition;
3. A1C contract-consistency PASS for the current canonical A1C manifest.

Only their conjunction may emit `a1c_execution_authorized=true`.

## Orthogonality of integrity and reproduction outcome

A1Q PASS means the A1R evidence DAG is authentic and internally coherent. It does not require the embedded A1R reproduction result to be PASS.

The authorization firewall therefore accepts these A1Q reproduction outcomes:

- `PASS`
- `NEGATIVE`

A valid A1R NEGATIVE is specifically a reason to run the independent A1C path. `INVALID` and `INDETERMINATE` are never execution-eligible.

## Current-subject binding

The firewall independently runs:

- `git rev-parse HEAD`
- `git rev-parse HEAD^{tree}`

and requires A1Q to name those exact identities.

It also recomputes the current environment-definition SHA-256 from:

- `nix/flake.nix`
- `nix/flake.lock`
- `de001a_execution_closure_v2.json`
- platform `x86_64-linux`
- runtime identity `nix (Nix) 2.34.7`

A1E must name that exact definition.

The canonical A1C manifest is read from the current checkout, hashed, and required to match the A1C contract receipt.

## Execution budget

Authorization is limited to one fixed-point executor process with:

- likelihood call budget = 1;
- sampler forbidden;
- minimizer forbidden;
- optimization forbidden;
- parameter mutation forbidden;
- network forbidden;
- CAMB forbidden;
- runtime package installation forbidden.

A static authorization receipt does not claim stateful global single-use. The future executor must bind the authorization-receipt SHA-256, record its actual likelihood-call count, and refuse any count other than exactly one.

## Receipt graph

```text
current exact HEAD/TREE ───────┐
                              │
A1Q PASS ──────────────────────┤
                              │
current env definition ── A1E PASS
                              │
current A1C manifest ─ contract PASS
                              │
current authorization spec ───┤
                              ▼
                    A1C EXECUTION AUTHORIZED
                              │
                              ▼
                  future one-call executor
```

The authorization receipt hashes every supplied prerequisite receipt plus the canonical manifest and specification.

## Authority boundary

A firewall PASS means only that A1C fixed-point execution is permitted under the frozen call budget.

It does not establish:

- an A1C likelihood result;
- A1R/A1C agreement;
- A1X qualification;
- optimizer authority;
- LambdaCDM validity;
- dynamic dark energy;
- any observational anomaly.

`scientific_claim=NONE` remains mandatory.
