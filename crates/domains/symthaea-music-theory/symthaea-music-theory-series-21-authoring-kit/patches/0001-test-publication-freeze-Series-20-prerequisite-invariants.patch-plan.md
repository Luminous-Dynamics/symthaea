# Patch 0001: test: publication freeze Series 20 prerequisite invariants

## Objective

Freeze verifier-owned policy matching, publication-event causality, pre-issuance mirror rejection, stable numeric persistence, and schema-role ordinals before adding resumption.

## Required implementation evidence

- Exact files and public symbols changed.
- At least one positive and one adversarial test unless documentation-only.
- Persisted models use fixed-width integer types and `deny_unknown_fields`.
- Canonical identity is independently recomputed during audit.
- Structural validity is not treated as external authentication.
- No mutation path accepts a cached authorization boolean.

## Acceptance

- Applies cleanly to the exact preceding tree.
- `git diff --check` passes.
- Public exports and schema roles are intentional and frozen.
- Full canonical Cargo/Nix verification remains mandatory.
