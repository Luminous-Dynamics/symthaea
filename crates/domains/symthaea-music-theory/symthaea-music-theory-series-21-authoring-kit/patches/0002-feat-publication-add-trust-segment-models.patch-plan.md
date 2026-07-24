# Patch 0002: feat: publication add trust segment models

## Objective

Add versioned trust-segment and segment-genesis records with content-derived identities and unknown-field rejection.

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
