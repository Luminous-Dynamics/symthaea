# Patch 0021: feat tooling add trust exhaustion report command

**Series:** 25

## Objective

Generate a deterministic technical retirement-trigger report without changing state.

## Intended changes

- Load exact incident, cycle, segment, authority, witness, quarantine, verifier, and preservation artifacts.
- Require caller-supplied expected retirement policy.
- Emit machine-readable satisfied, unknown, and unsupported conditions.

## Required tests

- Verify-only operation is non-mutating.
- Missing evidence cannot render safe.
- Outputs reproduce from identical inputs.

## Non-claims

- Does not decide retirement.
- Does not page or notify signers automatically.
