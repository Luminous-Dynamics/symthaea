# Patch 0019: feat add resumption plan and verify cli

**Series:** 31

## Objective

Expose a non-mutating operator path to construct and verify the slice inputs.

## Intended changes

- Accept explicit closure, certification, catalog, policy, signer, delegation, allowance, and publication files.
- Provide plan, signable-payload, and verify modes.
- Emit machine-readable reports.

## Acceptance evidence

- All modes are non-mutating.
- Ambiguous or missing inputs fail.
- Identical inputs reproduce identical outputs.

## Non-claims

- Does not contact signers.
- Does not reserve the first-mutation slot.
