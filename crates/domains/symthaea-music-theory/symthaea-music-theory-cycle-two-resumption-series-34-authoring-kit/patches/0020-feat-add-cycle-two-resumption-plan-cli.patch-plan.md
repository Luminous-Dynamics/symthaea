# Patch 0020: feat add cycle two resumption plan cli

**Series:** 34

## Objective

Expose non-mutating plan, signable-payload, and verify workflows.

## Intended changes

- Accept explicit cycle closure, certification, segment, catalog, policy, signer, delegation, allowance, and publication files.
- Emit canonical artifacts and machine-readable reports.
- Reject hidden defaults.

## Acceptance evidence

- All modes are non-mutating.
- Missing or ambiguous inputs fail.
- Identical inputs reproduce identical outputs.

## Non-claims

- Does not contact signers.
- Does not reserve the first-mutation slot.
