# Patch 0024: feat add trust exhaustion and retirement plan cli

**Series:** 35

## Objective

Expose non-mutating trigger evaluation, plan, signable-payload, and verify workflows.

## Intended changes

- Require explicit lifecycle history, current state, policy, capability inventory, archive policy, and signer files.
- Emit canonical reports and plans.
- Reject hidden defaults.

## Acceptance evidence

- All modes are non-mutating.
- Missing history renders unknown.
- Identical inputs reproduce identical outputs.

## Non-claims

- Does not contact signers.
- Does not commit retirement.
