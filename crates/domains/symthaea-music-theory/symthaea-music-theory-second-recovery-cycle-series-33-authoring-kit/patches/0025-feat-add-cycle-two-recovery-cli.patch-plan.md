# Patch 0025: feat add cycle two recovery cli

**Series:** 33

## Objective

Expose plan, candidate inspection, signable-payload, verify, dry-run, select, certify, and close workflows.

## Intended changes

- Require explicit frozen baseline, cycle ledger, policies, candidate set, quarantines, signer evidence, checkpoint, and certification inputs.
- Separate every state-changing mode.
- Emit machine-readable receipts and reports.

## Acceptance evidence

- Inspection, plan, verify, and dry-run modes are non-mutating.
- Stale state between steps fails.
- Interrupted writes leave no accepted partial state.

## Non-claims

- Does not contact signers.
- Does not create a new publication segment.
