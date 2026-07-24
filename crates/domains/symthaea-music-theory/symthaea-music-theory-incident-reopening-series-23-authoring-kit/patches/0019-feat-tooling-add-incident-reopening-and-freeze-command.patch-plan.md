# Patch 0019: feat tooling add incident reopening and freeze command

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Add the governed workflow that builds authorization and commits an exact segment freeze.

## Intended changes

- Load the challenge ledger, adverse report, active policies, closure, segment ledger, catalog head, quarantine state, and signer artifacts.
- Support plan, verify, dry-run, and transactional commit modes.
- Emit the reopening review package and lifecycle report.

## Required tests

- Stale head or authority changes between plan and commit fail.
- Interrupted writes cannot produce accepted partial state.
- Output path and overwrite protections are enforced.

## Non-claims

- Does not contact signers automatically.
- Does not begin a new recovery without separate Series 20–22 workflows.
