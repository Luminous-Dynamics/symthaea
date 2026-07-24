# Patch 0022: feat tooling add terminal retirement command

**Series:** 25

## Objective

Build, verify, and transactionally commit terminal retirement.

## Intended changes

- Support plan, signable-payload, verify, dry-run, commit, and disclosure-package modes.
- Require explicit current head, all active capabilities, and archive policy.
- Write results atomically with overwrite protection.

## Required tests

- Stale state between plan and commit fails.
- Interrupted writes leave no accepted partial retirement.
- Commit output passes cumulative audit and archive-only verification.

## Non-claims

- Does not contact signers.
- Does not destroy private keys automatically.
