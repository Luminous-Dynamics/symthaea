# Patch 0020: feat tooling add recovery cycle command

**Series:** 24

## Objective

Add an operator workflow for beginning, authorizing, committing, certifying, and closing a later recovery cycle.

## Intended changes

- Use explicit files and typed roles for every input.
- Provide plan, verify, dry-run, commit, certify, and close modes.
- Emit machine-readable receipts and complete evidence packages.

## Required tests

- Dry-run and verify-only modes are non-mutating.
- Stale state between plan and commit fails.
- Shell metacharacters and ambiguous output paths are rejected.

## Non-claims

- Does not contact signers automatically.
- Does not select a branch without explicit inputs.
