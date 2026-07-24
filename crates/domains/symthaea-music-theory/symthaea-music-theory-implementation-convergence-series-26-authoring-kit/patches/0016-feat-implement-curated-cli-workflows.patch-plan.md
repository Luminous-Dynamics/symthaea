# Patch 0016: feat implement curated cli workflows

**Series:** 26

## Objective

Turn the planned operator commands into one coherent no-shell command family.

## Intended changes

- Implement plan, signable-payload, verify, dry-run, commit, audit, archive, and disclosure modes for resumption, challenge, reopening, cycle recovery, and retirement.
- Use explicit input files, expected policies, output directories, and overwrite controls.
- Emit stable machine-readable reports and human summaries.

## Required tests

- Verify-only and dry-run modes are non-mutating.
- Interrupted writes leave no accepted partial output.
- Identical inputs reproduce byte-identical outputs.

## Non-claims

- Does not contact signers automatically.
- Does not select branches or governance outcomes implicitly.
