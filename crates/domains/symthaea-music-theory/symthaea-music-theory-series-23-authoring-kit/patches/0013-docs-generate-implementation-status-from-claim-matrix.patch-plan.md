# Patch 0013: Generate implementation status documentation

**Series:** 23

## Objective

Prevent prose from drifting ahead of executable truth.

## Intended changes

- Render human-readable status from the machine claim matrix.
- Check committed status documents against generated output.
- Link every demonstrated claim to evidence artifact identities.

## Required tests

- A hand-edited implementation claim fails CI.
- Unavailable architecture evidence is labeled precisely.
- Historical release claims remain immutable.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
