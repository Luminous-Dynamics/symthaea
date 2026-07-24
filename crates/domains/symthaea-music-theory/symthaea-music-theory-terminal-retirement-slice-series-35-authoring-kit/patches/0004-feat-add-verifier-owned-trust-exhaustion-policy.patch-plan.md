# Patch 0004: feat add verifier owned trust exhaustion policy

**Series:** 35

## Objective

Implement local, non-artifact-controlled criteria for considering terminal retirement.

## Intended changes

- Support maximum completed cycles, reopenings, unresolved verifier disagreements, compromised authority classes, forbidden quarantines, mandatory review, and configured immediate-retirement conditions.
- Use fixed-width counters and explicit unknown states.
- Bind policy identity into reports.

## Acceptance evidence

- Artifact-supplied weakening fails.
- Unknown history renders unknown rather than safe.
- Policy changes invalidate cached reports.

## Non-claims

- Does not prescribe universal thresholds.
- Does not retire the lineage automatically.
