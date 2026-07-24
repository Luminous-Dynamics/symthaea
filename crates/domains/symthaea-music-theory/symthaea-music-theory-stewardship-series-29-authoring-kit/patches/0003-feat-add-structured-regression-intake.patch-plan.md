# Patch 0003: feat add structured regression intake

**Series:** 29

## Objective

Turn real failures into reproducible maintenance inputs.

## Intended changes

- Add bounded reports for affected version, environment, operation, expected result, observed result, artifacts, privacy class, and reproduction status.
- Assign stable regression identities and severity classes.
- Separate unverified report, reproduced defect, expected rejection, documentation gap, and unsupported use.

## Required evidence

- Malformed or oversized reports fail safely.
- Duplicate reports link rather than multiply work.
- Private artifacts are excluded from public reports by default.

## Non-claims

- Does not treat report volume as defect severity.
- Does not accept arbitrary attachments without resource limits.
