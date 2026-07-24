# Patch 0013: Add stable resource failure dimensions to conformance

**Series:** 24

## Objective

Make limit failures independently reproducible without conflating them with malformed structure or bad signatures.

## Intended changes

- Add stable stages/codes for raw bytes, depth, count, canonical bytes, lineage work, external calls, archive expansion, files, output, timeout, and cancellation.
- Freeze positive and one-over-limit fixtures.
- Document whether a larger trusted profile may accept an otherwise valid artifact.

## Required tests

- Rust and independent verifier agree on every limit fixture.
- Earlier structural defects remain earlier than resource failures when applicable.
- Codes never include platform-dependent wording.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
