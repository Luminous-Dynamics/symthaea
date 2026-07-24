# Patch 0014: Freeze malicious-artifact and archive corpus

**Series:** 24

## Objective

Preserve regressions for known complexity and filesystem attacks.

## Intended changes

- Include nesting, declared-length, duplicate, sorting, cycle, signature-amplification, output-flood, zip/tar bomb, traversal, symlink, hardlink, device, and manifest-confusion cases.
- Store compact generators when full bomb payloads would be unsafe.
- Record expected stage, code, and maximum observed work.

## Required tests

- Every corpus case is deterministic.
- Corpus execution stays within CI budgets.
- A deliberately removed limit causes a focused regression failure.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
