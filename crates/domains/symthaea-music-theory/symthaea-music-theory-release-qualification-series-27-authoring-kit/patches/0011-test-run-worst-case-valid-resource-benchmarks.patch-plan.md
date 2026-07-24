# Patch 0011: test run worst case valid resource benchmarks

**Series:** 27

## Objective

Measure and cap verification work for the largest accepted artifacts and lifecycle histories.

## Intended changes

- Benchmark decoding, canonicalization, signatures, ledgers, archives, multi-cycle audit, terminal checkpoint, and independent-verifier subprocesses.
- Record CPU, memory, bytes read, calls, and elapsed time under exact policy.
- Define regression thresholds from observed evidence.

## Required tests

- Worst-case-valid fixtures complete within configured budgets.
- Resource regressions block qualification unless explicitly reviewed.
- Rejected hostile artifacts do not consume more than policy allows.

## Non-claims

- Does not promise identical performance on all hardware.
- Does not weaken semantic checks to meet benchmarks.
