# Patch 0016: Benchmark worst-case valid reference bundles

**Series:** 24

## Objective

Choose deployable limits using legitimate scale evidence rather than malformed inputs alone.

## Intended changes

- Generate valid bundles at threshold edges for records, events, lineage, signers, mirrors, vectors, and archive files.
- Measure wall time, allocations, peak RSS where available, canonical bytes, hashes, and external calls.
- Keep benchmark results advisory unless the release profile declares a hard budget.

## Required tests

- Reference bundles remain semantically valid.
- Regression thresholds are architecture-aware and documented.
- Benchmarks do not silently skip unavailable metrics.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
