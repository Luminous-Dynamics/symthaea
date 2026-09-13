# ADR-026 — RSK Monotonic Anchor

**Status**: Proposed
**Change Class**: A
**Date**: 2026-09-13

## Decision

RSK distinguishes local data integrity from monotonic continuity.

A separately authenticated external observation will bind one exact local tracker state using:

- namespace;
- epoch;
- sequence counter;
- canonical state digest;
- provider and trust identity;
- freshness evidence.

The local tracker and external observation must agree exactly before the anchored-state predicate is satisfied.

Any disagreement is non-operational for new authorization decisions. The verifier does not choose either side automatically and does not reset the external state.

State replacement or epoch change is handled only by the separate governed recovery process.

## Rationale

An older local snapshot can retain valid integrity metadata. A counter alone can also match a different state. Exact counter plus complete-state-digest agreement avoids both ambiguities.

## Reference scope

`scripts/rsk_monotonic_anchor.py` models comparison semantics only. It does not authenticate or mutate the external provider.

Production admission remains **DENIED / NOT YET ELIGIBLE**.
