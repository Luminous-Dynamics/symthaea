# ADR-034: RSK Cargo.lock Repair Lineage

**Status**: Proposed  
**Date**: 2026-09-13  
**Change Class**: A  
**Scope**: Replicator Safety Kernel (RSK) qualification only

## Decision

A Cargo-generated `Cargo.lock` repair must distinguish three exact byte sequences:

1. the pull-request base lockfile;
2. the lockfile committed at the exact candidate head;
3. the lockfile present after pinned Cargo resolves that exact head.

Two modes are admitted by the diagnostic tooling.

### Generation mode

If base and head lockfiles are byte-identical, pinned Cargo may generate a non-admissible repair candidate. The candidate must pass the RSK-only lock-delta qualifier against the base lockfile.

### Commit-verification mode

If the candidate head changes `Cargo.lock`, the base-to-head transition must pass the RSK-only lock-delta qualifier. Pinned Cargo must then leave the committed head lockfile byte-for-byte unchanged. Semantic equivalence, package reordering, or a second Cargo rewrite is insufficient.

## Rationale

A workflow that compares only the checked-out head to its post-Cargo state can generate a candidate safely, but cannot prove that a later committed lockfile is the exact reviewed base-to-candidate transition. Binding base, head, and post-Cargo bytes closes that lineage gap.

## Evidence boundary

The repair workflow remains non-admissible diagnostic evidence. A verified committed lockfile is only a prerequisite for later normal RSK qualification with `--locked` and exact receipt binding.

This ADR does not grant replication authority, production admission, or any physical-system capability. Production admission remains **DENIED / NOT YET ELIGIBLE**.
