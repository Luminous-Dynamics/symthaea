# Symthaea Music Theory Patch Series 19 Plan

**Date:** 2026-07-21
**Base:** Patch Series 18 / Git tree `878400a3db22439c56d97c804d659d8e14b2ddc8`
**Theme:** Rotatable checkpoint trust, authenticated gossip, and exact policy-to-head branch continuity

## Executive summary

Series 18 made one catalog head portable through checkpoints, one-step prefix
proofs, mirrors, and external checkpoint witnesses. It deliberately left the
witness policy fixed. That omission becomes operationally important as soon as
a witness key expires, an organization leaves, or a key is suspected of
compromise.

Series 19 adds a narrowly scoped evolution path:

1. append-only witness-policy epochs;
2. dual outgoing/incoming quorum authorization for every rotation;
3. authenticated observer gossip and portable conflict proofs;
4. exact multi-hop catalog lineage from the genesis policy activation through
   every rotation to the packaged head;
5. one continuity artifact that keeps authentication, branch continuity,
   gossip coverage, and conflict status as separate gates.

This remains evidence infrastructure, not a consensus implementation.

## Patch groups

### A. Witness-policy history and rotation

- Introduce versioned policy epochs and rotation payloads.
- Bind each policy activation to one exact catalog checkpoint.
- Require both outgoing and incoming thresholds for a transition.
- Reject no-op policy rotations.
- Preserve external signature verification and key custody boundaries.
- Split the implementation into model, integrity, and test modules before
  release.

### B. Authenticated checkpoint gossip

- Persist observer checkpoint statements with canonical signing bytes.
- Bind each statement to the observer's prior checkpoint and believed active
  policy epoch.
- Verify signatures through a caller-supplied interface.
- Extract portable rollback, same-height equivocation, and fork proofs.
- Preserve conflict-bearing ledgers as valid incident evidence while refusing
  conflict-free acceptance.

### C. Exact multi-hop catalog lineage

- Compose Series-18 direct consistency proofs without weakening them.
- Keep each intermediate catalog and checkpoint explicit.
- Reject identity changes, repeated checkpoints, or invalid direct hops.
- Require every policy activation checkpoint to appear in ordinal order.
- Require the lineage terminal state to equal the packaged catalog head.

### D. Combined continuity artifact

- Bind the catalog-head bundle, policy ledger, exact policy lineage, gossip
  ledger, and complete conflict-proof set.
- Re-authenticate head witnesses, policy rotations, and gossip statements.
- Reject mismatched active policies or gossip policy-epoch claims.
- Allow authenticated incident packages while distinguishing them from
  accepted conflict-free continuity.
- Record all trust limitations as canonical hashed data.

### E. Operator workflows

Add examples for:

- witness-policy genesis, planning, signing, rotation, and active lookup;
- authenticated gossip recording, verification, and conflict extraction;
- explicit catalog-lineage construction and audit;
- complete continuity-bundle build, audit, and verification.

The examples reuse one no-shell JSON-over-stdin external verifier adapter.

### F. Persistence and API governance

- Export only explicit new contracts through the crate root.
- Append new schema roles without renumbering Series-18 identities.
- Advance the schema registry through the lineage contracts.
- Freeze old and new role ordinals in regression tests.

### G. Adversarial and integration coverage

- Reject unchanged witness policies.
- Detect malformed or wrongly signed rotations.
- Detect observer rollback, equivocation, and forks.
- Reject gossip that names the wrong policy epoch.
- Reject policy histories not anchored to the packaged catalog branch.
- Exercise two actual catalog publications, a policy rotation, new-policy head
  witnesses, gossip, lineage, and a deliberately falsified lineage.

## Trust boundaries

Series 19 does not claim to establish:

- witness or observer independence;
- disjoint outgoing and incoming quorums;
- wall-clock freshness;
- universal gossip coverage;
- global non-equivocation;
- legal publication authority;
- key enrollment, compromise recovery, or private-key custody;
- compact Merkle consistency proofs;
- distributed consensus.

Those responsibilities remain external and are recorded as mandatory bundle
limitations.

## Landing order

1. Witness-policy models and integrity.
2. Gossip models, verification, and conflict extraction.
3. Initial continuity bundle and public API activation.
4. Schema registration and role freezing.
5. External verifier reuse and command-line tools.
6. No-op rotation and active-policy hardening.
7. Exact multi-hop lineage.
8. Continuity branch binding and adversarial integration test.
9. Module decomposition and panic-surface cleanup.
10. Release contract, application guide, and reproducibility bundle.

## Required canonical verification

In the project development shell:

```text
cargo fmt --all -- --check
cargo test -p symthaea-music-theory
cargo clippy -p symthaea-music-theory --all-targets -- -D warnings
```

Recommended focused checks:

```text
cargo test -p symthaea-music-theory publication
cargo test -p symthaea-music-theory witness_policy
cargo test -p symthaea-music-theory gossip
cargo test -p symthaea-music-theory lineage
cargo test -p symthaea-music-theory continuity
```

Static packaging checks are useful but do not replace Rust compilation.
