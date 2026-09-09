# Humanoid Verified Authority Import v0.1

Status: source-provenance import only; no humanoid execution authority is created by this document or commit.

## Purpose

The humanoid assurance lineage needs the existing cognition-free bounded-authority semantics and the existing verifier-owned Xenia authority proof without inventing a parallel robot-specific grant system.

This commit imports the minimum dependency slice from the existing authority lineage into the humanoid branch so later humanoid code can depend on the exact reviewed types rather than copying their semantics.

## Source lineage

Source branch: `agency/verified-authority-state-v0.1`

Source head: `9da533c672520b6292b28263d318a509b32639cc`

Common ancestor with the humanoid lineage at import time: `2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`

Imported directory trees:

- `crates/core/symthaea-authority` — tree `65e50842a966451784eb3311a0ad038c8c15c092`
- `crates/core/symthaea-action-checkpoint` — tree `ff01a857ce9b6994adb7d2103e2732be9007a744`
- `crates/bridges/symthaea-authority-time` — tree `88f710843b748eaddb7f8f6a21f99c5d64af1acf`
- `crates/bridges/symthaea-authority-state` — tree `8d8557c39e55bf24654142e9ad35b444759b2135`
- `crates/bridges/symthaea-xenia-authority` — tree `32d59664472bc19385b475e68875bcc49356ebdd`

These subtrees are imported byte-for-byte by Git tree identity. The import does not reinterpret or fork their semantics.

## Why this is not a merge commit

The authority and humanoid branches have diverged substantially. A synthetic merge parent that intentionally omitted unrelated authority-branch work would make Git history claim that work had been merged when it had not. This import therefore preserves source provenance explicitly while keeping later integration scope narrow.

## Authority boundary

The imported layers retain their existing separation:

- `CapabilityGrant` is bounded authority data and deterministic attenuation semantics;
- `VerifiedAuthorityTime` is a fresh challenge-bound time fact;
- `VerifiedAuthorityState` is a fresh threshold-authenticated epoch + negative-fact snapshot for one exact grant;
- `VerifiedXeniaCapability` is an affine verifier-owned proof binding the exact grant, executor workload, Xenia session/frontier, prior agent checkpoint, trusted time, and owned authority-state snapshot.

None of these objects is a humanoid motor command or a physical dispatch ticket.

## Non-claims

This import does not:

- qualify the imported authority lineage on the humanoid head;
- claim CI evidence transfers from the source SHA;
- create institutional policy or signing keys;
- grant humanoid goal or motor authority;
- reserve a grant use transactionally;
- bind authority to a finalized actuator command;
- replace protective or functional-safety authority.

All qualification must be rerun on the exact composed head.