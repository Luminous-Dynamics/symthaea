# HCP-MMP1 Lineage-B Scientific Input Snapshot v1

Status: **candidate qualification primitive; not integrated into transform execution**

Profile: `symthaea-hcpmmp-scientific-input-snapshot-v1`

## Problem

The current Lineage-B derivation verifies the fourteen run-manifest scientific input paths before Workbench execution and verifies them again afterward. That is a useful mutation detector, but it does not by itself prove that every Workbench read observed one immutable byte image throughout execution.

A source path can be changed, replaced, or transiently rewritten between the pre-execution hash and a later Workbench open. Final re-hashing detects many such changes, but the stronger execution theorem should be:

```text
committed scientific input bytes
        -> private content-verified snapshot
        -> transform execution
```

rather than:

```text
hash mutable operator paths
        -> execute against those same paths
        -> hash mutable operator paths again
```

## v1 snapshot theorem

For one already validated Lineage-B run manifest, the snapshot primitive:

1. accepts exactly the fourteen v1 scientific input roles;
2. obtains each source path only from that run manifest;
3. resolves the source path, opens it with `O_NOFOLLOW` and nonblocking semantics where supported, and requires the actual opened descriptor to be a regular file via `fstat()`;
4. copies into a fresh private snapshot directory;
5. creates each destination exclusively with mode `0600`;
6. closes the already-open source descriptor even if exclusive destination creation itself fails;
7. computes SHA-256 while copying and requires equality with the role's committed digest;
8. re-hashes the retained snapshot copy and requires the same digest;
9. fsyncs each copied file and the snapshot directory;
10. changes retained snapshot files to read-only mode `0400`;
11. on any build failure, requires partial-snapshot cleanup to complete and be confirmed before propagating failure; if cleanup cannot be confirmed, it raises an explicit cleanup failure and returns no receipt;
12. computes a path-independent digest over the canonical `(role -> committed SHA-256)` map.

The snapshot root remains mode `0700` so the owning process can clean it up safely.

The opened-descriptor check matters because `path is regular` and `object actually opened is regular` are not identical under a local path-replacement race. V1 validates the latter before reading any scientific bytes.

The descriptor-lifetime rule is separate: a race or pre-existing destination can make `O_EXCL` creation fail after the source has already been opened. That failure must not leak the source descriptor into a long-lived operator process.

The cleanup rule is likewise intentionally precise: a storage or permission failure can make physical deletion impossible. V1 does not claim otherwise. It guarantees that unconfirmed cleanup can never be represented as a successful `ScientificInputSnapshotV1`.

## What this closes

After a successful snapshot, later mutation or replacement of an operator source path cannot alter the already copied snapshot bytes. A source that has already drifted away from its committed digest cannot produce a successful snapshot. A path that resolves or races to a non-regular opened object cannot be consumed as a scientific input by this primitive.

The intended later integration is:

```text
load/validate run manifest
        -> current Workbench + input verification
        -> build_scientific_input_snapshot(...)
        -> Workbench consumes snapshot paths only
        -> existing post-execution source + Workbench revalidation
        -> semantic normalization
        -> retained evidence bundle
```

The final revalidation should remain even after snapshot integration. It is a conservative custody/provenance check in addition to execution isolation.

## Deliberate non-claims

This tranche is **not yet wired into `derive()`** and therefore changes no scientific execution result.

It does not establish:

- Workbench executable immutability;
- protection against a malicious same-UID process, compromised kernel, or hostile filesystem;
- guaranteed physical cleanup when the filesystem itself refuses cleanup;
- authorized acquisition of HCP/BALSA inputs;
- atlas correctness;
- execution independence;
- Lineage-A/Lineage-B independence;
- FMQ-010;
- neural alignment or consciousness evidence;
- retained evidence admission.

The `ScientificInputSnapshotV1` object is an authority-free local receipt. It is not serialized into scientific evidence and cannot establish scientific or governance authority.

## Qualification contracts

The dependency-free v1 suite covers:

1. exact fourteen-role closed-world snapshot + restrictive modes;
2. source mutation after snapshot cannot change captured bytes;
3. source path replacement after snapshot cannot retarget captured bytes;
4. pre-snapshot source corruption fails and removes the ordinary partial snapshot;
5. deliberately unconfirmed cleanup fails closed and returns no snapshot receipt;
6. an opened non-regular source object is rejected and leaves no valid snapshot;
7. destination-open failure closes the already-open source descriptor and preserves the pre-existing destination;
8. missing input role fails before snapshot creation;
9. an existing destination is never reused or overwritten;
10. snapshot content identity is independent of machine-local source paths.

Hosted qualification must also freeze descriptor-level regular-file checks, descriptor lifetime on destination-open failure, the confirmed-cleanup boundary, and absence of subprocess/network execution surfaces in this primitive.

## Promotion rule

Do not treat this primitive as part of the Lineage-B scientific generator until a separate integration tranche explicitly changes `derive()` to consume only its returned snapshot paths and that exact integrated head qualifies.
