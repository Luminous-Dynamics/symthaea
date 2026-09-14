# ADR-040: Bind exact Cargo metadata evidence in RSK lockfile lineage receipts

**Status**: Proposed
**Change Class**: A
**Scope**: Replicator Safety Kernel (RSK) Cargo.lock repair qualification

## Context

ADR-039 permits one narrow class of pre-existing package dependency-edge additions when pinned Cargo metadata proves that the source package is reachable from an RSK root and that the exact source-to-target edge exists in the resolved graph.

The executed #2850 diagnostic demonstrated that theorem successfully for the Cargo-generated lock transition containing:

`symthaea-replicator-safety -> sha2 0.10.9 -> digest 0.10.7 -> const-oid 0.9.6`.

However, the v2 lineage receipt records the derived proof path without cryptographically identifying the exact raw `cargo-metadata.json` bytes from which that proof was derived. A reviewer can inspect the retained artifact, but the receipt itself does not bind that evidence object.

The command-line verifier also previously permitted implicit discovery of a sibling `cargo-metadata.json`. That is convenient, but it makes the evidence dependency less explicit than a Class A verifier should allow.

## Decision

The lineage receipt schema advances to `symthaea.rsk.lockfile-repair-lineage.v3`.

Whenever Cargo metadata bytes are supplied, accepted and rejected receipts record an exact evidence binding containing:

- `present = true`;
- SHA-256 of the raw metadata bytes; and
- exact byte length.

When no metadata bytes are supplied, the binding records `present = false`, `sha256 = null`, and `bytes = 0`.

The binding is over the exact raw bytes consumed by the verifier. It is deliberately not a hash of parsed or canonicalized JSON. Semantically equivalent metadata serialized differently therefore has a different evidence identity.

The CLI no longer discovers a sibling metadata file. Workflows and human callers must pass `--cargo-metadata <path>` explicitly whenever metadata-backed edge proof is required. Missing metadata remains fail-closed through the existing `cargo_metadata_required` rejection.

The hosted diagnostic additionally retains an independent `cargo-metadata-sha256.txt` alongside the structured lineage receipt.

## Invariants

1. A metadata-backed exception cannot be accepted without explicit metadata bytes.
2. The receipt identifies exactly which raw metadata bytes were consumed.
3. Rejection receipts bind metadata bytes too when those bytes were available, including malformed metadata.
4. Metadata evidence does not grant authority, alter package identity, permit package additions, permit dependency removals, or relax ADR-039's RSK-reachability and exact-edge theorem.
5. The committed `Cargo.lock` remains the authority boundary for the later candidate tranche.
6. A metadata hash is provenance evidence only. It does not establish compilation, runtime admission, physical safety, or replication authority.

## Evidence separation

The #2850 hosted artifact remains evidence for the v2 theorem. This ADR does not retroactively claim that its v2 receipt bound metadata bytes.

The child tranche implementing this ADR must itself execute before a later lockfile-only candidate may rely on v3 receipt semantics. The intended sequence remains:

1. qualify the verifier and evidence-binding logic;
2. commit the exact Cargo-generated `Cargo.lock` candidate in a separate child;
3. require pinned Cargo byte-idempotence;
4. run normal RSK qualification with `--locked` and exact receipt binding.

Production admission remains **DENIED / NOT YET ELIGIBLE**.
