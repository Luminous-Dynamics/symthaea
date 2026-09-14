# ADR-039: Prove Cargo feature-unification edges in RSK lockfile repair lineage

**Status**: Proposed  
**Change Class**: A  
**Date**: 2026-09-14  
**Scope**: Replicator Safety Kernel (RSK) Cargo.lock repair qualification

## Context

The non-admissible RSK lockfile repair diagnostic at exact subject
`4699755dd84b89c7c1fa3792953564dfb9a3388d` executed and rejected the
Cargo-generated candidate because the v1 qualifier required every pre-existing
non-RSK package record to remain byte-identical.

The retained evidence shows a narrower transition:

- the three expected local RSK path packages are added;
- no pre-existing package identity, version, source, or checksum changes;
- no pre-existing package is removed;
- no new external package is introduced; and
- `digest 0.10.7` gains only the already-pinned dependency edge
  `const-oid 0.9.6`.

The same retained Cargo metadata resolves the path

`symthaea-replicator-safety -> sha2 0.10.9 -> digest 0.10.7 -> const-oid 0.9.6`.

This is consistent with Cargo feature unification for the RSK safety crate's
`sha2/oid` dependency. Treating all pre-existing record-byte changes as
equivalent would reject a dependency-edge repair that Cargo can independently
prove while also hiding the exact reason for rejection because the v1 lineage
tool emitted no structured report on failure.

## Decision

The v2 lockfile-delta theorem remains fail-closed but admits one narrowly
defined class of pre-existing package change: dependency-edge additions.

Such an addition is accepted only when all of the following are true:

1. the source package identity `(name, version, source)` already exists in the
   baseline lockfile and is unchanged;
2. every source-package field other than `dependencies` is unchanged;
3. no dependency edge is removed;
4. the dependency target already exists in the baseline lockfile;
5. Cargo metadata contains the exact resolved source-to-target edge;
6. the source package is reachable from an exact RSK package root in Cargo's
   resolved graph; and
7. the proof report records the complete RSK-root-to-target path.

The exception does **not** admit:

- package additions other than the exact expected local RSK path packages;
- package removals;
- version, source, checksum, or other package-record mutation;
- dependency-edge deletion;
- dependency-edge addition outside the RSK transitive closure;
- an edge absent from Cargo metadata;
- a dependency target not already pinned in the baseline lockfile; or
- ambiguous lockfile or metadata identities.

Cargo metadata is evidence for graph reachability only. It is not an authority
grant and cannot relax any other RSK admission invariant.

The lineage CLI will use an explicitly supplied metadata file when provided.
For compatibility with the existing diagnostic workflow, when that argument is
omitted it may consume only a sibling `cargo-metadata.json` next to the
post-Cargo lock snapshot. If neither is present, any pre-existing dependency
record change remains denied.

The lineage CLI must emit `lineage-report.json` on semantic rejection whenever
the requested output path is writable. Rejection receipts include a stable
reason code and the available base/head/post-Cargo lock digests.

## Evidence separation

This ADR changes only the verifier theorem and its tests. It does **not** commit
the generated Cargo.lock candidate. The verifier change must be qualified first
on its own exact head. Only a later child tranche may commit the exact
Cargo-generated candidate and require pinned Cargo to leave those bytes
unchanged.

The retained #2754 candidate remains non-admissible diagnostic evidence.
Production admission remains **DENIED / NOT YET ELIGIBLE**.

## Consequences

The repair qualifier can distinguish a proved Cargo feature-unification edge
from arbitrary dependency drift without creating a general allowlist. The
accepted report becomes more informative because every exceptional edge carries
an explicit resolved proof path, while failures leave a structured receipt for
postmortem analysis.

The trusted boundary is still conservative: if identity resolution is
ambiguous, metadata is malformed, reachability cannot be proved, an external
package would be newly introduced, or any non-dependency package field changes,
qualification fails closed.
