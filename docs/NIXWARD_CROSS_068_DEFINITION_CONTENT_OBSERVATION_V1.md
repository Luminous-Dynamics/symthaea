# Nixward CROSS-068 — systemd Definition Content Observation V1

## Purpose

CROSS-068 closes the gap between systemd source identity and byte-level definition provenance.

`FragmentPath` and `DropInPaths` identify the unit sources systemd loaded, but they are not content hashes. Current systemd documentation also exposes `NeedDaemonReload`, which indicates that the unit source has changed since systemd loaded the configuration. See the systemd D-Bus reference and current systemd man page: 
https://www.man7.org/linux/man-pages/man5/org.freedesktop.systemd1.5.html

## Observer boundary

The public observation API is unit-scoped:

`observe_service_definition_content(unit)`

Callers do not provide arbitrary filesystem paths.

The observer first resolves the canonical service unit through systemd and reads its exact FragmentPath/DropInPaths. Those paths are then passed to the observer-sealed content layer.

The content layer returns only:

- per-file BLAKE3 content digests;
- byte size;
- device/inode identity;
- mtime and ctime metadata;
- a deterministic overall commitment.

File bytes never enter the durable evidence structure.

## Exact-path safety

On Linux, the content reader uses `openat2(2)` with root-constrained path resolution and `RESOLVE_NO_MAGICLINKS`. Ordinary symlinks remain allowed because NixOS commonly exposes generated systemd sources through symlinked paths.

Path validation rejects empty paths, non-absolute paths, `.` and `..` components, embedded NUL bytes, and overlong paths.

The reader accepts regular files only and enforces a finite per-file size bound.

Linux `openat2(2)` explicitly provides resolver controls intended for restricting how potentially untrusted paths are resolved, including `RESOLVE_NO_MAGICLINKS`. Source: https://www.man7.org/linux/man-pages/man2/openat2.2.html

## Repeated-read stability

A single file read is not treated as atomic evidence.

For every source file, the observer:

1. opens one read-only file descriptor;
2. captures metadata;
3. hashes the complete file;
4. seeks to the beginning;
5. hashes the file again;
6. requires both content digests to match;
7. captures metadata again;
8. requires the metadata identity to match.

If content or metadata changes across the repeated observation, the commitment is rejected.

This is still an observation protocol, not a filesystem CAS primitive. A sufficiently capable attacker could construct changes that evade both sampled observations.

## systemd source rebinding

After the file hashes are complete, the observer re-reads:

- FragmentPath;
- DropInPaths;
- NeedDaemonReload;
- systemd manager unique owner.

The source identity must equal the pre-hash identity and NeedDaemonReload must remain false.

This prevents the following class of false evidence:

`systemd reports source A -> observer hashes A -> systemd switches to source B -> observer returns A as current definition evidence`

The observer instead fails closed when the source identity or manager incarnation changes during the observation.

## Manager incarnation

The commitment binds the systemd manager's unique D-Bus owner. This is provenance, not authorization. CROSS-063 carries the same manager-incarnation concept into Job evidence and durable receipts.

## Source identity, content identity, and configuration identity

The system now keeps these concepts separate:

`FragmentPath + DropInPaths` = systemd-reported source identity

`BLAKE3(file bytes)` = observed content identity

`NixOS generation / effect context` = declarative authorization provenance

A content digest must not be described as a generation identity, and a source path must not be described as a content hash.

## Claim ceiling

CROSS-068 does not provide:

- atomic filesystem snapshot semantics;
- cryptographic attestation of systemd;
- proof that content remained unchanged after the observation finished;
- proof that the same bytes were the bytes systemd parsed in an earlier transaction;
- content provenance for arbitrary files outside the systemd-reported source set.

It does provide a stronger, testable statement: at one bounded observation point, the exact source files reported by systemd were read twice through read-only descriptors, produced stable content digests, and remained associated with the same systemd source identity and manager incarnation.

## Qualification

The content observer is part of the read-only systemd observation boundary. Its constructor is observer-sealed and the qualification scanner prohibits filesystem write APIs in the module.

Exact-head GitHub Actions remain the qualification authority. Queued, pending, cancelled, mergeable, or static-only states are not PASS evidence.