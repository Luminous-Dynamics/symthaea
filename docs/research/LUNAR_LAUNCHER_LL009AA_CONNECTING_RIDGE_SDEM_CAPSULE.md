# LL-009AA — Connecting Ridge SDEM exact-byte capsule

LL-009AA creates the exact-byte foundation for using the independent 2026 Shape-from-Shading terrain products at Connecting Ridge.

NASA PGDA Product 104 states that these 5 m/pixel SDEMs use LROC NAC imagery to add terrain detail where LOLA-based LDEMs are limited by sparse sampling and smooth interpolated gaps. The Zenodo record contains a dedicated Connecting Ridge archive:

- `A3CLR22_6_Connecting_Ridge.zip`
- record `10.5281/zenodo.17954508`
- PGDA DOI `10.60903/gsfcpgda-sfs-a3clr`
- publisher-record MD5 `0ad982cf42319a300f2a2cff69d39322`

The archive includes multiple scientific products. LL-009AA deliberately does **not** infer their scientific roles from filenames.

## MD5 versus SHA-256

Zenodo publishes an MD5 checksum for the archive. AA uses that checksum only to establish that acquired bytes correspond to the publisher's record.

The acquisition/lock process then computes SHA-256 over the exact archive bytes. Only that SHA-256 is promoted as the cryptographic identity for downstream evidence.

No archive SHA-256 is guessed before acquisition.

## Modes

The script supports:

- `acquire`: HTTPS acquisition with allowlisted host/redirect validation, identity content encoding, Content-Length checking when supplied, publisher MD5 verification and computed SHA-256;
- `lock-local`: verify an archive obtained by another transfer path and emit the same source-lock semantics;
- `extract`: safely materialize the archive into a new directory and emit a complete deterministic member manifest;
- `verify`: offline exact-file-set replay against that manifest;
- `self-test`: dependency-free synthetic safety campaign.

## Archive safety

Before extraction AA validates every ZIP entry and fails on:

- absolute, dot or parent-traversal paths;
- backslash paths that could change interpretation across platforms;
- duplicate normalized paths;
- encrypted entries;
- symlink/device/special-file entries;
- configured nested-archive suffixes;
- entry-count limits;
- per-file and total uncompressed-size limits;
- excessive compression ratios.

Extraction occurs under a temporary sibling directory and is atomically renamed only after all members have been streamed and hashed successfully.

## Exact member manifest

For every regular extracted file AA records:

- normalized relative path;
- ZIP CRC32;
- compressed and uncompressed byte counts;
- SHA-256 of the exact extracted bytes.

The manifest also binds the archive SHA-256, publisher MD5, exact byte count, source lock and extraction policy. Offline verification requires the exact complete set of files: missing, modified, or unexpected files fail.

## Role boundary

`role_assignment = intentionally_unresolved_until_ll009ab`

This is important. The archive is expected to contain SDEM elevation, SDEM–LDEM differences, image coverage and other products, but filenames alone are not scientific semantics. LL-009AB should operate on the real AA member manifest, select exact member SHA-256 values, validate raster metadata and only then assign evidence roles.

## Why this improves LL-009R

Product 104 is a methodologically distinct terrain reconstruction based on NAC image shading while retaining the LOLA geodetic reference. That makes it useful for testing terrain detail missed in interpolated LOLA gaps and for independent uncertainty calibration.

AA itself makes neither claim. It only makes the bytes and archive membership reproducible enough for later cross-method scientific evidence.

## Local logic evidence

The exact committed script passed Python compilation and the dependency-free synthetic self-test. The campaign verifies deterministic extraction/replay, wrong-publisher-MD5 rejection, exact member hashing, extracted-file tamper detection, path-traversal rejection and symlink rejection.

No real 2.2 GB Connecting Ridge archive was downloaded in this environment, so no real archive SHA-256 or member manifest is claimed yet.
