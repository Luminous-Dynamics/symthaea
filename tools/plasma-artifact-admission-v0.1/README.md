# Plasma Artifact Admission v0.1

**Status:** Standalone fail-closed reference library. Unit tests are authored; Rust compilation and hosted CI are pending exact-head validation.

## Paradigm

A scientific artifact is not admissible merely because a DOI resolves or a download button exists. Admission has separate dimensions:

1. **Identity:** exact record DOI, version, metadata URI and retrieval date.
2. **Rights:** the dataset's own license and action-specific permissions. Article licensing is not inherited by a linked dataset.
3. **Fixity:** repository-declared checksums are source claims. Local SHA-256 must be computed from lawfully acquired bytes and recorded separately.
4. **Scientific qualification:** format inspection, units, data semantics, split lineage and reproduction are later steps. This crate does not claim them.

Unknown rights never become permission. A repository MD5 never counts as locally computed SHA-256. Repeated repository checksums across differently named files trigger review, not a claim that the files are byte-identical.

## Current 2025 record fixture

The metadata-only fixture records what the visible Zenodo page displayed on 2026-11-19 for DOI 10.5281/zenodo.14915239:

- Linear_Landau_Damping_training_data.mat
- Noninear_Landau_Damping_training_data.mat
- Both are displayed as 41.0 MB on the page.
- Both are displayed with the same repository MD5: 6370e709ce0a825148917fabc8638a2a.
- The visible record page shows a Rights / License label but no license value.

The nonlinear filename is reproduced exactly as shown by the record page ("Noninear"), not silently corrected to "Nonlinear". The displayed 41.0 MB is rounded, so the fixture intentionally leaves exact byte size null. These are metadata observations, not byte-level findings. The duplicate MD5 may be a metadata transcription problem, a duplicated object, or another repository issue; it does not prove the two contents are equal. Do not download, execute, transform, redistribute or train on these assets until rights are verified and the two objects' independent SHA-256 digests are established.

The 2025 PNAS article's license is a separate legal object. It does not establish the license of the Zenodo dataset.

Primary sources:
- Zenodo record: https://zenodo.org/records/14915239
- PNAS article: https://doi.org/10.1073/pnas.2419073122
- Zenodo guidance on record metadata and versioning: https://help.zenodo.org/docs/deposit/about-records/
- Zenodo principles on dataset usage licenses: https://about.zenodo.org/principles/

The 2026 online-FNO dataset is a separate record, DOI 10.5281/zenodo.22800112. It must receive its own version-specific metadata, file, digest and rights audit; no 2025 file identity or license is inherited by it.

## Dispositions

- Blocked: one or more hard gates fail, including unverified rights, malformed identity, acquired files without local SHA-256 evidence, or quarantined files.
- ReviewRequired: a repository metadata anomaly exists, such as duplicate declared checksums across distinct filenames.
- MetadataOnly: declarations are coherent, rights are verified, but file bytes have not been acquired and locally hashed.
- FixityEvidenceDeclared: every acquired file declares a local SHA-256. This is only a declaration check; the crate does not read or hash bytes and does not establish scientific validity.

Every action is independently permission-gated: local inspection, executing research, creating derivatives, redistribution and commercial use. A verified license identifier alone is insufficient; every permission field must be explicit. The code does not parse legal text or decide whether a license interpretation is correct.

## Run

From the repository root:

    cargo fmt --manifest-path tools/plasma-artifact-admission-v0.1/Cargo.toml -- --check
    cargo test --manifest-path tools/plasma-artifact-admission-v0.1/Cargo.toml --locked
    cargo clippy --manifest-path tools/plasma-artifact-admission-v0.1/Cargo.toml --locked --all-targets -- -D warnings

The JSON schema and metadata-only fixture are checked in CI using a pinned JSON Schema validator.

## Explicit exclusions

No network retrieval, downloader, auto-license inference, data redistribution, MAT parser, model training, PDE solver, plasma control or experimental procedure is implemented. A passing schema or Rust unit test is a software-contract result only.
