# Metal Measurement Evidence Adapter

A Rust-first, fail-closed validator for source-bound manufacturing measurement datasets. The first profile is the historical NIST AM Bench 2025 challenge AMB2025-02 (PBF-L IN718 tensile testing). Its original submission deadline was August 29, 2025; this repository uses it as a public scientific-evaluation profile, not an active challenge submission. This crate intentionally does not bundle NIST data.

## Status boundary

The current source manifest is metadata-only. Artifact-specific terms and local-use permissions have not been reviewed, so it declares no local artifacts and the tool rejects artifact verification until both manifest-level and per-artifact flags have been reviewed. Future artifact entries must bind source object ID/revision, access date, terms URL, byte count, digest, and explicit local-use/bundling permissions. A DOI is provenance, not a license grant.

The NIST source register separates the AMB2022-01 modeling-description, thermography, microstructure, residual-strain/stress/deflection, and serial-sectioning/XCT DOI records. These are candidate upstream references, not a claim that every file is used; exact feature/field selection must be reconciled to the official submission template. The source profile reports eight miniature tensile specimens excised from the same original AMB2022-01 specimen, AMB2022-718-AMMT-B7-P4. Do not row-randomize those sibling specimens into train/test and call the result independent-build validation. The manifest records that lineage as one held-out target group.

## Commands

From the repository root:

    cargo test --manifest-path tools/metal-measurement-bench/Cargo.toml
    cargo run --manifest-path tools/metal-measurement-bench/Cargo.toml -- validate-manifest tools/metal-measurement-bench/schemas/amb2025-02-source-manifest-v0.1.json

After artifact-specific local-use terms have been reviewed and exact file identities, sizes, and SHA-256 digests have been recorded:

    cargo run --manifest-path tools/metal-measurement-bench/Cargo.toml -- verify-artifacts MANIFEST.json /path/to/lawfully-obtained/files

Inspect a specific manifest-declared CSV after every artifact in the manifest has been placed under the artifact root:

    cargo run --manifest-path tools/metal-measurement-bench/Cargo.toml -- inspect-csv MANIFEST.json ARTIFACT_DIR relative/path.csv

Inventory a manifest-declared ZIP archive without extracting any member:

    cargo run --manifest-path tools/metal-measurement-bench/Cargo.toml -- inspect-zip MANIFEST.json ARTIFACT_DIR relative/path.zip

Reconcile exact source-native row IDs and an explicit mapping file against the manifest-declared CSV:

    cargo run --manifest-path tools/metal-measurement-bench/Cargo.toml -- reconcile-csv MANIFEST.json ARTIFACT_DIR MAPPING.json relative/path.csv

For a normalized observation set:

    cargo run --manifest-path tools/metal-measurement-bench/Cargo.toml -- validate-records MANIFEST.json OBSERVATIONS.json ARTIFACT_DIR

The `inspect-zip` command inventories safe member paths, compressed/uncompressed sizes, CRC-32 metadata and directory flags from a verified archive without extracting members. It rejects absolute/traversal/backslash paths, repeated or dot path components, Windows-reserved or normalization-ambiguous names, symlinks, duplicate names, case-colliding paths and archives with more than 100,000 members. The receipt is bound to the exact outer archive SHA-256. Member CRC-32 values are not cryptographic per-file digests, and this inventory is not a license decision or a semantic data parse.

Machine-readable contracts:
- `tools/metal-measurement-bench/schemas/measurement-manifest-v0.1.schema.json`
- `tools/metal-measurement-bench/schemas/observation-set-v0.1.schema.json`
- `tools/metal-measurement-bench/schemas/audit-receipt-v0.1.schema.json`
- `tools/metal-measurement-bench/schemas/csv-mapping-v0.1.schema.json`

The CSV inventory is deliberately structural: it records the exact CSV digest, parsed-record count, ordered header names and per-column empty/nonempty counts. It rejects blank/colliding headers and inconsistent row widths. It does not infer measurement units, semantics, specimen IDs or upstream record IDs, and must not be reported as a complete row-level reconciliation.

The `inspect-csv` command parses the verified byte buffer (after rechecking its declared size and digest) and emits a source-bound CSV structure inventory: exact artifact SHA-256, record count, header names/positions, empty/nonempty cell counts, and strict quote/duplicate-header rejection. It does not infer units, semantic meaning, specimen identity, or upstream row IDs. Parsed row ordinals are not claimed to be source-native identifiers.

The `reconcile-csv` command accepts an explicit mapping contract. It checks that the chosen CSV path is manifest-declared, that the source-native row-ID header and every mapped header exactly exist, that source row IDs are nonblank and unique, that canonical field IDs are unique, and that each mapped field has an explicitly declared raw unit, role, and missing-cell policy. **Every header other than the row-ID column must either be mapped or listed under `excluded_columns` with a nonblank reason.** A hidden/unaccounted-for column blocks reconciliation. The receipt binds exact artifact and mapping bytes, header digest, ordered source-row-ID digest, row counts, per-field present/missing counts, and per-excluded-column empty/nonempty counts to a versioned receipt. Mapped fields declare a value kind (`text` or `finite_number`) and an exact list of source-specific missing tokens. Malformed/non-finite numbers are rejected; an undeclared `NA`/sentinel is not assumed to mean missing. Empty cells or declared sentinels either fail under `reject` or remain explicitly counted under `preserve_missing`; they are never imputed. Explicit exclusions remain visible with a reason and source-cell counts. The command does not parse numeric values, validate the physical meaning of units, infer specimen lineage, or produce model observations. Exact source headers and units must be filled from an inspected real source artifact, not guessed.

Record rows must bind the manifest's source-object ID and upstream row ID as well as the DOI and source-field identity. The exact source object must be declared, term-reviewed, locally authorized, and linked to a declared artifact whose byte count and SHA-256 are verified before record validation. The current `reconcile-csv` stage proves the declared row identities and mapped headers exist in the verified bytes and accounts for every parsed row in its receipt. It does not yet define dataset-specific row exclusions or feature transformations; those must be explicit in a later profile-specific adapter. Raw numeric values and units, and censoring bounds, are preserved without replacing censored values with made-up point estimates. A normalized value requires a normalized unit and explicit conversion note. Repeated source fields must maintain one variable identity and raw unit; each normalized variable uses a consistent normalized unit. Uncertainty, censoring, missingness, specimen lineage, source DOI, and calibration/held-out outcome role stay explicit. The validator rejects undeclared source DOIs, duplicate record IDs, unjustified normalization, unexplained unknown censoring, and calibration target outcomes from the held-out target lineage.

## What a successful receipt means

It establishes only that the declared metadata/record contract passed and, when applicable, that local artifact bytes match manifest SHA-256 and sizes. It is not model-accuracy evidence, material qualification, process capability, or safety approval. Model scores require a separate frozen task-specific evaluation and independent metric recomputation.
