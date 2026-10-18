# Metal Measurement Evidence Adapter

A Rust-first, fail-closed validator for source-bound manufacturing measurement datasets. The first profile is NIST AM Bench 2025 challenge AMB2025-02 (PBF-L IN718 tensile testing). This crate intentionally does not bundle NIST data.

## Status boundary

The current source manifest is metadata-only. Artifact-specific terms and local-use permissions have not been reviewed, so it declares no local artifacts and the tool rejects artifact verification until both manifest-level and per-artifact flags have been reviewed. Future artifact entries must bind source object ID/revision, access date, terms URL, byte count, digest, and explicit local-use/bundling permissions. A DOI is provenance, not a license grant.

The source profile reports eight miniature tensile specimens excised from the same original AMB2022-01 specimen, AMB2022-718-AMMT-B7-P4. Do not row-randomize those sibling specimens into train/test and call the result independent-build validation. The manifest records that lineage as one held-out target group.

## Commands

From the repository root:

    cargo test --manifest-path tools/metal-measurement-bench/Cargo.toml
    cargo run --manifest-path tools/metal-measurement-bench/Cargo.toml -- validate-manifest tools/metal-measurement-bench/schemas/amb2025-02-source-manifest-v0.1.json

After artifact-specific local-use terms have been reviewed and exact file identities, sizes, and SHA-256 digests have been recorded:

    cargo run --manifest-path tools/metal-measurement-bench/Cargo.toml -- verify-artifacts MANIFEST.json /path/to/lawfully-obtained/files

For a normalized observation set:

    cargo run --manifest-path tools/metal-measurement-bench/Cargo.toml -- validate-records MANIFEST.json OBSERVATIONS.json

The machine-readable observation contract is at
`tools/metal-measurement-bench/schemas/observation-set-v0.1.schema.json`.

Record rows preserve the source-field identity, raw numeric values and units, and censoring bounds without replacing censored values with made-up point estimates. A normalized value requires a normalized unit and explicit conversion note. Repeated source fields must maintain one variable identity and raw unit; each normalized variable uses a consistent normalized unit. Uncertainty, censoring, missingness, specimen lineage, source DOI, and calibration/held-out outcome role stay explicit. The validator rejects undeclared source DOIs, duplicate record IDs, unjustified normalization, unexplained unknown censoring, and calibration target outcomes from the held-out target lineage.

## What a successful receipt means

It establishes only that the declared metadata/record contract passed and, when applicable, that local artifact bytes match manifest SHA-256 and sizes. It is not model-accuracy evidence, material qualification, process capability, or safety approval. Model scores require a separate frozen task-specific evaluation and independent metric recomputation.
