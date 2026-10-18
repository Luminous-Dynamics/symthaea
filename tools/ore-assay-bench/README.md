# Ore Assay Benchmark Validator

Standalone Rust validator for a frozen synthetic assay-semantics and process-accounting corpus.

From the repository root:

    cargo fmt --manifest-path tools/ore-assay-bench/Cargo.toml -- --check
    cargo test --manifest-path tools/ore-assay-bench/Cargo.toml
    cargo run --quiet --manifest-path tools/ore-assay-bench/Cargo.toml -- \
      docs/tech/ore-assay-benchmark-corpus-v0.1.json \
      --source-commit "$(git rev-parse HEAD)"

The validator tests below-detection semantics, dry/wet assay bases, elemental versus oxide-equivalent grades, sample/lot lineage, analytical QC, mineral-phase mass fractions, explicit dry-mass output closure and source provenance.

The eight scenarios are synthetic. "validated" means the implementation reproduced the frozen software expectations; it is not an assay, resource estimate, mineral identification, metallurgical recovery prediction, mining authorization, or economic feasibility result.

No real assay data are bundled. Reference sources and license review are documented separately.
