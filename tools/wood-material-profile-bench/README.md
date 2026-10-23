# Wood Material Profile Benchmark

Standalone Rust validator for a frozen synthetic wood-species/property-applicability corpus.

From the repository root:

    cargo fmt --manifest-path tools/wood-material-profile-bench/Cargo.toml -- --check
    cargo test --manifest-path tools/wood-material-profile-bench/Cargo.toml
    cargo run --quiet --manifest-path tools/wood-material-profile-bench/Cargo.toml -- \
      docs/tech/wood-material-profile-corpus-v0.1.json \
      --source-commit "$(git rev-parse HEAD)"

The validator checks profile source identity, external-source license references, species/specimen/lot scope, moisture/conditioning basis, units, grain direction, evidence level and required measurement references.

The corpus is entirely synthetic and includes one candidate-ranking positive control plus seven deliberate negative controls. A validated result tests software semantics only; it is not botanical identification, wood grading, a structural allowable, a kiln schedule or an engineering qualification.

No third-party density or property records are bundled. Candidate datasets require file-specific license and attribution review before ingestion.
