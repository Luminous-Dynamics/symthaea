# ORE-VOI-001 benchmark runner

This small Rust crate compares four one-step measurement-selection policies on a frozen synthetic contract:

- fixed-priority order;
- maximum declared uncertainty reduction;
- estimated net decision gain (estimated gross gain minus measurement cost);
- an evaluator-only oracle using realized synthetic gains.

Run the known-answer tests and corpus validator:

cargo fmt --manifest-path tools/ore-voi-bench/Cargo.toml -- --check
cargo test --manifest-path tools/ore-voi-bench/Cargo.toml
cargo run --quiet --manifest-path tools/ore-voi-bench/Cargo.toml -- docs/tech/ore-voi-benchmark-corpus-v0.1.json --source-commit "$(git rev-parse HEAD)"

The executable emits a JSON receipt that binds exact corpus bytes and the source commit. Costs and gains are synthetic integer units, not currency, energy measurements, recovery estimates, or real plant economics.

This v0.1 is a public deterministic policy-contract fixture, not a blinded scientific challenge. The evaluator-only field is separated from the selector functions, but its values are present in the public corpus for reproducibility. Do not use this version to claim predictive performance on unseen physical data.