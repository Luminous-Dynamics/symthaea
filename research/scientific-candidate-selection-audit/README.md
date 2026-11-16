# Scientific candidate-selection audit v0

Dependency-free Rust reference for the evaluator-blind candidate-selection audit tracked by Symthaea #12489.

## Run

    cargo fmt --manifest-path research/scientific-candidate-selection-audit/Cargo.toml -- --check
    cargo test --locked --manifest-path research/scientific-candidate-selection-audit/Cargo.toml --all-targets
    cargo clippy --locked --manifest-path research/scientific-candidate-selection-audit/Cargo.toml --all-targets -- -D warnings
    cargo run --locked --manifest-path research/scientific-candidate-selection-audit/Cargo.toml --bin selection_audit_fixture

The selector API sees only SelectorPacket; evaluator scores are accepted by audit_submission only after a frozen SelectionTrace. The package audits per-metric Pareto domination/regret and keeps failed/missing outcomes in the denominator. It does not implement a Pareto optimizer or candidate generator.

Important: this API boundary is not an OS sandbox. Production runs must isolate evaluator artifacts and hidden simulator state with separate process/storage permissions. The upstream SciLaws dataset is not vendored; its per-task licenses must be checked individually.

See the research contract at docs/research/SCIENTIFIC_CANDIDATE_SELECTION_AUDIT_V0_2026-11-16.md and issue #12489.
