# Scientific candidate-selection audit v0

Dependency-free Rust reference for the evaluator-blind candidate-selection audit tracked by Symthaea #12489.

## Run

    cargo fmt --manifest-path research/scientific-candidate-selection-audit/Cargo.toml -- --check
    cargo test --locked --manifest-path research/scientific-candidate-selection-audit/Cargo.toml --all-targets
    cargo clippy --locked --manifest-path research/scientific-candidate-selection-audit/Cargo.toml --all-targets -- -D warnings
    cargo run --locked --manifest-path research/scientific-candidate-selection-audit/Cargo.toml --bin selection_audit_fixture

The selector policy receives only a canonical SelectorView containing one row per exact ready-artifact SHA-256. The raw SelectorPacket retains all generation attempts; exact artifact aliases are not separately selectable. Duplicate artifact rows with conflicting development metrics, eligibility or evidence references fail closed. Evaluator scores are accepted once per canonical artifact by audit_submission only after a frozen SelectionTrace. The receipt reports raw attempt states (ready, generation failure, execution failure, timeout, invalid, rejected, incomplete) separately from distinct artifacts, selectable candidates and evaluator outcomes. Ready candidates must have a valid model-artifact SHA-256. The upstream pool-manifest producer supplies candidate_pool_sha256; the caller must verify its binding to the packet because this crate only validates the digest shape and trace echo. Identical artifact digests are grouped and counted separately from attempts; this is not semantic equivalence detection. The package audits per-metric Pareto domination/regret and keeps failed/missing outcomes in the denominator. It does not implement a Pareto optimizer or candidate generator.

Important: this API boundary is not an OS sandbox. Production runs must isolate evaluator artifacts and hidden simulator state with separate process/storage permissions. The upstream SciLaws dataset is not vendored; its per-task licenses must be checked individually.

See the research contract at docs/research/SCIENTIFIC_CANDIDATE_SELECTION_AUDIT_V0_2026-11-16.md and issue #12489.
