# SENSE-TAC-001B Rust reference oracle

This standalone Rust tool checks the 30-case synthetic tactile transducer/front-end corpus. It is a reference/evaluator program, not production sensor firmware.

## Checks

- Parses JSON with duplicate-object-member rejection.
- Validates every fixture against the closed v0.1 JSON Schema vocabulary used by the tracked corpus.
- Pins all 30 fixture IDs, family identities, and fault kinds.
- Recomputes the declared two-channel synthetic signal transforms, range effects, cross-axis coupling, time/clock relations, replay identity, shared-reference effects, calibration/installation mismatches, and representation-loss/alias cases.
- Independently derives the expected evidence, task, independence and representation dispositions from each fault kind.
- Includes mutation tests for missingness-as-zero, saturation, common-mode independence inflation, replay conflicts, HDC aliases, schema drift, timing confusion and self-test blind spots.

## Run

\`\`\`sh
cargo fetch --locked --manifest-path tools/sense-tac-001b-oracle/Cargo.toml
cargo fmt --manifest-path tools/sense-tac-001b-oracle/Cargo.toml --all -- --check
cargo test --locked --offline --manifest-path tools/sense-tac-001b-oracle/Cargo.toml
cargo clippy --locked --offline --manifest-path tools/sense-tac-001b-oracle/Cargo.toml --all-targets -- -D warnings
cargo run --locked --offline --manifest-path tools/sense-tac-001b-oracle/Cargo.toml
\`\`\`

## Claim ceiling

A passing run proves only that this exact synthetic corpus is structurally valid and that the independent reference logic reproduces its frozen known answers. It does not estimate physical sensor accuracy, uncertainty, fault prevalence, diagnostic coverage, calibration traceability, task superiority, or human safety. No candidate hardware result is imported into this oracle.
