# Battery formation × charge-policy co-design audit v0.1

Dependency-free Rust reference auditor for factorial formation/charging campaign manifests.

It checks exact design/formation/charge identities, full factor-pair coverage, reference-arm identity, cell/lot/line uniqueness, declared holdout-group isolation, prediction-time cutoff, right-censoring/competing-failure semantics and evidence-class/metrology requirements.

## Run from repository root

    cargo metadata --manifest-path tools/battery-formation-charge-audit/Cargo.toml --locked --no-deps
    cargo fmt --manifest-path tools/battery-formation-charge-audit/Cargo.toml -- --check
    cargo test --manifest-path tools/battery-formation-charge-audit/Cargo.toml --locked
    cargo clippy --manifest-path tools/battery-formation-charge-audit/Cargo.toml --all-targets --locked -- -D warnings

The fixture is entirely synthetic. It validates contract behavior, not electrochemical performance or a physical factorial study.
