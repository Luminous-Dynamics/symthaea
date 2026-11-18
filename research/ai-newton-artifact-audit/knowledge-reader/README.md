# AI-Newton native knowledge reader

A dependency-free Rust parser for the bounded [Knowledge] ... [end] section used by the pinned AI-Newton public artifact snapshot.

## Contract

- Preserves every parsed record's exact raw line and 1-based source line number.
- Classifies identifiers as ordinary symbols, C_<digits> concept entries, and R_<digits> native-rule entries.
- Preserves duplicate identifiers rather than merging them.
- Recognizes only the native suffixes "is conserved" and "is zero"; other rule suffixes are retained as Other.
- Stops at [end]; it does not parse later [Experiments] sections.
- Fails closed on missing headers, missing end markers, malformed records, and malformed C_* / R_* identifiers.
- Does not parse/evaluate mathematical expressions, certify conservation, or promote upstream labels into proof status.

## Run

    cargo fmt --manifest-path research/ai-newton-artifact-audit/knowledge-reader/Cargo.toml -- --check
    cargo test --locked --manifest-path research/ai-newton-artifact-audit/knowledge-reader/Cargo.toml --all-targets
    cargo clippy --locked --manifest-path research/ai-newton-artifact-audit/knowledge-reader/Cargo.toml --all-targets -- -D warnings

This parser is only the first bounded reader. General/specific model JSON, object-model semantics, memory and policy checkpoints, logs, expression semantics and independent scientific verification remain separate future tasks.

## Upstream source and license

This parser is authored for the Symthaea audit and is not copied from AI-Newton. The upstream artifact inventory and MIT license identity are pinned in ../manifest.v0.json.
