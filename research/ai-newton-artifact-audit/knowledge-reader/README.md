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


## Pinned upstream syntax preflight

The pinned upstream file `data/test_cases/example_1/1/1_knowledge.txt` (Git blob SHA `b8e90a8fce1ee5af37f09f9d4f2c6d17e6086dc5`, 137,926 bytes) was inspected in full with an independent source-structure preflight:

- 5,102 total lines; 152 records in the [Knowledge] section.
- 9 ordinary symbols, 83 C_* concept entries, 60 R_* native-rule entries.
- Native rule suffixes: 36 "is conserved", 24 "is zero".
- 0 malformed records and 0 duplicate identifiers in this file.

This checks that the reader's bounded syntax model matches this pinned file. It does not verify the mathematical correctness of the expressions or native rule labels. The Rust tests have not yet been executed locally; the PR workflow is the exact-head implementation gate.
