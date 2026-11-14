# Minimal evidence benchmark reference tests

This standalone Cargo package contains a small Rust reference evaluator for the synthetic minimal-evidence corpus. It is deliberately outside the production Symthaea workspace and does not provide a production API or a source/provenance service.

From the repository root, run:

    cargo test --manifest-path benchmarks/engineering-change-impact/minimal-evidence-reference/Cargo.toml

The tests independently compute:
- candidate-visible admissible provenance roots per mandatory predicate;
- all sufficient independent-root sets in each evaluator-frozen source inventory;
- source-scope and provenance unknown states;
- common-cause collapse and derivative-not-independent reason codes;
- paired FLIP and INVARIANT relations;
- integrity of the 16 preregistered mutation-control entries.

The corpus currently contains eight synthetic cases, six executable before/after intervention pairs and sixteen preregistered mutation controls. The reference code is intended to qualify fixture semantics, not score an AI system. It is not a population reliability estimate, an engineering assurance case, or a safety qualification.

The Rust tests are authored but must not be described as passed until executed against the exact branch head. No CI result, performance result, external validation or physical engineering result is implied by this package.
