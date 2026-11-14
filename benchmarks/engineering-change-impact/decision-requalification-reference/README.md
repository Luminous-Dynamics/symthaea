# Decision requalification reference tests

Standalone Rust reference tests for the synthetic K0/K1 evidence lifecycle benchmark.

From the repository root:

    cargo test --manifest-path benchmarks/engineering-change-impact/decision-requalification-reference/Cargo.toml

The package independently checks:
- decision dependency-manifest and event-inventory SHA-256 bindings;
- candidate-visible event manifest binding;
- as-of K0 versus current K1 evaluation;
- late-arriving event non-retroactivity;
- dependency-scoped invalidation;
- unknown scope, incomplete inventory and unknown event ordering;
- the eight preregistered intervention relations;
- uniqueness and integrity of the mutation-control manifest.

The package is outside the production workspace. It contains no event ingestion, trust root, temporal ledger, configuration service or authority API.

The corpus is synthetic. Tests are authored but not claimed as executed until compiled and run at the exact branch head.
