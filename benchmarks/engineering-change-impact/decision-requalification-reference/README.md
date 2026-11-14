# Decision requalification reference tests

Standalone Rust reference tests for the synthetic K0/K1 evidence lifecycle benchmark.

From the repository root:

    cargo test --manifest-path benchmarks/engineering-change-impact/decision-requalification-reference/Cargo.toml

The corpus currently contains eleven synthetic cases, thirteen decision outcomes, nine paired interventions and twenty preregistered mutation controls. The package independently checks:
- decision dependency-manifest and complete owner event-inventory SHA-256 bindings;
- separate K0 and K1 candidate-visible event manifest bindings;
- as-of K0 versus current K1 evaluation;
- late-arriving event non-retroactivity;
- dependency-scoped invalidation;
- relevant owner events missing from candidate-visible materialization;
- unknown scope, incomplete inventory and unknown event ordering;
- nine preregistered intervention relations;
- uniqueness and integrity of the 20-control mutation manifest.

The package is outside the production workspace. It contains no event ingestion, trust root, temporal ledger, configuration service or authority API.

The corpus is synthetic. Four Rust tests are authored but are not claimed as executed until compiled and run at the exact branch head.
