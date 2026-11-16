# SciLaws candidate-selection audit harness

A small, dependency-pinned Rust harness for compiling and testing the **exact repository source files** used by the SciLaws-Real audit:

- `crates/domains/symthaea-architecture-engineering/src/pareto_objectives.rs`
- `crates/domains/symthaea-architecture-engineering/src/scilaws_candidate_selection.rs`

The source is imported by path from `src/lib.rs`. It is not copied or forked. The harness isolates those modules from unrelated compile errors elsewhere in the stacked architecture-engineering crate.

## Reproduce

```sh
cargo fmt --manifest-path research/scilaws-candidate-selection-audit-harness/Cargo.toml -- --check
cargo test --locked --manifest-path research/scilaws-candidate-selection-audit-harness/Cargo.toml --all-targets
cargo clippy --locked --manifest-path research/scilaws-candidate-selection-audit-harness/Cargo.toml --all-targets -- -D warnings
```

The pinned lockfile is local to this harness. This does not qualify the full architecture-engineering crate or whole Symthaea workspace.

## Scope

The imported SciLaws module audits a frozen Real-track submission using separate native S_N and S_V objectives, hard contract/anti-hacking feasibility constraints, evaluator-only outcomes, failure denominators and per-metric regret. It reuses the shared Pareto implementation. It does not select candidates, access the live selector from the evaluator, or implement a new optimizer.

Parallel S_S remains a separate workstream tracked by #12522.

See [PR #12492](https://github.com/Luminous-Dynamics/symthaea/pull/12492), [issue #12489](https://github.com/Luminous-Dynamics/symthaea/issues/12489), and the [research contract](../../docs/research/SCILAWS_EVALUATOR_BLIND_SELECTION_AUDIT_V0_2026-11-16.md).
