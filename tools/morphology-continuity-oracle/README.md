# Morphology Continuity Oracle v0.1

Dependency-free Rust reference evaluator for separating **current morphology** from **intended target identity** under damage.

## Run locally

```sh
cargo fmt --manifest-path tools/morphology-continuity-oracle/Cargo.toml -- --check
cargo test --manifest-path tools/morphology-continuity-oracle/Cargo.toml
cargo clippy --manifest-path tools/morphology-continuity-oracle/Cargo.toml --all-targets -- -D warnings
cargo run --quiet --manifest-path tools/morphology-continuity-oracle/Cargo.toml
```

## The identifiability fixture

This is a tiny evaluator, not a learned controller. The canonical table has 17 occupied voxel cells. The chair shares the same seat/legs and adds a four-cell backrest. Removing that backrest from a chair yields exactly the same observed cell set as an intact table.

Given that table-shaped observation and a deletion budget of four cells, both the table and chair are compatible histories. An exact observation-only classifier returns `table`, but that does not prove the historical target was a table. With no target evidence, the evaluator returns `AbstainAmbiguousHistory`. A claimed identity is separately scored against evaluator-only ground truth, so a compatible but false claim cannot hide behind a perfect current-shape score.

The deterministic JSON receipt includes the source commit, source tree SHA, and pinned request SHA-256. The fixture request is frozen in `canonical_request_v0_1.txt`; the dedicated workflow independently checks its SHA-256 before tests. The target-conditioned template score is an upper-bound oracle, not a learned result; the CLI's no-memory fixture does not invoke it.

## Scope and limitations

- Compatibility assumes cell deletions only, under an explicit maximum missing-cell budget. It is not a posterior probability.
- This fixture proves a narrow identifiability fact; it does not solve target-memory integrity, learn a repair policy, or certify constraints.
- Later controllers should compare observation-only graph/WL, local NCA, target-conditioned template oracle, symbolic identity memory, and HDC-bound target memory using the same inputs and resource budgets.
- Report current-shape classification, target identity, restoration quality, unsafe additions/deletions, abstention, and resource use separately.
- This is not a biological model, physical robot validation, or a claim of state-of-the-art performance.

## Research basis

Moreno et al., “Smart cellular bricks for decentralized shape classification and damage recovery,” *Nature Communications* 17, 5932 (2026-07-13): https://doi.org/10.1038/s41467-026-75166-7

Research issue: https://github.com/Luminous-Dynamics/symthaea/issues/8401
