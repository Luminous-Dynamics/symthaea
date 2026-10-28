# SCI-ETC-REPLICATION-001

Research protocol and machine-readable trial design for a budget-matched replication of *Explore, Then Commit* on NewtonBench.

- Main protocol: `docs/research/SCI-ETC-REPLICATION-001.md`.
- Run manifest schema: `research/science-etc-replication/schema/run-manifest.schema.json`.
- Non-scoring example: `research/science-etc-replication/fixtures/run-manifest.example.json`.
- Tracking issue: [#8804](https://github.com/Luminous-Dynamics/symthaea/issues/8804).

## Why replicate rather than copy the headline

The October 2026 paper reports major reductions in measurements and magnitude-based RMSLE for a bundled explore-then-commit configuration on 576 NewtonBench trials. It also notes unequal baseline/planner batch constraints, inconsistent symbolic-recovery gains, random acquisition competitiveness, incomplete prediction artifacts that can still yield finite RMSLE, and limits to zero-noise direct-output tasks. See https://arxiv.org/abs/2610.07620.

The replication therefore separates measurement-matched and interaction/compute-matched comparisons, uses eight reference arms (two native baselines plus variance/pairwise/random acquisition with and without interpreter access), and reports coverage-complete prediction error, native RMSLE, judge-based Symbolic Accuracy, measurement use, total resource cost and failures independently.

## Fixture and schema limitations

The checked-in manifest is **protocol fixture only**. Its benchmark revision, judge/model snapshots, paired seed digests and analysis-plan digest are deliberately unpinned placeholders. It cannot be used to score any policy or substantiate a performance claim.

The schema enforces strict object shapes and basic types. A later Rust validator must additionally check cross-field invariants that JSON Schema cannot adequately express by itself, including unique arm IDs, completeness of the eight-arm grid, model/task-slice links, contrast-arm references, sorted budget caps, pinned snapshot requirements for qualified runs, and zero-placeholder digest rejection.

## Proposed implementation order

1. Pin the upstream NewtonBench revision, evaluator/metric code and judge identity; reproduce a public reference artifact.
2. Validate the manifest and freeze an explicit task × arm × budget-profile × rollout-seed matrix.
3. Run a baseline-only pilot and estimate variance/failure rates.
4. Freeze powered confirmatory contrasts and multiplicity handling.
5. Run equal-measurement-budget and equal-end-to-end-budget campaigns separately.
6. Extend to noisy, embedded, operator-held-out and sealed refresh tasks only after the narrow replication is qualified.

No model performance result is included in this protocol tranche.