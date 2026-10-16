# nano-policy-prototype

A standalone, research-only Rust prototype for **pre-query evidence selection**. It is deliberately outside the Symthaea production workspace and does not implement physics, an optical solver, a surrogate model, or a validated Symthaea integration.

## Why it exists

Photonic inverse-design work already has domain-native forward solvers and candidate-generation strategies. The policy hypothesis is narrower: under a fixed budget of expensive solver calls, can a deterministic pre-query selector choose more useful candidates than random, goal-only, uncertainty-only, or diversity-only selection?

External benchmark context:
- [MEEP-gated nanophotonic splitter study](https://github.com/pberlizov/nanophotonics-inverse-design) includes equal-budget MEEP comparisons and retained negative results.
- [MOCLIP measured-spectrum inverse design](https://doi.org/10.1038/s41467-026-76714-x) supplies a separate measured geometry/spectrum retrieval task.
- [NIST active-learning benchmark for materials optimization](https://www.nist.gov/publications/benchmarking-active-learning-strategies-materials-optimization-and-discovery) emphasizes equal-budget comparisons on experimentally characterized material spaces.

Do not combine these tasks or their labels/metrics into one leaderboard.

## Supported policies

- `random`: deterministic seeded-random ordering, stable under reordering the JSON candidate array.
- `goal_only`: lowest **pre-query predicted** goal error.
- `uncertainty_only`: highest pre-query uncertainty estimate. This is not a calibrated probability by itself.
- `diversity_only`: greedy maximin distance in the declared feature space.
- `hybrid`: greedy weighted sum of normalized goal fit, uncertainty, low estimated cost, and maximin diversity.

The current policies are transparent baselines, not claims that hybrid selection is superior. The weights and feature-space version must be frozen before scored evaluation.

## Interface and information boundary

The binary reads one JSON request from stdin and writes one JSON response to stdout. Errors go to stderr and exit non-zero.

- Request schema: `nano-policy-input-v1`; output schema: `nano-policy-output-v1`.
- Features must use a frozen, recorded transformation and lie in [-1, 1]. The feature space includes a lowercase SHA-256 digest of the preprocessing specification.
- Every feature vector must have the same dimension and declared feature-space semantics.
- `predicted_goal_error`, `uncertainty` and `estimated_cost` must be estimates available *before* querying the true oracle. For signals used by the policy, the request must include the matching estimator/method ID and SHA-256 digest. Unknown fields are rejected during JSON parsing; oracle labels and post-query observations are not part of the request schema.
- `eligible=false` or `already_queried=true` excludes a row. Duplicate IDs, invalid values, missing required signals and over-budget batch sizes are rejected, not silently repaired.
- `anchors` are predeclared feature vectors for already evaluated/reference candidates in the same feature space. Never include hidden outcomes in anchors.
- The selector cannot certify optical performance. A downstream domain-native forward solver (e.g. MEEP) must evaluate the chosen designs and its output must be separately recorded.

## Build and test (not yet executed in this environment)

From this directory, after installing a compatible Rust toolchain:

```bash
cargo generate-lockfile
# Review and commit Cargo.lock together with the resolved toolchain identity.
cargo fmt --check
cargo test --locked
cargo run --locked -- < examples/request.json
```

The current runtime did not have Cargo/rustc/rustfmt available, so the lockfile has not yet been generated and no compilation or tests have run. Do not claim a build/test pass until the lockfile is committed and the exact commands succeed. This prototype was authored without cargo/rustc available in the current runtime, so do not claim it compiles or that its tests have passed until the exact revision is built and the test output is captured.

## Benchmark design

1. Freeze candidate pool, feature transform/model/checkpoint, split groups, target and constraints.
2. Compare random, nearest-goal, uncertainty, diversity, conventional domain-native policy, and the hybrid policy at equal oracle-call and compute budgets.
3. Use only pre-query fields in this process. Reveal solver outputs after candidate selection and record all query failures and invalid/duplicate/rejected candidates.
4. Keep groups/replicas of the same physical geometry in a single split. For MOCLIP, prove released subset rows' official train/validation/test membership before any pretrained-checkpoint test claim.
5. Report Success@K and best-so-far target error by solver budget; include wall time, costs, invalid fraction, unique geometry count, robustness to process/geometry perturbations and independent re-evaluation.
6. Predeclare the practical-effect margin and confidence-interval method before opening held-out evaluation.
7. A simulation-benchmark pass is not fabrication verification or a safety qualification.

## Status

This is a new research tool prototype in the Symthaea nanotechnology workstream. No solver was run, no data or checkpoint was loaded, and no performance improvement is claimed.
