# GEOM-001C — Dual-Projection Cognitive Trajectories

## Status

Measurement-integration tranche under #3132, stacked on GEOM-001B exact head `b562f86e16e4acf6cc17f61c7ba9a055d9620a75`.

This tranche connects the preregistered native cognitive-state projection to the GEOM-001 Fisher-Rao observatory. It does not introduce a new state representation and does not select between the two GEOM-001B projections.

## Input authority

Input is an ordered sequence of native Symthaea thought vectors. Every sample must independently satisfy GEOM-001B:

- exactly 32 coordinates;
- all values finite;
- non-zero projection mass.

If sample `k` fails projection, the entire trajectory fails and reports `k`. Invalid samples are never skipped, repaired, interpolated, or replaced.

## Mandatory dual path

For each native sample `x_t`, GEOM-001B produces:

- primary L1-magnitude simplex coordinates `p_t`;
- mandatory squared-energy sensitivity coordinates `q_t`.

GEOM-001C feeds the full `p_0...p_T` and `q_0...q_T` sequences into two independent `GeometricEmergenceObservatory` instances.

The result contains two complete `TrajectoryMetrics` records:

- `primary_l1`;
- `sensitivity_energy`.

There is deliberately no combined scalar, average, rank, or built-in robustness verdict.

## Failure semantics

Projection failures retain exact sample index and underlying GEOM-001B error.

Geometry failures retain which measurement family failed:

- `PrimaryGeometry`;
- `SensitivityGeometry`.

A one-sample trajectory therefore fails as insufficient geometry rather than being assigned an arbitrary zero-length scientific result.

## Frozen controls

### Stationary control

Repeating the same native state must yield, under both projections:

- path length = 0;
- endpoint displacement = 0;
- excess path = 0;
- geodesic efficiency = 1.

### Per-sample scale/sign invariance

Each native sample may be multiplied by its own non-zero scalar, including a negative scalar. Because both GEOM-001B projections are globally scale/sign invariant per sample, the complete geometric report must remain unchanged.

This matters because changes in native HDV projection amplitude alone must not masquerade as changes in trajectory geometry.

### Fixed coordinate permutation

Applying the same coordinate permutation to every native sample must preserve all descriptive Fisher-Rao metrics. A common relabeling of the 32 dimensions is not a cognitive effect.

### Time reversal

Reversing the sample sequence must preserve the current descriptive metrics:

- total path length;
- endpoint displacement;
- geodesic efficiency;
- excess path length;
- mean, variance, and maximum step length.

These observables are intentionally direction-agnostic. A later directional/causal trajectory statistic must be introduced as a distinct measurement, not inferred from GEOM-001C.

### Projection distinction

A concentrated two-coordinate example must produce non-zero geometry under both projections while giving different path lengths. This proves that mandatory sensitivity reporting can reveal projection dependence.

## Scientific reporting rule

Every target-system experiment using this adapter must retain both metric families.

If an intervention appears under the primary L1 representation but not under squared-energy sensitivity, or vice versa, the result is **projection-sensitive** until a preregistered statistical concordance rule says otherwise.

Do not:

- choose the projection with the larger effect;
- average the projections into a new score;
- reinterpret disagreement as replication;
- tune either projection after lesion outcomes are known.

## Claim boundary

Allowed:

> The intact and lesion trajectories differed under the L1 projection by X and under the squared-energy sensitivity projection by Y.

Allowed:

> The finding was projection-sensitive.

Not allowed:

> The preferred projection proves a geometric effect after the other projection failed.

Not allowed:

> Fisher-Rao trajectory geometry is itself consciousness.

Not allowed:

> A shared geometric pattern establishes a physical gravity-consciousness relation.

## Next gate

After GEOM-001B/001C and the intervention lineages qualify, a separate run protocol may collect real `CycleResult.thought_vector` sequences from matched intact/lesion/sham/rescue executions. That protocol must consume this frozen adapter rather than reimplementing projection or geometry locally.
