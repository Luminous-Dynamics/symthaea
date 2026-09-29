# HDC-LTC Liquid Resolution Trajectory Matrix

Status: research specification; no production semantic change.

## Why this exists

The HDC-LTC liquid state is a ContinuousHV. Adaptive resolution therefore cannot be validated by static hypervector similarity alone. The liquid update computes state-dependent time constants and gating from the state/input, so a dimension conversion can perturb the dynamical trajectory even when instantaneous representation error is small.

This matrix is the empirical companion to #6531, #6536, #6541, #6560, and #6561.

## Resolution ladder

1K, 2K, 4K, 8K, 16K, 32K, 64K.

Every strict source -> target transition is evaluated, plus same-dimension identity controls.

## Experimental arms

1. identity: evolve entirely at target dimension.
2. reference: evolve entirely at the declared high-resolution reference.
3. convert_then_evolve: convert source state/parameters/input to target, then evolve.
4. evolve_then_convert: evolve at source resolution, then convert.
5. round_trip: high -> low -> high, then continue evolving.
6. legacy_dilate: existing ContinuousHV::dilate, retained only as a baseline/negative control.

## Metrics

### Static representation
- normalized L2 error
- cosine similarity
- pairwise similarity-order preservation
- binding equivariance error
- permutation equivariance error
- bundle distortion

### Liquid dynamics
At each step:
- state error
- equilibrium-state error
- effective tau error
- gating/sigma error
- one-step prediction error
- cumulative trajectory divergence
- long-horizon boundedness

Use both fixed dt and an irregular deterministic dt sequence.

### Hysteresis
For D_hi -> D_lo -> D_hi record:
- reconstruction error immediately after round trip
- trajectory divergence after 1, 10, 100, and 1000 further updates
- tau/gating recovery
- whether learned parameters remain equivalent
- whether snapshots remain valid

Round-trip loss must not be interpreted as recoverable merely because the target dimension can later be expanded.

### Cost
Record separately:
- resident bytes
- bytes moved during conversion
- allocation count
- conversion latency
- liquid evolution latency
- peak temporary memory
- regenerated/materialized representation cost

Unknown measurements remain unknown; unsupported metrics are not_applicable, never zero.

## Determinism

Every fixture must carry:
- fixture version
- source/target representation identity
- source/target dimension
- conversion-family identity/version
- seed
- input trajectory seed
- dt schedule seed/version
- configuration identity
- metric version

Identical fixture identity must reproduce identical measurements within declared floating-point tolerance.

## Important architectural constraint

Do not implement adaptive resize by constructing a new HdcLtcUnifiedNetwork and calling that a resolution transition.

A valid transition must explicitly define how these are transformed:
- neuron state
- weight hypervector
- input mask
- tau modulator
- gate weight
- gate bias
- momentum state
- layer bindings
- layer outputs
- historical snapshots
- Fourier/evolution clock state

If any component cannot be converted safely, the transition must fail closed or explicitly reset that component under a named policy.

## Candidate ordering

1. Identity control.
2. Existing ContinuousHV::dilate as legacy negative control.
3. Prefix/fold research projections where dimension divisibility permits.
4. Deterministic orthogonal/Hadamard-family projections.
5. Nested orthogonal bases.
6. Learned/task-specific maps only after representation-neutral evidence.

No candidate is production-safe merely because it minimizes static vector error.

## Acceptance

The resulting machine-readable matrix must make it possible to answer, per workload:

> What is the lowest-cost liquid representation that preserves the declared trajectory-quality target?

There is no universal winning dimension or conversion family.