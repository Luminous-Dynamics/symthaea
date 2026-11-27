# Particle-measure Wasserstein stability fixtures v0.1

This fixture establishes an exact, fixed-domain **one-dimensional interval W1** comparison primitive.

For probability measures on a line:
`W1(mu,nu) = integral |F_mu(x)-F_nu(x)| dx`.
The implementation sorts and coalesces atoms at construction, then sweeps both cumulative distribution functions in O(n+m).

The fixed physical interval is supplied explicitly. It is not inferred from either particle cloud's extrema. Input weights are finite and non-negative, normalized to a probability measure by the constructor. This is a one-species positive measure; signed charge / multiple species must be represented separately.

The known-answer cases cover:
- permutation invariance;
- exact co-located particle split / merge;
- particle-count change that preserves the exact weighted measure;
- a controlled movement of 0.25 probability mass by 0.1 coordinate units, giving W1=0.025.

This metric is **not** the periodic-torus W1 metric and is not a multidimensional transport solver. The current schema and fixture do not measure a field-operator error or establish a field stability constant. They provide a precise input-measure distance for future output-error-vs-W1 experiments.

The associated implementation is in the Rust crate `WeightedEmpiricalMeasure1D`; its constructor validates fixed interval membership, normalizes weights using max-scaled sums, and aggregates co-located atoms. Unit tests exercise these four cases plus malformed domains / weights and large finite weight scales.


## Executable Rust primitive

The public Rust type `WeightedEmpiricalMeasure1D`:
- requires a caller-supplied finite, strictly increasing physical interval;
- rejects particles outside that interval and invalid / all-zero weights;
- normalizes non-negative weights using a max-scaled sum to avoid overflow;
- drops zero-mass atoms and aggregates co-located atoms;
- stores atoms in sorted physical-coordinate order;
- evaluates W1 by a two-pointer cumulative-distribution sweep in O(n+m), with compensated accumulation.

The focused Rust unit suite consumes the checked-in fixture directly and verifies every declared W1 known answer. Further tests cover weight renormalization, zero-mass atoms, large finite weights, symmetry and a triangle-inequality control.

The implementation deliberately refuses to use particle-cloud extrema as a coordinate transform. The domain and units are explicit inputs. It does not yet implement torus W1, multidimensional optimal transport, or a W1-to-Coulomb-field theorem.


## Periodic torus distance

A separate companion contract now covers geodesic W1 on a 1D torus:
- schema: `particle-measure-torus-w1-v0.1.schema.json`;
- fixture: `fixtures/particle-measure-torus-w1-known-answer-v0.1.json`;
- implementation: `WeightedEmpiricalMeasure1D::wasserstein1_periodic`.

The periodic metric uses the circular CDF formula with a length-weighted median offset and returns physical distance scaled by the declared period. It identifies the endpoints and measures the shortest seam-crossing arc. Its current implementation sorts the combined wrapped supports, so it is O((n+m) log(n+m)); the interval metric remains the O(n+m) CDF sweep.

Use interval W1 for open-boundary cohorts and torus W1 only when the physical boundary contract explicitly declares periodic topology. Neither is multidimensional transport.
