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
