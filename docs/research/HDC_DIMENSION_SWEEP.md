# HDC Dimension Sweep Evidence

The dimension observatory describes analytic scaling expectations; this sweep contract adds deterministic empirical measurements without treating dimensionality as a proxy for task quality.

## Sweep identity

A DimensionSweepSpec is the semantic experiment definition:

- schema version
- ordered dimensions
- deterministic seed
- samples per dimension

Its canonical bytes are hashed with the versioned domain symthaea:hdc-dimension-sweep. Execution provenance is intentionally outside the identity.

## Measurements

For each dimension the deterministic sweep records:

- mean absolute cosine between independent vectors
- maximum absolute cosine observed
- mean absolute cosine between a bound vector and one operand
- mean cosine between a three-vector bundle and a bundle member

These are structural HDC measurements, not task accuracy or a claim that one dimension is globally preferable.

## Interpretation boundary

The sweep can establish whether implementation behavior follows expected concentration/scaling patterns under a fixed generator. It cannot establish the minimum sufficient dimension for a real task.

Task-level qualification should therefore add:

1. a task/data identity,
2. task-specific quality metrics,
3. resource measurements,
4. the same experiment identity and manifest machinery used by other research evidence.

This preserves the distinction between geometric capacity, computational cost, and application quality.

## Default matrix

The default sweep covers 1K through 256K:

- Canonical: 1K, 2K, 4K, 8K, 16K, 32K, 64K
- Exploratory: 128K, 256K

The default sample count is 32 per dimension. Larger studies should change samples_per_dimension, which necessarily changes the sweep identity.
