# HDC Coupled Resonator Cleanup-Temperature Sweep

## Purpose

This research harness isolates the softmax cleanup temperature used by ResonatorNetwork. The production default remains unchanged (temperature = 0.1).

The experiment is deliberately separate from dimension and representation-geometry experiments. It asks a narrower question:

> How does cleanup temperature affect two-factor resonator factorization when the codebook, query fixtures, dimensions, and solver seeds are held fixed?

The 2026 resonator literature treats cleanup as a meaningful part of resonator dynamics and reports that cleanup nonlinearities can change both capacity transitions and terminal failure modes. This harness therefore records correct convergence, spurious convergence, and non-convergence separately rather than reducing them to a single accuracy score.

## Default matrix

- dimensions: 2048, 8192
- codebook sizes: 4, 8
- temperatures: 0.05, 0.10, 0.20
- query-noise weights: 0.0, 0.20
- solver seeds: 0x51, 0xA7, 0xD3
- queries per target pair: 1
- maximum iterations: 16
- cells: 2 × 2 × 3 × 2 = 24
- resonator trials per cell: n² × 3

The exhaustive reference uses the same bound query and searches all n² ordered pairs. Because the HDC binding used here is commutative, the primary factorization correctness measure is symmetry-aware unordered joint correctness.

## Metrics

Each cell records:

- exhaustive reference accuracy and mean margin
- factor-X correctness
- factor-Y correctness
- symmetry-aware joint correctness
- correct convergence
- spurious convergence
- non-convergence
- mean iterations
- converged-only mean iterations
- mean factor margin
- mean terminal joint constraint energy

The evidence also records the exact temperature ladder, solver seeds, scenario revision, specification digest, and artifact digest.

## Methodological boundaries

This is **not**:

- a universal optimal-temperature claim
- a dimension ranking
- a combined score across dimensions
- a claim that lower or higher temperature is intrinsically better
- a comparison of unrelated cleanup nonlinearities

Temperature is treated as one controlled dynamical parameter. Any observed trade-offs remain empirical properties of this fixture family.

## Reproducibility

The fixture codebook and noisy queries are deterministic. Solver initialization is seeded through solve_system_seeded. The canonical specification is hashed into spec_identity; serialized evidence is hashed into artifact_digest.

Run the evidence example with:

    cargo run -p symthaea-core --example hdc_resonator_temperature

The CI workflow validates the schema, matrix identity, trial counts, outcome partition, metric bounds, and evidence digests.
