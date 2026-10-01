# HDC Resonator Geometry × Cleanup Temperature Interaction

This research harness crosses two previously isolated variables in the coupled two-factor resonator:

- representation geometry, generated through the existing shared-component probability fixture and measured by realized pairwise similarity;
- cleanup temperature, the existing ResonatorConfig softmax-cleanup parameter.

The interaction study is deliberately separate from the geometry-only and temperature-only sweeps. It asks whether terminal resonator behavior changes when representation separation and cleanup sharpness are varied together.

## Default qualification matrix

- dimensions: 2,048 and 8,192
- codebook sizes: 4 and 8
- shared-component probabilities: 0.0, 0.50, 0.75
- cleanup temperatures: 0.05, 0.10, 0.20
- query noise weights: 0.0 and 0.20
- solver seeds: 0x51, 0xA7, 0xD3
- queries per target pair: 1
- maximum iterations: 16

This produces 36 cells and 864 resonator trials.

## Measurements

Each cell records:

- realized mean and maximum absolute pairwise codebook similarity;
- exhaustive n² reference accuracy and margin;
- factor-level X/Y correctness;
- symmetry-aware joint correctness;
- correct convergence, spurious convergence, and non-convergence;
- mean and converged-only iteration counts;
- mean factor margin;
- terminal coupled constraint energy.

The same deterministic codebook and corrupted query are supplied to both the exhaustive reference and resonator solver. Bipolar binding is commutative, so the primary joint correctness measure is symmetry-aware rather than pretending X/Y labels are identifiable from a shared codebook.

## Interpretation boundary

This is a controlled interaction surface, not a composite score, ranking, or universal temperature optimum. A cell is not selected as a winner. Geometry and cleanup dynamics remain independently observable.

The 2026 resonator cleanup study reports that cleanup nonlinearities alter capacity and terminal failure modes, while the 2026 representation study reports that insufficient separation can produce spurious resonator assignments. This harness tests the interaction between those two mechanisms without conflating them.

## Reproducibility

The complete protocol is committed into a SHA-256 specification identity. Fixture generation, solver seeds, dimension/codebook ladders, temperature ladder, noise ladder, and iteration budget are all part of that identity. The example emits machine-readable JSON evidence and the qualification workflow validates the matrix cardinality, trial counts, outcome partition, metric bounds, and identity fields.

Production resonator defaults are unchanged.
