# HDC Resonator Representation-Separation Sweep

This harness isolates one representation-geometry mechanism in the coupled two-factor resonator problem: how much pairwise overlap exists between symbols in the shared codebook.

## Research question

The baseline two-factor harness uses independently generated bipolar codewords. This layer deliberately varies a **shared-component probability** while measuring the realized codebook geometry.

The generation parameter is not treated as the correlation itself. Each codeword independently copies a shared latent bipolar component with the configured probability and otherwise uses an independent component. The evidence reports the realized:

- `mean_abs_pairwise_similarity`
- `max_abs_pairwise_similarity`

This distinction matters because the realized finite-dimensional geometry is the experimental quantity, not the requested generation parameter.

## Coupled task

For each target pair `(X, Y)` the harness constructs `X ⊛ Y`, adds the configured query corruption, and gives that same observed query to:

1. exhaustive `n²` pair retrieval;
2. the coupled two-factor resonator.

Bipolar binding is commutative, so exhaustive and resonator joint correctness are **symmetry-aware**: `(X, Y)` and `(Y, X)` represent the same bound product. Factor-level X/Y decoding remains separately reported as a diagnostic.

## Metrics

The harness keeps representation geometry and inference outcomes separate:

- `mean_abs_pairwise_similarity`
- `max_abs_pairwise_similarity`
- exhaustive unordered accuracy and margin
- factor X accuracy
- factor Y accuracy
- unordered joint accuracy
- correct convergence
- spurious convergence
- non-convergence
- mean iterations
- converged-only mean iterations
- mean factor margin

No composite score or universal dimension ranking is introduced.

## Why this experiment

Recent 2026 work on HDC representation design reports that factorization becomes ambiguous when encoded values are highly correlated and that reducing correlation can restore identifiability. The same work emphasizes that cognitive retrieval and factorization need more exclusive representations than learning-oriented encodings.

Recent resonator work also treats correct convergence, spurious convergence, and non-convergence as distinct terminal states rather than collapsing them into one accuracy number.

References:

- Poduval et al. (2026), *Optimal hyperdimensional representation for learning and cognitive computation*, Frontiers in Artificial Intelligence 9.
- Yeung, Poduval, and Imani (2026), *A comparative study of nonlinear cleanup rules in resonator networks*, Frontiers in Artificial Intelligence 9.

This harness tests the representation-geometry mechanism inside Symthaea's current bipolar/shared-codebook resonator implementation; it does not establish that any particular correlation level is universally optimal.

## Default CI qualification

The bounded qualification matrix is:

- dimensions: 2K, 8K;
- codebook sizes: 4, 8;
- shared-component probabilities: 0.00, 0.50, 0.75;
- query corruption: 0.00, 0.20;
- three deterministic solver seeds;
- 16 maximum solver iterations.

That is 24 cells. Larger dimensions and geometry ladders remain extended research surfaces.

## Reproducibility

The evidence identity commits the dimension, codebook-size, shared-component, query-noise, solver-seed, fixture, query-count, and iteration ladders. The artifact digest is emitted separately.

Focused validation:

`cargo test -p symthaea-core --lib resonator_separation_harness`

Evidence example:

`cargo run -p symthaea-core --example hdc_resonator_separation`
