# HDC Noise Robustness Task Harness

## Purpose

This is the second task family in the HDC dimension evidence chain. It is intentionally independent from the baseline synthetic prototype-retrieval harness.

Instead of asking only whether a fixed signal is separable, it sweeps **perturbation severity at every dimension**. This exposes a two-dimensional response surface:

`dimension × noise severity → accuracy + mean decision margin`

The fixture is synthetic and deterministic. It is evidence about this controlled task family, not a general benchmark or a universal dimension recommendation.

## Protocol

Four deterministic class prototypes are created for every resolution. Each held-out query uses the same prototype/noise construction across the dimension ladder:

`query = (1 - noise_weight) × prototype + noise_weight × deterministic_noise`

The default noise ladder is:

- 0.10
- 0.20
- 0.35
- 0.50
- 0.65
- 0.80

The default dimension ladder is 1K through 256K. Each cell contains 32 queries per class (128 total).

The seed, protocol, dimension ladder, noise ladder, and sample count are committed to a canonical specification identity.

## Why this is separate evidence

The baseline harness asks whether one controlled signal survives dimensional scaling. This harness asks how the same representation behaves as signal quality is progressively degraded.

The two should not be collapsed into one score. A later analysis can compare response curves, margins, uncertainty, and resource/performance measurements while preserving task identity.

## Interpretation boundary

This harness can identify:

- noise regimes in which this synthetic task becomes less separable;
- whether response curves differ materially across dimensions;
- changes in decision margin as perturbation increases;
- deterministic regressions in the task implementation.

It cannot establish that a dimension is optimal, universally robust, or representative of production HDC workloads.

The next independent task family should probe **associative cleanup**, where the representation must recover a bound/bundled item rather than classify a query against directly supplied prototypes. Sequence/order retrieval can then test positional structure.

## Reproducibility

The fixture is deterministic for task-quality outputs. Performance timing is intentionally excluded from this module so the semantic robustness surface remains stable across machines.

