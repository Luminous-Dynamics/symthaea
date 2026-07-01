# Experiment Matrix Reports

Alpha.6 adds an experiment matrix runner for repeated binding comparisons across dimensions and noise levels.

The purpose is not to prove quantum advantage. The purpose is to make local probes less anecdotal by asking whether a pattern survives multiple dimensions, multiple noise settings, and multiple deterministic seed replicates.

## What the matrix records

Each matrix cell records:

- dimension
- noise level
- replicate count
- classical noisy similarity mean
- phase noisy similarity mean
- correlation noisy similarity mean
- paired effect-size summaries already produced by the comparative runner

## Recommended use

Start small:

- dimensions: 128, 256, 512
- noise: 0.0, 0.05, 0.10, 0.20
- trials: 8 or 16
- replicates: 4 or 8

Then increase dimensions and replicates only after local verification is passing.

## Interpretation rule

A matrix trend is a hypothesis generator, not a conclusion. Any promising pattern should be rerun with a larger replicate count, an explicit negative control, and a written claim boundary.
