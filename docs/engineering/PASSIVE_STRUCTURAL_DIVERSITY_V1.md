# Passive Structural Diversity v1

`symthaea-passive-diversity` provides a deliberately simple novelty measure over
deterministic CSG fingerprints.

`fingerprint_hamming_distance(a, b)` returns the fraction of the 256 digest bits that
differ. `novelty_against()` averages that distance against a reference population.

## Why this is useful

Performance-only search tends to collapse toward nearby designs. Structural novelty gives
the optimizer another signal: explore a different topology even when its current score is
slightly worse, because it may reveal a different mechanism or a different failure mode.

## Epistemic boundary

The digest is a structural identity, not a physical similarity metric. A high Hamming
distance does not imply different mechanical behavior, and a low distance does not imply
equivalent physics.

The metric is therefore a search heuristic for exploration/diversity, not a substitute for
simulation or measurement.

## Intended combination

`fitness + Pareto dominance + topology novelty + failure-memory`

This is especially relevant to inverse design, where one target response can correspond to
many valid structures and where diversity can expose multiple mechanically distinct routes
to the same function.