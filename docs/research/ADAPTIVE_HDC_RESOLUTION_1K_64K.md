# Adaptive HDC Resolution: 1K–64K

Status: experimental research specification.

## Resolution ladder

| Dimension | Binary bytes | Relative work vs 16K |
|---:|---:|---:|
| 1,024 | 128 | 0.0625x |
| 2,048 | 256 | 0.125x |
| 4,096 | 512 | 0.25x |
| 8,192 | 1,024 | 0.5x |
| 16,384 | 2,048 | 1x |
| 32,768 | 4,096 | 2x |
| 65,536 | 8,192 | 4x |

The relative-work column is a first-order byte/bit-operation model, not a measured performance prediction.

## Research question

What is the minimum HDC resolution that preserves a declared task-quality target?

The answer must be measured separately for each workload. A larger hypervector is not assumed to be better.

## Compatibility experiment

Compare four ways to derive lower resolutions from a common 64K master representation:

1. Prefix/truncation.
2. Deterministic orthogonal/Hadamard-submatrix projection.
3. Independent seeded random projection.
4. Explicit nested basis construction.

For each family, measure:

- similarity preservation;
- binding/unbinding behavior;
- permutation behavior;
- bundling fidelity;
- retrieval rank stability;
- interference/collision rate;
- noise tolerance;
- compositional depth;
- promotion/demotion error.

A projection is not considered compatible merely because it produces vectors of the requested size.

## Adaptive controller

The controller should target a declared quality threshold and select the smallest sufficient resolution.

Candidate escalation signals:

- retrieval margin below threshold;
- rising collision/interference rate;
- bundling saturation;
- increasing compositional depth;
- sequence length;
- prediction error;
- uncertainty;
- memory/latency/energy budget.

Candidate demotion signal:

- quality margin remains above target for a stability window.

Promotion and demotion must be deterministic under identical inputs, configuration, and seed.

## Benchmark matrix

Every run should record:

- git revision;
- Rust/toolchain version;
- CPU and ISA;
- OS;
- dimension;
- representation/projection family;
- operation/workload;
- seed;
- warm/cold state;
- repetitions;
- latency distribution;
- allocations;
- peak memory;
- CPU cycles where available;
- energy measurement method where available;
- quality metric;
- retrieval margin.

Export machine-readable records so Pareto frontiers can be regenerated without scraping terminal output.

## Safety boundary

The production BinaryHV type is currently a fixed 16,384-bit value with many downstream consumers assuming 2,048 bytes. The first implementation should therefore avoid changing its ABI or silently accepting smaller vectors.

Instead, introduce an explicit experimental representation/resolution layer. Only promote a design into BinaryHV after compatibility and downstream-audit evidence exists.

## Initial hypotheses

- Lower dimensions will materially reduce memory traffic and improve cache residency.
- Quality will saturate for some tasks well below 16K.
- Some tasks will require 16K or above.
- 64K may provide additional interference/compositional capacity, but its 4x bit-work and 4x binary storage versus 16K must be justified by measurable quality gains.
- A compatible projection family can make adaptive resolution substantially safer than ad-hoc truncation.

These are hypotheses, not results.

## Exit criteria

The research is complete when Symthaea can answer, for a declared workload and quality target:

> What is the cheapest resolution and representation that meets the target?

No universal dimensionality recommendation should be inferred from a single benchmark.