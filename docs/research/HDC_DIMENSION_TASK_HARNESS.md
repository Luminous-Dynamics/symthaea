# HDC Dimension Task Harness

## Purpose

The dimension observatory and structural sweep establish how hyperspace geometry changes with resolution. The dimension frontier manifest establishes how task quality, performance, and resource evidence can be joined without copying measurements.

This harness supplies the first executable task layer between those contracts.

It evaluates a deterministic synthetic nearest-prototype retrieval task at:

- 1K, 2K, 4K, 8K, 16K, 32K, 64K
- 128K and 256K exploratory resolutions

The same fixture, seed, class count, query protocol, representation, and model revision are held constant. Only the continuous-HV resolution changes.

## Task

Four deterministic prototype hypervectors represent four classes. Each held-out query is:

`query = 0.80 * class_prototype + 0.20 * deterministic_noise`

Inference selects the prototype with maximum cosine similarity.

The default run evaluates four queries per class, giving 16 held-out queries per dimension.

This is deliberately a controlled representation-scaling experiment, not a claim about general ML accuracy or a substitute for a public benchmark dataset.

## Evidence emitted

Each dimension produces three existing evidence records:

1. **Task quality** — accuracy over the held-out synthetic queries.
2. **Performance** — measured wall-clock time for the complete dimension-specific task run plus explicit logical-byte accounting.
3. **Resource** — declared continuous-f32 vector/resident working-set requirements and budget qualification.

Each record is content-addressed by SHA-256 and referenced by a `DimensionFrontierManifest` row. The frontier contains references only; it does not copy or rank measurements.

Execution provenance remains separate from semantic task identity.

## Interpretation boundary

The harness can answer questions such as:

- whether this controlled task remains separable at a given resolution;
- how measured execution time changes across the same workload;
- how logical memory requirements scale;
- where a later analysis might investigate saturation or diminishing returns.

It cannot, by itself, establish that a dimension is optimal, universally sufficient, or superior for another task.

The correct next layer is a small family of independent task fixtures with different signal structures (for example associative cleanup, sequence/order retrieval, and noisy classification). A dimension-specific conclusion should require agreement across those task families rather than a single synthetic workload.

## Reproducibility

The fixture has a versioned scenario identifier, explicit seed, fixed dimension ordering, fixed protocol, and canonical specification identity.

Performance measurements are intentionally not treated as deterministic semantic outputs: elapsed time depends on the execution environment and is carried as provenance-bearing evidence.

## CI

The HDC resolution qualification workflow executes the harness, validates all nine dimension rows, checks evidence status and digest shape, and uploads the machine-readable JSON artifact.

The workflow does not turn task results into a pass/fail dimension ranking.
