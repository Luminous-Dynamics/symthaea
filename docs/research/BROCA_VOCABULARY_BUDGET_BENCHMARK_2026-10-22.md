# Broca Vocabulary Budget Benchmark (2026-10-22)

## Purpose

This benchmark isolates the cost of the current Broca weight-tied cosine scorer as the vocabulary grows. It calls `LanguageController::compute_logits` directly, rather than timing a mocked dot product, so the current compiled path (scalar/Rayon and/or Candle when enabled) is the system under test.

It measures the efficiency side of [issue #8290](https://github.com/Luminous-Dynamics/symthaea/issues/8290). It does not change vocabulary policy, pruning behavior, token IDs, checkpoints, or production defaults.

## Measurement design

- Default HDC dimension: 16,384; f32 vector rows.
- Vocabulary sizes: 64, 256, 512, 1,024, 2,048.
- Four deterministic query vectors, two warm-up rounds, nine measured rounds per size.
- Per round, wall time is measured around four calls to `compute_logits`; reported per-query times are nearest-rank p50 and p95 across nine round samples (with nine samples, p95 is the maximum sample and should be read as a smoke tail statistic, not a stable production-tail estimate).
- The receipt also stores every sample, tokens scored/s at p50, embedding coordinates scored/s, raw embedding bytes lower bound, fixture digest, source revision when supplied, build mode, OS, architecture, logical CPU count, and compiled feature/backend mode.
- Run with `--release` and keep the machine, power mode, build features, and other workload conditions constant when comparing candidates.

## Scope and limitations

This first instrument measures the unmodified dense all-token scorer and raw embedding rows. It does not claim end-to-end generation latency, GPU speedup, peak resident memory, allocation costs, cache-transfer time, tokenization quality, or semantic quality. Raw embedding bytes exclude allocator overhead, network state, temporary logits and any duplicate/normalized accelerated cache.

The receipt records compiled Candle/Rayon feature flags, but the active Candle device/fallback choice is not currently exposed by this API. Do not compare scores across machines or builds as if the environments were identical. The benchmark should be extended with explicit backend/device reporting once the controller exposes that metadata.

## Research hypothesis

If per-step scoring time and raw storage grow approximately linearly with the vocabulary, adding token rows forever will eventually be dominated by the coverage benefit of each token. This motivates a vocabulary budget policy that allocates entries to terms that reduce held-out fragmentation or improve task utility—not merely frequent inflections that can be composed from a base form and a transformation.

The 2026 paper [Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic](https://aclanthology.org/2026.findings-acl.1618/) composes many surface forms from base-form and transformation vectors, freeing 10–40% of vocabulary slots in its tested LLM settings. Its architecture and claims do not transfer directly to Broca, but it provides a serious industry comparator.

## Comparison plan

1. Capture a baseline receipt for the current dense scorer.
2. Add a candidate sparse/hierarchical scoring implementation only behind a separate feature/flag. Verify its logits against dense scoring on exhaustive small cases and random larger cases before measuring speed.
3. Compare append-only vocabulary growth and compositional reallocation at matched memory and latency budgets.
4. On a provenance-labelled, source-separated held-out corpus, measure bytes per supported lexical item, mean fragments per term, byte-level loss or task quality, old-vocabulary regression, throughput, p50/p95 latency, raw/peak memory, and exact checkpoint-ID round-trip.
5. Keep at least five deterministic seeds where initialization is stochastic, report paired confidence intervals, and never trade away old behavior silently.
6. A claim of superiority requires a Pareto improvement (quality/coverage and efficiency) over the dense baseline and strong constituent/subword baselines, not merely faster scoring.

## Acceptance and safety gates

- Original row IDs and checkpoint mapping remain unchanged unless a separate, versioned migration is explicitly selected.
- Compaction/retirement is opt-in, reversible, and migration-tested.
- Sparse/hierarchical scoring must match dense logits within a preregistered tolerance on the exact old-token path, or its deviations must be documented and separately quality-evaluated.
- Report memory lower bounds separately from measured RSS/peak allocations.
- Performance receipts must include exact source revision and build/runtime settings.

## Validation boundary

The source and Cargo registration are committed on the PR branch. Rust compilation, formatting, runtime execution, and benchmark receipts have not been verified in this environment. Numbers in future receipts should be treated as local measurements, not vendor-neutral industry results.
