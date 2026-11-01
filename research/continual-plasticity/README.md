# Continual Plasticity Replay Baselines (v0.1)

This standalone candidate-side crate implements two deliberately simple
continual-learning replay baselines:

- fixed-capacity FIFO: retain the newest items and evict the oldest;
- reservoir admission: sample a uniformly selected seen-item index using an
  externally supplied bounded random draw.

The caller supplies the bounded draw explicitly. Reservoir admission accepts a
draw in `[0, seen_items)`; sampling uses partial Fisher-Yates draws in
`[0, retained_items-i)`. The crate does not use modulo on arbitrary random
words, so the benchmark harness must provide an unbiased bounded draw and
record the deterministic draw trace. This is reproducibility, not a claim that
the draw source is secure or statistically qualified.

Each item binds an ID, candidate-visible sequence index, task ID, payload
digest, payload bytes and replay work units. Sequence indices must be
contiguous. Timestamps are monotonic. A currently retained item ID cannot be
inserted twice. Payload digests and sizes must be nonzero.

Accounting is explicit:
- retained items and payload bytes;
- retained byte-time in byte-microseconds, integrated piecewise at observation
  times and finalized with `finish`;
- eviction and non-retention counts;
- replay selection count, logical replay payload bytes, and declared replay work.

Sampling is without replacement within one call and checks total replay work
before committing the accounting update. Admission validates all resource
arithmetic before mutating the buffer. An invalid input or budget overrun
leaves buffer state unchanged.

## Limits

This is a reference structure, not a complete continual-learning algorithm.
It does not implement task-balanced quotas, semantic retrieval, consolidation,
or learned replay prioritization. Replay work units are declared task costs,
not physical CPU/GPU or energy measurement. The evaluator must measure actual
outcomes and hardware costs separately.

No Rust build, test, formatting, Clippy, or benchmark has run for this draft.
See tracking work item NEURO-SILICON-010 and
`docs/research/CONTINUAL_PLASTICITY_REPLAY_BASELINES_V0_1.md`.
