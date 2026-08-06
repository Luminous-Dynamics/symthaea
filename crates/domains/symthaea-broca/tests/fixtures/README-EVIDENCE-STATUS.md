# Evidence status of the fixtures in this directory

## `measurement-baseline-v1/`, `measurement-candidate-v1/` — VOID as quality evidence

Both directories carry `"evidence_level": "measured"` / `"measured": true`. Treat that label
as **false**. They were produced before commit `1016956a47` (2026-07-25), which fixed
`BrocaGenerator::from_checkpoint_struct` rebuilding the tokenizer as
`BpeTokenizer::default_minimal()` (~256 tokens) regardless of the vocab the checkpoint was
trained with — so restored token embeddings, indexed by *training-time* IDs, were decoded
against the wrong reload-time vocabulary. Every Broca number recorded through that path
measures the bug, not the model.

Two further reasons not to read them as quality data, independent of the vocab bug:

- `cases: []` in both — there is no per-case evidence behind the aggregates.
- `num_cases: 2` / `total_exercises: 4`. Whatever these are, they are not a suite.

Nothing in the repo reads these files (verified 2026-07-28: no reference from any `.rs` or
`.sh`). They are kept, not deleted, because they remain a useful *regression stimulus* — see
below.

## `eval-canonical-v1.jsonl` — still valid

Input data, not results: 60 hand-curated `(channels, target_text)` cases. Unaffected by the
vocab bug, which corrupted decoding rather than the corpus. This is the suite to run for the
first honest post-fix baseline.

## Why the void artifacts are worth keeping

The Jul-10 distillation pair (`data/models/broca-distilled-20260710-224444.eval-*.json`,
untracked) records `avg_coherence: 1.0` — identical before and after, across all 8 intent
buckets — through a ~9,870x perplexity regression (2,244 → 22,155,560). That makes it an
excellent fixture for testing that a coherence metric actually *responds*: any instrument
that reports the same value on both halves of that pair is not measuring anything.

As of 2026-07-28 `EvalResult` carries `coherence_samples`, `min_coherence`, and
`coherence_stddev` precisely so this failure is visible in the evidence itself — a
`min_coherence` of 1.0 with `coherence_stddev` of 0.0 says "pinned" immediately, where the
mean alone read as a healthy score.

See `SYMTHAEA_BROCA_IMPROVEMENT_PLAN_2026-07-28.md` §2 for the full reconstruction.
