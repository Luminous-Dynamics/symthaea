# Aesthetic Truth Patch Series

This series converts `symthaea-aesthetic` from a promising heuristic feedback seed into a more auditable critic without requiring an immediate downstream rewrite.

## Application order

1. **Separate intrinsic quality from prediction surprise**
   - Adds `AestheticScore::intrinsic_composite()` and `surprise_against()`.
   - Updates expectation from intrinsic evidence only.
   - Calibrates the Birkhoff log-ratio instead of clamping every ratio above one.
   - Validates or fail-safely sanitizes feedback configuration.

2. **Make aesthetic memory durable and count-safe**
   - Adds versioned, validated JSON memory.
   - Adds observable `try_load` and atomic `try_save` paths.
   - Separates session counts from persisted lifetime counts.

3. **Ground taste learning in labeled preferences**
   - Adds explicit absolute and pairwise preference training.
   - Keeps human, population, analyst, and self-critic provenance counts separate.
   - Retains the old `train(score)` method only as a deprecated, auditable self-critic path.

4. **Make cross-modal hue math circular and fail-safe**
   - Corrects wraparound averaging and hue warmth discontinuities.
   - Removes the duplicate feedback blender implementation.
   - Reframes the equal chromatic hue wheel as a creative mapping rather than historical or universal truth.

5. **Learn harmony preference from contrastive evidence**
   - Compares quality when each harmony is active versus inactive.
   - Prevents always-active harmonies from looking valuable merely through prevalence.
   - Keeps human and self-evaluation evidence in separate persistence channels.

6. **Add deterministic calibration diagnostics**
   - Exposes a bounded grid sweep suitable for CI and evidence bundles.
   - Makes future score saturation regressions measurable.

## Compatibility notes

The existing `AestheticScore`, `AestheticTracker`, `AestheticMemory`, `TasteModel`, and synesthesia entry points remain available. The memory schema gains defaulted fields and upgrades legacy unversioned JSON on successful load.

`AestheticTracker::evaluation_count()` now explicitly means the current session. Use `total_evaluation_count()` for the persisted lifetime total.

The legacy `TasteModel::train(score)` remains source-compatible but is deprecated because it trains against the crate's own formula. New code should use `train_observation` or `train_pairwise`.

## Recommended downstream integration

Creative evaluators should populate order, complexity, harmony, and calibrated Birkhoff evidence, then call `compute_composite()` only after an explicit novelty value is available. Stateful callers should normally pass the score to `AestheticTracker::process`, which derives prediction surprise without feeding novelty back into the expectation baseline.

Human studies should prefer pairwise comparisons and preserve rater, task, modality, intent, and confidence metadata outside this compact online model. The model's source counts are an audit signal, not a replacement for a full study ledger.

## Verification status

The patch series has passed whitespace/diff validation, delimiter parsing, deterministic numerical checks, and source-level consistency review in the generation environment. A Rust toolchain was unavailable there, so workspace compilation, clippy, and the full test suite must be run in the parent Symthaea workspace before merge.
