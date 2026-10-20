# Melothaea Performance Counterfactual Audit v1

**Status: regression contract authored; execution and listening not yet verified.** This audit tests whether style-selected performance dialects affect real performed events, not whether the resulting music is better or historically authentic.

## Paradigm shift: check the behavior, not just the label

A test of a dialect enum and its scale factors can pass even if the selected route never reaches the actual performed notes. The additional regression test therefore follows the production seam:

1. Choose Mazurka or Toccata and its real `CompositionSpec`.
2. Compose one symbolic score for a fixed seed and intent.
3. Resolve the same ensemble once.
4. Call the spec-driven performance path twice to prove deterministic replay.
5. Compare it to a `ClassicalRubato` counterfactual while holding score, ensemble, swing, return-color policy, and cognitive state fixed.
6. Require at least one matched note onset to move for each of eight fixed seeds.
7. Assert finite, positive frequency/duration, nonnegative finite start time, finite velocity in range, and nonempty event streams.

The cohort uses seeds `[0, 1, 7, 13, 29, 42, 73, 101]` for both styles. Note counts are not forced to match: the performance layer includes a monophonic overlap guard that may legitimately remove a note if jitter causes a collision. Common event prefixes are compared for onset movement.

## What this test proves—and does not

A passing test would establish that the declared dialect has a deterministic, observable timing effect along the real performance path and that the exercised outputs satisfy basic event-integrity constraints. It would **not** establish that the effect is large enough, musically correct, more expressive, stylistically recognizable, or preferred by listeners.

The scale factors remain provisional engineering hypotheses. Mazurka timing has not been calibrated on a reviewed performance corpus; Toccata timing has not been calibrated against keyboard-performance data. Expert judgment and listening remain necessary.

## Evaluation plan after mechanical validation

- Render matched Mazurka/Waltz/Minuet and Toccata/Classical/Baroque Suite cohorts with score length, seed set, renderer and mastering policy pinned.
- Include a dialect-on vs same-score ClassicalRubato counterfactual so the performance effect is isolated from compositional differences.
- Publish per-sample feature rows: score/audio digests, note/event counts, inter-onset intervals, timing deviation, duration distributions, velocity/dynamic range, peak and loudness, warnings and failures.
- Reject non-finite events, unexpected empty outputs, clipping, invalid durations, nondeterministic replay, or a declared control with zero observed effect.
- Build a level-matched blind listening pack only after these analytic gates pass. Ask about timing fit, style recognition, coherence, expression and preference separately.
- Hold out seeds/prompts for tuning and final evaluation. Report all tested arms, not only the winner; do not compare best-of-N output to a one-shot baseline without a matched candidate budget.

## Focused verification

```bash
cargo fmt --manifest-path crates/domains/symthaea-muse/Cargo.toml -- --check
cargo test -p symthaea-muse --features theory --lib theory_realize::tests::style_dialect_changes_reach_real_performed_events_across_fixed_seeds
cargo test -p symthaea-muse --features theory --lib theory_realize::tests::dialect_derivation_maps_every_style_preset
```

No test, render, external benchmark, or listener result is claimed here. Skipped CI remains unknown, not a pass.
