# Melothaea Counterfactual Performance Metrics v1

**Status: report harness authored; execution is not yet verified.** This tranche adds a public performance-only override and a deterministic JSON benchmark for the Mazurka/Toccata dialect experiments. It measures implementation effects, not musical merit.

## Research motivation

Modern music evaluation emphasizes blinded human preferences and prompt-matched comparisons. MIREX 2026's CMI-RewardBench compares automated preference predictors against listener preferences; its results page reports 73.9% core-set accuracy for the leading submission, 72.7% for SongEval average, and 65.9% for the CMI-RM baseline. These results evaluate reward models—not music generators—and form a suitable reference for a later Melothaea critic evaluation.

The Artificial Analysis music methodology uses blind pairwise voting, prompt-matched comparisons, loudness normalization, confidence intervals, and separately reported instrumental/vocal leaderboards. Its stated default is one track per prompt, with consistent settings where providers expose them. MIREX 2026's symbolic-generation task instead focuses on a four-bar prompt plus twelve-bar continuation and evaluates prompt coherence, creativity, structure and overall musicality. These are distinct tasks; their scores should not be conflated.

Sources:
- MIREX 2026 CMI-RewardBench results: https://music-ir.org/mirex/wiki/2026:Music_Evaluation_via_CMI-RewardBench_Results
- Artificial Analysis, Music Generation Benchmarking Methodology: https://artificialanalysis.ai/methodology/music
- MIREX 2026 Symbolic Music Generation: https://www.music-ir.org/mirex/wiki/2026:Symbolic_Music_Generation
- Grötschla et al., *Benchmarking Music Generation Models and Metrics via Human Preference Studies* (ICASSP 2025): https://arxiv.org/abs/2506.19085

## Paradigm shift: controlled interventions with machine-readable evidence

A named timing dialect and a constants test do not prove that the choice reaches actual performance events. The report harness compares the style-derived dialect against explicit `ClassicalRubato` for the same generated score. It holds score digest, seed, spec, ensemble-selection rule, swing, return-color policy and cognitive state fixed.

A public `perform_with_spec_and_dialect` API supplies the controlled performance-only intervention without synthesizing audio. The existing `perform_with_spec` remains the automatic style-derived path and delegates to the same internal implementation, preventing a second performance code path from drifting.

The benchmark covers:
- Styles: Mazurka and Toccata.
- Seeds: `[0, 1, 7, 13, 29, 42, 73, 101]`.
- Reproducibility: exact event signature replay.
- Evaluator negative control: repeat the exact same `ClassicalRubato` arm; the report fails unless its replay is bit-identical, every event is matched, and measured onset change is zero. Unit tests also verify that the ordered aligner detects a 1-second systematic timing shift instead of dropping those notes as unmatched.
- Provenance: optional source revision from `MUSE_SOURCE_REVISION` or `GITHUB_SHA`, plus the `rustc --version` string when available; missing data is reported as null rather than guessed.
- Integrity: nonempty output and finite, in-range event fields.
- Counterfactual: actual style-selected dialect versus `ClassicalRubato`, same score digest.
- Matching: order-preserving longest-common-subsequence alignment by voice identity and exact frequency. No fixed time window is used, so large cumulative rubato shifts remain measurable; unmatched events are reported.
- Pair counts: matched and unmatched events in each arm are reported rather than hidden. Both arms must preserve at least 90% of events under ordered pitch/voice alignment; lower coverage fails the mechanics gate rather than permitting a sparse false-positive comparison.
- Effect size: onset shift distribution (mean, median, p95), changed-onset fraction (a shift must exceed 0.1 ms), plus median duration, inter-onset interval statistics and event validity counts.

Usage:

```bash
cargo fmt --manifest-path crates/domains/symthaea-muse/Cargo.toml -- --check
cargo run --release -p symthaea-muse --features theory \
  --example performance_counterfactual_benchmark -- /tmp/melothaea-counterfactual.json
```

The command emits 16 case rows (2 styles × 8 seeds) to a versioned JSON report and exits nonzero if any case fails its mechanical gates. The report explicitly records `audio_rendered: false` and `human_quality: "not_measured"`.

## Interpretation boundary

A mechanics pass means:
1. identical automatic-path replay for the same score and seed;
2. nonempty and valid performed event streams in both arms;
3. at least one matched onset differs from the baseline.

It does **not** show that the timing shift is perceptually meaningful, historically authentic, stylistically recognizable, or preferred. The matching procedure aligns note pitch sequences while preserving order; repeated pitches can remain ambiguous, so it is a descriptive heuristic rather than formal note identity. The 90% match-coverage threshold is a mechanical guard, not an artistic-quality score. The report must not be used as a reward model.

## Next gates

1. Run format, compile and the exact report harness against the exact PR head.
2. Archive JSON output with source commit, Rust toolchain, and score digests.
3. Render the same cohorts with the actual and control dialects and measure audio integrity, loudness, peak, dynamics, and render warnings under the same renderer.
4. Build randomized, level-matched blind comparisons. Keep style recognition, rhythmic fit, coherence, expression, and overall preference as separate questions.
5. Tune timing parameters on a development set only; evaluate once on held-out seeds/prompts.
6. Later evaluate an automated critic on held-out pairwise listener preferences using accuracy, calibration, subgroup breakdowns, and confidence intervals.

No benchmark run, render, human rating, or CI pass is claimed here. Skipped checks are not passing evidence.
