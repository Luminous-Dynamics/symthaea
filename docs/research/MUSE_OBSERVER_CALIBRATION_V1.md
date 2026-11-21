# MUSE observer calibration v1 — false-support and selective-risk evaluation

**Status:** the source-group family-risk extension passed exact-head formatting, all 18 tests, Clippy, CLI and Draft 2020-12 schema/fixture validation at commit `280c5e6524f44f9d01319e16f7ae53d3bf2da31d`. No real-audio calibration or benchmark win is claimed.

**Tracking:** [#12998 — Calibrate acoustic pitch evidence and benchmark false support](https://github.com/Luminous-Dynamics/symthaea/issues/12998).

## Objective

The #11564/#11617 spectral observer reports fundamental and harmonic peak-to-sideband salience in dB. These are descriptive features, not probabilities, event-presence decisions or musical correctness. A strong harmonic can arise from an octave collision, another part, resonance or timbre. Calibration must establish how much evidence a feature value carries on a source-disjoint cohort and when the system should abstain.

This tool calibrates one frozen scalar feature at a time. V1's feature is `fundamental_salience_db`. It does not classify a note as absent; it can only emit **support** above a conservative threshold. All other cases abstain.

## Three-way split protocol

The evaluator requires three disjoint splits, grouped by source work/recording lineage:

1. **Fit:** train a monotone isotonic mapping from raw dB feature to an event-presence probability.
2. **Threshold selection:** choose the lowest support threshold that satisfies the configured false-support ceiling on measured negative controls. This split cannot update the fitted calibrator.
3. **Held-out:** report the final score once, without fitting the calibrator or choosing the threshold on it.

The source group ID, source WAV digest and source MIDI digest may not cross the three splits. The group ID must identify the actual composition/work, recording session or source lineage—not the individual note. A 1,000-row note-level split of one recording is leakage, not an independent test.

The selected threshold uses **measured negative examples only**. Unmeasured negative examples remain abstentions and cannot dilute the measured false-support denominator. Held-out output separately reports false-support over all negatives and over measured negatives.

## Input evidence and claim boundary

Each row binds:
- observer ID/version and feature ID;
- sample ID and source group ID;
- exact source audio and MIDI SHA-256 digests;
- observation receipt SHA-256;
- reference/annotation receipt SHA-256 and reference type;
- split, measurement state and adjudicated event-present label;
- negative family for every absent-event case.

The input dataset also binds a frozen annotation protocol and a rights-clearance receipt.

**Important:** the dataset SHA-256 is an unkeyed content commitment. It detects changes but does not authenticate a human annotator, rights grant or upstream observer. The calibration importer must admit only #11617 observation receipts that have been replay-verified against the exact WAV/MIDI bytes. The calibration crate itself does not replay WAV/MIDI and does not turn caller-provided receipt strings into proof. Human-adjudicated labels need their own signed/versioned annotation evidence.

Rows whose observer measurement is `no_measurable_spectral_energy`, `insufficient_duration`, `truncated_by_audio`, `fundamental_above_nyquist` or `insufficient_frequency_resolution` carry no raw score and always abstain. They remain in coverage and abstention denominators.

## V1 model and report

- Isotonic regression provides a monotone mapping. It is not a multi-feature classifier and does not solve harmonic attribution.
- The model hash binds only the fit split, observer/version, feature and protocol/rights receipts. The threshold-selection split and held-out split have separate hashes in the output.
- The output reports Brier score, fixed-bin ECE, ROC-AUC, average precision, support precision, event recall, false-support rates over all and measured negatives, measurement coverage, support coverage and abstention rate. It reports held-out false-support counts/rates separately for each negative family, so octave collision, missing fundamental and cross-part contamination cannot hide inside an aggregate. The family report also counts distinct measured-negative source groups and distinct groups with at least one false support; note rows from one work do not inflate the independent denominator.
- A deterministic, seeded cluster bootstrap resamples whole source groups—not individual note rows—and reports percentile 95% intervals for support precision, both false-support rates (all negatives and measured negatives), event recall, support coverage and Brier score. The contract freezes the seed, replicate count and minimum held-out group count; a work cap bounds the resampling computation.
- Each negative family reports a one-sided nominal 95% Wilson score upper bound over independent source groups, plus `qualified_at_target`, `insufficient_evidence` or `observed_risk_exceeds_target`. This is a Wilson score bound, not an exact Clopper-Pearson interval. Bounds are marginal per family, not simultaneous family-wise coverage. Underpowered families must remain `insufficient_evidence`; zero note-level false supports do not prove a low group-level risk.
- `verify_calibration_bundle_v1` deterministically re-runs fitting, threshold selection and held-out scoring from the exact sealed dataset and contract; it rejects a modified model, threshold or metric report.
- If no threshold satisfies the false-support ceiling, the result emits no support threshold and all cases abstain. It never relaxes the ceiling to obtain a positive coverage number.

## Synthetic fixtures

- `docs/research/fixtures/muse-observer-calibration-dataset-v1.synthetic.json`
- `docs/research/fixtures/muse-observer-calibration-contract-v1.synthetic.json`

The dataset contains 39 synthetic rows across fit, threshold-selection and held-out groups, including explicit insufficient-frequency-resolution rows. Group IDs are independent of split names, and the fixture has seven held-out source groups for the cluster-bootstrap test. Placeholder digests are synthetic identifiers. This fixture exercises schema, group leakage, fit/threshold/test separation, and output replay. It is not real audio, measured instrument data, an independently adjudicated cohort or evidence of calibration quality.

## Prospective targets — not achieved

On a held-out, rights-cleared, work/performer/patch-disjoint measured cohort:
- at least 95% support precision at 80% or greater support coverage;
- measured-negative false-support rate no greater than 1%;
- ECE no greater than 0.05 after calibrating on a disjoint fit split;
- publish Brier, ROC-AUC and average precision with source-group-clustered confidence intervals;
- zero false supports on deterministic octave-collision, missing-fundamental, wrong-part and cross-part contamination negatives;
- report every observer measurement state, including abstentions, and stratify by instrument, patch, register, dynamics, SNR and overlap.

These targets require enough independent source groups for reliable clustered intervals; the 39-row synthetic fixture is only a unit test and must not be used to claim them. As a conservative planning heuristic for a 1% family risk ceiling, roughly 299 independent negative source groups with zero failing groups are needed for a one-sided 95% exact-binomial upper bound below 1%; the v1 Wilson score field is explicitly labelled as an approximate score bound, and each family must be sized and reported independently.

## Competitive and benchmark context (20 November 2026)

The benchmark must not conflate three different tasks:

- **Score-planned generation:** YuE2's September 12 WildSongBench comparison covers 192 prompts and 17 settings. YuE2 best-of-8 reports 6.9632 SongBench average versus Suno v5 at 6.8721, but metric leadership varies by column, and candidate-selection budgets differ. This is a generation-quality reference, not a score-to-audio observer benchmark. Sources: https://map-yue2.github.io/ and https://huggingface.co/datasets/m-a-p/WildSongBench/blob/main/benchmark/results.md
- **Audio editing:** the ICASSP 2027 Audio Editing Challenge lists 2,000 public MMAE examples and 17,741 rubrics, plus separate 500-example hidden Single Model and Agent tracks. It ranks by conjunctive Exact Match Rate. The official deadline is November 25, 2026; registration closed October 8 and Agent weights had to be public before November 1. No MUSE registration or eligibility is assumed. Source: https://audio-editing-challenge.github.io/
- **Production workflow:** Suno Studio 2.0 includes MIDI/piano-roll editing, automation, effects, synth and chat-based production; Moises Studio offers collaborative multitrack/MIDI, generated editable parts, stem separation and visible/reversible Agentic Mode actions. Chat, MIDI, stems and collaboration alone are not defensible MUSE differentiators. Sources: https://help.suno.com/en/articles/13670529 and https://moises.ai/newsroom/product-announcements/moises-launches-studio-collaborative-workspace/

SMEBench's human validation is an additional warning against equating automatic scores with experienced instruction-following: 17 participants gave 236 ratings; target quality averaged 4.23/5, source alignment 4.65/5 and instruction alignment 3.99/5. Three operations did not reliably exceed the midpoint for instruction alignment. Source: https://arxiv.org/abs/2610.36715

## Implementation

- Package: `tools/muse-observer-calibration/`
- Library: `src/lib.rs`
- CLI: `cargo run --manifest-path tools/muse-observer-calibration/Cargo.toml -- evaluate DATASET.json CONTRACT.json`
- Replay command: `cargo run --manifest-path tools/muse-observer-calibration/Cargo.toml -- verify DATASET.json CONTRACT.json BUNDLE.json`
- Schemas: `docs/research/schemas/muse-observer-calibration-*.schema.json`
- Exact-head workflow: `.github/workflows/muse-observer-calibration.yml`

Eighteen unit tests are authored and passed on the exact PR head, covering monotonic isotonic fitting, threshold selection, unmeasured abstention, group/artifact leakage, duplicate case IDs, invalid/non-finite feature values, no-safe-threshold behavior, held-out separation, negative-family attribution, source-group risk bounds, bundle tampering and checked-in fixture replay. The exact-head workflow also passed formatting, Clippy with warnings denied, CLI help smoke and Draft 2020-12 schema/fixture validation. This qualifies the implementation and synthetic fixture only; no real-audio calibration result is claimed.

## Next

1. Run the exact-head workflow and repair any compile/format/test issue without weakening fail-closed behavior.
2. Build a rights-cleared observer-row importer that calls #11617 replay on exact WAV/MIDI inputs before dataset admission.
3. Add real measured renderer/performer cohorts and blind adjudication; keep synthetic, renderer and real-instrument strata separate.
4. Extend the synthetic/renderer/real cohort and run leave-one-patch-out evaluation; cluster-bootstrap intervals are implemented, but the synthetic fixture is not large enough to support a research claim.
5. Compare the calibrated observer against pinned YIN/pYIN and permitted learned F0/transcription baselines on identical stems and labels.
6. Register the calibration receipt as an independent component in #12985 only after the source-group risk extension qualifies. A calibration receipt may support an acoustic claim; it cannot replace symbolic correctness, render execution integrity, score-to-audio conformance or human acceptance.
