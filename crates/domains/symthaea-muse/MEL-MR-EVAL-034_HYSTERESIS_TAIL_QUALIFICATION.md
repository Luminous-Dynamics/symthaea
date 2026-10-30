# MEL-MR-EVAL-034 — calibrated hysteresis and explicit tail qualification

**Status:** versioned EVAL-017 v0.2 segmenter implemented; six new segmenter tests authored; EVAL-025 adapter upgraded to v2; no compilation, runner, real FluidSynth render or empirical support-quality result claimed.

## Why this is separate from EVAL-017 v0.1

EVAL-017 v0.1 must remain replay-compatible. It independently tests each PCM output window/channel against the absolute and no-op-calibrated thresholds, using RMS **or** spectral excess, and merges all raw exceedance windows. The v0.2 work adds temporal post-processing without rewriting the v0.1 report or threshold semantics.

The relevant sound-event detection literature treats event onset/offset post-processing as a material part of measurement. Two-threshold hysteresis, minimum duration and gap handling can change interval boundaries and event-based F-scores. DCASE's event-based metrics also use explicit boundary collars; PSDS/t-PSDS provide complementary threshold-sweep views.

TimingBench (August 2026) is a useful adjacent music-temporal methodology: it includes machine-readable event schemas and a deterministic detector cascade across 250 prompts. Its own detector median localization error varies by task family (reported range 8–218 ms); its independently human-annotated validation set was still under construction. It is a method reference, not a gold-standard editing benchmark.

References:
- DCASE sound-event detection post-processing review: https://doi.org/10.1561/116.00000051
- Evaluation of post-processing algorithms for polyphonic SED: https://arxiv.org/abs/1906.06909
- TimingBench: https://taan.ai/research/rushing-or-dragging/
- DCASE 2023 sound event detection results: https://dcase.community/challenge2023/task-sound-event-detection-with-soft-labels-results

## Paradigm shift: preserve all three temporal facts

Do not collapse measurement, segmentation and safety padding into one interval list:

1. **Raw measured exceedance** — exact EVAL-017 v0.1 threshold results and merged windows.
2. **Hysteresis-qualified support** — temporal state-machine output using explicit onset/release thresholds and persistence.
3. **Conservative tail guard** — separately recorded padding after measured/hysteresis support, bound to a frozen profile and explicitly *not* called measured influence.

EVAL-017 v0.2 preserves per-window RMS and spectral values, thresholds and boolean decisions so any temporal boundary can be explained and replayed.

## Implemented v0.2 API

In `src/counterfactual_influence.rs`:

- `InfluenceThresholdCombination`: explicit `AnyMetricExceeds` (default, preserves v0.1 raw support) or `BothMetricsExceed` (strict ablation).
- `InfluenceSegmentationConfig`: threshold combination, release ratio, release persistence count, tail guard frame count, optional tail-profile digest.
- `InfluenceWindowSegmentation`: independent onset/release RMS and spectral booleans, combined decisions, per-channel hysteresis state and thresholds.
- `CounterfactualInfluenceSegmentedReport`: embeds the exact v0.1 report, the full window measurements, raw intervals, hysteresis intervals, tail guard intervals, conservative support intervals, clipping flag and v0.2 receipt.
- `measure_counterfactual_influence_segmented`: new v0.2 measurement path.
- `verify_counterfactual_influence_segmented`: replays the v0.1 measurement, raw window trace, hysteresis, tail guard and report digest from original renderer-bound inputs and the verified no-op calibration.

The old `measure_counterfactual_influence` and `verify_counterfactual_influence` keep their v0.1 report semantics.

## Hysteresis semantics

For each channel independently:

- onset threshold is the existing calibrated RMS and spectral threshold;
- release threshold for each metric is `max(absolute_floor, calibrated_onset_threshold × release_threshold_ratio)`;
- onset and release decisions combine the two independent metric booleans using the declared `InfluenceThresholdCombination`;
- once active, the state remains active for the configured number of consecutive below-release windows;
- the first below-release window beyond the persistence count closes the segment at the end of the last retained window;
- channel intervals are unioned only after temporal segmentation.

The default release ratio is 0.5 and persistence is two windows. These are initial research defaults, not proven optimal values. The strict `BothMetricsExceed` policy is included as an ablation so precision/recall trade-offs can be measured rather than hidden.

## Tail guard semantics

- `tail_guard_frames=0` means no additional guard.
- A non-zero guard requires `tail_guard_profile_sha256`, which commits the report to the declared renderer/effect-tail calibration profile.
- Each guard interval begins at a hysteresis interval's exclusive end and extends by the configured number of output frames, clipped only for reporting.
- The report preserves the un-clipped requested end through `tail_guard_was_clipped`. If the requested guard passes the rendered PCM horizon, the flag becomes true even if the emitted guard interval is shorter or empty.
- The EVAL-025 v2 adapter propagates that flag and **refuses planner handoff** if any event's tail guard was clipped.
- Padding is a conservative safety margin, not measured influence. The referenced profile digest does not itself authenticate its publisher or prove it is correct; the caller must resolve the profile and qualify it on held-out renderer/effect configurations.

## EVAL-025 adapter v2 integration

The support adapter now consumes only replay-verified v0.2 segmented reports. Each per-event evidence record retains raw measured, hysteresis, guard and conservative intervals. EVAL-019 receives only the conservative union. The aggregate receipt commits to all four interval categories and the tail clipping state.

The effects-group routing domains remain diagnostic only. The complete downstream/master/plugin/output graph is still unknown, so route completeness remains false and EVAL-019 must globally broaden pair scope. A clipped tail guard blocks handoff independently of that incomplete-routing rule.

## New source tests authored

1. Two below-release windows are retained, the next closes, and a later onset starts a new interval.
2. Between-onset-and-release measurements stay active.
3. `AnyMetricExceeds` and `BothMetricsExceed` produce distinct decisions for RMS-only evidence.
4. Tail guard is reported separately from hysteresis and conservative support.
5. A guard extending past the output horizon sets the clipping flag.
6. Invalid release ratios, persistence counts, unbounded guards and non-zero guards without a valid profile digest reject.
7. The segmented receipt changes when raw/hysteresis/guard fields change.

(There are six segmenter tests in the current file; the combination-mode ablation is bundled into the temporal-segmentation coverage.)

## Benchmark plan

A proper held-out support corpus should provide:
- source event identities and exact intervention receipts;
- raw per-window RMS and spectral values;
- no-op repeatability controls;
- exact PCM output differences and effect-tail end points;
- SoundFont, renderer build, effects-group topology, sample rate and output format strata;
- composition/source lineage for clustered resampling.

Compare:
1. MIDI-envelope-only support;
2. v0.1 raw-window RMS-OR-spectral support;
3. RMS-only and spectral-only raw ablations;
4. v0.2 `AnyMetricExceeds` hysteresis;
5. v0.2 `BothMetricsExceed` hysteresis;
6. v0.2 hysteresis plus a renderer/effect-specific tail profile;
7. exact waveform-difference intervals as an empirical ceiling only.

Report interval precision/recall/F1/IoU, onset/offset error, missed influence energy outside support, false-positive support duration per minute, interval fragmentation, support recall at fixed guard-frame budgets, worst-group results and p50/p95 compute/memory.

Do not treat windows/channels/retries as independent observations. Use source composition/lineage clusters for confidence intervals.

## Qualification targets (targets, not results)

- 100% replay rejection for changed raw per-window metrics, thresholds, hysteresis configuration, tail profile, intervals or clipping status.
- Zero misses on deterministic influence fixtures above the declared measurement floor.
- Zero tail-handoff authorization when the requested guard is clipped.
- Zero false-independent route classifications on shared-master, plugin, send/return and incomplete-route controls.
- Held-out source compositions, SoundFonts and effect topologies before promotion.
- Publish raw, hysteresis and guard distributions separately; no average-only success claim.

## Remaining requirements

- The current tail-profile digest is a binding to a declared profile ID, not a registry verification or empirical qualification. Build a signed/profile-resolved calibration registry before using non-zero guards in production.
- The v0.2 hysteresis code is deterministic and replayable but has not been compiled or executed.
- The adapter still supports only EVAL-017's currently admitted Remove intervention. Other intervention classes remain rejected until their MIDI/event preservation gates are implemented.
- Full FluidSynth downstream/master/plugin routing remains unverified, so planner route scope must remain global.

## Validation boundary

The code and source tests are authored only. The current environment has no cargo, rustc, rustfmt, FluidSynth binary, SoundFont or rendered benchmark corpus. No real support precision/recall, optimal threshold, tail profile qualification, false-prune rate, speedup or SOTA claim is asserted.