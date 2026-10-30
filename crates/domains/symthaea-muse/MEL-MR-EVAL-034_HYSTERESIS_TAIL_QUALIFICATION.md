# MEL-MR-EVAL-034 — calibrated hysteresis and explicit tail qualification

**Status:** versioned EVAL-017 v0.2 segmenter implemented; seven new segmentation-focused tests authored; EVAL-025 adapter upgraded to v2 with eight focused adapter tests; no compile, runner, real FluidSynth render or empirical support-quality result claimed.

## Why this is separate from EVAL-017 v0.1

EVAL-017 v0.1 must remain replay-compatible. It independently tests each PCM output window/channel against absolute and no-op-calibrated thresholds, using RMS **or** spectral excess, then merges all raw exceedance windows. This work adds temporal post-processing without rewriting v0.1 report semantics.

Sound-event detection practice treats onset/offset post-processing as part of measurement. Two-threshold hysteresis, minimum duration and gap handling can change interval boundaries and event-based F-scores. DCASE's event-based metrics use boundary collars; PSDS/t-PSDS provide complementary threshold-sweep views.

TimingBench (August 2026) is a useful adjacent music-temporal methodology: it includes machine-readable event schemas and a deterministic detector cascade across 250 prompts. Its detector median localization error varies by task family (reported range 8–218 ms); its independently human-annotated validation set was still under construction. It is a method reference, not a gold-standard editing benchmark.

References:
- DCASE sound-event detection post-processing review: https://doi.org/10.1561/116.00000051
- Evaluation of post-processing algorithms for polyphonic SED: https://arxiv.org/abs/1906.06909
- TimingBench: https://taan.ai/research/rushing-or-dragging/
- DCASE 2023 sound event detection results: https://dcase.community/challenge2023/task-sound-event-detection-with-soft-labels-results

## Paradigm shift: preserve three temporal facts

Do not collapse measurement, segmentation and safety padding into one interval list:

1. **Raw measured exceedance** — exact EVAL-017 v0.1 threshold results and merged windows.
2. **Hysteresis-qualified support** — temporal state-machine output using explicit onset/release thresholds and persistence.
3. **Conservative tail guard** — separately recorded padding after measured/hysteresis support, bound to a frozen profile and explicitly *not* called measured influence.

EVAL-017 v0.2 preserves per-window RMS and spectral values, thresholds and boolean decisions so interval boundaries can be explained and replayed.

## Implemented v0.2 API

In `src/counterfactual_influence.rs`:

- `InfluenceThresholdCombination`: explicit `AnyMetricExceeds` (default, preserves v0.1 raw semantics) or `BothMetricsExceed` (strict ablation).
- `InfluenceSegmentationConfig`: combine rule, release ratio, release persistence count, tail guard frame count and optional tail-profile digest.
- `InfluenceWindowSegmentation`: independent onset/release RMS and spectral booleans, combined decisions, per-channel hysteresis state and thresholds.
- `CounterfactualInfluenceSegmentedReport`: exact v0.1 report, full window measurements, raw intervals, hysteresis intervals, tail guard intervals, conservative support intervals, clipping flag and v0.2 receipt.
- `measure_counterfactual_influence_segmented`: new v0.2 measurement path.
- `verify_counterfactual_influence_segmented`: replays the v0.1 measurement, raw window trace, hysteresis, tail guard and receipt from original renderer-bound inputs.

The old `measure_counterfactual_influence` and `verify_counterfactual_influence` keep their v0.1 report semantics.

## Hysteresis semantics

For each channel independently:
- onset threshold is the calibrated RMS and spectral threshold;
- release threshold for each metric is `max(absolute_floor, calibrated_onset_threshold × release_threshold_ratio)`;
- onset and release decisions combine the independent metric booleans using the declared `InfluenceThresholdCombination`;
- once active, the state remains active for the configured count of consecutive below-release windows;
- the first below-release window beyond that count closes the segment at the end of the last retained window;
- channel intervals are unioned only after temporal segmentation.

Defaults are a 0.5 release ratio and two persistence windows. They are research defaults, not proven optima. `BothMetricsExceed` is provided as an ablation so its precision/recall trade-off can be measured rather than hidden.

## Tail guard semantics

- `tail_guard_frames=0` means no additional guard.
- A non-zero guard requires `tail_guard_profile_sha256`, binding the report to a declared renderer/effect-tail calibration profile.
- Each guard interval begins at the hysteresis interval's exclusive end and extends by the configured number of rendered output frames.
- The report preserves `tail_guard_was_clipped` if the requested guard extends past the rendered PCM horizon, even when the emitted guard interval is shorter or empty.
- EVAL-025 adapter v2 propagates that flag and refuses planner handoff if any event guard was clipped.
- Padding is a conservative safety margin, not measured influence. A profile digest is only a binding; callers must resolve and qualify the referenced profile on held-out renderer/effect configurations.

## EVAL-025 adapter v2 integration

The support adapter consumes replay-verified v0.2 segmented reports. Each event evidence record retains raw measured, hysteresis, tail-guard and conservative intervals. EVAL-019 receives only the conservative union. The aggregate receipt commits to all interval categories and tail clipping state.

The adapter returns an opaque verified aggregate token only after replaying all original per-event EVAL-017 inputs. Its planner-support accessor returns an error when the tail guard was clipped.

Effects-group domains remain diagnostic only. The complete downstream/master/plugin/output graph is still unknown, so routing completeness remains false and EVAL-019 must globally broaden pair scope.

## New segmenter tests authored

1. Below-release windows are retained for the configured persistence, then close the segment.
2. Measurements between onset and release thresholds keep a segment active.
3. `AnyMetricExceeds` and `BothMetricsExceed` produce distinct results on RMS-only evidence.
4. Tail guard is separate from hysteresis and conservative support.
5. A guard extending past the output horizon sets the clipping flag.
6. Invalid ratios, persistence counts, unbounded guards and non-zero guards without a valid profile digest reject.
7. The v0.2 receipt changes when raw/hysteresis/tail data changes.

An existing renderer-bound influence test was also extended to execute v0.2 and verify the replay path and tamper rejection.

## Benchmark plan

Build a controlled support corpus with exact source event IDs, intervention receipts, no-op controls, output PCM, effects tails, SoundFont/renderer/effects topology and source-lineage metadata.

Compare:
1. MIDI-envelope-only support;
2. v0.1 raw-window RMS-OR-spectral support;
3. RMS-only and spectral-only raw ablations;
4. v0.2 `AnyMetricExceeds` hysteresis;
5. v0.2 `BothMetricsExceed` hysteresis;
6. v0.2 hysteresis plus renderer/effect-specific tail profile;
7. exact waveform-difference intervals as an empirical ceiling only.

Report interval precision/recall/F1/IoU, onset/offset error, missed influence energy outside support, false-positive support duration per minute, interval fragmentation, support recall at fixed guard-frame budgets, worst-group results, repeatability false alarms and p50/p95 compute/memory.

Use source composition/lineage clusters for confidence intervals. Do not treat windows, channels or retries as independent observations.

## Qualification targets (targets, not results)

- 100% replay rejection for modified raw window metrics, thresholds, hysteresis configuration, tail profile, intervals or clipping status.
- Zero misses on deterministic influence fixtures above the declared measurement floor.
- Zero planner authorization when a requested tail guard is clipped.
- Zero false-independent route classifications on shared-master, plugin, send/return and incomplete-route controls.
- Held-out source compositions, SoundFonts and effect topologies before promotion.
- Publish raw, hysteresis and guard distributions separately; no average-only success claim.

## Remaining requirements

- Tail-profile digest is a binding, not a registry verification or empirical qualification. Add signed/profile-resolved calibration before production use of non-zero guards.
- The v0.2 code is deterministic and replayable but has not been compiled or executed.
- EVAL-017 currently admits Remove interventions only. Other intervention classes remain rejected until operation-specific MIDI and preservation gates are implemented.
- Complete FluidSynth downstream/master/plugin routing remains unverified; planner route scope must remain global.

## Validation boundary

The code and source tests are authored only. The current environment has no cargo, rustc, rustfmt, FluidSynth binary, SoundFont or rendered benchmark corpus. No real support precision/recall, optimal threshold, tail-profile qualification, false-prune rate, speedup or SOTA claim is asserted.