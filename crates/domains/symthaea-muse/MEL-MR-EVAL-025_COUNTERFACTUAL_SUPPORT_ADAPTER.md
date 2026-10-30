# MEL-MR-EVAL-025 — replay-verified FluidSynth support adapter

**Status:** adapter v2 consumes replay-verified EVAL-017 segmented v0.2 reports; eight focused source tests authored; no compile, test, real FluidSynth render or dataset run claimed.

## Objective

Convert replay-verified EVAL-017 per-event acoustic-influence reports into the exact EVAL-019 `EventInteractionSupport` input shape. This is the bridge from counterfactual acoustic measurements to planner evidence.

## Data path

1. A renderer adapter creates a baseline FluidSynth render and a separate remove-one-event counterfactual for every source Score event.
2. Each render is bound to its exact Score, MIDI bytes, performed-event map artifact, WAV/decoded PCM, renderer executable/SoundFont hashes, process exit and ordered argv using `RendererExecutionReceipt`.
3. `verify_renderer_audio_binding_for_score` reconstructs the performed-event map from MIDI bytes and returns an opaque `VerifiedRendererAudioBinding`.
4. At least three independent no-op rerenders establish the per-window/channel repeatability floor. `verify_repeatability_calibration` replays this calibration and returns an opaque `VerifiedRepeatabilityCalibration`.
5. `measure_counterfactual_influence_segmented` preserves the v0.1 raw report and every per-channel/window RMS/spectral measurement, then applies a versioned v0.2 hysteresis policy. `verify_counterfactual_influence_segmented` replays raw measurements, threshold decisions, hysteresis and tail guard from the exact source Score, maps, schedules, receipts, audio buffers and calibration.
6. `verify_counterfactual_event_support` refuses to create planner evidence unless the v0.2 replay in step 5 passes. It binds raw, hysteresis, tail-guard and conservative intervals to the canonical source event ID and typed effects-group assignment.
7. `assemble_counterfactual_support_adapter_report` requires exactly one event record for each source Score index, then produces an opaque aggregate verification token.
8. `verify_counterfactual_support_adapter_report` independently replays all EVAL-017 reports before returning that token from a serialized report. Only the opaque verified aggregate exposes `planner_event_supports()` for EVAL-019.

## Implemented changes

### Receipt-bound effects-group identity

`VerifiedRendererAudioBinding` now exposes an optional effects-group count parsed from the exact `-o synth.effects-groups=N` pair in the replayed FluidSynth argv. Missing, malformed, duplicate or out-of-range settings produce no usable count. The existing renderer execution profile hashes ordered argv, so this setting is also bound into the renderer configuration digest.

The adapter rebuilds and replay-verifies the EVAL-029 `FluidSynthEffectsGroupAssignment` for values 1–128, then maps every performed MIDI channel for a source event to its observed effects-group domain. Baseline and counterfactual tokens must have the same explicit effects-group count and renderer configuration.

### Support interval and digest contract (adapter v2)

- `raw_measured_intervals` preserves the v0.1 independent-window RMS-OR-spectral exceedance intervals.
- `hysteresis_support_intervals` uses per-channel onset/release segmentation with an explicit persistence count.
- `tail_guard_intervals` contains only explicit post-hysteresis padding; it is not measured influence.
- `conservative_support_intervals` is the union of hysteresis and tail guard, and this—not the raw v0.1 interval set—is what populates EVAL-019 `acoustic_support`.
- `acoustic_support` is copied from `CounterfactualInfluenceSegmentedReport.conservative_support_intervals`.
- Starts remain inclusive and ends exclusive.
- All interval sets use `base_report.total_frames`: the rendered PCM output frame count. It can include release/reverb tails beyond input MIDI duration.
- Per-window onset and release RMS/spectral booleans and the declared combination rule are all retained in the v0.2 trace.
- A non-zero tail guard requires a content-addressed tail-profile SHA-256; the referenced profile must be resolved and qualified against the renderer/effects topology by the caller.
- If the requested guard exceeds `base_report.total_frames`, the event and aggregate report set `tail_guard_was_clipped=true`. The opaque verified planner token refuses `planner_event_supports()` rather than handing truncated support to EVAL-019.
- Every event receives a domain-separated SHA-256 support evidence digest, even when its measured interval list is empty.
- Empty intervals plus digest mean the full output scan replayed and found no windows above the declared repeatability/absolute thresholds. It does not prove zero physical influence.
- The per-event digest commits to the v0.1 and v0.2 influence receipts, source Score, event identity, baseline/counterfactual render receipts/audio, repeatability calibration, segmentation configuration (including the RMS/spectral combination rule and tail-profile digest), all four interval sets, clipping state, renderer profile, effects-group assignment, output format/horizon and route-domain metadata.
- The aggregate receipt commits to the canonical ordered event records and common baseline/render context.
- The serialized report is not itself a planner authorization. A caller must replay it to obtain `VerifiedCounterfactualSupportAdapterReport`.

### Route completeness deliberately remains false

The adapter emits diagnostic domains of the form `fluidsynth:effects-group:N`, but it does **not** mark routing complete. The FluidSynth effect-group mapping describes the reverb/chorus groups only. It does not prove the absence of shared master processing, plugins, output mixing, or other shared stages.

Every event therefore has `routing_complete = false`, and the aggregate says the same. EVAL-019 must conservatively expand pair scope globally rather than prune pairs on the basis of incomplete routing. This preserves the useful measured acoustic support while preventing an unproven route graph from being used as an exclusion certificate.

## Adversarial source tests authored

Eight adapter tests cover:

- An empty measured support still has a mandatory evidence digest and retains the `NoEffectAboveNoiseFloor` disposition.
- Changing a measured interval invalidates the per-event evidence digest.
- EVAL-029 modulo channel-to-effects-group mapping is replay-checked for 1/4/16/128-capable assignments.
- Aggregate output sorts events canonically, preserves the output frame horizon and marks the route graph incomplete.
- Missing/duplicate source event records reject.
- Mixed renderer configuration/effects topology across per-event scans rejects.
- An existing renderer receipt replay test now asserts that effects-group count is recovered from the exact argv.

## Current EVAL-017 limitation

The existing `validate_intervention` currently admits only `EventIntervention::Remove`; other enum variants remain reserved and reject. This adapter does not widen that operation gate. Transpose/onset/duration/velocity counterfactuals require operation-specific MIDI and preservation validators before they can be admitted.

## Next qualification steps

1. Compile and run the focused Rust tests on a Rust-enabled runner.
2. Execute the same baseline + three no-op calibration + one remove-one-event pair for every source event with real FluidSynth and a pinned SoundFont.
3. Verify every event has a replayable report and every support digest matches the exact source Score/MIDI/map/renderer profile.
4. Measure the 1/4/16/128 group matrix on matched pairs; report support intervals and cross-channel interaction reduction independently.
5. Keep routing incomplete until the complete FluidSynth route graph, including shared downstream/master paths, is measured.
6. Only then benchmark planner cost/recall and false-prune rate against EVAL-024's NoProbe, UniformRandom, TemporalOnly, RoutingOnly, EVAL-019, OracleTopK and Exhaustive policies.

## Validation and claim boundary

The adapter code and eight focused tests are authored, but this environment has no `cargo`, `rustc`, `rustfmt`, FluidSynth binary, SoundFont or benchmark audio. CI status will be reported as skipped unless it actually executes. No real per-event support measurement, held-out tail-profile qualification, routing completeness, pruning gain, runtime improvement or SOTA result is claimed.