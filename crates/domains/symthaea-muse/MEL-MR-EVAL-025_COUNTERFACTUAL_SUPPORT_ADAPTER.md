# MEL-MR-EVAL-025 — replay-verified FluidSynth support adapter

**Status:** adapter and seven focused source tests authored; EVAL-035 upgrades the influence report to v0.2 and the adapter/evidence hash schemas to v2, with separately replayable raw and temporally qualified support. No compile, test, real FluidSynth render or dataset run is claimed.

## Objective

Convert replay-verified EVAL-017 per-event acoustic-influence reports into the exact EVAL-019 `EventInteractionSupport` input shape. This is the bridge from counterfactual acoustic measurements to planner evidence.

## Data path

1. A renderer adapter creates a baseline FluidSynth render and a separate remove-one-event counterfactual for every source Score event.
2. Each render is bound to its exact Score, MIDI bytes, performed-event map artifact, WAV/decoded PCM, renderer executable/SoundFont hashes, process exit and ordered argv using `RendererExecutionReceipt`.
3. `verify_renderer_audio_binding_for_score` reconstructs the performed-event map from MIDI bytes and returns an opaque `VerifiedRendererAudioBinding`.
4. At least three independent no-op rerenders establish the per-window/channel repeatability floor. `verify_repeatability_calibration` replays this calibration and returns an opaque `VerifiedRepeatabilityCalibration`.
5. `measure_counterfactual_influence` computes raw enter-threshold crossings and versioned hysteresis-qualified support against that calibration. `verify_counterfactual_influence` replays the per-window trace, temporal policy and output intervals from the exact source Score, maps, schedules, receipts, audio buffers and calibration.
6. `verify_counterfactual_event_support` refuses to create planner evidence unless step 5 passes. It binds the evidence to the canonical source event ID and typed effects-group assignment.
7. `assemble_counterfactual_support_adapter_report` requires exactly one event record for each source Score index, then produces an opaque aggregate verification token.
8. `verify_counterfactual_support_adapter_report` independently replays all EVAL-017 reports before returning that token from a serialized report. Only the opaque verified aggregate exposes `planner_event_supports()` for EVAL-019.

## Implemented changes

### Receipt-bound effects-group identity

`VerifiedRendererAudioBinding` now exposes an optional effects-group count parsed from the exact `-o synth.effects-groups=N` pair in the replayed FluidSynth argv. Missing, malformed, duplicate or out-of-range settings produce no usable count. The existing renderer execution profile hashes ordered argv, so this setting is also bound into the renderer configuration digest.

The adapter rebuilds and replay-verifies the EVAL-029 `FluidSynthEffectsGroupAssignment` for values 1–128, then maps every performed MIDI channel for a source event to its observed effects-group domain. Baseline and counterfactual tokens must have the same explicit effects-group count and renderer configuration.

### Support interval and digest contract

- `support.acoustic_support` is copied from `CounterfactualInfluenceReport.qualified_support_intervals`; planner input never consumes raw isolated threshold crossings.
- `raw_measured_support` separately retains the raw `measured_influence_intervals` for audit and diagnosis.
### Support interval and digest contract

- EVAL-025 adapter version is now v2; event-evidence and aggregate-report hash domains are also v2.
- The planner-facing acoustic support is derived from EVAL-017 qualified_support_intervals, not raw threshold-positive windows.
- raw_measured_support separately retains raw measured intervals for audit. Both interval sets are validated against the exact output-frame horizon and committed by the per-event digest.
- The v2 adapter does not authorize older v1 receipts as though they used the new support semantics.

- Starts remain inclusive and ends exclusive.
- All intervals use `report.total_frames`: the rendered PCM output frame count. It can include release/reverb tails beyond input MIDI duration.
- EVAL-017 v0.2 stores the full ordered per-window/per-channel trace, enter and release thresholds, raw detection, release detection, qualified membership, and bridged-gap flags.
- The replay-bound `InfluenceSupportPolicy` has a release ratio, minimum active-window count, maximum bridged gap and explicit tail-extension frame bound. Tail extension is conservative post-processing, not an observed-effect claim; the applied frame count is reported and clamped to the exact output horizon.
- Measured-empty scans remain empty even when a nonzero tail-extension policy is configured.
- Every event receives a domain-separated SHA-256 support evidence digest, even when its measured interval list is empty.
- Empty intervals plus digest mean the full output scan replayed and found no windows above the declared repeatability/absolute thresholds. It does not prove zero physical influence.
- The per-event digest commits to the influence report receipt (including policy and trace), source Score, event identity, baseline/counterfactual render receipts/audio, repeatability calibration, renderer profile, effects-group assignment, output format/horizon, raw measured intervals, qualified intervals and route-domain metadata.
- The aggregate receipt commits to the canonical ordered event records and common baseline/render context.
- The serialized report is not itself a planner authorization. A caller must replay it to obtain `VerifiedCounterfactualSupportAdapterReport`.

### Route completeness deliberately remains false

The adapter emits diagnostic domains of the form `fluidsynth:effects-group:N`, but it does **not** mark routing complete. The FluidSynth effect-group mapping describes the reverb/chorus groups only. It does not prove the absence of shared master processing, plugins, output mixing, or other shared stages.

Every event therefore has `routing_complete = false`, and the aggregate says the same. EVAL-019 must conservatively expand pair scope globally rather than prune pairs on the basis of incomplete routing. This preserves the useful measured acoustic support while preventing an unproven route graph from being used as an exclusion certificate.

## Adversarial source tests authored

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

1. Compile and run the focused Rust tests on a Rust-enabled runner, including the EVAL-035 hysteresis/gap/tail controls.
2. Execute the same baseline + three no-op calibration + one remove-one-event pair for every source event with real FluidSynth and a pinned SoundFont.
3. Verify every event has a replayable report and every support digest matches the exact source Score/MIDI/map/renderer profile.
4. Measure the 1/4/16/128 group matrix on matched pairs; report support intervals and cross-channel interaction reduction independently.
5. Keep routing incomplete until the complete FluidSynth route graph, including shared downstream/master paths, is measured.
6. Only then benchmark planner cost/recall and false-prune rate against EVAL-024's NoProbe, UniformRandom, TemporalOnly, RoutingOnly, EVAL-019, OracleTopK and Exhaustive policies.

## Validation and claim boundary

The adapter code and seven focused tests are authored; EVAL-017 currently contains fifteen source tests including seven temporal support controls, but this environment has no `cargo`, `rustc`, `rustfmt`, FluidSynth binary, SoundFont or benchmark audio. CI status will be reported as skipped unless it actually executes. No real per-event support measurement, routing completeness, pruning gain, runtime improvement or SOTA result is claimed.