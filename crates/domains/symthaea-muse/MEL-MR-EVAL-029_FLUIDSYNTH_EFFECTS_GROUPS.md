# MEL-MR-EVAL-029 — FluidSynth effects-group editability experiment

**Status:** renderer control is implemented as an explicit experimental parameter; no FluidSynth render comparison has been run in this environment.

## Hypothesis

FluidSynth defaults to one effects group, with all voices rendered through a shared reverb and chorus. Its `synth.effects-groups` setting permits 1–128 groups. The documentation says effects-group assignment follows MIDI channels; if group count is below MIDI-channel count, channel indices wrap modulo the group count. Values above one increase CPU use.

Primary sources:
- https://www.fluidsynth.org/api/settings_synth.html
- https://github.com/FluidSynth/fluidsynth/blob/master/doc/fluidsettings.xml

Changing effect topology may reduce some cross-channel effects coupling and thereby make edits more compositional. This is a falsifiable hypothesis, not an independence guarantee. Separate effect groups do not prove the absence of shared master processing or other signal paths.

## Implementation in this increment

- `render_midi_to_wav_observed` remains backward-compatible and explicitly selects one effects group.
- `render_midi_to_wav_observed_with_effects_groups` allows an explicit 1–128 group count.
- `render_midi_to_wav_with_effects_groups` provides a non-evidence convenience wrapper.
- The exact `synth.effects-groups=N` setting is included in ordered command arguments, which the renderer process observation commits.
- `effects_group_for_midi_channel` exposes the documented zero-based modulo assignment for channels 0–15.
- `build_effects_group_assignment` produces a typed, domain-separated SHA-256 receipt for the complete 16-channel assignment. The exact assignment is included in the returned `FluidSynthRenderObservation`.
- `verify_effects_group_assignment` rebuilds the assignment and rejects a mutated mapping or digest.
- Invalid group counts and zero sample rates are rejected before child process execution.

The channel→effects-group mapping only characterizes the synth effects assignment. It must not be emitted as a complete route graph by itself.

## Experimental matrix

Use the same serialized Score, MIDI bytes, event map, SoundFont digest, FluidSynth executable digest, sample rate, gain, mastering settings, reverb/chorus values and environment snapshot.

| Condition | Effects groups | Expected channel mapping |
|---|---:|---|
| Compatibility | 1 | all channels → group 0 |
| Partial split | 4 | channel → channel mod 4 |
| Per-channel | 16 | channels 0–15 → groups 0–15 |
| Upper setting | 128 | channels 0–15 → groups 0–15; extra groups unused |

The 128-group condition is an optional stress condition, not presumed to improve quality or independence.

For each compatible FluidSynth build/SoundFont pair:
1. Confirm the effective setting is accepted; capture process exit, exact argv, executable/SoundFont digests and output receipts.
2. Render paired EVAL-018 baseline/remove-A/remove-B/remove-AB coalitions for same-channel, different-channel/same-group, and different-group event pairs.
3. Measure contextual PCM residuals, per-channel interaction, tail length, repeatability floors and output layout.
4. Include deliberately shared downstream processing / unknown-route controls; unknown topology means incomplete routing.
5. Measure wall time, CPU, peak memory and output file size.
6. Run blinded, level-matched comparisons and operation-specific target/preservation checks. Do not infer perceptual equivalence from PCM interaction reduction.

## Primary reported measurements

- Per-pair contextual interaction norm `||Y_all - Y_-A - Y_-B + Y_-AB||_2`, normalized separately by declared main-effect energy only when the denominator is nonzero.
- Distribution of cross-channel vs same-channel interactions, stratified by shared vs separated effect group.
- Tail-support duration and interval recall against exact controlled counterfactual PCM differences.
- Render latency p50/p95, CPU time, peak memory and failure rate.
- Change in target success, protected-property preservation, and blinded listener preference.
- Worst SoundFont / FluidSynth version / topology result; no average-only win claim.

Do not collapse these into a single weighted number. Report the interaction reduction together with compute overhead and perceptual changes.

## Qualification gates

- 100% of known invalid parameter values reject before spawning.
- 100% of captured receipts bind the exact selected group count in argv.
- Known channel assignment fixtures match modulo semantics.
- Zero false-independent classifications in shared-master, limiter, plugin and incomplete-route negative controls.
- Any cross-channel interaction reduction must replicate on held-out SoundFonts and at least one held-out effects topology.
- Perceptual quality must be non-inferior within a preregistered margin before the configuration can become a default.
- Added synthesis CPU/memory cost must be reported; no assumption that group isolation is a net system win.

## Limitations

This first increment exposes the parameter and its deterministic channel mapping. It does not yet capture the full FluidSynth internal routing graph, prove all versions/builds accept every setting, establish a reduction in causal interaction, or qualify a user-facing preset. The real adapter must keep routing incomplete wherever shared downstream stages cannot be verified.
