# MEL-MR-EVAL-035 — replayable hysteresis and tail-aware support qualification

**Status:** EVAL-017 v0.2 support-policy schema, window trace and deterministic qualification implemented; seven focused temporal-policy tests authored; EVAL-025 adapter updated to planner-consume qualified intervals. No Rust compilation/test run, real FluidSynth render, SoundFont matrix or measured support-precision result claimed.

## Finding

EVAL-017 previously made independent per-window decisions using absolute thresholds and per-window no-op repeatability floors, then merged adjacent positive intervals. That can fragment a continuous reverb/release tail when a noisy window briefly falls below threshold, and it can turn a one-window impulse into planner support.

EVAL-025 had been copying those raw intervals into EVAL-019. A diagnostic threshold crossing was therefore being treated as a planner candidate region without temporal qualification.

## Paradigm shift

Treat counterfactual support as a temporal inference problem with separate, replayable layers:

1. **Raw measurement:** per-window RMS, peak and spectral deltas plus calibrated enter thresholds.
2. **Temporal qualification:** a versioned hysteresis policy, episode duration and bounded gap bridging.
3. **Conservative tail policy:** explicit post-episode support extension, reported separately from measured windows.

The detector never rewrites the raw trace. The planner only receives qualified support; researchers retain the raw measurement and full trace to understand what the policy changed.

## EVAL-017 schema changes

- Bump the report to mel-mr-eval-017-counterfactual-influence-v0.2.
- Bump the influence receipt hash domain to v2.
- Add InfluenceSupportPolicy:
  - version mel-mr-eval-017-support-policy-v1;
  - release threshold ratio in millionths, strictly between 0 and 1;
  - minimum release-positive windows to accept an episode;
  - maximum consecutive below-release windows that may be bridged;
  - tail-extension frame bound, rejected if larger than the exact output-frame horizon.
- Retain the full ordered per-channel/window trace, including measured metrics, enter thresholds, release thresholds, raw enter decisions, release decisions, accepted support membership and bridged-gap flags.
- Preserve measured_influence_intervals as merged raw enter-threshold-positive intervals.
- Add qualified_support_intervals as the post-policy mask. This is the only support interval set a downstream planner may consume.
- Record qualified and bridged channel-window counts and actual unique tail-extension frames applied.
- Drive InfluenceDisposition from qualified support, not raw isolated crossings.

The default policy has a 0.75 release/enter threshold ratio, minimum one active window, at most one bridged gap window and zero tail extension. Callers may select a stricter minimum duration or conservative tail bound explicitly. Different policies are part of the report commitment and cannot be silently substituted during replay.

## Detector semantics

The policy is applied independently to each audio channel:

1. An episode begins only on a raw enter-threshold crossing.
2. Once active, lower release thresholds maintain the episode.
3. Consecutive below-release windows are held as pending gaps. If another release-positive window arrives before the configured gap bound is exceeded, those intervening windows are marked as bridged.
4. If the gap bound is exceeded, the episode closes at the last release-positive window; trailing below-release windows are not support.
5. An episode is accepted only if it contains the configured minimum number of release-positive windows. A rejected spike remains in raw measurements but creates no qualified support.
6. Qualified per-channel intervals are unioned before tail extension. The frame count added is calculated after union so the same stereo/shared frame is not counted once per channel.
7. Tail extension is applied only after a qualifying episode, and is clamped to the rendered output-frame horizon. The number of unique additional frames is recorded.
8. A measured-empty scan never becomes non-empty because of a tail policy.

A tail extension is a conservative declared support bound, not observed physical influence. A supported interval is an empirical region in the declared paired-render experiment, not exclusive sample ownership or proof of complete route topology.

## EVAL-025 adapter v2 changes

- Bump the EVAL-025 adapter and receipt hash domains to v2.
- CounterfactualEventSupportEvidence.raw_measured_support retains converted raw intervals for audit.
- EventInteractionSupport.acoustic_support now comes from qualified_support_intervals.
- Structural receipt checks bound both raw and qualified intervals by each event's output-frame horizon.
- The original replay of EVAL-017 still validates policy, per-window trace, raw intervals, qualified intervals, tail counts and all source/renderer/PCM evidence before an opaque planner token can be returned.
- Routing remains explicitly incomplete. Effects-group mapping is not a complete FluidSynth downstream/master/plugin/output route graph; EVAL-019 must continue expanding incomplete routing globally.

## Adversarial tests authored

Seven focused temporal detector tests were added:
1. A high enter crossing followed by lower release-positive windows remains one episode even after falling below enter.
2. An isolated high window remains raw evidence but is rejected when minimum support duration is two.
3. A single bounded below-release gap bridges; a gap longer than the limit closes the episode.
4. Tail extension is clamped to the exact output frame horizon and reports the unique added frames.
5. Measured-empty support remains empty even under a non-zero tail-extension policy.
6. Channels are qualified independently before their intervals are unioned.
7. Invalid release ratios, minimum-window counts and tail bounds fail closed.

The existing renderer-backed influence test now asserts that report trace length equals the calibrated window/channel count and that tampering with the support policy or a per-window decision invalidates the report receipt. The adapter test also contains a raw-spike/empty-qualified-support negative control: the raw interval is preserved in evidence, while the planner support remains empty.

## Qualification metrics

For held-out renderer/SoundFont/effects-group strata, publish:
- Raw and qualified support precision/recall against controlled counterfactual PCM deltas.
- Onset/offset localization error before and after qualification.
- False support rate on measured-empty and isolated-spike controls.
- Miss rate and tail cutoff on long release/reverb controls.
- Number of bridged windows, accepted/rejected episodes, and tail-extension frames.
- Planner cost/recall curves from EVAL-024, including false-prune rate and cost@95% recall.
- Worst renderer, SoundFont and topology, not just an aggregate average.

Do not set final numeric superiority thresholds from the development data. Freeze thresholds and confidence intervals after calibration, then evaluate composition/source-lineage-held-out scenes.

## Integration order

- EVAL-017 / #9093: paired influence measurement plus versioned support qualification.
- EVAL-025 / #9218: adapter v2 routes qualified support to EVAL-019 and retains raw evidence.
- EVAL-024 / #9217: compare planner utility at equal actual render cost.
- EVAL-029 / #9275: effects-group count is a renderer stratum, not a claim of complete independence.
- EVAL-031 / PR #9303: benchmark split lineage isolation.
- EVAL-032 / PR #9312: judge reliability.

## Validation and claim boundary

The EVAL-017 Rust code and tests and EVAL-025 adapter source are authored on the EVAL-035 branch. The current environment has no cargo, rustc, rustfmt, FluidSynth binary, SoundFont or benchmark data; no code has been compiled/executed and no real render or measured accuracy result exists. CI and PR Governance are only considered passes if they actually execute.

The detector estimates temporal support for a declared intervention, renderer profile and calibration. Hysteresis reduces threshold flicker; minimum-window gating rejects isolated spikes; gap bridging handles short interruptions; tail extension adds an explicit conservative bound. None establishes sample ownership, universal physical causality, perceptual salience or complete routing.
