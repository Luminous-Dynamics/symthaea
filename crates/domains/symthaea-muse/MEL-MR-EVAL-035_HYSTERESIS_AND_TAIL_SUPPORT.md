# MEL-MR-EVAL-035 — Replayable Hysteresis and Tail-Aware Support Qualification

**Status:** first temporal-qualification tranche authored on 12 November 2026. The source trace, hysteresis policy, per-channel episode detector, support-adapter handoff and synthetic controls are present. Pinned Rust qualification is pending; no compiled test, real FluidSynth, SoundFont-matrix or held-out calibration result is claimed.

**Issue:** [#9381 — replayable hysteresis and tail-aware counterfactual support qualification](https://github.com/Luminous-Dynamics/symthaea/issues/9381)

**Integration:** MEL-MR-EVAL-017 counterfactual influence report → this temporal support qualification → MEL-MR-EVAL-025 adapter → MEL-MR-EVAL-019 sparse interaction planner.

## Why this is a separate qualification layer

EVAL-017 measures per-window sample-domain and spectral deltas against a repeat-render noise floor. Its raw threshold crossings are useful diagnostic evidence, but independent thresholding has no temporal memory:

- a single transient can look like a complete influence interval;
- a real release/reverb tail can flicker above and below one threshold;
- a short sub-threshold gap can split one continuous episode;
- a conservative post-episode tail prior is not a measured acoustic effect.

This change keeps the raw measurement and adds a separate temporal inference result. It does not change the paired-render intervention or its renderer/PCM evidence boundary.

## Implemented

### Versioned policy

A new `InfluenceSupportPolicy` is included in `InfluenceAnalysisConfig`, which is already committed into the repeatability calibration and influence report:

- fixed-point release ratio in millionths, strictly between zero and one;
- minimum release-positive windows required to qualify an episode;
- maximum consecutive sub-release windows that may be bridged;
- explicit tail-extension bound in output frames.

The policy is included in the content-addressed calibration/report identity. The calibration and influence report schema versions are bumped to v0.2.

### Complete replayable trace

The influence report now retains every ordered per-window/per-channel measurement, including:

- raw measured RMS, peak-sample and spectral-magnitude deltas;
- enter thresholds from absolute floors and per-window no-op calibration;
- release thresholds derived from the frozen fixed-point ratio;
- raw enter-threshold crossing;
- release-threshold decision;
- qualified-episode membership;
- whether the window was an interior bridged gap.

The original `measured_influence_intervals` remains the merged raw enter-threshold-positive diagnostic output.

### Hysteresis-qualified episodes

The per-channel state machine:

1. starts only at a raw enter-threshold crossing;
2. remains active while release-positive windows continue;
3. bridges only the configured number of consecutive sub-release windows;
4. qualifies only when the minimum release-positive-window count is met;
5. records the last actually measured release-positive window separately from any post-measurement extension.

It returns both channel-specific episodes and a merged qualified interval vector.

### Tail extension

Tail extension is an explicit conservative post-processing bound. It:

- starts after the last actually measured release-positive window;
- is clamped to the exact rendered output-frame horizon;
- records the measured end, final end and actual frames applied per channel episode;
- records aggregate extension frames applied;
- never produces support when the measured trace is empty;
- rejects a configured bound larger than the rendered horizon.

A non-zero extension is not proof that the renderer physically influenced those extra frames. Topology-specific statistical tail profiles remain a separate calibration programme (#9337).

### Adapter handoff

EVAL-025 now copies only `qualified_support_intervals` into the EVAL-019 planner support vector. Raw threshold-positive intervals remain in the EVAL-017 report for audit. The adapter still binds and replays the complete EVAL-017 report, and routing stays incomplete until the full downstream/master route graph is independently observed.

### Dispositions

- `EffectDetected`: at least one qualified temporal episode exists.
- `InsufficientQualifiedSupport`: raw enter crossings exist, but no episode met the minimum support-window criterion.
- `NoEffectAboveNoiseFloor`: no raw enter crossing exists.

This avoids calling a one-window spike either a qualified support episode or a clean no-effect result.

## Synthetic controls authored

Nine direct temporal-state-machine tests cover:

1. one isolated enter spike is retained raw but rejected as planner support;
2. sustained release-positive windows survive after enter-threshold crossing ceases;
3. a one-window sub-release gap is bridged and explicitly labelled;
4. a gap longer than the configured allowance closes the episode;
5. raw spikes do not enter qualified support;
6. tail extension clamps at the rendered output horizon and reports actual frames;
7. measured-empty support remains empty under a non-zero tail policy;
8. channels qualify independently before interval union;
9. invalid release ratios, zero minimum duration and excessive tail bounds reject.

Additional report replay assertions reseal a deliberately modified support trace or policy and confirm replay rejects the altered report. The adapter negative fixture contains a raw positive interval but an empty qualified interval and confirms only qualified support reaches the planner representation.

These are source-authored synthetic controls. They are not yet compiled or executed.

## Limits and non-claims

- Windows are overlapping analysis windows. The trace order is exact and replay checked; support time resolution remains bounded by the configured window/hop.
- The release ratio is a deterministic state-machine policy, not an empirically optimal universal threshold.
- The tail extension is a bounded prior, not an acoustic observation or proof of source-sample ownership.
- Effects-group assignment is only a partial FluidSynth routing subgraph. `routing_complete=false` remains mandatory; EVAL-019 must widen candidate pairs when the complete path is unknown.
- No real FluidSynth/SoundFont/effects-topology campaign, held-out calibration, perceptual salience or SOTA comparison has been completed.

## Qualification targets — proposals, not results

| Measure | Target |
|---|---:|
| Isolated spike promoted to planner support under minimum-duration policy | 0 |
| Known sustained-tail fixtures lost by hysteresis | 0 |
| Measured-empty traces extended into non-empty support | 0 |
| Tail extension beyond output-frame horizon | 0 |
| Deliberate policy/trace/count/interval/receipt tampering accepted on replay | 0 |
| Held-out support containment after calibration | >=95%, with confidence intervals |
| Held-out false-support rate | <1% target, reported per renderer stratum |
| Onset/offset support localization | p50/p95, reported in frames and milliseconds |
| Cost | CPU time, memory and render count reported separately |

The 95% containment and <1% false-support figures are stretch targets to be calibrated on development compositions and tested on composition-lineage-held-out splits. They are not current results.

## Next research steps

1. Build the EVAL-035 topology-specific tail-profile registry (#9337) using held-out composition lineages and an upper confidence bound on tail-duration quantiles.
2. Add real FluidSynth captures with reverb, chorus, multiple effects groups and shared master-bus interactions.
3. Integrate the completed EVAL-025 adapter with EVAL-019/#9217 and measure matched-budget pair recall, false candidate discovery and actual renders consumed.
4. Extend the factorial interaction evaluator (#9133/#9155/#9213) to report raw-vs-qualified support sensitivity across additive, pairwise and higher-order fixtures.
5. Only after held-out qualification, use the influence graph as a planner prior. Final musical edit acceptance still requires independent target-change and protected-content verification.
