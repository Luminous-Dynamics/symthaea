# MEL-MR-EVAL-018 — Pairwise Musical-Event Interaction

**Status:** first deterministic factorial measurement primitive authored on 12 November 2026. It is not connected to live FluidSynth/NativeMuse orchestration yet. Eight Rust tests are authored; no compiled or executed pass is claimed.

**Research issue:** [#9132 — Measure pairwise musical-event interactions with factorial render interventions](https://github.com/Luminous-Dynamics/symthaea/issues/9132)

**Stack:** [#9090 — FluidSynth process observation](https://github.com/Luminous-Dynamics/symthaea/pull/9090) → [#9093 — Counterfactual event acoustic influence](https://github.com/Luminous-Dynamics/symthaea/pull/9093) → this factorial interaction layer.

## Research question

When two performed events A and B coexist, how much of their joint acoustic consequence is not explained by their two separate removal counterfactuals?

A single-event influence map from MEL-MR-EVAL-017 is conditional on the musical context. It cannot be safely added to another event's map unless independence has been established. Overlap, phase, voice interaction, shared effects, routing and mastering can all invalidate naive addition.

## Factorial design

For one frozen source score, renderer profile and pair of event IDs:

- Y00 = R(S)
- Y10 = R(S without A)
- Y01 = R(S without B)
- Y11 = R(S without A or B)

For a frozen feature representation Phi:

J_AB = Phi(Y00) - Phi(Y10) - Phi(Y01) + Phi(Y11)

This is a second-order finite difference in the selected representation. Its meaning depends on Phi.

The implementation deliberately reports two different representations:

1. **PCM-domain interaction:** the signed sample-domain second difference, measured as normalized RMS and peak per time window and channel.
2. **Spectral-magnitude feature interaction:** the second difference of one-sided windowed FFT magnitudes, normalized by the sum of the four coalition magnitudes.

A spectral-magnitude residual can be nonzero for phase interference even when the PCM mix is exactly additive. It must not be called a nonlinear renderer result. A PCM residual above a calibrated repeat-render floor is stronger evidence of a non-additive output in the tested PCM representation, but is still conditional on the exact renderer, timing, random-stream and artifact controls.

## Implemented first tranche

The new src/pairwise_event_interaction.rs module provides:

- Exact source event inventory with sorted, unique IDs.
- Four explicit coalitions with exact membership:
  - Y00 contains every source event;
  - Y10 omits A and only A;
  - Y01 omits B and only B;
  - Y11 omits A and B and no other event.
- Each coalition binds:
  - exact performed-schedule digest;
  - performed-event-map receipt digest;
  - renderer execution receipt digest;
  - exact PCM buffer digest, sample rate, channels and frame count.
- Each quartet's four schedule digests must be distinct. Every repeat quartet must reuse the same four schedule and event-map commitments while using different renderer receipts, preventing one receipt from being reused to impersonate an independent execution.
- Primary and repeat layouts must match exactly. Non-finite samples, missing repeats, malformed hashes, malformed event inventories, changed coalition membership, changed schedule/map commitments, duplicate execution receipts and mismatched output layouts reject.
- Per time-window/channel:
  - normalized PCM interaction RMS and peak;
  - maximum repeat-quartet PCM residual deviation;
  - normalized spectral-magnitude interaction;
  - maximum repeat-quartet magnitude-feature residual deviation;
  - separate noise-adjusted thresholds and detections.
- Three representation outcomes plus no-detect and abstention:
  - DetectedPcm
  - DetectedMagnitudeOnly
  - DetectedBoth
  - NoDetectableInteraction
  - Abstain
- Domain-separated report digest and a replay verifier that recomputes the full report from the exact quartet and repeats.

The module binds the renderer and event-map receipt digests, but it does not verify those receipts or parse the event maps itself. Before calling this function, the orchestration layer must independently replay each receipt, verify each event map against its exact MIDI/schedule, verify the retained descendants' identity and performance fields, and verify that all renders used a common renderer profile and the declared random-stream strategy.

## Repeat-render control

A repeat quartet is one complete, independent rerun of all four coalitions under the same four schedule and event-map identities. The maximum difference between the primary quartet's interaction residual and each repeat quartet's residual forms the per-window noise floor.

This is intentionally a conservative first gate, not a statistical confidence interval. Production calibration should use multiple repeat quartets across independent base compositions and report the distribution of residuals, not just its maximum. Common random numbers should be used when a renderer supports reliable stream control; independently seeded runs should be retained as a separate stochastic-noise condition.

## Eight authored deterministic tests

1. Linear additive PCM control: the PCM second difference remains below the configured floor.
2. Nonlinear cross-term fixture: a synthetic multiplicative cross-term creates detectable PCM interaction.
3. Magnitude-only phase/feature interaction: the report distinguishes a magnitude residual from PCM non-additivity.
4. Repeat-quartet variation: measured repeat deviations enter the noise floor.
5. Missing calibration: the verdict abstains while retaining diagnostics.
6. Invalid coalition membership and receipt reuse: both reject.
7. A repeat quartet with changed event-map commitment rejects.
8. A repeat quartet with a mismatched frame layout rejects.

Tests are authored, not reported as compiled or executed.

## Scientific boundaries

- This is not a universal nonlinear-system detector. It measures a finite difference in a selected output representation.
- A detected spectral-magnitude interaction can reflect phase interference or representation nonlinearity without PCM non-additivity.
- A nonzero PCM residual can arise from stochastic divergence, unpaired random streams, timing drift, renderer mismatch or mastering-state mismatch. Those are experimental failures unless explicitly represented and calibrated.
- A zero second difference in one feature/window does not prove that the pair is musically independent in every context.
- The report is not a source-separation mask, sample-ownership attribution, perceptual salience score or authorization to edit.
- The event-map digests are commitments. The caller must verify exact retained descendants, tracks/channels, pitch, onset/end, frame projection and all non-target event invariants.
- One analysis resolution is used in this first tranche; multi-resolution and perceptual features remain future additions.

## Proposed benchmark matrix

| Family | Expected diagnostic |
|---|---|
| Linear additive independent stems | PCM residual at numerical floor |
| Linear summation with overlapping partials | PCM residual at floor; possible magnitude-only residual |
| Fixed linear convolution/reverb | PCM residual at floor when the exact same linear operator and state are used |
| Shared compressor/limiter | PCM residual can be nonzero; calibrate by operating region |
| Saturation/nonlinear distortion | PCM residual should be detectable above a calibrated floor |
| Same-pitch overlapping notes | Identity must remain separate; compare phase and note-release strata |
| Chord and cross-voice overlap | Interaction support can differ by channel/time |
| Non-overlap with long release tails | Distinguish scheduled non-overlap from acoustic tail overlap |
| Seed mismatch / non-repeatable effects | Noise floor rises or the experiment must abstain |
| Wrong coalition / altered protected event | Structural rejection before feature analysis |

## Proposed targets — research goals, not results

- 100% exact coalition identity on deterministic symbolic fixtures.
- PCM residual below a frozen floating-point tolerance on linear additive controls.
- At least 95% detection sensitivity on held-out calibrated nonlinear positive controls while keeping false positives at or below 1% on held-out additive controls.
- Calibration and coverage reported by renderer, effect topology, event density and overlap class.
- No classified result without sufficient repeats, a frozen calibration ID and valid exact-head artifact receipts.
- Report render count, wall time, CPU, peak memory and report size per tested event pair.

The proposed 95% / 1% targets are not achieved results. They must be frozen on development/calibration splits and reported with independent composition-level uncertainty. Repeated seeds on the same musical fixture are not independent composition replications.

## Integration sequence

1. Integrate a coalition compiler over the exact performed-event maps from #9037/#9093.
2. Use #9090 process observations and #9064 artifact receipts; independently verify every receipt before passing its digest to this module.
3. Create matched deterministic FluidSynth quartets and repeat quartets. Keep NativeMuse separate until it provides a schedule-bound execution receipt.
4. Build linear, magnitude-only, reverb, compressor, limiter and saturation controls with held-out compositions.
5. Add multi-resolution features and uncertainty intervals after the one-resolution exact evidence path is qualified.
6. Feed interaction measurements to the #9185 locality-constrained edit repair planner only as a prediction/prior. The planner must still render and independently verify the actual proposed edit.

## Relationship to external benchmarks

SMEBench and MMAE remain the right external comparisons for requested edit execution and unrelated-content preservation. They do not provide the four-coalition ground truth needed to establish event-pair interaction. Report those product-edit metrics independently from this research microbenchmark; do not collapse them into a composite score or claim SOTA from the synthetic controls.
