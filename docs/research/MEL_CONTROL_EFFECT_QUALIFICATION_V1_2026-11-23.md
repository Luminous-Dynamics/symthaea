# MUSE control-effect qualification v1

**Date:** 23 November 2026  
**Parent:** [#10196 — capability-qualified musical realization backends](https://github.com/Luminous-Dynamics/symthaea/issues/10196)  
**Composed with:** #12841, #13348, #13349 and #13358

## Research shift

A renderer accepting `BPM=120` is not proof that the output follows 120 BPM. An API acknowledgement proves request handling, not control application. A renderer-reported effective setting is still not proof that the waveform changed. A waveform change is not proof that a musician perceived the intended musical effect.

The evidence ladder is therefore:

1. **Declared capability** — a versioned backend manifest claims a control exists.
2. **Accepted request** — a specific intervention request was parsed.
3. **Applied setting** — the backend reports the effective value.
4. **Independent output effect** — a capture-bound observer measures the target effect and collateral drift.
5. **Human-perceived effect** — musicians/listeners recognize the intended change and consider it useful.

Do not promote evidence from one layer into a stronger layer by assumption.

## Implemented Rust contract

The first executable slice is `crates/domains/symthaea-muse/src/control_effect.rs`, exported as `symthaea_muse::control_effect`.

It defines:

- `BackendCapabilityManifestV1`: backend/model/adapter identity, backend identity digest, typed controls, support class, value range, unit, target observation metric, minimum effect, sham-equivalence tolerance, collateral-drift bound and evidence references.
- `ControlInterventionRequestV1`: request ID, counterfactual pair ID, active/sham/reversal condition, exact manifest and backend identity digests, baseline control value, requested value, expected observed-effect direction, baseline capture digest, baseline closure digest and baseline observer-manifest digest.
- `ControlExecutionReceiptV1`: exact request/backend linkage, applied/accepted/ignored/rejected/unsupported/failed status, effective value and output capture/closure/observer-manifest references.
- `ControlOutputObservationV1`: capture recorder and event-observer identities, producer identity, exact baseline/output capture digests, separate baseline/output closure and observer-manifest digests, separate structural-eligibility states for both windows, observer method version, exact target-metric contract digest, observer-qualification and metric-calibration receipt digests, baseline/output metric values with separate uncertainty radii, and collateral drift with uncertainty.
- `evaluate_control_effect_v1`: derives the signed target effect as output metric minus baseline metric, adds the baseline/output uncertainty radii conservatively, validates the reference chain, and evaluates the resulting uncertainty interval. The caller cannot supply a trusted precomputed delta.
- `tools/muse-control-effect-qualifier` imports the exact production `control_effect.rs` by relative path as a public module. The isolated package runs the same 19 unit tests and warnings-denied Clippy on that exact source without inheriting unrelated Clippy warnings across the wider legacy `symthaea-muse` crate. There is no copied/forked evaluator implementation. The root workspace lock explicitly includes the harness package and its already-pinned serde/serde_json/sha2 dependency graph.
- `qualify_counterfactual_control_triplet_v1`: qualifies a capability only when a matched active intervention produces the predicted effect, a sham remains inside its predeclared equivalence band, and a reversal moves the control value to the opposite side of the same baseline and produces the opposite observed effect. All three arms must share the exact baseline capture, baseline closure and baseline observer-manifest digests.

All numeric values use fixed-point thousandths. A sham is not just "no API call"; it is a randomized request with the same pipeline and baseline control value, designed to expose drift, hidden state, session carryover and accidental side effects.

## Verdict semantics

| Verdict | Meaning |
|---|---|
| `CapabilityUnsupported` | Manifest or execution says the control is unavailable |
| `NotExecuted` | Request was rejected or failed |
| `Unobserved` | No independent measurement or capture closure is incomplete/unavailable |
| `ObservedEffectVerified` | Target effect's entire uncertainty interval clears the predeclared minimum effect in the expected direction |
| `ObservedEffectInconclusive` | The interval does not establish the minimum effect or a strong opposite effect |
| `ObservedEffectContradictory` | The measured interval establishes an effect in the opposite direction |
| `ShamNoChangeVerified` | The worst-case interval lies inside the sham-equivalence band |
| `ShamLeakage` | The no-op/sham produced an effect outside its equivalence band |
| `ObservedEffectUnattributed` | Output changed as expected but the backend did not attest applying the requested setting |
| `CollateralDriftExceeded` | Non-target disturbance exceeds its predeclared upper bound |
| `HardControlValueMismatch` | A hard guarantee reports an effective value different from the requested value |

An accepted request without independent output evidence is `Unobserved`, never successful. An ignored control followed by an audio change is `ObservedEffectUnattributed`, not proof the control worked.

## How this composes with existing work

- [#12841](https://github.com/Luminous-Dynamics/symthaea/issues/12841) owns the factorial symbolic control surface, contrasts and invariance sets. Do not create another factorial generator sweep.
- [#10196](https://github.com/Luminous-Dynamics/symthaea/issues/10196) owns backend capability manifests, routing, guarantees, downgrade and abstention.
- [#13348](https://github.com/Luminous-Dynamics/symthaea/pull/13348) owns output-observation closure and the distinction between detector-reported events and unobserved windows.
- [#13349](https://github.com/Luminous-Dynamics/symthaea/issues/13349) owns the live independent loopback/MIDI observer and clock calibration.
- [#13358](https://github.com/Luminous-Dynamics/symthaea/pull/13358) binds immutable backend/adapter/runtime identity across RMI counterfactual arms.
- [#13254](https://github.com/Luminous-Dynamics/symthaea/issues/13254) remains the signer-role/authenticated provenance and trusted execution boundary.

This module derives the signed delta from the observer-reported baseline and output metric values, but it does not reprocess captured waveform bytes to recompute those values. It checks metric-contract and receipt digest linkage plus structural eligibility passed from the observation-closure evaluator. It does **not** authenticate the receipts, validate the calibration study, validate event-detector sensitivity/specificity, independently redo #13348's loopback closure arithmetic, or prove that a declared backend identity is the binary that physically ran. Those require the linked owners above.

## Initial conformance suite

Twenty-three synthetic Rust tests currently exercise:

- active effect verified from an independent output measurement;
- accepted API request without an observation remains unobserved;
- ignored control with a coincident output change is not attributed to that control;
- sham leakage detection;
- incomplete baseline or output observation cannot create a positive/negative label;
- mismatched baseline closure reference rejected;
- mismatched baseline observer manifest across counterfactual arms rejected;
- intervention request without baseline closure/manifest receipts rejected;
- capture/event observer identity separation from producer;
- output capture and observer-manifest receipt linkage;
- collateral drift rejection;
- baseline/output uncertainty aggregation that correctly downgrades a nominal effect to inconclusive;
- metric-contract digest substitution rejection;
- missing metric-calibration receipt rejection;
- tampered capability-manifest payload rejection;
- hard-control effective-value mismatch;
- contradictory observed effect;
- successful matched active/sham/reversal triplet;
- reversal with the wrong observed direction does not qualify;
- cross-arm baseline mismatch is rejected.

These are synthetic evaluator controls. The click-track fixture generates PCM and measures its own onset spacing, but it is not a commercial renderer, physical loopback capture, calibrated production observer, audio event-detector qualification, or live Melothaea output measurement.

## Prospective benchmark targets

| Track | Initial target | Stretch |
|---|---:|---:|
| Declared capability conformance probes | 100% version-pinned | 100% across supported backend versions |
| Unsupported exact-control requests silently accepted | 0 | 0 |
| Active effect direction on confirmatory trials | ≥95% | ≥99% |
| Sham false-positive rate | ≤5% | ≤1% |
| Reversal direction success | ≥95% | ≥99% |
| Scored conditions with complete independent output evidence | 100% | 100% |
| Non-target drift | Within predeclared task-specific equivalence bounds | Same with narrower bounds |
| Control-release persistence/carryover | Report decay time and residual drift per backend | Predictable and reversible |

These are prospective gates. Confidence intervals must be clustered by session/musician and backend version; the discovery panel must not double as the confirmatory holdout.

## Control dimensions to probe

- Tempo: requested BPM vs independently estimated BPM and beat-phase stability.
- Tonal center: requested key/scale vs chroma/key posterior and chord-root distribution.
- Note density: requested events/bar vs onset density and polyphony.
- Brightness/timbre: requested control vs spectral centroid/roll-off, with loudness controlled.
- Groove: target accent pattern vs onset/accent F1 and phase.
- Role entry: requested instrument/role onset vs independent role/event detection.

For each dimension, declare a target metric, units, effect direction, minimum meaningful effect, uncertainty method and collateral-invariance set **before** execution. A control may be nonlinear or interaction-dominated; do not force monotonicity when the data disagree.

## Current industry comparison

- Artificial Analysis AA-Music v1.1 is a strong finished-track blind-preference reference, with separate vocal and instrumental boards and Bradley–Terry/Elo confidence intervals. Its task is not control-effect conformance.
- Taan TimingBench measures prompted timed events within ±500 ms, a useful timing-generation comparator but not beat-phase or live reciprocal interaction.
- Daydream reports parameter-change-to-sound latency and independent chord, pitch, onset and groove detectors. Its reported 81 ms is a parameter-to-audio endpoint, not musician input-to-acoustic-onset.
- ICASSP 2027 MMAE separates instruction following, preservation consistency and exact match for audio edits. The public resource and hidden test have separate roles; the current official page extends the final submission deadline to 28 November 2026, but registration closed 8 October.
- ElevenLabs Music inpainting inserts selected stored-song audio-reference chunks unchanged while generating other chunks. This is a useful targeted-edit comparator; exact preservation of the referenced byte region and perceptual transition quality should be scored separately.

The strategy is not to claim Melothaea already beats these systems. It is to expose a missing evaluation axis: did the requested control actually change the output in the intended way, without collateral damage, and can an independent observer demonstrate that?

## Next increment

After this contract is qualified, add a version-pinned mock renderer that intentionally:
1. applies BPM correctly;
2. accepts but ignores BPM;
3. reports an effective value that differs from the request;
4. applies a control only after a delay;
5. leaks a sham effect through session carryover;
6. applies the target effect but disturbs a protected dimension;
7. changes model/adapter identity between active and reversal arms.

Then connect one real symbolic/MIDI renderer and one streaming audio backend only after each has a conformance manifest and a real independently observed output path.
