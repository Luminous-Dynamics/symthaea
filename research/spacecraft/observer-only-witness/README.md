# Spacecraft Observer-Only Temporal Perception Witness v0.1

Parent: [#14063 — observer-only spacecraft temporal perception witness](https://github.com/Luminous-Dynamics/symthaea/issues/14063)

Companion benchmark: [Luminous-Dynamics/symtropy#7828 — CFI-Space](https://github.com/Luminous-Dynamics/symtropy/issues/7828)

## Executive decision

Use Symthaea's `symthaea-vision-manifold` as a **temporal anomaly-observation candidate**, not as a pose estimator, source authenticator, probability engine, or controller.

The intended contribution is an additional, separately attributable witness that learns/represents how a sensor stream evolves and reports when current patch-level observations depart from its frozen temporal expectations. It may increase scrutiny, mark a stream as suspect, or request an evidence-gathering action through an independently authorized policy. It must not grant permission to accept a pose or command a maneuver.

## Why this is the right Symthaea substrate

The existing crate implements:
- patch-based spatial encoding into `ContinuousHV` with deterministic seeded basis construction;
- CfC manifold state evolution and next-frame prediction;
- patch-level novelty from current/previous HDC patch similarity;
- optional coarse-to-fine prediction error;
- a delayed-horizon evaluator that compares forecasts against later observations;
- a checked raw-frame interface and telemetry including prediction residual, coherence, surprise/attention entropy and runtime.

This is more directly relevant than wiring the whole cognitive loop, language stack, consciousness indicators or an unrestricted active-inference agent into the perception path.

The existing `dream_ahead` implementation uses identity/permute-style abstract evolution and must not be confused with a validated rigid-body spacecraft simulator. The HDC latent is not metric SE(3). The default frame-level prediction error and FEP/free-energy fields are not calibrated probabilities.

## Proposed component boundary

Create a narrow crate, e.g. `symthaea-spacecraft-observer`, depending only on `symthaea-vision-manifold` and minimal serialization/hash utilities. Do not depend on the root `symthaea` cognitive loop.

### Input contract

Every observation is bound to:
- `episode_id`, `target_id`, `session_id`, `trajectory_id`, `frame_id`, monotonic sequence index;
- acquisition timestamp, time unit, clock profile/revision and timestamp-validity evidence;
- modality, sensor identity, frame dimensions, channel layout, crop/ROI digest;
- camera/rig calibration digest, preprocessing and encoder-config digest;
- fixed model/checkpoint/seed/profile digest;
- raw source bytes digest, source-evidence reference and rights class;
- fixed episode horizon, expected frame budget and declared missing-frame policy.

Input validation must complete before the manifold is updated. A rejected frame does not advance the sequence clock, temporal history, learning state or forecast evaluator.

### Witness contract

Proposed type: `TemporalPerceptionWitnessV0_1`.

Required fields:
- schema version, observer profile ID and implementation/build digest;
- all episode/frame identity and source/profile digests echoed from validated input;
- disposition: `WARMING_UP`, `OBSERVED`, `INPUT_REJECTED`, `OUT_OF_SCOPE`, `MISSING_HISTORY`, `INTERNAL_ERROR`;
- forecast horizon(s), warm-up count, patch-grid dimensions and valid patch count;
- per-patch fixed-point novelty / prediction residual map with explicit quantization, range and rounding;
- global representation residual, coherence, and delayed-horizon skill with sample counts and baseline comparator;
- input modality, geometry and time profile;
- encode/evolve/total latency; memory/energy only when actually measured;
- explicit constant claim flags:
  - `observer_only=true`
  - `learning_frozen=true`
  - `raw_frame_retained=false`
  - `actuation_authority=false`
  - `pose_probability_claim=false`
  - `source_authentication_claim=false`
  - `physical_truth_claim=false`.

A value is not a cryptographic digest or probability merely because it is deterministic or normalized. HDC similarity and free-energy are internal representation diagnostics only.

## Observer-only invariants

A dedicated profile must establish these properties by construction and tests:

1. **No learning on evaluation data.** Weights, time constants, feature weights, adaptive thresholds and any optimiser state are frozen before the first episode frame. State and weight digests are unchanged after the episode.
2. **No raw-frame retention.** The present manifold stores `last_observed_frame` in its active state and in checkpoints. Add an explicit retention policy and prove that observer mode stores no raw bytes in live state, scene memory or checkpoint. No post-hoc deletion as a substitute for non-retention.
3. **No action execution.** Skip/bypass FEP `select_action`, behaviour mapping, geodesic/path selection, intent/swarm, dream replay, scene-memory writes and external side effects in observer mode. If FEP perception is retained for diagnostics, it cannot choose or mutate a behaviour.
4. **No silent stale output.** Use `VisionManifold::try_new` and `observe_frame_checked`. A malformed frame must yield an error/disposition, never a previous telemetry snapshot labelled as current.
5. **No hidden profile changes.** Pin HDC dimension, patch size/grid, feature channels, temporal-binding setting, predictive-hierarchy setting, horizon schedule, seed, model weights and software build for each scored episode.
6. **No implicit dilation.** Disable automatic dimensionality adaptation; memory preflight and profile identity are immutable for the scored episode.
7. **Explicit warm-up and time continuity.** Cold start, modality switch, out-of-order frame, duplicate frame, dropped frame, timestamp gap, clock revision change and horizon expiration have typed statuses.
8. **Bounded data handling.** Validate image byte count, dimensions, ROI and per-patch output count; reject non-finite telemetry; define integer conversion, saturation and overflow.
9. **No cross-modal conflation.** Each modality keeps its own temporal context, identity and calibration. A common observer implementation does not prove independence between modality streams.
10. **Reproducible receipts.** Bind exact input bytes/profile to output; independent challenger re-computes the witness from the same frozen fixture and checks canonical output parity.

The current `VisionManifold` has a private `learning_frozen` field (initialized false), a raw-frame retention field, and an FEP action-selection / vision-behaviour mapping path inside `observe_encoded`. These are implementation gaps for this profile, not features to work around at the adapter layer.

## Research questions / ablations

| Question | Arms |
|---|---|
| Is HDC patch structure useful beyond cheap novelty? | RGB residual, optical flow, patch cosine novelty, HDC patch representation |
| Does CfC learn temporal continuity beyond persistence? | persistence, linear/AR, CfC-only, HDC+CfC |
| Does coarse-to-fine prediction help? | hierarchy off/on, identical compute budget |
| Does temporal binding help? | off/on, fixed sequence group splits |
| Which patch scales transfer? | 4/8/16/32 px, fixed memory and compute budget |
| Do motion/color channels add robust evidence? | luminance-only, motion-only, color/opponent, combined |
| Is depth useful or harmful under poor sensing? | no depth, calibrated measured depth, missing/uncertain depth controls |
| Does online adaptation inflate evaluation? | frozen confirmation only; unfrozen development-only arm |
| Is the signal calibrated? | raw residual, held-out calibration on independent units, risk-coverage and silent-miss analysis |

Baselines:
- previous-frame persistence and robust pixel residual;
- optical-flow/motion baseline where data supports it;
- lightweight ConvLSTM / temporal CNN if compute matched;
- independent RGB/event pose estimators (pose truth scored separately);
- existing Symtropy P1 consistency gate and conformal sequence-region methods.

Do not compare a paper's published score with this sidecar as though the experiments were head-to-head. Use exact datasets, splits, preprocessing, hardware, budget and tolerances for comparative claims.

## Promotion protocol

| Stage | Required evidence | Claim ceiling |
|---|---|---|
| P0 contract | schema, deterministic IDs, validation, invariants | interface only |
| P1 synthetic | finite known-answer fault cases; Rust/challenger parity; negative controls | scorer/contract semantics |
| P2 public real | lawful data, independent labels, target/session/trajectory splits | named-dataset performance |
| P3 blind | evaluator custody, no test-time learning, no label leakage | blind performance on declared corpus |
| P4 HIL | exact sensors/clock/compute, injected clock/extrinsic/dropout faults, resource/thermal logs | HIL profile only |
| P5 physical | mission-specific independent validation and safety case | only exact physical/operational scope |

Metrics are per fault family and non-compensatory:
- detection recall at fixed 0.1%, 1% and 5% false-alarm rates;
- per-family AUROC/AUPRC as diagnostics;
- delay to detection in frames and milliseconds;
- risk-coverage, silent-miss and false-abstention rates;
- sequence-level failure rate and complete-horizon outcomes;
- p50/p95/p99 latency, peak RAM, energy/frame and thermal envelope;
- reproducibility, rejected-input coverage and exact receipt parity.

The benchmark unit is the target × session/trajectory × sensor rig/configuration, not the frame. Every counterfactual sibling of an episode stays in the same split.

## Initial aspirational goals (not results or standards)

For a finite, explicitly enumerated development/confirmation fault suite:
- zero silent accepts in deterministic known-answer critical fault cases;
- ≥95% recall at a 1% false-alarm point on the declared fault families;
- ≤5% false abstention on qualified nominal episodes;
- ≥90% complete-sequence detection/coverage target only after a separate, valid calibration protocol;
- report detection latency and compute on fixed edge hardware.

These should be tightened only after baseline distributions, denominators, confidence intervals and mission-specific tolerances are measured. The zero-count finite test target is not a universal zero-risk claim.

## Integration with Symtropy

Symtropy consumes the witness as a separately typed `TEMPORAL_NOVELTY` / `TEMPORAL_PREDICTION_RESIDUAL` observation. It must not:
- average the witness with pose estimates;
- treat it as a covariance or confidence multiplier;
- infer sensor independence;
- authorize maneuver or physical actuation;
- clear a mismatch in #7813 or replace independent-unit calibration #7801.

Permitted downstream effects are conservative: request re-observation, flag a frame/sequence for investigation, increase review priority, or return an explicit insufficient-evidence state. Any sensing action is only an advisory proposal until separately authorized by the existing action/evidence gates.

The public Symtropy repository currently defines `stubs/symthaea` as a minimal standalone type-compatible stand-in; its `vision-manifold` feature is empty. That stub is not the real upstream `symthaea-vision-manifold`. Integration therefore needs a versioned artifact/IPC contract or a consciously pinned upstream dependency and build provenance; a feature flag alone is not integration.

## Claim ceiling

This is an integration specification. It makes no pose-estimation, probability-calibration, sensor-independence, source-authentication, physical-truth, HIL, flight-readiness or SOTA superiority claim. HDC/CfC telemetry is an additional signal to test; Symtropy's independent geometry, clock, common-cause, pose-region and action-admission gates remain authoritative.
