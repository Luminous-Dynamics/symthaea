# Spacecraft Active Observation: Foveation + Information Gain v0.1

Parent: [#14078 — qualify foveation and information-gain active observation](https://github.com/Luminous-Dynamics/symthaea/issues/14078)

Companion programme: [Symtropy CFI-Space #7828 / #7829](https://github.com/Luminous-Dynamics/symtropy/pull/7829)

## Decision

Use more of Symthaea, but separate three capabilities that must not be conflated:

1. **Temporal observation:** HDC/CfC forecasts flag unexpected changes in a frozen observer profile. Candidate implementation: #14063 / #14064 / #14072.
2. **Compute foveation:** allocate bounded recognition/analysis compute to selected regions in an already captured image.
3. **Active sensing:** choose a future acquisition time, exposure, sensor or viewpoint because its predicted evidence is more informative for the current task.

The current `symthaea-foveation` is only a candidate for (2). It is not already an active-information-gain controller for (3), and its current ventral backend is a stub. Neither component should be wired into spacecraft authority yet.

## Source audit and blockers

At `main` on 2026-11-27, review of `crates/domains/symthaea-foveation` found:

- `FoveationManager::on_saliency` binds request frame ID/time from the current frame buffer.
- `on_frame` may overwrite that single frame buffer while requests are pending.
- `tick` later pops the old request but extracts pixels from the *current* frame buffer, then keeps the old request's frame ID/time. A crop can therefore contain frame B's pixels while being labelled with frame A's identity.
- The manager stores a raw full-frame buffer; the pending `BinaryHeap` has no explicit queue-capacity limit or source-frame lease release contract.
- The ventral stub creates OCR/object/caption strings from a sampled-pixel hash and emits fixed confidence constants (0.3–0.5). That output is not semantic recognition or calibrated confidence.
- The optional real-perception feature/dependency was removed/archived in the crate manifest; the real-backend code is only a documented adoption path.
- `modulate(ne, da)` dynamically changes the surprise threshold and concurrency. This is a resource heuristic, not expected information gain or a safety policy.

**Do not integrate the current foveation manager into CFI-Space as a qualified recognizer.** First remove the identity mismatch and separate the stub from any performance claim.

## Research context

The 15 October 2026 preprint *Illumination-Aware Active Perception for Spacecraft Inspection* (https://arxiv.org/abs/2610.22335) motivates evaluating task-relevant image information instead of treating geometric line-of-sight visibility as sufficient. Its learned image-grounded illumination measure reports improvements in its own simulator; CFI-Space must reproduce a matched baseline before using its figures as a comparison.

The 2026 review of uncertainty-aware spacecraft perception also identifies active uncertainty reduction as an open challenge, alongside calibration, OOD detection, fault detection and deployment evidence:
https://doi.org/10.1016/j.actaastro.2026.07.027

## Distinguish compute allocation from sensing

### A. Compute allocation (first scope)

The camera has already captured the image. The system selects which ROI(s) receive higher-resolution crop processing, OCR/part recognition or additional model passes. This does not change what the camera observed and does not create new viewpoint information.

### B. Acquisition planning (second scope)

The system requests another frame, a different exposure, a different sensor or a different measurement time. These may produce genuinely new evidence. The first experiments remain advisory and non-actuating.

### C. Physical viewpoint selection (out of scope here)

Changing chaser translation, attitude, range or approach trajectory is a physical control action. It requires separately qualified dynamics, GN&C, safety margins, keep-out-zone and retreat-authority evidence. A foveation or FEP score cannot authorize it.

## Hierarchical information model

Use separate uncertainty variables:
- (S): latent target/camera/relative-navigation state;
- (Theta): observation and dynamics parameters (clock, exposure, extrinsics, sensor noise, reflectance, calibration);
- (M): model structure (occlusion, illumination, target geometry, sensor-fault and competing scene explanations).

For candidate observation action (a) and possible outcome (O), define an action-conditioned predictive model (p(Omid S,Theta,M,a)). When the joint distribution is actually specified, the information terms can be decomposed:

[
I(S,Theta,M;Omid a)
= I(M;Omid a)
+ I(Theta;Omid M,a)
+ I(S;Omid M,Theta,a).
]

This is a chain-rule identity, not permission to add unrelated entropy deltas. Missing likelihoods, posteriors or conditional dependencies must remain explicitly unavailable.

A research-only action score can combine task risk, sensing/compute cost and expected information:

[
J(a)=mathbb E[L_{task}mid a]
+lambda_c C(a)+lambda_r R(a)
-lambda_i I(S,Theta,M;Omid a).
]

All terms require declared units, horizons, model evidence and validation. Surprise and novelty remain separate baselines; they are not substitutes for mutual information.

## Required comparison arms

For compute allocation and acquisition planning, report separate experiments.

1. No active policy / fixed compute allocation.
2. Random ROI or round-robin.
3. Highest HDC surprise (current baseline).
4. Motion / optical-flow priority.
5. Image-grounded illumination / component visibility policy.
6. Pose-observability or Fisher-information greedy policy.
7. State information gain.
8. Parameter information gain.
9. Model-structure information gain.
10. Hierarchical state + parameter + structure information policy with task loss and explicit cost/risk.

For every arm, hold image episodes, candidate-visible metadata, ROI/crop dimensions, compute budget, deadline, hardware and preprocessing constant. Do not compare paper-reported metrics from unmatched splits or hardware as a head-to-head result.

## Metrics and benchmark targets

Report:
- realized reduction in pose/part-state error and calibrated uncertainty;
- predicted-versus-realized information gain, ranking regret and calibration;
- visibility/illumination/occlusion quality for small critical components;
- fault-family detection and time-to-detection under clock, calibration, translation and common-cause faults;
- simultaneous episode coverage and sequence silent misses;
- critical-region recall and false ROI selection at fixed compute budget;
- starvation/fairness, dropped requests and deadline misses;
- p50/p95/p99 latency, peak memory and energy/frame on named hardware;
- retained raw-byte count and source/crop identity mismatch count.

Proposed initial targets (not achieved results or mission standards):
- zero source-frame/crop identity mismatches on deterministic contract suites;
- zero unbounded queue growth under declared load envelopes;
- at least 95% recall for declared critical regions at the same fixed ROI budget as baselines;
- at least 10% improvement in realized task-relevant uncertainty reduction or defect/part coverage at equal compute cost before claiming a policy advantage;
- no increase in unsafe pose acceptance; no confidence promotion from low surprise;
- all missed deadlines and unknown model predictions become explicit non-observations.

These goals must be refined using baseline distributions and independent-unit denominators. A zero-count synthetic test is not a universal safety statement.

## Frame-bound request contract

Every ROI request binds:
- source frame ID, sequence, acquisition timestamp, time unit and source byte digest;
- sensor, modality, clock, calibration and preprocessing revisions;
- image dimensions, channel layout, exact ROI bounds and crop-policy digest;
- request ID, episode ID, source-independent target/session identity, deadline and cancellation reason;
- exact crop digest and the source frame identity repeated in the result.

Implement either:
1. an immutable frame lease keyed by source identity and bounded by byte/capacity/deadline limits, with explicit release/zeroization; or
2. a caller-supplied source frame for each dispatch, with byte digest and metadata revalidation before cropping.

A pending request must never crop from a replacement `frame_buffer`. Stale, cancelled, disconnected, incomplete or expired work is a typed non-observation, not a stale recognition result.

## Integration with the temporal observer

The frozen HDC/CfC observer can supply an additional region-novelty cue and forecast residual, but it may not promote pose validity. Keep at least four separate fields:
- representation novelty;
- temporal prediction residual;
- task-relevant expected information gain;
- verified physical/geometric observability.

They are not interchangeable and must each be evaluated/calibrated separately.

FEP's isolated hierarchical information-gain primitive is a candidate for experiments, not proof that the production `ActiveInferenceAgent::select_action` already implements correct action-conditioned expected-free-energy selection. The existing FEP-guided WSP research notes that production action scoring did not establish action-dependent epistemic value and the standalone model-information-gain primitive is not wired to a full active policy.

Do not call the FEP action selector from the spacecraft observer. The experiment policy is a separate advisory process; physical actuation and camera-platform movement stay outside this design.

## Staged qualification

- **P0:** frame-bound request identity, queue lifecycle, deadline and retention contracts.
- **P1:** fix stale-frame/crop mismatch, bound queue capacity and source-frame leases, test overwrite-before-dispatch, stale completion, cancellation, duplicate identities and digest substitution.
- **P2:** replace/qualify the ventral backend; no fixed stub confidence in any performance result.
- **P3:** frozen synthetic CFI-Space policy experiments, independent scorer and matched baselines.
- **P4:** rights-cleared public-real data, independently checked part/pose/illumination labels and grouped holdouts.
- **P5:** HIL sensor/clock/exposure/dropout/compute-overload tests. Physical viewpoint policies require a separate mission-specific assurance case.

## Claim ceiling

This specification makes no claim of semantic recognition performance, calibrated information gain, physical observability, pose validity, active-viewpoint safety, HIL/physical qualification, flight readiness or SOTA superiority. Symthaea foveation remains a research candidate until its provenance, retention, backend and evaluation contracts pass.
