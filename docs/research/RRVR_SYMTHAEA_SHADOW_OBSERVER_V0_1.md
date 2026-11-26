# RRVR–Symthaea Shadow Recovery Observer V0.1

**Status:** research/implementation constitution, 2026-11-26. This defines a bounded integration and evaluation plan. It reports no observer benchmark result, simulator campaign, RoboRecover reproduction, HIL result, or physical robot result.

**Tracking issue:** [RRVR-SX-001 / #13764](https://github.com/Luminous-Dynamics/symthaea/issues/13764)

**RRVR counterparts:** [Symtropy #7583](https://github.com/Luminous-Dynamics/symtropy/issues/7583), [#7586](https://github.com/Luminous-Dynamics/symtropy/issues/7586), [#7595](https://github.com/Luminous-Dynamics/symtropy/issues/7595), [#7598](https://github.com/Luminous-Dynamics/symtropy/issues/7598).

## 1. Thesis

RRVR remains the independent execution-integrity and evidence-qualification layer. Symthaea adds a read-only cognitive observer that reasons about trajectories, predicts decision-relevant outcomes, recognizes its own uncertainty, and recommends the next useful observation.

The observer has **no actuator, dispatch, permit, endpoint-fence, effect-verdict, or authoritative evidence-ledger interface**. Its outputs are hypotheses and predictions. The native robot owner remains the only action-authority owner.

Keep these four questions distinct:

1. **Cognitive prediction:** What might be happening, what may happen next, and what observation would distinguish explanations?
2. **Execution integrity:** Which commands, attempts, authority epochs, queue generations, and endpoints were actually involved?
3. **Physical adjudication:** Did the intended effect occur, and what independent observations support that conclusion?
4. **Decision authority:** Is a new action currently permitted by the robot's native owner?

No layer silently substitutes for another.

## 2. Existing implementation inventory

Prefer reuse, but do not infer a capability from a crate name.

| Existing component | Appropriate first use | Boundary / required validation |
|---|---|---|
| symthaea-core HDC and PredictiveHdcEncoder | Encode observations, state/event descriptors and trajectory fragments; retrieval and novelty features | Must beat simple encodings and baselines on held-out trajectories; an embedding is not a dynamics model |
| symthaea-fep | Advisory prediction-error, uncertainty and observation-attention experiments | No action authority; calibrate against known outcomes |
| symthaea-digital-twin | Store telemetry, prediction residuals, epistemic/aleatoric uncertainty | Current ingestion takes caller-supplied expected value/tolerance; it does not establish a physical state or independently predict dynamics |
| symthaea-control | Kalman/LQG/MPC as baselines or explicit model-based estimators | Record estimator assumptions, process/noise models and operating envelope |
| symthaea-humanoid | Separate push/fall-recovery locomotion test lane | Do not pool biped locomotion with manipulation recovery or claim RoboRecover equivalence |
| decision-predictability-benchmark | Existing finite-world EVSI and prior-free robust decision-value evaluator | Reuse; do not create a competing EVSI engine. The evaluator does not execute probes |
| symthaea-formal-safety | Record proposed assurance obligations and evidence references | Current convenience discharge changes status when a reference is attached; that alone is not proof checking |

**Explicit non-use:** symthaea-vision-manifold PredictiveCodingHierarchy.dream_ahead currently evolves its coarse state using a placeholder identity-like permutation. It must not be described or benchmarked as a qualified physical world model. The frame encoder's measured residual/novelty may be tested as a feature, subject to held-out visual evaluation.

Do not pull in the monolithic cognitive loop for the first integration merely because it is comprehensive. Begin with an isolated crate and small trait surfaces.

## 3. Architecture

Immutable RRVR episode/evidence snapshot
→ strict input and decision-cut validator
→ read-only observation normalizer
→ parallel feature paths: HDC/trajectory features, visual/state novelty, and model-based estimators
→ calibrated predictive ensemble
→ OOD and selective-prediction gate
→ advisory hypothesis report
→ decision-value observation planner (EVSI when justified; robust minimax otherwise)
→ advisory next-observation proposal
→ RRVR/native owner independently decides.

The observer reads an immutable snapshot at a declared decision cut. It cannot append authoritative evidence, adjudicate Applied/NoEffect, change a replay audit, or authorize a retry. A report links to the exact snapshot digest; it is not itself evidence that the snapshot's contents are physically true.

## 4. Input and temporal semantics

Consume only versioned RRVR records and raw artifact references admitted by the relevant evidence domain. Every observation retains:

- scenario, task, policy, checkpoint, attempt, and trajectory-cluster identity;
- event time, observation/acquisition time, receiver ingest sequence, source clock/domain, and time uncertainty;
- camera/modality, frame index, image dimensions/encoding/transfer assumptions where applicable;
- pose/state values, coordinate frame, units, uncertainty and calibration references;
- raw artifact digest, parent/lineage references, and evidence snapshot digest;
- host replay status and exact replay-profile digest;
- simulator, robot, controller, sensor and rendering profile identities;
- missing, stale, delayed, reordered, duplicated, unknown and disputed states.

Do not collapse event time, observation time and ingest time into one timestamp. Do not use receiver wall-clock time to overwrite source event time. Do not regularize or interpolate an irregular stream without emitting a separately identified transformation and preserving the original stream.

Only records available at the decision cut may be visible to the observer. Hidden simulator state, evaluator labels, future frames, intervention outcomes, test-set normalization, and post-cut evidence are inaccessible. Invalid, stale, wrong-scope or incomplete inputs yield typed abstention/rejection, not silent repair.

## 5. Output contract

A versioned observer report contains:

1. **Input binding:** evidence snapshot digest, decision-cut identity, source episode/attempt/cluster, exact observation IDs and model/configuration/checkpoint digests.
2. **Representation:** trajectory descriptors, HDC vector/profile identity, nearest analogue IDs and retrieval distances.
3. **Prediction:** horizon-indexed state/event/recovery distributions; baseline predictions; per-regime calibration profile.
4. **Uncertainty:** epistemic and aleatoric contributions, out-of-distribution status, calibration support and coverage limitations.
5. **Anomaly:** residuals, novelty, detection latency and the exact observations responsible.
6. **Hypotheses:** candidate causal explanations, supporting/contradicting observation IDs, assumptions, and a clear hypothesis-only label.
7. **Selective output:** predict / abstain / OOD / insufficient-input disposition with reason codes.
8. **Observation advice:** candidate next observation, expected decision value or robust worst-case risk reduction, acquisition cost, deadline and resource budget.
9. **Resource receipts:** measured CPU/GPU time, memory, observation count, data bytes, energy where measured, and p50/p95/p99 latency.

The schema must not contain action capabilities, permit IDs, endpoint-fence claims, effect adjudication, physical-observation claims, or fields that RRVR can interpret as authorization.

## 6. Decision-focused observation selection

Use the existing decision-value evaluator rather than maximizing entropy or mutual information by default.

- Use **EVSI** only when prior probabilities, outcome likelihoods, conditional-dependence assumptions, action losses and observation costs are defensible and frozen.
- Use the existing **prior-free robust minimax** evaluator where only set-valued outcome supports and worst-case losses are justified.
- Keep EIG/mutual information, covariance distance, EVSI and robust loss reduction as distinct quantities.
- A high-information observation with no downstream decision value is a required negative control.
- A low-information observation that changes the preferred decision or prevents a critical error is a required positive control.
- The observer proposes an observation. It cannot initiate acquisition or override permissions, resource limits or deadlines.

The existing finite-world evaluator is not a probe executor. The independent execution layer must later report whether a proposed observation was permitted, acquired, delivered, complete and independently observed.

## 7. Evaluation lanes

### Lane A — process-model software qualification

Use frozen RRVR deterministic fault-harness traces and known-truth synthetic streams. Keep the evidence domain process_model. This lane tests parser, time semantics, digest binding, anomaly features, calibration plumbing, decision-cut enforcement, OOD abstention and deterministic output. It is not simulator evidence.

### Lane B — simulator qualification

Use independently captured simulator records, not process-model traces relabelled as simulator. Require simulator-native reset/action/replay receipts, pinned simulator and model profiles, raw observation digests, and actual simulator state/outcome labels kept evaluator-side. First integrate with the simulator already used for RRVR; do not create a new unconnected toy world.

### Lane C — host benchmark reproduction

For RoboRecover, first reproduce the host protocol and metrics from the authors' exported archive, checkpoints/weights, and required simulator assets when those materials are actually available. Do not use an issue fixture as a dataset substitute.

Retain the corrected replay structure:

- 20 difficulty-screening trials under the host's four reference policies;
- select the three replay attempts using the recorded seed-42 selection procedure and bind all selected IDs/indices to the screening ledger;
- for each selected attempt, compare five original/replayed pre-recovery frame pairs and independently measure end-effector and object positions;
- compute mean SSIM over the five frames for each attempt; require each of the three attempts to meet the host per-attempt criterion and pose thresholds;
- do not require every individual frame score to exceed 0.99 unless a separate rule says so.

For RoboTwin, keep the repository verifier as a separate, stricter profile: reconstruct initial state, compare configured joint/pose/rotation/gripper state, require exact action-step count, then compare all requested camera videos across every required frame. These results must not be replaced by the paper's three-attempt SSIM spot check.

Only after host fidelity passes should native-recovery and RRVR-overlay arms be paired from the same replay-qualified checkpoint.

### Lane D — separate humanoid locomotion

Use symthaea-humanoid push/fall recovery to stress state prediction and anomaly detection for locomotion. Report it separately from object manipulation and benchmark recovery.

## 8. Baselines, ablations and metrics

### Predictors

Where appropriate and faithfully reproducible, compare persistence, constant velocity/acceleration, Kalman/LQG, AR, ESN, NG-RC/NGRC-MDS, compact GRU/LSTM/TCN/state-space models, HDC-only, CfC/LTC-only, HDC+CfC/LTC, and a causal-discrimination hybrid. A privileged oracle model is a non-deployable upper bound, never a candidate score.

All arms receive identical observations, training data, tuning budget, inference horizon and resource accounting. Mark mismatched assumptions non-comparable rather than forcing a rank.

### Observation policies

Compare fixed cadence, event/threshold-triggered, uncertainty-triggered, entropy/EIG, EVSI and robust minimax selection. Ablate HDC, temporal predictor, causal-world discrimination, calibration, analogue retrieval, observation selection and abstention.

### Separate endpoints

- prediction: horizon-wise proper scores, error curves, event Brier/log score;
- uncertainty: coverage, reliability, calibration error and worst-regime/OOD coverage;
- selective safety: risk-versus-coverage, unsupported-prediction rate and abstention;
- decisions: paired regret, missed-event exposure, false interventions, decision deadline success and worst-regime loss;
- observations: net EVSI / robust minimax reduction, acquisition cost, latency and deadline adherence;
- anomaly: detection probability, false alarms and detection delay;
- causal hypotheses: precision/recall only against independently authored, held-out causal labels; do not score prose similarity;
- generalization: held-out scenario, task, trajectory cluster, parameter family, robot/controller and simulator profile;
- resources: p50/p95/p99 latency, CPU/GPU time, memory, sensor samples, bytes and measured energy.

There is no weighted aggregate that allows better utility to compensate for a hard-safety regression.

## 9. Statistical controls

- Freeze test roster, episode IDs, observation cut, model and checkpoint hashes, calibration data, metrics, exclusions and stopping rules before confirmatory runs.
- Split by scenario, task, trajectory cluster, parameter family and simulator/configuration profile. Never randomly split adjacent frames from the same episode across train and test.
- Preserve paired scenario inputs but keep realized per-arm outcomes distinct.
- Bootstrap whole clusters, not frames or episodes independently where repeated episodes share a trajectory.
- Preserve every scheduled episode, including invalid, missing, disputed and unresolved strata. Report exact denominators.
- Do not use the three host replay attempts as 3 independent evaluation scenarios. They are within-scenario replay qualification attempts.
- Independently recompute aggregate metrics from stored predictions and evaluator-side outcomes.

## 10. Initial gates and targets

These are proposed acceptance targets, not measured results:

1. 100% of observer outputs bind to exact snapshot, decision cut, model, calibration and configuration digests.
2. 100% rejection/abstention for stale, malformed, future-leaking, wrong-scope or incomplete observations in the frozen adversarial corpus.
3. Zero hidden-state and future-label leakage in negative controls.
4. Deterministic repeated execution produces byte-identical canonical reports under a pinned toolchain/profile.
5. No calibration pass may rely only on an aggregate: report per regime, task, cluster and OOD stratum.
6. At least one preregistered task shows lower paired decision regret than the strongest eligible baseline, with uncertainty intervals excluding no improvement and no material hard-safety regression; otherwise publish the null result.
7. Report complete resource vectors and inference overhead; do not claim faster/cheaper without matched target hardware and workloads.
8. A second implementation independently recalculates report metrics from frozen predictions and outcome records.

The first objective is a trustworthy observer, not beating a robotics task-success leaderboard by optimizing a surrogate.

## 11. Implementation sequence

1. Define a minimal read-only episode-observation trait and strict versioned input schema. Add decision-cut and clock-domain validation first.
2. Build a deterministic trajectory feature baseline using HDC and transparent state features; compare against persistence and constant-velocity baselines before introducing learned dynamics.
3. Add anomaly / OOD and abstention reports. Freeze model IDs, feature scaling and calibration data.
4. Add one predictive baseline at a time (Kalman/model-based, AR, then CfC/LTC/HDC hybrids only where a real temporal model exists).
5. Wire the existing EVSI and robust minimax evaluator to permitted observation candidates, without exposing evaluator truth to the candidate.
6. Add the independent metrics recomputation tool and cluster-aware inference outputs.
7. Connect the observer in shadow mode to RRVR simulator episode records, while RRVR's existing qualification remains unchanged.
8. Only after simulator evidence and host replay fidelity are established, preregister the confirmatory RoboRecover overlay.

Every stage preserves the claim ceiling: software → process model → simulator → HIL → physical. No tier promotion by metadata.

## 12. Research references and comparison targets

- RoboRecover, [arXiv:2609.28952](https://arxiv.org/abs/2609.28952): recovery from execution deviations, fixed host splits, replay fidelity.
- LIBERO-RECOVER, [arXiv:2609.05178](https://arxiv.org/abs/2609.05178): recovery-level stratification.
- SafeVLA-Bench, [leaderboard](https://safevla.org/leaderboard.html): task success separated from safety / successful-but-unsafe.
- VLA-FixBench, [ICML 2026](https://icml.cc/virtual/2026/poster/10012345): failure diagnosis and recovery under static, dynamic and real-world tasks.
- RoboCasa, [leaderboard](https://robocasa.ai/leaderboard.html): multi-task manipulation success and held-out composite tasks.
- Symthaea, [Decision-Relevant Predictability benchmark constitution](https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/research/DECISION_RELEVANT_PREDICTABILITY_CHAOS_BENCHMARK_V0_2026-11-17.md): prediction beyond pointwise horizon, decision regret, calibration, and full resource accounting.
- Symthaea, [Decision-Relevant Predictability evaluator](https://github.com/Luminous-Dynamics/symthaea/tree/main/research/decision-predictability-benchmark): implemented finite-world EVSI and prior-free robust decision-value primitives.

The external benchmarks are complementary, not numerically interchangeable. RRVR's proposed differentiation is a policy-independent measurement of execution-integrity preservation under recovery, with independent effect adjudication and calibrated uncertainty—not a claim to replace task-success or VLA-safety leaderboards.
