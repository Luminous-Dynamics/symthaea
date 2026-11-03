# SAM Reference-Model Triangulation Protocol v0.1

**Status:** research protocol proposal; documentation-only  
**Scope:** simulation and offline evaluation for Symthaea Autonomous Mobility (SAM)  
**Owners:** Symthaea benchmark/model-discrepancy work; Symtropy execution and simulation-credibility work  
**Protocol ID:** `SAM-REF-TRIANGULATION-001`

## 1. Problem statement

A controller can appear robust because it is evaluated against a flawed world model. A second model does not automatically solve this problem: different model names, source repositories, artifact digests, or simulator processes do not establish independent evidence.

Models may share code, equations, parameter sources, calibration data, training data, labels, numerical methods, sensor assumptions, or simplifying assumptions. An optimizer can exploit a defect shared by every evaluator. Agreement among correlated models is not independent confirmation.

The protocol therefore changes the question from:

> Does the controller succeed in the simulator?

to:

> Which independently supported claims about this outcome survive explicit model-applicability limits, common-mode analysis, independent checks, and observed evidence?

This protocol does not add a simulator, a reference physics engine, a safety authority, or a second evidence ledger. It composes existing owners:

- Symthaea #10257: horizon-conditioned model discrepancy and planner-exploitation audit.
- Symthaea #10252: cross-model action-ranking stability.
- Symthaea #10256: trajectory prediction-set coverage and sharpness.
- Symthaea #10083: action-conditioned, model-relative avoidability.
- Symthaea #10043: counterexample-guided scenario discovery and holdout policy.
- Symthaea #6247: road-vehicle dynamics ownership.
- Symthaea #6283: benchmark measurement validity, transfer, common-mode dependencies and claim ceilings.
- Symtropy #1172/#1173: deterministic execution and context-of-use / simulation credibility.
- Symtropy #3387: exact bounded scenario execution and replay.
- Symtropy #3471: causal ego-action-conditioned actor response.
- Symtropy #3385: cluster-aware inference when its estimand and sampling design are valid.

## 2. Separate the claims

Never compress these into a single `valid`, `credible`, or `safe` bit.

| Claim | What it establishes | What it does not establish |
|---|---|---|
| Input integrity | The supplied inputs and identities are intact under the declared digest scheme | Authenticity, correctness, or independence |
| Execution integrity | The named software executed and produced a replayable result | Numerical accuracy or physical fidelity |
| Numerical verification | The implementation meets specified analytic, convergence, and invariant checks | That its equations represent the target vehicle/world |
| Model applicability | Evidence supports using this model within a stated context and envelope | Applicability outside that context |
| Physical validation | Model outputs agree with independently measured physical observations within declared uncertainty | Universal validity or safety |
| Decision robustness | A decision remains acceptable across the admitted model/evidence set and declared uncertainty | That the admitted set contains every real-world condition |
| Safety/operational qualification | A separate authorized process has met its requirements | This protocol does not provide this qualification |

A deterministic replay can be wrong. A numerically stable solver can be wrong. A validated model can be inapplicable to a new surface, vehicle, sensor stack, weather regime, or decision consequence.

## 3. Reference evidence is a portfolio, not a champion oracle

Classify evidence by how it was obtained and what it can support. A higher numbered layer is not automatically more authoritative; applicability and measurement uncertainty govern the claim.

### R0 — Exact analytic oracles

Use closed-form or mathematically bounded fixtures where the domain permits them: constant-velocity crossing, stopping-distance bounds under explicit assumptions, 1-D/2-D kinematics, conservation/invariant checks, and exhaustive finite-action cases.

Required: assumptions, units, solution domain, singular/boundary behavior, tolerance derivation, and at least one negative control.

Claim ceiling: verifies a narrow mathematical contract, not vehicle physics or real-world behavior.

### R1 — Independent implementation or solver

Use a separately implemented solver, code path, or engine for selected matched cases.

Record common-mode roots explicitly: governing equations, numerical method, shared libraries, parameter tables, calibration sources, training data, preprocessing, labels, and authorship/review lineage. Independence is a vector of disclosed relationships, not a boolean inferred from a different digest.

Required: common input contract, independently prepared fixtures, matched boundary cases, convergence/precision analysis, and discrepancy disposition.

Claim ceiling: agreement supports cross-implementation consistency only for the tested cases. Shared assumptions remain shared.

### R2 — Context-qualified higher-fidelity simulation

Use a more detailed simulator only after its context of use has been assessed under Symtropy #1173. Record the exact model profile, relevant capabilities, known approximations, numerical guards/clamps, solver settings, physics callbacks, mesh/contact providers, and unsupported semantics.

Required: suitability for the particular measurand and decision; model-to-model discrepancy characterization; numerical-integrity and replay receipts; independent checks against R0/R1 where possible.

Claim ceiling: more detailed simulation is not physical truth. Do not promote a result merely because a simulator is marketed or described as high fidelity.

### R3 — Empirical observations and instrumented data

Use rights-cleared, provenance-bearing logs, controlled measurements, or independently collected trajectories with known observation and label limitations.

Record sensor calibration, sampling frequency, timestamp synchronization, coordinate transforms, localization reference, occlusion/censoring, annotation protocol, instrument uncertainty, and missingness.

Required: a declared target population / sampling frame, measurement model, quality exclusions frozen before outcome inspection, and split isolation by route, scene, recording session, geography, actor/source and other common roots as appropriate.

Claim ceiling: evidence is limited to the sampled population and instrument/annotation context. Observational correlation does not by itself identify counterfactual actor response.

### R4 — Low-consequence physical validation

Only after the prior gates, use independently reviewed, low-consequence, controlled experiments with explicit operating restrictions, calibrated instrumentation, a separate safety process, and stop conditions.

This protocol does not authorize a vehicle test, actuator interface, public-road experiment, or human-subject study. Any such work requires the applicable approvals and qualified personnel.

## 4. Common-mode dependency graph

For every claimed independent comparison, preserve a machine-readable dependency manifest or a stable reference to one. At minimum, assess these dimensions:

1. **Equations and assumptions:** dynamics, contact, friction, actor behavior, sensor and observation models.
2. **Implementation:** source lineage, reused functions, libraries, solver kernels, discretization, and compiler/runtime dependencies.
3. **Parameters:** source, calibration procedure, fitted range, tuning exposure, and version.
4. **Data:** training, calibration, validation, labels, maps, scenario generation, and split lineage.
5. **Measurements:** sensor hardware, clock synchronization, coordinate frames, localization, annotation, and filtering.
6. **Evaluation:** metric implementation, event predicate, evaluator identity, post-processing, and report code.
7. **People/process:** shared review, manual corrections, selection decisions, and benchmark exposure.

For each pair of evidence sources, classify every dimension as `shared`, `partially_shared`, `independent_with_evidence`, or `unknown`, with a rationale and references. Unknown is not independent.

Do not calculate a universal independence score. Do not multiply nominal confidence values or count correlated sources as separate votes. Common-mode uncertainty must remain visible in the interpretation.

## 5. Matched discrepancy experiment

Each row must bind the same decision-time context wherever a paired comparison is claimed:

- scenario and failure-family identity;
- actor policy and observation-availability history;
- initial state and coordinate/frame contract;
- vehicle, tire, surface, actuator and dynamics profile;
- environment and exogenous disturbance stream;
- ego action history and exact candidate action payload;
- forecast issuance time, horizon and measurement definition;
- predictor, reference execution, evaluator and artifact identities;
- sampling split, ODD/shift regime, and execution receipt.

A comparison is paired only when all fields required by its estimand match. A mismatch is rejected or reported as unpaired; it is never silently averaged.

For each forecast horizon, report signed bias and absolute residual distributions (median, p90, p95 and worst adequately sampled family). Report residual growth only across genuinely paired horizons for the same scenario, actor-policy, action history, state and disturbance context.

Preserve missing, censored, unsupported, timed-out, and incomplete outputs as such. Missing reference execution is not zero residual, no-event, or a successful prediction.

## 6. Planner-exploitation audit and outcome blinding

The reference/evaluator path must not leak its future outcomes into the planner.

1. Freeze planner, model, action candidate generator, objective, constraints, and scenario corpus.
2. Generate a finite feasible candidate set using planner-visible information only.
3. Record the planner's ranked actions and commit its selected action before reference execution outcomes are revealed.
4. Evaluate the same finite candidate set under each admitted reference profile from the matched state and exogenous stream.
5. Retain all feasible, infeasible, unknown, timed-out and incomplete outcomes.
6. Compare predicted and realized utility, action-ranking reversals, selected-action regret, and optimistic/pessimistic bias.
7. Preserve the raw trace and replay identity; independently recompute the report.

Use common random numbers for matched branches only where the environment semantics support it. The shared disturbance stream must be identical up to the point where the ego intervention causally changes the future.

A finite candidate set can establish only a finite-set comparison. It does not prove global optimality. A search timeout or failed optimizer does not prove an action is unavailable or a collision unavoidable.

## 7. Disagreement semantics

Do not resolve disagreement by majority vote.

Each evidence source returns a result scoped to its declared context. The joint interpretation must allow at least:

- `concordant_within_declared_tolerance`: all applicable sources agree within preregistered tolerances on the specified measurand;
- `discordant`: applicable sources disagree beyond tolerance;
- `reference_inapplicable`: a source's context-of-use does not cover the query;
- `insufficient_evidence`: missing data, weak measurement, underpowered family, or incomplete reference;
- `unsupported_semantics`: required physics/actor/sensor behavior is not implemented or measured;
- `indeterminate`: the available evidence cannot support the requested conclusion.

A disagreement triggers diagnosis: inspect common-mode roots, input alignment, numerical error, measurement uncertainty, model assumptions, and applicability. Preserve the disagreement as an artifact and counterexample. Do not automatically select the most conservative model unless its conservatism and domain are independently justified.

## 8. Falsification and negative controls

The first executable corpus should include:

- exact analytic motion and stopping fixtures with boundary cases;
- independent implementation agreement and a deliberately perturbed reference;
- identical code with different digests (must not count as independence);
- different code sharing the same parameter source (common-mode disclosed);
- a hidden shared bias that causes correlated model agreement;
- an intentionally optimistic model that causes planner rank reversal;
- perfect-model and known horizon-growth controls;
- sensor timestamp skew, coordinate-frame mismatch, missing observations and actor occlusion;
- same scenario ID with changed action history or actor policy (must reject pairing);
- missing/timed-out reference execution (must remain incomplete);
- future-outcome leakage and evaluator access negative controls;
- fresh-process exact replay and declared portable-tolerance replay;
- holdout-family exposure and post-result tuning controls.

The evaluator must detect deliberately injected model optimism when the reference experiment is applicable. It must not mislabel a shared or inapplicable reference as independent ground truth.

## 9. Benchmark ladder and scorecards

Keep the benchmark tracks separate and publish the full denominator, split, hardware, model identity, uncertainty, and runtime.

| Track | External / internal comparison | Primary measures |
|---|---|---|
| Motion forecasting | Waymo Open Motion, Argoverse 2 Motion | minADE/minFDE/miss rate plus horizon residuals and prediction-set coverage/size |
| Interactive prediction | Waymo Sim Agents where applicable; Symtropy #3471 synthetic actors | causal response, interaction-conditioned prediction, reaction delay and hidden-state/future-command noninterference |
| Closed-loop planning | CARLA Leaderboard 2.1; WOD-E2E where its protocol is reproducible | route completion, infraction-weighted score, task outcome, comfort and runtime; do not equate open-loop metrics with route safety |
| Distribution shift | Fail2Drive paired routes; separately frozen SAM ODD/failure families | paired ID/OOD degradation, worst-family results, recovery and abstention |
| Safety measurement | NIST AV measurement-science tiers | surrogate measures, outcome prediction, counterfactual choice and cross-scenario diagnosis as distinct tiers |
| Model validity | R0–R4 reference portfolio and Symtropy #1173 | analytic agreement, numerical convergence, common-mode exposure, measurement uncertainty and context-of-use |
| Runtime assurance | SAM RTA and independent fallback baselines | detection/switch latency, intervention, false intervention, fallback availability, deadline misses and no-admissible-fallback |

External leaderboard scores are reference points only. Reproduce the published protocol, sensor privileges, splits, compute constraints and evaluator before making a comparison. A motion-forecasting leaderboard is not a closed-loop planner comparison. A rater-preference score is not a safety metric. A simulator route score is not a real-world crash-rate estimate.

For Fail2Drive, retain the 100 matched ID/OOD route pairs (200 routes) and 17 reported shift scenario classes as the paper defines them. Do not call 200 routes 200 independent pairs.

## 9.1 External benchmark reference snapshot (checked 2026-11-03)

These are external reference points, not SAM results. Recheck the exact board, protocol, checkpoint, and date before making a leaderboard claim.

| Benchmark | Surfaced reference snapshot | Comparison boundary |
|---|---|---|
| Argoverse 2 multi-agent motion forecasting test | SOTA2's surfaced test board lists QCNeXt2026.07 at 0.50 m average minADE (K=6), 1.02 m average minFDE (K=6), and 0.13 actor miss rate (K=6). | Third-party leaderboard aggregation; open-loop forecasting, not interactive closed-loop planning or calibrated set coverage. |
| Waymo Open Sim Agents Challenge (WOSAC) | The surfaced 2025 board lists SMART-R1 at realism score 78.58, kinematics 49.44, interactivity 81.10, map adherence 92.01 and minADE 1.2885; the board was updated about eight months before this check. | A simulator-agent realism benchmark; its dimensions should remain separate, and this dated third-party board is not a November 2026 verified current leader. |
| WOD-E2E planning | SOTA2 lists DriveMA-4B (2026.05) at RFS Overall 8.079, RFS Spotlight 7.169, ADE(5s) 2.67 m and ADE(3s) 1.166 m; that page reports it was updated five months ago. | A dated rater-preference / open-loop evaluation snapshot, not a November 2026 verified current leader or safety metric. |
| Waymo Open Dataset challenges | Waymo states it is not hosting formal challenges in 2026; the public leaderboards remain active. Its 2025 challenge tracks included E2E driving, Sim Agents, Interaction Prediction and Scenario Generation. | Reproduce the published tracks and protocols; do not describe them as an active 2026 competition. |
| Fail2Drive | The 2026 paper reports 200 routes across 17 new scenario classes; the project describes these as 100 matched ID/OOD route pairs. The evaluated methods showed a 22.8% average success-rate drop under shift. | The unit is 100 matched pairs (200 routes), not 200 independent pairs. It is a closed-loop generalization benchmark, not a physical crash-rate study. |
| CARLA Leaderboard | Leaderboard 2.1 is the current official evaluation version. Its driving score is route completion multiplied by an infraction penalty; submissions may not use privileged simulator ground truth. Version 2.1 changed the infraction aggregation after early-stopping strategies exploited the previous scoring design. | Pin CARLA and Leaderboard 2.1 exactly. Scores from 2.0 and 2.1 should not be compared as if the evaluator were unchanged. |

The CARLA early-stop episode is an important benchmark-design lesson: every aggregate score needs adversarial incentive tests. Include controls for stopping early, route truncation, slow driving, safe-but-useless immobilization, and trading away rare vulnerable-road-user performance to improve average completion.

Sources:
- [Argoverse 2 multi-agent motion forecasting test leaderboard — SOTA2](https://www.sota2.com/research/sota/multi-agent-motion-forecasting-on-argoverse-2-av2-test)
- [Waymo Open Sim Agents Challenge methodology](https://waymo.com/research/the-waymo-open-sim-agents-challenge/) and [surfaced 2025 leaderboard snapshot](https://www.sota2.com/research/sota/motion-simulation-on-waymo-open-sim-agents-challenge-2025)
- [WOD-E2E planning leaderboard — SOTA2](https://www.sota2.com/research/sota/autonomous-driving-planning-on-wod-e2e)
- [Waymo Open Dataset challenge status](https://waymo.com/open/challenges/)
- [Fail2Drive paper](https://arxiv.org/abs/2604.08535) and [project repository](https://github.com/autonomousvision/fail2drive)
- [CARLA Leaderboard 2.1 evaluation criteria](https://leaderboard.carla.org/evaluation_v2_1/)

## 10. Prospective gates (targets, not results)

These targets must be preregistered with exact baseline, dataset version, sample design and compute budget before execution:

- **Reference qualification:** every reference used for a consequential ranking has a context-of-use record, disclosed common-mode graph, uncertainty/limitations, and at least one independent negative control. Unknown roots remain unknown.
- **Discrepancy:** at matched compute, seek at least 20% lower p95 residual on a frozen OOD family set versus the strongest faithfully reproduced baseline, while limiting in-distribution p95 regression to 1% relative. Report signed bias as well as absolute residual.
- **Coverage:** on adequately powered, family-disjoint exchangeable holdouts, assess compatibility with preregistered 90% and 95% marginal targets. At matched coverage, seek at least 20% lower median/p95 set measure than a frozen heuristic-tube baseline without worse adequately sampled worst-family coverage.
- **Planning shift:** on paper-faithful Fail2Drive reproduction, seek at least 25% relative reduction in paired OOD success-rate degradation, with ID success no more than 1 percentage point worse. Publish absolute percentage-point changes and uncertainty.
- **Decision quality:** at matched compute and fixed feasible action sets, seek at least 10% lower median and p95 counterfactual regret versus the strongest reproduced planner, without more infeasibility or deadline violations.
- **Interaction:** pass every causal noninterference control before synthetic interactive-agent results are counted; then measure transfer only against separately sourced and rights-cleared actor observations.
- **Runtime:** publish p50/p95/p99 latency, memory, energy, deadline misses and monitor overhead on named hardware.

These are research objectives. None is a currently measured SAM result, safety threshold, or qualification criterion.

## 11. Execution order

1. **Protocol and lineage:** admit the reference portfolio, freeze measurands and common-mode dependency manifests. Compose #1173 and #6283.
2. **Analytic oracle:** build and independently check small motion, crossing, stopping and finite-action fixtures.
3. **Discrepancy evaluator:** implement #10257 as an offline evaluator over supplied, context-bound rows. Do not add another prediction model or simulator.
4. **Interactive execution:** implement the minimal causal actor vertical slice under Symtropy #3471 and execute through #3387.
5. **Held-out exploitation:** inject optimistic model defects and test whether planner choice and regret audit expose them.
6. **External reproduction:** reproduce selected AV2/Waymo, CARLA 2.1 and Fail2Drive baselines with split/rights/compute controls.
7. **Physical bridge:** only after context-of-use, instrumentation and approval gates; low-consequence experiments are not deployment.

## 12. Promotion and claim boundaries

A result may advance only when its exact source head, inputs, protocol, artifacts, evaluator, reference lineage, raw observations, denominators, uncertainty and independent replay are retained.

A passing analytic fixture does not validate vehicle physics. Independent code does not prove independent assumptions. Simulator agreement does not prove physical fidelity. Marginal conformal coverage does not prove conditional or OOD coverage. An action that is best among finite candidates is not globally optimal. A model-relative avoidability result is not real-world causation.

This protocol establishes no ISO 26262 / ISO 21448 compliance, no roadworthiness, no collision-risk reduction, no public-road authority, and no SAE Level 4/5 capability.

## References and owner links

- [Symthaea #10257 — horizon-conditioned discrepancy and planner exploitation](https://github.com/Luminous-Dynamics/symthaea/issues/10257)
- [Symthaea #10252 — cross-model ranking stability](https://github.com/Luminous-Dynamics/symthaea/issues/10252)
- [Symthaea #10256 — trajectory prediction-set coverage](https://github.com/Luminous-Dynamics/symthaea/issues/10256)
- [Symthaea #6283 — measurement validity and benchmark evidence graph](https://github.com/Luminous-Dynamics/symthaea/issues/6283)
- [Symtropy #1173 — context-of-use and simulation credibility](https://github.com/Luminous-Dynamics/symtropy/issues/1173)
- [Symtropy #3387 — deterministic scenario execution/replay](https://github.com/Luminous-Dynamics/symtropy/issues/3387)
- [Symtropy #3471 — causal interactive actor response](https://github.com/Luminous-Dynamics/symtropy/issues/3471)
- [NIST Measurement Science for Automated Vehicles](https://www.nist.gov/programs-projects/measurement-science-automated-vehicles-0)
- [CARLA Leaderboard 2.1 evaluation](https://leaderboard.carla.org/evaluation_v2_1/)
- [Fail2Drive (2026)](https://arxiv.org/abs/2604.08535)
- [Waymo Open Dataset](https://waymo.com/open/)
- [Argoverse 2](https://www.argoverse.org/av2.html)
- [WOD-E2E, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_WOD-E2E_Waymo_Open_Dataset_for_End-to-End_Driving_in_Challenging_Long-tail_CVPR_2026_paper.html)
