# ROB-FEP-THERMO-001 — active inference under physical thermodynamic constraints

Status: research protocol and synthetic known-answer corpus. No controller, physical qualification or FEP-superiority claim.

Parent: [Symthaea #11496](https://github.com/Luminous-Dynamics/symthaea/issues/11496).

## Research question

Does a calibrated, action-conditioned active-inference policy improve thermally constrained mission decisions—especially sensing, diagnosis, cooldown, retreat and future-option preservation—over strong Bayesian, MPC, rule-based and thermal-aware RL baselines?

The benchmark combines two different domains of reasoning:

- **FEP / expected free energy:** beliefs, predictive observations, preferences and decision-relevant information.
- **Physical thermodynamics:** electrical and mechanical energy, heat generation/transfer, node temperatures, actuator derating and cooling.

They must exchange evidence, not share a fictitious unit system.

## Essential distinction: variational free energy is not Helmholtz free energy

For a variational distribution q(s), a common variational-free-energy expression is:

`F[q] = E_q[ln q(s) - ln p(o,s)]`

With natural logarithms this is an information-theoretic quantity reported in nats. It is a bound on model evidence under the stated probabilistic construction. An expected-free-energy objective is a policy-evaluation quantity; the exact decomposition and units depend on the chosen formulation and implementation.

Physical Helmholtz free energy is instead `A = U - T S`, a thermodynamic state function with energy units. Battery energy, mechanical work, heat and entropy production have their own physical boundaries and units.

Do not add FEP free-energy scores to joules, or assume that minimizing a variational score automatically minimizes physical energy, heat, entropy production or mission risk. A conversion between information and energy requires a specified statistical-mechanical system and assumptions; an arbitrary controller weight is not such a mapping.

Also distinguish:
- entropy of a belief distribution;
- thermodynamic entropy of a physical system;
- expected information gain about a hypothesis;
- expected reduction in downstream decision loss (EVSI);
- a heuristic novelty bonus.

These quantities are related in some formulations but are not interchangeable.

## Current Symthaea source audit — 10 November 2026

The public `main` versions were fetched and inspected:

- `crates/core/symthaea-fep/src/free_energy.rs` — blob `f4bb1e0f8629f9f36280f0f70350bee5c6d7bf1d`.
- `crates/core/symthaea-fep/src/agent.rs` — blob `678a951162e0c56c8e9522ffe6307c4d00a7d4ac`.
- `crates/core/symthaea-fep/src/generative_model.rs` — blob `a442654634f99a783e82582516a88a7feae2a8ed`.
- `crates/core/symthaea-fep/src/types.rs` — blob `0a106e4e158e0337573b8bafeb59e25903349d41`.

In the current path:
1. `predict_next_state(state, action)` applies an action-conditioned transition to the predicted mean.
2. The predicted precision update uses the same current precision and transition-precision formula for every action.
3. Mode probabilities are copied unchanged.
4. `HiddenState::entropy` uses the continuous precision vector and dimensionality; it does not read the mean or mode probabilities. `mode_entropy` is a separate method and `total_uncertainty` combines continuous and discrete terms, but the current EFE epistemic selector calls `entropy`, not `total_uncertainty`.
5. `compute_epistemic_value` is predicted continuous entropy minus current continuous entropy.

Therefore, for a fixed current belief/model, the present entropy-difference epistemic term is action-invariant. The predicted mean and pragmatic prediction-error term may vary by action. The novelty bonus can also vary, but novelty is not expected information gain.

This reproduces the source-level audit in [#8391](https://github.com/Luminous-Dynamics/symthaea/issues/8391). It is not a runtime-performance claim.

The decision-focused EVSI kernels in [#11122](https://github.com/Luminous-Dynamics/symthaea/issues/11122) are a valuable foundation, but the issue records that the live `ActiveInferenceAgent` integration and grounded action-conditioned predictive-observation adapter remain open.

**Do not increase the epistemic weight or claim active exploration until action-conditioned outcome distributions, Bayesian updates and this negative control are independently qualified.**

## Architecture and ownership

### Symthaea — belief and proposal

- infer beliefs / hypothesis sets over terrain, contact, actuator, thermal and fault state;
- generate candidate observation / diagnostic actions;
- estimate action-conditioned predictive observation distributions;
- compute pragmatic, epistemic, ambiguity, novelty and EVSI terms separately;
- propose mission-level intents (continue, slow, probe, cool, retreat, requalify, safe-stop);
- learn from episodes only through quarantined model-update proposals and held-out validation.

HDC, CfC/LTC and FEP are candidate mechanisms, not guaranteed winners. Keep Bayesian, set-membership, recurrent and deterministic baselines.

### Symtropy — physical consequences

Consume existing qualified owners:
- #4044 energy boundaries;
- #4179 actuator loss maps;
- #4206 actuator dynamics;
- #4511 recovery topology;
- #4549 mission-sequence state carryover;
- #4588 observation-constrained future options;
- #4610 coupled thermal dynamics (qualification still pending).

Symtropy evaluates candidate consequences and retains physical histories. It does not certify its own dynamics as physical truth.

### Independent authority

The cognitive executive emits bounded proposals. Existing action projection, robust viability, state-dependent assurance, runtime assurance and HAL decide admissibility. No FEP score, novelty, Phi, posterior confidence or model-update claim can grant or extend actuation authority.

## Decision contract

For candidate action a, maintain separate records for:

- pragmatic / task preference value;
- action-conditioned expected information gain (EIG);
- decision-focused expected value of sample information (EVSI);
- ambiguity / model uncertainty;
- novelty heuristic;
- physical electrical energy and compute cost;
- per-node thermal trajectory and actuator derating;
- hard constraints, recovery topology and safe-stop;
- evidence status, source identities, calibration and deadline.

A high-EIG observation can have zero EVSI when it does not change the optimal downstream decision. A low-EIG observation can have high EVSI when it resolves a rare but consequential fault hypothesis. Probe selection should prioritize the latter when conservative decision-value gain justifies its costs and all hard constraints pass.

Use EVSI / downstream regret as the primary diagnostic-value evidence. Report EIG separately.

Use hard constraints, lexicographic priorities and Pareto reporting rather than an arbitrary weighted scalar that lets a good belief score compensate for overtemperature or loss of a recovery route.

## Thermodynamic state and accounting

Consume physical state from the Symtropy thermal and energy owners. Keep these quantities distinct:

- signed battery-terminal energy;
- DC-bus and actuator electrical input/loss;
- mechanical joint work;
- heat generation per node;
- inter-node and environment heat transfer;
- cooling-system power;
- temperature and thermal uncertainty per node;
- time-to-derating and cooldown;
- energy/heat-balance residual where the control volume is qualified.

A candidate thermal network may have the general form:

`C_i dT_i/dt = P_{loss,i} + Σ_j G_{ji}(T_j-T_i) - Q_{cool,i}`

but parameters, topology, source boundaries and model class require evidence. The node manifest is embodiment-specific; modeled ambient or internal states are not automatically measured sensor channels.

Do not optimize thermodynamic entropy production or exergy until the physical boundary, environment reference, heat-transfer model, instrumentation and uncertainty are qualified. Report such quantities separately from variational free energy.

## Factorial benchmark

Compare at equal information, compute, action and physical-resource budgets:

| Arm | Decision mechanism | Physical thermal/energy constraints |
|---|---|---|
| A | Deterministic rules / finite-state executive | Fixed/nominal baseline |
| B | Deterministic rules / finite-state executive | Qualified thermal and actuator constraints |
| C | Bayesian / set-membership belief + EVSI | Fixed/nominal baseline |
| D | Bayesian / set-membership belief + EVSI | Qualified thermal and actuator constraints |
| E | Current FEP path (legacy audited baseline) | Fixed/nominal baseline |
| F | Corrected action-conditioned FEP + EVSI | Fixed/nominal baseline |
| G | Corrected action-conditioned FEP + EVSI | Qualified thermal and actuator constraints |
| H | Thermal-constrained MPC / HQP | Same qualified plant and sensor budget |
| I | Thermal-aware RL / residual policy | Same qualified plant and sensor budget |
| J | GenAIF-style generative trajectory model (diffusion / CVAE / flow matching) with action-conditioned likelihoods | Same plant and inference budget; manipulation-domain comparison first |
| K | Robust min–max MPC / viability-filtered controller | Same qualified physical plant, disturbance set and action budget |
| L | Evaluator-only clairvoyant oracle | Upper bound only; never deployable |

Add HDC-only, CfC/LTC-only and combined ablations only where matched compute and data are available.

Keep A1, Go2, ANYmal, humanoid and manipulator comparisons in separate platform strata.

## Synthetic known-answer cases

`fixtures/known_answers_v0_1.json` contains:
1. Current FEP action-invariance canary.
2. High information gain but zero EVSI: probe must be declined when it costs resources.
3. Low information gain but positive EVSI: a fault probe changes the optimal downstream decision and should be selected when its decision cost is justified.
4. Hard thermal constraint: a low-FEP-score action is rejected when any declared node exceeds its limit.
5. Unit-separation trap: variational free energy and physical energy cannot be added without an explicit justified mapping.
6. Missing thermal evidence: no positive physical feasibility claim.

The cases are synthetic. They test arithmetic and semantic separation, not a physical robot, FEP superiority or thermal-model validity.

## External research comparators

- Qian et al., A1 thermal-aware locomotion, arXiv:2603.01631: over 27 minutes versus approximately 7 minutes baseline under that paper's 3 kg payload conditions.
- Wan et al., A1 residual thermal policy, arXiv:2605.27046: separate experiment, over 13 minutes versus approximately 5 minutes nominal baseline.
- De Benedittis et al., *Keep It Cool*, IEEE TCST (20 Aug 2026), DOI 10.1109/TCST.2026.3721130: temperature-constrained HQP/MPC with a redundant planar manipulator hardware validation.
- Matsumoto et al., *Goal-Directed Latent Variable Inference*, first published 23 Sep 2026, DOI 10.1111/tops.70081: relevant active-inference / humanoid line. The article reports physical robot data collection, but key plan evaluations were simulated trajectories; its authors note generated trajectories were not executable in real time on the physical robot within the available compute. Treat it as a representation/planning comparator, not physical active-inference locomotion validation.
- Sweeney et al., *Decision, Inference, and Information*, *Entropy* 28(1), 2026, DOI 10.3390/e28010001: formal relationships between expected free energy and decision/information frameworks, with assumptions and representation constraints.
- Aguilera et al., *How particular is the physics of the free energy principle?*, *Physics of Life Reviews* (2022), DOI 10.1016/j.plrev.2021.11.001: critical analysis of assumptions and restricted applicability in studied linear systems.
- Active Inference Institute's robotics review identifies open gaps in high-dimensional real-time EFE scaling, model specification, robustness under distribution shift and standardized head-to-head benchmarks. Treat this as an ecosystem review, not a controlled benchmark result.

These papers do not form one leaderboard. A1 thermal endurance, planar-manipulator temperature control, humanoid active inference and theoretical equivalence results answer different questions.

## Metrics

**Belief / inference**
- Brier score, log score, calibration / coverage and selective abstention;
- EIG predicted versus realized;
- EVSI predicted versus realized decision-risk reduction;
- downstream decision regret and action-selection stability;
- action-invariance negative-control behavior.

**Physical thermodynamics**
- battery discharge, returned and net energy under exact boundary;
- electrical actuator input, joint work and loss residual;
- per-node peak and upper-bound temperature;
- time-to-derating, cooldown and derating duration;
- cooling and compute energy separately;
- heat/energy-balance residual with propagated uncertainty where qualified.

**Mission / recovery**
- mission completion, useful work, tracking and elapsed time;
- observation-constrained common continuations;
- recovery routes, failure-domain cut sets and safe-stop feasibility;
- diagnosis delay and physical probe costs;
- adverse contact, fall, slip and impact events by category;
- compute p50/p95/p99/worst observed latency and deadline misses;
- Unknown/incomplete-evidence rates.

The mission run is the statistical unit. Preserve aborted, failed, Unknown and missing-evidence cases in the denominator. Report paired differences and confidence intervals.

## Prospective success gates (not achieved)

- Current action-invariant epistemic behavior reproduced and protected by a regression canary.
- Corrected action-conditioned EIG/EVSI agrees with independent exact enumeration on bounded fixtures.
- High-EIG / zero-EVSI probes are declined when they have nonzero cost/risk.
- Low-EIG / positive-EVSI probes are preferred over more informative but decision-irrelevant probes when the frozen loss model justifies it.
- Thermal-aware FEP does not violate per-node hard limits or lose terminal safe-stop evidence.
- The combined FEP + thermal arm improves the pre-registered decision-quality / recoverability Pareto frontier over the strongest matched single-mechanism baseline, without adverse-category regression.
- HDC/CfC/LTC/FEP components remain only when they add held-out value at equal information and compute.
- No cognitive result changes actuation authority.

## Staging and claim ceiling

F0: source audit / negative-control reproduction.
F1: action-conditioned likelihoods, posterior updates and exact EVSI verification.
F2: consume qualified thermal/energy state from Symtropy.
F3: offline factorial replay.
F4: simulator shadow mode (no action application).
F5: instrumented joint / quadruped shadow mode (cognition advisory only).
F6: separate RTA/HAL and physical qualification review.

A successful finite corpus proves only the finite calculation. It does not establish physical thermal-model validity, continuous-time safety, FEP superiority, arbitrary-fault tolerance, consciousness or robot readiness.

No motor command or authority is created by this protocol.


## Entropy / exergy extension

The benchmark now has a dedicated design addendum: [entropy_exergy_extension.md](./entropy_exergy_extension.md). It makes the policy boundary explicit: do not minimize Shannon entropy or physical entropy production as a universal objective. Use decision-focused EVSI for the value of information, physical entropy/exergy only with qualified boundaries and assumptions, and observation-constrained future mobility as the mission-level viability criterion.

The physical-accounting work is tracked separately in [Symtropy #4631](https://github.com/Luminous-Dynamics/symtropy/issues/4631). It composes with #4044 (energy), #4179 (actuator loss maps), #4610 (coupled thermal model) and #4588 (future mobility). It owns no second energy ledger, thermal authority or controller.


The entropy/exergy analytical controls are also machine-readable:
- [fixtures/entropy_exergy_v0_1.json](./fixtures/entropy_exergy_v0_1.json)
- [fixtures/entropy_exergy_v0_1.schema.json](./fixtures/entropy_exergy_v0_1.schema.json)

They are seven synthetic known-answer cases, not empirical or physical-model qualification. The physical-accounting implementation owner is [Symtropy #4631](https://github.com/Luminous-Dynamics/symtropy/issues/4631).


## Competitive landscape and proposed targets

The November 2026 comparator matrix, ablations, sequential-mission design and proposed (not achieved) targets are in [competitive_landscape_2026-11.md](./competitive_landscape_2026-11.md). The comparison keeps RAIBO2, A1, ANYmal, manipulator and simulation results in separate strata and introduces GenAIF and test-time physical-diagnosis papers as additional active-inference comparators.


## Evidence-to-action qualification

The six-gate evidence-use benchmark is now specified in [evidence_to_action_gates.md](./evidence_to_action_gates.md) with eight synthetic known-answer cases and a strict schema:
- [fixtures/evidence_use_v0_1.json](./fixtures/evidence_use_v0_1.json)
- [fixtures/evidence_use_v0_1.schema.json](./fixtures/evidence_use_v0_1.schema.json)

It composes with #11122 and #11338: it does not create another EVSI or active-sensing engine.


## Real-time computation and dynamic-world evaluation

The new cross-cutting real-time benchmark is tracked in [Symthaea #11561](https://github.com/Luminous-Dynamics/symthaea/issues/11561) and summarized in [realtime_computation_extension.md](./realtime_computation_extension.md). It distinguishes frozen-world upper bounds from wall-clock real-time operation and charges compute/observation latency against changing physical state.
