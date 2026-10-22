# External Runtime-Assurance Benchmark Protocol v1

**Status:** preregistered research protocol; not an implemented adapter, benchmark result, or safety certification.  
**Parent baseline:** robust-viability prototype qualified on synthetic data at `e7bd8e2a4fd65d57edc5b83eee6b4c43b9ce3a1f`.  
**Purpose:** define a credible path from the scalar synthetic allostatic-control evaluator to independently scored, physics-based control environments without implying that the current prototype already transfers.

## 1. Research question

Can capability-preserving runtime assurance reduce constraint violations under bounded model error while retaining more task capability than a global worst-case safety margin—and can the system honestly identify when its certificate assumptions do not apply?

The central design rule is to separate three claims that are often conflated:

1. **Prediction:** the nominal model predicts a safe transition.
2. **Conditional certificate:** the transition is safe for every state/action effect in a declared uncertainty set.
3. **Observed outcome:** the independently evaluated plant actually stayed within constraints.

A passing synthetic test supports only the contract exercised by that test. It does not validate the external environment model, an OOD detector, a physical plant, or superiority to a published controller.

## 2. Benchmark order and task adapters

### Phase A — Safe-Control-Gym

Use the public [Safe-Control-Gym repository](https://github.com/learnsyslab/safe-control-gym) as the first external integration target. It provides physics-based CartPole and Quadrotor environments, symbolic dynamics, constraints, disturbance hooks, and reference controller/safety-filter implementations. Before any result is published, resolve the exact Git commit, environment configuration, dependency lock, simulator settings, control rate, and enabled constraints. Do not treat a moving `main` branch or a package name alone as a reproducible version.

Run two tasks in this order:

- **CartPole:** a lower-cost integration test for state-dependent uncertainty, action-conditioned envelopes, constraint margins and multi-step fallback.
- **Quadrotor:** a higher-dimensional stress test for coupled state/action uncertainty, actuator effects, and reachable-set volume.

The current Symthaea prototype uses scalar reserve/load state. **Do not map those two scalars onto CartPole or Quadrotor and call that transfer.** Implement task-specific adapters from each environment's real state, action, transition, and constraint functions. The adapter must expose what the controller knows separately from what the evaluator uses as the actual plant.

### Phase B — Safety-Gymnasium

After Phase A is reproducible, add a [Safety-Gymnasium](https://safety-gymnasium.readthedocs.io/en/latest/) navigation/manipulation task to test whether results generalize beyond classic control. Keep this as a distinct task family; do not pool raw cost units or imply direct equivalence between its cost signals and physical state constraints.

## 3. Comparators

Use the same task, simulation step, action bounds, observation access, episode horizon, disturbance tape, and compute budget for all compatible methods.

| ID | Policy | Role |
|---|---|---|
| B0 | Task-native reference controller (pinned PID/LQR/iLQR/MPC where supported) | Strong task-performance reference |
| B1 | Unshielded task-native policy | Failure/utility reference only; never a safety claim |
| B2 | Point-prediction shield | Tests value of uncertainty accounting |
| B3 | Global interval/worst-case shield | Conservative robust baseline |
| B4 | State-and-action-conditioned uncertainty shield | Capability-preservation candidate |
| B5 | Always-reachability policy | Conservative runtime-assurance reference |
| B6 | Calibrated score-triggered predictive/reachability switch | Current architecture under evaluation |
| B7 | Residual-aware monitor plus reachability fallback | Tests post-action evidence in addition to novelty scores |
| B8 | Safety-constrained active model discrimination | Tests safe information gathering and post-probe viable-set gain |
| B9 | Task-compatible MPSC safety filter | Native strong safety-filter baseline in Safe-Control-Gym tasks that support it |
| B10 | Task-compatible CBF / neural CBF safety filter | Competing certificate/filter family; use only when task and constraints match |
| B11 | Conformalized SLS-MPC | Strong uncertainty-aware reachable-set comparator; requires pinned reproduction/adaptation |
| B12 | GPU-SLS robust MPC | High-performance reachability/control comparator; requires pinned reproduction/adaptation and separate hardware disclosure |
| O0 | Evaluator-only oracle with knowledge of the shift and actual model | Diagnostic upper bound; never eligible as a deployable baseline |

Include a published algorithm such as SODA-MPC or conformalized SLS-MPC in a **headline comparison only** after its method is either reproduced on the same task or adapted with a documented mathematical mapping, pinned implementation, matched information access, and comparable tuning/compute budget. SODA-MPC's published demonstration uses learned pedestrian prediction in autonomous driving; its reported results are not directly comparable to CartPole or Quadrotor. The 2026 conformalized SLS-MPC work uses state/control-dependent uncertainty and volume-optimized reachable sets; it is a key methodological reference, not evidence that this prototype implements or beats it.

## 3A. Active falsification and model-update evaluation

Add a distinct experiment arm for safety-constrained active model discrimination. The candidate action must be certified against a **single conservative hull of all still-plausible models**, with a common safe continuation. Do not union model-specific safe-action sets when their future continuations differ.

Compare passive global/state-action robust shielding with:
- random selection among certified actions;
- a task-greedy certified action;
- maximum worst-pair prediction-box separation among certified actions;
- a principled information-gain method where a validated observation likelihood is available.

Charge every probe against a predeclared mission/exploration cost budget. For each probe, retain the before/after hypothesis set, observed transition, uncertainty geometry, runtime cost, task reward and safety outcomes. After the measured transition, retain only model hypotheses consistent with the predeclared observation-noise set; recompute the reachable hull and viable-action set from those hypotheses.

Required outcomes:
- Report the fraction of probes that are actually informative, the posterior hypothesis-count/uncertainty reduction, and the change in certified-action availability. Selecting a probe is not itself evidence that uncertainty fell.
- Record model-family falsification when the observation is inconsistent with every candidate. The result is an explicit unknown/uncertified state: do not silently reset to the prior, use the closest candidate without justification, or issue a certificate.
- Separate true-plant membership (known to the evaluator) from membership of the observation's compatibility set (known only from noisy measurement). Compatibility is not proof that a candidate is the true plant.
- Report false model eliminations caused by noise, unidentifiable model pairs, and cases where a safely informative action does not exist.
- Compare methods after a fixed probe/interaction budget and at matched violation bounds. The primary benefit metric is the subsequent safety–capability frontier (e.g. false vetoes, viable-action availability and task reward), not the raw count of selected probes.

For the maximum-separation baseline, use its name accurately: a min-over-model-pairs distance between prediction sets is a geometric distinguishability proxy. Do not call it mutual information or Bayesian information gain without specifying and validating a probabilistic observation model.

## 3B. Assumption-bound certificate lifecycle

Treat each permission to execute a certified action as a short-lived, context-bound capability. Bind it to the state revision, dynamics revision, constraint/limits revision, calibration revision, action-authority revision, monitor epoch, support-evidence digest, decision step and certified horizon. The runtime must reject a cached certificate when any bound field changes, when the lease expires, when the clock moves backward, or when the caller requests a longer horizon than was certified.

The external adapter must also exercise time-of-check/time-of-use behavior: change each bound field between certificate validation and action application and verify that the action cannot proceed on the stale permission. A negative result is "certificate invalid / recertification required", not a claim that the action itself was necessarily unsafe. The runtime then follows its separately verified fallback/no-action protocol.

Report stale-reuse attempts and rejections by reason, false acceptance of stale certificates, and p50/p95/p99 validation overhead. Revision identifiers and evidence digests must be generated from the actual configuration/evidence path; arbitrary constant identifiers in a test fixture validate only the state machine, not real provenance. If a digest changes due to irrelevant log append operations, define and pin the exact evidence-bundle scope rather than weakening the equality check silently.

## 4. Separation of data and information

Use separate, disjoint episode seeds for model fitting, uncertainty/calibration, threshold selection, and final evaluation. Never tune an envelope, alarm threshold, or controller hyperparameter on evaluation results.

Initial seed allocation (300 episodes per task/condition/policy in the public evaluation panel):

- calibration: seeds 100000–100999 (1,000 nominal episodes per task for monitor calibration);
- nominal test: seeds 200000–200299;
- within-envelope disturbance: seeds 300000–300299;
- parameter/dynamics shift: seeds 400000–400299;
- sensor/actuator shift: seeds 500000–500299;
- compound shift: seeds 600000–600299;
- stealth-score negative control: seeds 700000–700299.

For paired policy comparisons, all methods receive the same initial conditions and exogenous disturbance tape within a given test seed. Seed ranges must be changed if the pinned environment or task consumes randomness in a way that breaks paired comparability; record that decision before seeing results. A seed is a reproducibility key, not by itself proof that episodes are statistically independent.

Calibrate the monitor on nominal episode-level scores only. If using split conformal thresholds, state the nonconformity score, quantile convention, target false-alarm level, exchangeability assumption, and calibration sample count. Report nominal calibration and held-out false-alarm results separately. Calibration coverage does not imply robustness to distribution shift.

## 5. Predeclared perturbation ladder

1. **Nominal:** unmodified parameters and documented nominal disturbances.
2. **Within-envelope:** disturbances sampled from the uncertainty model the certificate claims to cover.
3. **Parameter/dynamics shift:** mass, inertia, drag, thrust, actuator gain, or another task-relevant parameter shifts beyond the declared predictor envelope.
4. **Observation/actuator shift:** sensor noise/bias, delay, saturation, or action execution mismatch. Keep hidden truth available only to the independent evaluator.
5. **Compound shift:** at least two independently specified perturbation families occur together.
6. **Stealth-score negative control:** alter the actual plant while keeping the monitor score paired to its nominal value/distribution. This is intentionally undetectable to a monitor whose only input is that score. It must expose the architectural blind spot, not be scored as a failed surprise detector. Evaluate whether a separate residual/constraint monitor can provide useful evidence, and list the sensor assumptions that make that possible.

Each shift must have a machine-readable parameter range, sampling rule, affected transition terms, envelope-membership oracle, and safe-constraint oracle. The evaluator's plant and labels must not leak into the policy input.

## 6. Required metrics

Report metrics per task, condition, policy, safety bound, and seed panel before any aggregate.

| Dimension | Required outputs |
|---|---|
| Safety frequency | Episode violation probability; number of violating episodes / episodes; one-sided 95% binomial upper confidence bound when zero violations are observed |
| Severity | Normalized violation magnitude among violating episodes; integrated normalized violation exposure; maximum excursion; time-to-first-violation |
| Conditional certification | False-certificate count and rate: controller issued a certificate, but the independently evaluated actual transition violated a declared constraint |
| Envelope validity | Fraction of transitions inside predictive and fallback envelopes; certificates attempted while their assumptions were invalid; unknown/invalid status must never be silently recorded as inside |
| Capability | Return/reward, task success, safe-action availability, false-veto rate among actions independently labeled safe, no-certified-action rate |
| Detection | Episode false-alarm rate, shift miss rate, time-to-alarm, fraction of alarms that occurred before the first violation |
| Runtime | p50/p95/p99 decision latency, worst-case search work, timeout/exhaustion rate, fallback duty cycle |
| Calibration | Nominal held-out coverage / false-alarm rate, score distributions by condition, and calibration/test provenance |
| Reproducibility | Environment + dependency + policy revisions; full seed manifest; trace/schema version; source and output digests; byte-identical deterministic reruns where determinism is expected |

Also publish empirical CDFs/histograms and per-condition results. Average cumulative cost alone can hide violation frequency and severity. The SafeRLEval work proposes complementary violation-rate and normalized-deviation metrics; use its published definitions when reporting them, rather than relabeling custom metrics as equivalent.

For any zero-violation claim, include its denominator and a confidence interval. For example, zero observed violations over 300 genuinely independent test episodes has an approximate one-sided 95% upper bound near 1% under a binomial model; it is not proof of zero risk, and correlation or seed reuse weakens that interpretation.

## 7. Correctness gates versus performance targets

### Mandatory protocol/correctness gates

- A certificate's uncertainty-set assumptions, time step, action bounds and constraints are explicit and checked.
- The adapter's transition and constraint calculations are cross-checked against the pinned environment; actual-plant evaluation is implemented independently from candidate policy logic.
- Calibration, tuning, validation and final evaluation seeds are disjoint.
- Invalid/non-finite inputs, missing calibration, exceeded search budgets, unknown envelope membership, or an empty fallback set produce an explicit uncertified/no-action outcome—not a silent fallback to the stale predictive policy.
- Horizon-1 output matches the one-step evaluator for the same model; certified first-action sets do not grow when the horizon is extended; intermediate violations cannot be excused by later recovery.
- Every result row carries the policy ID, scenario ID, seed, task revision, certificate status, envelope-membership statuses, actual constraint outcome and runtime work.
- Independent analysis recomputes aggregate metrics from raw trace rows and verifies all denominators.

### Proposed capability targets (not achieved results)

- At matched observed violation budget, reduce false vetoes against the global-margin shield by at least 20% in the declared in-envelope cells.
- Keep nominal task-return loss within 5% of the strongest compatible safe baseline, while publishing the full safety–capability Pareto frontier rather than optimizing one headline score.
- On the declared in-envelope test panel, target zero observed false certificates and report an uncertainty bound; one observed false certificate is reported, not averaged away.
- For score-detectable shifts, target lower violation exposure than predictive-only control without exceeding the pre-registered nominal false-alarm budget.
- For the stealth-score control, do **not** require a score-only detector to detect an information-free shift. Require explicit miss reporting and test whether independent residual evidence reduces risk.
- Report whether p95 control latency fits the task's control deadline. Any overhead target must be fixed before runs and must not be bought by weakening safety checks.

These targets are hypotheses and engineering gates, not universal SOTA claims. If an algorithm cannot certify any action, its observed safety may be good while its utility is unacceptable; show both.

## 8. Statistical plan and comparison rules

- Pair policies by episode seed and disturbance tape where possible.
- Publish raw results and task/condition stratification. Use paired bootstrap confidence intervals for reward and severity differences; use exact or Wilson binomial intervals for episode-level violation rates.
- Declare a primary safety metric and a primary capability metric before running the held-out panel. Correct for multiple comparisons across the predeclared policy family or label those comparisons exploratory.
- Never compare one method's best seed to another method's average. Report all seeds and a robust summary (median and interval).
- Do not combine different units or tasks in a single raw mean. If a normalized cross-task score is useful, publish the normalization reference and retain native per-task results.
- Freeze all code, configuration, checkpoint, environment revisions and input hashes before unsealing held-out results.
- Distinguish “implementation reproducibility”, “statistical evidence”, “formal/soundness argument under assumptions”, and “physical validation”; success in one category does not entail success in another.

## 9. Evidence bundle and record of claims

Each run bundle should include:

- a protocol manifest and exact environment/dependency lock;
- one raw transition trace per policy/seed, including observed state, action proposed, action applied, predictor envelope, fallback envelope, certificate/monitor decisions, actual next state, actual constraint margins, elapsed decision time and work counters;
- a machine-readable summary with explicit numerators and denominators;
- a source/config/trace digest receipt and command line;
- an independent verifier result and its own version/digest;
- a written exception record for missing tasks, nondeterminism, discarded episodes, or deviations from the protocol.

The trace producer must not be its sole verifier. A successful workflow status is evidence that a particular revision ran the defined software gates; it is not proof of physical safety or external validity.

## 10. Literature and benchmark references

- Safe-Control-Gym source and task capabilities: https://github.com/learnsyslab/safe-control-gym
- Safety-Gymnasium documentation: https://safety-gymnasium.readthedocs.io/en/latest/
- Contreras, Shorinwa & Schwager (L4DC 2025), *Safe, Out-of-Distribution-Adaptive MPC with Conformalized Neural Network Ensembles*: https://proceedings.mlr.press/v283/contreras25a.html
- Srinivasan, Leeman & Chou (L4DC 2026), *Safety Beyond the Training Data: Robust Out-of-Distribution MPC via Conformalized System Level Synthesis*: https://proceedings.mlr.press/v331/srinivasan26a.html
- Spoor, Plaat & Moerland (2026), *Evaluation Metrics for Safe Reinforcement Learning* / SafeRLEval: https://arxiv.org/abs/2609.15315 and https://github.com/lindsayspoor/SafeRLEval

## Exit criterion

Do not claim a field-level improvement until the pinned external adapters exist, the native and robust baselines are reproduced under matched conditions, the held-out comparisons are published with uncertainty, and an independent verifier agrees with the reported raw traces. The first objective is a benchmark other researchers can falsify—not a result that is easy for our own implementation to pass.
