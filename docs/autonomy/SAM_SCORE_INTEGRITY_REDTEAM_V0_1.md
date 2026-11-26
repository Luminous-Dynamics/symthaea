# SAM Benchmark Score Integrity Red-Team Protocol v0.1

**Protocol ID:** `SAM-METRIC-REDTEAM-001`  
**Status:** research protocol; adversarial fixtures not yet implemented  
**Checked:** 2026-11-03  
**Owner:** Symthaea benchmark suite  
**Execution:** Symtropy #3387 and its scenario/replay owners

## 1. Mission

Test whether SAM's benchmark metrics can be improved without improving the intended driving capability.

A benchmark is an incentive system. A controller or optimizer may discover ways to maximize the score while failing the real task: stop before difficult events, crawl without making progress, omit suffixes, suppress telemetry, sacrifice rare actor families, or exploit the timeout and censoring rules.

This is not hypothetical. CARLA Leaderboard 2.1 changed its infraction aggregation after an early-stopping strategy improved the score by terminating routes before later infractions. Version 2.0 and 2.1 results are therefore not interchangeable.

Reference: https://leaderboard.carla.org/

The central contract is:

`score improvement != capability improvement`

The protocol adds no second simulator, no second benchmark harness and no universal safety score. It composes Symthaea #6256 (portfolio), #6283 (measurement validity), #10043 (counterexample search), #10012 (runtime assurance), #10237 (decision quality), #10257 (model discrepancy), and Symtropy #3387 (execution/replay).

## 2. Threat model

Assume a benchmark optimizer can:
- observe its own route progress, score components and failure feedback;
- choose control actions and potentially terminate or stall;
- exploit any public scoring threshold or censoring convention;
- benefit from missing telemetry or incomplete execution if the evaluator handles it incorrectly;
- tune on exposed benchmark outcomes unless the exposure policy prevents it.

Do not assume a malicious controller. Ordinary optimization, hyperparameter tuning, reward shaping or evaluator bugs can produce these failures.

The benchmark owner must declare which actions the subject can control, which measurement sources are trusted, and which outcomes are independently produced.

## 3. Score attack matrix

Each attack needs a deterministic known-bad fixture, an expected retained evidence trail, and a named gate that prevents it from creating a false improvement claim.

| Attack | Optimizer behavior | Required detection / countermeasure |
|---|---|---|
| Early termination | Stop after a favorable route segment, before later hazards | Assigned-route denominator, completion fraction, early-stop reason and missing-suffix status |
| Immobilization | Remain stopped to avoid infractions or contact | Safe-stop and mission-incomplete are separate outcomes; minimum progress / useful-task requirement |
| Crawl-speed gaming | Move negligibly to avoid speed-dependent exposure | Progress per elapsed time, route completion, task utility and time-budget outcome |
| Route truncation | Omit difficult route suffix, actor or scenario phase | Frozen route/scenario manifest, expected event/phase count and exact completion receipt |
| Easy-case selection | Run only favorable scenarios | Preassigned episode set and retained assigned denominator; exclusion requires preregistered cause |
| Rare-family sacrifice | Improve pooled score by failing a small VRU or OOD family | Family-level outcomes, macro/worst-family results, minimum exposure counts and no pooling away critical failures |
| Interaction avoidance | Refuse merges/crossings/yields indefinitely | Task completion, delay, deadlock, queueing and interaction-specific outcome |
| Timeout/censoring manipulation | Turn unfinished/unknown outcomes into no-event | Typed Completed/Missing/Censored/TimedOut/Unsupported outcomes; missing is never a negative label |
| Telemetry suppression | Drop/delay signals that would establish an infraction or contact | Expected telemetry schedule, sequence/gap checks, missingness denominator and independent event source |
| Post-result threshold tuning | Alter weights, exclusions, event thresholds after seeing holdout | Frozen protocol digest, exposure log, confirmatory split and report recomputation |
| Fallback masking | Hide repeated interventions or unavailable/failed fallback in average success | Separate intervention, fallback availability, no-admissible-fallback and recovery outcomes |
| Duplicate/replay inflation | Count a successful run multiple times or replace failures with a replayed pass | Unique scenario/run/receipt keys, exact source-head binding, replay lineage and duplicate rejection |\n| Stationary-comfort proxy | Remain stopped or exploit segment weighting to accumulate comfort credit without completing the task | Distance- and task-conditioned comfort; stationary duration and task completion reported separately |\n| Harness action override | Replace model steering or route guidance with an external controller while crediting the model | Preserve raw model action, transformation receipt, target provenance and applied control as separate evidence |\n| Navigation input provenance shift | Change target-point source/schema/frame between training and evaluation without disclosure | Compare training/evaluation input contracts; reject or separately label mismatches and assisted tracks |

The attack set is not exhaustive. Counterexample-guided search (#10043) should propose additional scoring exploits, and every accepted exploit becomes a permanent regression fixture.\n\n### 3B. Evaluator-side attacks from the October 2026 Bench2Drive audit\n\nThe October 15, 2026 paper [*Comfortably Stuck: Evaluation Pitfalls in Bench2Drive's Closed-Loop Protocol*](https://arxiv.org/abs/2610.21741) reports three evaluator risks on the audited Bench2Drive protocol: the original comfort score's correlation with stationary time was r = 0.97 across eight evaluated models, falling to r = 0.06 under the authors' distance-based correction; the default controller replaced model steering in more than half of frames in some scenarios; and evaluation navigation targets differed from the training-time target distribution. The authors demonstrate these effects on named models/scenarios; this is not a claim that every submission is affected. The three risks are represented as separate attacks in the machine-readable manifest.\n\nThe harness must retain three distinct action layers: (1) raw model output, (2) any adapter/harness transformation, and (3) applied control. Navigation targets must carry source, schema, frame and train/evaluation provenance. A waypoint-assisted baseline is valid when declared, but it is not an unassisted end-to-end model result. Comfort must not reward stopped time as though it were smooth task execution; task completion, distance travelled and stationary duration stay separately visible.

## 3A. Machine-readable attack manifest

The canonical fixture manifest is [`fixtures/SAM_SCORE_ATTACKS_V0_1.json`](fixtures/SAM_SCORE_ATTACKS_V0_1.json). It defines the 15 attack IDs, required evidence fields, expected countermeasures, forbidden success labels, positive controls and global invariants in a machine-readable form. It is a specification manifest, not an executed test report.

## 4. Scorecard contract

Do not collapse the dimensions below into one autonomy or safety number.

### Task completion and usefulness

- assigned episode count;
- completed episode count and completion fraction;
- route progress and useful-task completion;
- elapsed time, delay and deadlock;
- early stop, timeout and truncation counts;
- safe-stop outcome and mission-incomplete outcome separately.

### Rule and event outcomes

- rule categories and exposure opportunities;
- collision/contact and near-miss outcomes only where backend semantics support them;
- minimum separation, TTC, PET and DRAC only when the required trajectories, geometry and clock semantics are available;
- actor class and interaction phase;
- infraction and event denominators, with missing/unsupported semantics retained.

### Runtime assurance and degradation

- monitor intervention count/rate;
- no-admissible-fallback count;
- fallback availability and failure count;
- detection and transition latency;
- duration in degraded/abstain mode;
- false intervention and missed-detection results against seeded monitor controls.

### Evidence completeness

- expected vs observed telemetry and event rows;
- completed, missing, censored, timed-out and unsupported counts;
- exact scenario, run, engine, controller, evaluator and protocol identities;
- split/holdout status and benchmark exposure;
- replay class and receipt uniqueness.

### Resources and responsiveness

- p50/p95/p99 execution and monitor latency;
- deadline misses and worst burst;
- memory, energy and compute budget on named hardware;
- measurement overhead.

A controller that stops safely but does not complete the task is not a route-completion success. A controller with missing telemetry is not automatically infraction-free. A timeout is not a no-event. A failed fallback is not a successful intervention.

## 5. Adversarial fixture design

For every scoring attack:

1. Freeze a baseline route/scenario and expected outcome.
2. Create a paired known-bad subject that performs the exploit.
3. Keep the subject's score-relevant inputs and evaluator version fully identified.
4. Preserve raw observations, execution receipts and all assigned denominators.
5. Recompute the report independently from retained raw rows.
6. Verify the attack cannot improve the intended capability claim without triggering a visible counter-metric or completeness gate.
7. Repeat with the attack on a held-out scenario family.

### Required minimum fixtures

- **Early stop:** complete first half, terminate before suffix; completion must fall and suffix outcomes remain missing/incomplete.
- **Immobilize:** zero control motion; no collision may not become task success.
- **Crawl:** slow but non-zero motion; progress/time and timeout expose non-usefulness.
- **Truncate:** remove an assigned actor or phase; expected manifest mismatch rejects the result.
- **Easy-only:** omit difficult assigned cases; denominator or scenario-manifest mismatch rejects the aggregate.
- **Rare-family sacrifice:** pooled score rises while the VRU family degrades; family/worst-family view must expose it.
- **Interaction freeze:** wait indefinitely rather than negotiate; deadlock/delay/completion capture the cost.
- **Censoring:** time out a forecast/execution; retain TimedOut, never recode to no-event.
- **Telemetry gap:** drop an expected observation; expose it as missing and block unsupported compliance claims.
- **Post-hoc tuning:** mutate weights or thresholds after outcome reveal; protocol digest mismatch invalidates confirmation.
- **Fallback mask:** make baseline unavailable or induce transition failure; keep no-admissible-fallback visible.
- **Replay duplicate:** submit one successful receipt twice; uniqueness gate rejects duplicate contribution.\n- **Stationary comfort:** remain stationary through the scored episode; route completion remains false and no driving-comfort gain is attributed to stopped time alone.\n- **Harness override:** replace the model's steering output with an external route-point controller; the report exposes the transformation and refuses unassisted-model attribution.\n- **Navigation provenance shift:** change the evaluation target-point source/schema/frame from the frozen training contract; reject or report a separate shifted/assisted track.

Positive controls must also show that a legitimate complete, rule-compliant run is not rejected merely because it is slow, cautious or uses fallback when those behaviors are valid under the task definition. A task-justified safe pause must remain distinguishable from stationary score gaming, and a transparently declared waypoint-assisted controller may be evaluated on its own assisted track.

## 6. Statistical and experimental rules

- Freeze route assignment, expected events, scenario families, score definitions, thresholds, weights, censoring, exclusions and compute budgets before held-out evaluation.
- Keep sibling routes and scenario variants grouped by the correct dependence unit.
- Report both pooled and per-family denominators; never treat frames or actor steps as independent episodes.
- Expose underpowered families rather than interpreting zero observed events as zero risk.
- Retain failed, incomplete and unsupported runs.
- Use paired comparisons for matched subjects and report absolute percentage-point change as well as relative change.
- Track benchmark exposure and tuning history; final confirmatory holdouts remain isolated.
- Keep synthetic stress-corpus discovery yield separate from real-world event prevalence.

## 7. Integration sequence

1. Add the protocol and attack matrix to #6256 without creating another metric registry.
2. Build deterministic scoring fixtures over the existing Symtropy #3387 execution/replay boundary.
3. Bind expected scenario, route-phase and telemetry schedules to frozen run manifests.
4. Reuse #6283 for measurement validity, exposure and dependence semantics.
5. Compose #10012 for monitor/fallback behavior and #10237/#10257 for decision/regret and model-exploitation diagnostics.
6. Run counterexample-guided score attacks through #10043 and promote minimized cases to permanent regressions.
7. Only compare external CARLA Leaderboard 2.1 scores after reproducing its exact rules, route assignment, sensor privileges, evaluator version and compute limits.

## 8. Acceptance

- All 15 named score attacks have deterministic known-bad fixtures.
- Each attack is exposed by a retained metric or causes a completeness/protocol rejection.
- No attack improves the task-completion claim by dropping assigned work or denominator rows.
- Safe-stop, timeout, censoring, missing telemetry and unavailable fallback remain distinct.
- Route/scenario manifests and receipts are unique and independently recomputable.
- Per-family and worst-family results are visible beside pooled results.
- Thresholds/weights/exclusions are frozen before held-out outcomes.
- The protocol resists all known fixtures without invalidating legitimate cautious or fallback behavior.

## 9. Prospective goals (not results)

- Zero accepted duplicate scenario/run/receipt identities.
- Zero expected route-phase or telemetry slots silently omitted from a claimed complete run.
- Zero censored/timed-out/unsupported outcomes recoded as negative or successful.
- Zero post-result score configuration changes accepted as confirmatory evidence.
- Every known score exploit either causes a completeness rejection or remains visible as a separate failed task dimension.

These are protocol acceptance goals, not measured SAM capabilities or operational safety thresholds.

## 10. Claim ceiling

Passing these fixtures establishes only that the named SAM benchmark version resists the listed score-manipulation attacks under the declared execution and measurement assumptions. It does not prove resistance to all exploits, physical safety, real-world crash-risk reduction, ISO 26262 / ISO 21448 qualification, roadworthiness, deployment readiness or SAE Level 4/5 capability.

## References and owner links

- [Symthaea #6256 — multi-benchmark road/autonomy suite](https://github.com/Luminous-Dynamics/symthaea/issues/6256)
- [Symthaea #6283 — measurement validity and benchmark evidence graph](https://github.com/Luminous-Dynamics/symthaea/issues/6283)
- [Symthaea #10043 — counterexample-guided discovery](https://github.com/Luminous-Dynamics/symthaea/issues/10043)
- [Symthaea #10012 — runtime assurance](https://github.com/Luminous-Dynamics/symthaea/issues/10012)
- [Symthaea #10237 — counterfactual decision quality](https://github.com/Luminous-Dynamics/symthaea/issues/10237)
- [Symthaea #10257 — model discrepancy and planner exploitation](https://github.com/Luminous-Dynamics/symthaea/issues/10257)
- [Symtropy #3387 — deterministic scenario execution/replay](https://github.com/Luminous-Dynamics/symtropy/issues/3387)
- [CARLA Leaderboard 2.1](https://leaderboard.carla.org/)\n- [Comfortably Stuck: Evaluation Pitfalls in Bench2Drive's Closed-Loop Protocol (October 15, 2026)](https://arxiv.org/abs/2610.21741)
