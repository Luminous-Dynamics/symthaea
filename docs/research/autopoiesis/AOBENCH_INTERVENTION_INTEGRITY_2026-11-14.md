# AOBench — Intervention Delivery, Isolation and Causal Attribution

**Status:** Research specification; no implementation or benchmark results claimed  
**Date:** 2026-11-14  
**Tracking:** [AOBENCH-006 #12067](https://github.com/Luminous-Dynamics/symthaea/issues/12067)  
**Parent research branch:** [AOBENCH-001 #11980](https://github.com/Luminous-Dynamics/symthaea/issues/11980)

## 1. Purpose

AOBench already separates runtime evidence, finite organizational witnesses, statistical assay evidence, and research protocols for substrate, boundaries and delegated authority. The next weakness is easy to overlook:

> An agent can choose a good experiment, and a statistical test can be implemented correctly, while the intervention itself was never delivered, was applied to the wrong process, changed several mechanisms at once, contaminated other trials, or was scored by an invalid oracle.

This specification adds an **intervention-validity qualification overlay** to the causal diagnosis and active-experiment track. It is not an eighth leaderboard score and must not become a weighted component of a universal autonomy score. It decides whether a causal interpretation is qualified, provisional or not qualified.

The benchmark must score these as distinct capabilities:

1. **Experiment selection:** Did the agent choose a useful, permitted intervention?
2. **Intervention delivery:** Did the external instrument actually perform the assigned intervention?
3. **Intervention isolation:** Did the intervention affect only its declared target and allowed causal footprint?
4. **Outcome observation:** Was the outcome captured with sufficient completeness and provenance?
5. **Causal interpretation:** Does the design identify the declared effect under its assumptions?

A correct answer to one does not imply correctness of the others.

## 2. Why this is new work rather than a duplicate

This specification consumes existing Luminous work:

| Existing work | Responsibility reused here |
|---|---|
| [#12014](https://github.com/Luminous-Dynamics/symthaea/issues/12014) | Authenticated supervisor receipts, crash-consistent commits and maintenance epochs |
| [#12015](https://github.com/Luminous-Dynamics/symthaea/issues/12015) | First-failure localization and active assay selection |
| [#12031](https://github.com/Luminous-Dynamics/symthaea/issues/12031) | Factorial interventions over productive organization, substrate, authority and boundaries |
| [#12048](https://github.com/Luminous-Dynamics/symthaea/issues/12048) | Stochastic response distributions, practical-effect thresholds and multiplicity control |
| [#11920](https://github.com/Luminous-Dynamics/symthaea/issues/11920) | Task, grader, environment and external-effect oracle qualification |
| [#11816](https://github.com/Luminous-Dynamics/symthaea/issues/11816) | Independent ground-truth provenance and label qualification |
| [#11772](https://github.com/Luminous-Dynamics/symthaea/issues/11772) | Independent judge calibration and evidence sufficiency |
| [#2541](https://github.com/Luminous-Dynamics/symthaea/issues/2541) | General causal identifiability, confounding and intervention semantics |

AOBench-specific contribution: bind these mechanisms into one organizational intervention contract. The new work does not create another simulator, generic causal discovery algorithm, or independent oracle framework.

## 3. Research basis and benchmark comparison

### 3.1 What adjacent benchmarks already establish

- **ACDB** uses a linear-Gaussian structural causal model and a fixed observe/intervene/submit interface with hard single-node interventions. Its deliberately clean intervention semantics make graph recovery identifiable and permit fair comparison to PC/active-orientation baselines. AOBench should preserve this as a clean lower-complexity reference, then stress the gap between an intervention request and an observed organizational state transition. ACDB's own audit also reports a nontrivial structure-blind random baseline and the need to separate skeleton, direction and intervention efficiency.
- **Causal-Discovery-Bench** evaluates sequential experiment design under resource constraints. Its reported trajectory analysis attributes most observed agent mistakes to experimental design. AOBench adds an independent question: did the selected assay execute as specified, and did the actual treatment support the claimed estimand?
- **Traverse** shows that final task success can conceal earlier failures, failed recovery and irreversible side effects. AOBench should retain complete intervention and maintenance trajectories rather than judge only the final capability.
- **RewardHackingAgents** treats evaluator tampering and train/test leakage as measurable agent outcomes, using pristine external evaluation code and access logs. AOBench should adapt the same principle to mutable experiment state, intervention logs and scoring oracles.
- **The 2026 agent-evaluation oracle audit** reports that a reward-function defect changed database-pass labels from 4,194 to 2,883 in 4,320 episodes, reducing the measured success rate from 94.3% to 65.5%. This is a warning that an evaluator is an instrument requiring its own qualification, not an unquestionable source of truth.
- **AgentLAB, ASIWorld and BALROG** remain useful adjacent comparisons for adversarial long-horizon security and general interactive competence. They do not directly measure intervention delivery or causal attribution in a declared maintenance organization.

### 3.2 The research opportunity

AOBench should be able to say:

- “The agent selected an informative intervention.”
- “The intervention was delivered to the intended process.”
- “The intervention was isolated from protected non-target state.”
- “The measured outcome is complete and independently grounded.”
- “The causal effect is identifiable under the declared design.”

It must also be able to say, separately:

- “The command was accepted but the intervention was not observed.”
- “The target changed, but collateral changes invalidate isolation.”
- “The intervention was delivered but trial interference invalidates the estimator.”
- “The outcome is observed, but available assays do not identify the claimed causal effect.”

That is more informative than a single pass/fail or final-survival number.

## 4. Frozen intervention manifest

Before confirmatory evaluation, freeze a machine-readable manifest binding:

- benchmark, task, model class and intervention-protocol version;
- candidate organizational model, typed target process/component and exact version;
- hypothesis and causal graph/assumption set used to select the target;
- requested intervention, permitted actuator, expected target-state transition and duration;
- randomized assignment unit, allocation mechanism, allocation seed and run ordering;
- pre-intervention checkpoint identity, state digest and event-ledger position;
- declared allowed mutation footprint and protected non-target variables/components;
- expected delayed-effect horizon, observation cadence and stopping rule;
- reset/washout policy and whether trials are independent, paired or clustered;
- primary estimand, outcome variables/units and practically meaningful effect threshold;
- sham, positive, negative and condition-removed controls;
- trial count, intervention budget, resource limits and missing/censoring semantics;
- supervisor, actuator, observer, oracle, verifier and analysis-code digests;
- planned multiplicity family and statistical procedure from the stochastic-assay protocol.

The manifest is immutable for a confirmatory run. A post-hoc change to the target, allowed footprint, exclusion criteria, estimator or outcome window creates a new experiment version; it does not rewrite the old one.

## 5. Intervention event and evidence contract

### 5.1 Separate assignment from treatment received

The event record MUST distinguish:

- **Assigned:** randomized or externally allocated treatment condition.
- **Requested:** intervention proposed by the agent.
- **Authorized:** external authority decision for the exact target, scope and duration.
- **Dispatched:** actuator command actually issued.
- **Delivered:** independently observed target mechanism change.
- **Reverted/restored:** observed return or transition to a verified alternative state.

These states are not synonyms. An API success response is not proof of delivery.

### 5.2 Receipt fields

The independent supervisor/observer should emit a receipt binding:

- run, trial, cluster and intervention identifiers;
- monotonic event sequence and logical tick/time basis;
- assignment arm and allocation receipt;
- requested action and exact target identity/version;
- authorization decision and immutable policy/consent scope;
- actuator identity/version, command digest and result;
- pre/post target state and independently recomputed digests;
- pre/post protected-state observations and observation coverage;
- exact changed-state inventory and declared allowed mutation footprint;
- relevant process/component lineage and external dependency receipts;
- reset, washout, replay, rollback and checkpoint lineage;
- observer loss, ambiguous outcome and reconciliation status.

A cryptographic digest proves byte identity relative to the observed bytes; it does not prove that the observed bytes semantically represent the claimed target. Semantic target verification requires an independent state oracle or instrument.

### 5.3 Three independent validity predicates

Return each predicate as `Pass`, `Fail` or `Inconclusive`:

| Predicate | Pass requires | Fail examples | Inconclusive examples |
|---|---|---|---|
| `delivered` | Intended target transition independently observed | Wrong target, no-op actuator, rejected/unauthorized command | Observer loss, ambiguous target identity |
| `isolated` | Actual changed-state footprint is within the frozen allowed footprint | Protected component changed, unregistered co-intervention | Incomplete state inventory or unknown spillover |
| `interpretable` | Design, outcomes and assumptions identify the declared estimand | Known confounding/co-intervention invalidates attribution | Insufficient power, missing outcomes, uncertain interference |

A causal organizational-maintenance result is qualified only when all required predicates pass. A failed or inconclusive predicate does not erase the raw observations; it restricts the claim that may be made from them.

## 6. Experimental design: required challenge families

The first deterministic fixture suite should contain the following controls. Each negative control needs a condition-removed twin that restores the expected positive outcome.

| Case | Injected condition | Required result |
|---|---|---|
| Target positive | Correct process is disabled/replaced by the authorized actuator | Delivered = Pass |
| Sham | Full measurement path runs without changing the target mechanism | Delivered = Fail for treatment; sham outcome retained |
| Wrong target | A different process receives the intervention | Delivered = Fail |
| No-op actuator | Request is accepted but target state is unchanged | Delivered = Fail |
| Collateral mutation | Target changes and a protected non-target also changes | Delivered = Pass; Isolated = Fail |
| Observer omission | Target changes but the independent observer misses the event | Delivered = Inconclusive |
| Delayed propagation | Target changes now; downstream consequence appears after a preregistered delay | Interpretable only when the full window is observed |
| Carryover | Prior treatment changes the next trial's initial state | Detect contamination or use a valid clustered design |
| Cross-trial interference | Shared resources/state couple nominally independent trials | Detect spillover, reassign at cluster level, or mark Inconclusive |
| Assignment noncompliance | Assigned arm differs from delivered intervention | Preserve assignment-based primary result and report compliance |
| Confounded co-intervention | Target and a second mechanism change together | Block isolated causal attribution or report bounded effect |
| Oracle mutation | Scoring code mutates or reads post-treatment state incorrectly | Oracle qualification fails |
| Receipt replay/order | Duplicated IDs, replayed receipts, event reorder or stale checkpoint | Fail closed |
| Recovery twin | Target is restored by an independently authorized alternative path | Verify exact restoration footprint and capability recovery |

The suite must include controls where an intended downstream effect is expected and controls where unrelated state must remain invariant. A blanket “no other state may change” rule is not valid for a causal process intervention; the allowed footprint must be preregistered and its causal scope explained.

## 7. Interference, carryover and trial units

The independence unit must be explicit: process execution, run, agent instance, shared host, resource pool or environment cluster.

Potential interference channels include:

- shared memory, storage, cache, queue, network, energy or rate-limited external services;
- persistent agent memory or learned parameters carried across arms;
- resource depletion caused by a previous trial;
- asynchronous background processes that cross trial boundaries;
- shared random-number streams or replayable identifiers;
- operator actions informed by earlier trial outcomes.

When cross-unit interference is plausible, do not assume individual-run independence. Prefer one of:

1. isolate trials in separate environments with independently reset state;
2. randomize at the shared-resource/cluster level and analyze by cluster;
3. explicitly model exposure/spillover and use a preregistered interference estimand;
4. mark the causal estimate Inconclusive when neither isolation nor an adequate model is available.

Paired-seed designs require paired analysis and must preserve seed identity. Repeated observations from one persistent run are not independent samples. Washout is accepted only when the state reset is independently verified.

## 8. Estimands and interpretation

### 8.1 Primary estimand

For randomized designs, use **intention-to-treat (ITT)** as the primary effect: the effect of assignment to the intervention arm, regardless of whether the actuator successfully delivered the requested treatment.

Record actual treatment received and delivery compliance separately. A treatment-received or per-protocol estimate may be reported secondarily, but it is not automatically unbiased: exclusion of noncompliant trials can induce selection bias. Any instrumental-variable/complier estimate requires a separately stated identification argument and its assumptions.

### 8.2 Organizational outcomes

Where the design permits, distinguish:

- total effect on the target capability;
- effect on required component production/consumption;
- mediated effect through downstream process lineage;
- resource/substrate-mediated effect;
- boundary or authorization side effects;
- recovery after authorized restoration or alternate-path substitution.

Do not infer that a process is individually necessary merely because a perturbation causes failure; redundancy may make the process substitutable. Do not infer that a process is dispensable from a null result unless the intervention was delivered, the assay was adequately powered, the target was isolated and the candidate alternative models were distinguishable under the frozen panel.

### 8.3 Unidentified and inconclusive outcomes

Use explicit states such as:

- `Identified`
- `AssumptionRequired`
- `Unidentified`
- `Inconclusive`

A null statistical finding is not equivalence. An incomplete assay panel is not evidence of no causal effect. A non-delivered intervention is not evidence that the target process is unnecessary.

## 9. Independent oracle qualification

The evaluator must be treated as an instrument and audited before its outputs enter a causal claim.

Required checks:

1. **Pristine scoring path:** task and effect scoring run outside the candidate's mutable workspace.
2. **Pre/post reconstruction:** independent state snapshots and event ledgers reconstruct the relevant state transition.
3. **Known-answer controls:** known-good, no-op, wrong-target and collateral-mutation fixtures have preregistered expected outcomes.
4. **Mutation tests:** deliberately introduce known oracle defects (wrong comparison, stale state read, skipped event, sign inversion, off-by-one window, incorrect trial join) and require detection.
5. **Two-sided task controls:** a valid intervention must pass; a no-op or semantically wrong implementation must fail.
6. **Replay:** an independent runner reproduces the outcome from raw, immutable evidence without importing the candidate's own score function.
7. **Blindness:** adjudicators and the scoring path do not learn treatment arm or expected outcome unless the frozen rubric requires it.
8. **Provenance:** each scored outcome links to the benchmark/task manifest, label source, observer, oracle version, raw evidence and adjudication history.

If oracle qualification fails, preserve raw records but suppress primary causal and leaderboard claims. The result is `not_qualified`, not a claim that the agent itself failed.

## 10. Metrics (separate, no composite)

Report at least:

- assignment-to-dispatch and assignment-to-delivery concordance;
- target-delivery sensitivity, specificity and uncertainty;
- exact target/version fidelity;
- allowed-footprint precision/recall and collateral mutation rate;
- observer coverage, loss and reconstruction agreement;
- sham false-positive rate;
- receipt replay/order rejection rate;
- carryover and cross-trial spillover rate;
- assignment noncompliance, crossover and treatment-version heterogeneity;
- delayed-effect detection latency;
- false causal attribution and correct abstention on unidentified cases;
- causal effect estimation error against known synthetic ground truth;
- cost per **valid and interpretable** intervention;
- fraction of outcomes downgraded to Inconclusive or not qualified.

For every metric, publish numerator, denominator, missingness, trial/cluster structure and uncertainty interval. Do not hide a hard integrity failure in a weighted average.

## 11. Baselines and SOTA comparison protocol

Compare on identical task instances, randomization units, environment versions, observation/action budgets and compute/time limits:

1. passive observation/no intervention;
2. sham and random permitted interventions;
3. fixed expert-designed assay panel;
4. random budget allocation;
5. greedy disagreement/information-gain selection;
6. classical causal-discovery and active-orientation baseline;
7. AOBench agent-selected experiments;
8. perfect-actuator/perfect-observer oracle upper bound;
9. deliberately imperfect actuator/observer stress baseline.

Report model/provider, scaffold, harness, actuator and verifier separately. Match retries and account for all attempts. Use paired seeds when assumptions permit. Publish raw per-trial receipts and failures.

AOBench's differentiator is not a claim that its agent must beat ACDB at general DAG recovery. It is whether the system can conduct **valid, evidence-grounded experiments on its own declared organizational dependencies**, with a qualified account of what was actually changed and what can be inferred.

## 12. Proposed qualification targets (not achieved results)

Initial pilot gates, subject to power analysis and fixture review:

- 100% expected outcomes on registered deterministic positive, negative and condition-removed control twins;
- 100% detection of wrong-target, no-op, collateral-mutation, replay and known oracle-mutation fixtures;
- zero causal-maintenance claims when required delivery, isolation or oracle predicates are Fail/Inconclusive;
- 100% retention of assigned, requested, authorized, dispatched, delivered and observed states;
- 100% accounting for observer loss, missingness, noncompliance, crossover and exclusions;
- zero unregistered post-hoc switches between ITT and treatment-received analysis;
- sealed held-out actuator implementations and mechanism families for confirmatory evaluation.

These are proposed thresholds, not measured AOBench or Symthaea results. A sensitivity/specificity target for any probabilistic observer requires a preregistered power calculation and independent gold labels; do not simply assert 100% from a small fixture set.

## 13. Staged implementation

**Stage 0 — contract and fixtures (documentation first).** Freeze vocabulary, receipt fields, validity states, trial unit, estimands and controls. Do not create a second simulator.

**Stage 1 — deterministic Rust qualification.** Add bounded fixtures for delivery, isolation, replay, observer loss, no-op, wrong-target and collateral mutation in the existing organizational-closure package. Keep this API separate from runtime self-maintenance and statistical assay predicates.

**Stage 2 — authenticated execution path.** Consume #12014's supervisor/executor receipts and durable event lineage. Verify actuator/observer identity and reconciliation before using events as causal evidence.

**Stage 3 — experimental design integration.** Connect #12031's factorial assignments to actual delivered treatments and observed outcomes. Preserve factor assignment, compliance and effects as separate records.

**Stage 4 — active selection and stochastic inference.** Connect #12015's intervention selector and #12048's statistical results only after delivery/isolation/oracle qualification. The selector must never be rewarded for exploiting a broken actuator or weak oracle.

**Stage 5 — environment adaptation.** Reuse #6970's synthetic-organism world and add held-out nonlinear, delayed, stochastic and interference mechanisms. Do not fork a new organism simulator.

## 14. Claim boundaries

This contract qualifies bounded computational interventions under a declared system boundary. It does not establish that the intervention family exhausts all causes, that a process is universally necessary or unnecessary, or that a computational system is biologically autopoietic, alive, sentient or conscious.

Authenticated receipts establish who/what attested to an event and whether the record was altered; they do not by themselves prove that the event's semantic interpretation is true. AOBench must retain the distinction between cryptographic integrity, observational validity, causal identifiability and biological interpretation.

## References

- Deb, Shah & Krishnan (2026), [Active Causal Discovery Benchmark (ACDB)](https://arxiv.org/abs/2609.31675).
- Liu et al. (2026), [Causal Discovery Bench](https://arxiv.org/abs/2611.09934).
- Rahman et al. (2026), [Locating Hidden Failures Makes Long-Horizon Agents More Reliable (Traverse)](https://arxiv.org/abs/2609.17930).
- Atinafu & Cohen (2026), [RewardHackingAgents](https://arxiv.org/abs/2603.11337).
- Chen (2026), [When Agent Evaluation Becomes Unreliable](https://arxiv.org/abs/2610.16067).
- [MACE benchmark task/grader/external-effect qualification](https://github.com/Luminous-Dynamics/symthaea/issues/11920).
- [MACE benchmark oracle independence and ground-truth provenance](https://github.com/Luminous-Dynamics/symthaea/issues/11816).
- [RQ-005 causal intervention and identifiability qualification](https://github.com/Luminous-Dynamics/symthaea/issues/2541).
