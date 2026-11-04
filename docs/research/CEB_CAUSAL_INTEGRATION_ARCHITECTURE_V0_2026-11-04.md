# CEB causal integration architecture V0 — 2026-11-04

**Status:** design-only architecture contract; not an implementation or qualification result.  
**Parent:** [CEB-CAUSE-INTEGRATION-001 #10387](https://github.com/Luminous-Dynamics/symthaea/issues/10387)  
**Protocol:** [CEB causal debugging V0](https://github.com/Luminous-Dynamics/symthaea/blob/research/ceb-causal-debugging-2026-11-04/docs/research/CEB_CAUSAL_DEBUGGING_V0_2026-11-04.md)  
**Claim ceiling:** this document specifies how a future CEB adapter should preserve evidence authority, causal identification and prospective forecasts. It does not establish that Symthaea currently performs causal engineering diagnosis.

## 1. Decision

Build a **CEB-specific, append-only investigation ledger adapter** that composes existing benchmark and metareasoning semantics. Do not create:

- a second general value-of-computation predictor or scheduler;
- a second active-probe engine;
- a new CEB run-receipt or DSSE format;
- an inference path that upgrades a plausible hypothesis, dream output, correlation or passing patch into a confirmed cause.

The adapter is a data and validation boundary. It records what the system believed, what it predicted before acting, what action was taken, what was actually observed, how the diagnosis changed, and how a repair was independently graded. It does not itself execute experiments, grant authority, authenticate a runner, or decide that an observation is true.

## 2. Source audit: what exists today

The following audit is against the repository's `main` source revision inspected on 2026-11-04. The source file is `crates/core/symthaea-core/src/hdc/consciousness_feedback_dynamics.rs`.

### 2.1 CausalDreamIntegrator is an exploratory candidate generator

The current `CausalHypothesis` contains only:

- `cause: String`;
- `effect: String`;
- `prior: f64`;
- optional HDC relationship encoding;
- a priority value.

It has no task identity, source revision, applicability domain, competing-hypothesis set, evidence references, evidence ancestry, calibrated posterior, identification state, intervention record or falsification history.

`extract_from_causal_mind` is explicitly a no-op placeholder. Hypotheses are manually queued. `next_hypothesis` orders that queue by priority; it does not select an engineering experiment from observed causal evidence.

### 2.2 Dream-derived “evidence” is not causal evidence

The current counterfactual-dream path maps `DreamResolution` variants to fixed numeric values, combines that number with mean fragment coherence, and calls the result evidence. Its `bayesian_update` method is a fixed-rate linear interpolation:

```text
updated = prior + 0.3 * (dream_score - prior)
```

This is not a Bayesian posterior update: it does not define a likelihood model, condition on an observed intervention, or account for the probability of the observation under competing hypotheses. The result is useful as an exploratory internal heuristic, but must not be described as causal identification or calibrated probability.

Separately, `FeedbackProcessor::process_counterfactual_dream` creates `CausalDiscovery` records from dream fragments using fragment valence and divergence. Those values are properties of the generated dream representation, not measurements of an external engineering system.

**Integration rule:** a dream-generated idea may seed a CEB `HypothesisProposed` event. It may not populate `EvidenceObserved`, `InterventionObserved`, or `CausalClaimIdentified` without independent evidence.

### 2.3 Existing research contracts are not yet proven Rust APIs

The current repository search found issue-level research contracts, not stable Rust types that a CEB adapter can simply import:

| Existing work | Reusable semantics | Missing / do not claim |
|---|---|---|
| [#5476 prospective VoC](https://github.com/Luminous-Dynamics/symthaea/issues/5476) | Forecast before outcome; separate forecast and realized outcome; task/profile binding; uncertainty, predicted information gain, error reduction, resource cost and deadline risk; missing counterfactuals remain missing | A repository-wide `CognitiveValueForecastV1` / `CognitiveValueOutcomeV1` Rust API was not found in the inspected source search |
| [#5478 causal metareasoning](https://github.com/Luminous-Dynamics/symthaea/issues/5478) | Matched-budget treatment arms; intact/lesion/sham/rescue controls; negative controls; prefer simpler policy on a null result | A typed, CEB-specific ablation identity and event contract were not found |
| [#8286 active probe design](https://github.com/Luminous-Dynamics/symthaea/issues/8286) | Distinguish interventions, probe costs, disturbance and exact small-instance oracles; existing pilot semantics | The issue identifies a Python Boolean-mechanism pilot (100 seeded two-input instances), not an established Rust typed-action API or full multi-node SCM engine |
| [#8721 scientific tool allocation](https://github.com/Luminous-Dynamics/symthaea/issues/8721) | Prospective tool allocation and cost-to-target analysis; reuse #5476/#5478/#8286 | It is a scientific task application, not the CEB debugging scheduler |
| [#10115 CEB V0](https://github.com/Luminous-Dynamics/symthaea/pull/10115) | Existing CEB task and run-receipt profile | Do not add another execution receipt |
| [#10314 general run receipt](https://github.com/Luminous-Dynamics/symthaea/pull/10314) | Generic benchmark run accounting | A content hash is not a signature, runner identity or isolation proof |
| [#10167 / #10212 verifier work](https://github.com/Luminous-Dynamics/symthaea/issues/10167) | DSSE, subject binding, primary/regrade consistency and deterministic verification goals | Receipt verification does not provision a trustworthy runner |
| [#10064 isolation prerequisite](https://github.com/Luminous-Dynamics/symthaea/issues/10064) | Hard boundary before executing task-controlled code | Remains a prerequisite; no CEB candidate/build/test execution is authorized by this design |

The adapter must therefore implement an explicit CEB boundary and use stable references to these contracts. It must not pretend that an issue specification is already a callable library API.

## 3. Architecture and authority boundaries

The proposed flow is:

```text
CEB task + pinned source/configuration digests
                 |
                 v
        Hypothesis ledger
                 |
        forecast sealed first
                 |
                 v
      typed experiment proposal
                 |
      authorized isolated executor  <--- blocked by #10064 today
                 |
                 v
         immutable observation
                 |
                 v
      hypothesis update / abstain
                 |
                 v
          repair proposal
                 |
                 v
       independent primary grade
                 |
                 v
       pristine independent regrade
```

The ledger is the reasoning trace. The existing CEB/generic receipt and DSSE layers remain the execution-attestation boundary. The ledger may be referenced as a task artifact by digest only through a compatible existing receipt field; if no compatible field exists, resolve that in #10370 before extending a receipt contract. Do not silently add a fourth receipt schema.

### 3.1 Separate six kinds of authority

Use explicit authority labels, not one generic `verified` boolean:

1. `source_declared`: literal source/configuration content;
2. `syntax_derived`: parser-derived structure without semantic resolution;
3. `compiler_resolved`: facts established by a pinned compiler or language server;
4. `runtime_observed`: measured execution with named instrumentation and environment;
5. `intervention_observed`: measured result after a controlled intervention with a declared assignment and control strategy;
6. `inferred` / `unresolved`: candidate reasoning or insufficient/conflicting evidence.

Compiler-resolved evidence can establish types and symbol resolution; it does not by itself establish runtime causality. Runtime observation can establish a repeatable association; it does not by itself identify a causal effect. Intervention evidence must still state its assumptions, controls, confounders and external validity.

### 3.2 Task-scoped hypotheses

Each hypothesis should bind:

- stable hypothesis ID and exact causal claim;
- CEB task ID, task revision, source-tree digest and configuration-world digest;
- applicability domain and known OOD boundaries;
- explicit competing hypotheses, including a null/no-fault alternative;
- prior value plus prior source and interpretation (do not conflate subjective confidence with a calibrated probability);
- supporting and contradicting evidence IDs;
- evidence ancestry / dependence groups, so duplicated logs or tests are not counted as independent;
- unresolved confounders and identification assumptions;
- current status: `candidate`, `correlational`, `intervention_supported`, `identified_under_assumptions`, `falsified`, `non_identifiable`, `out_of_domain` or `unresolved`.

Hypothesis updates are new ledger events. Do not overwrite the original prior or erase a rejected alternative. A hypothesis can remain unresolved after a repair passes.

### 3.3 Typed action semantics

The adapter must distinguish at least:

| Action kind | Question it can answer | Does not establish by itself |
|---|---|---|
| Observation / read-only probe | What is present in source, state, telemetry or trace? | Effect of changing a cause |
| Hard intervention (`do(X=x)`) | What changes when a declared variable is externally set? | Generality beyond tested assignment and environment |
| Mechanism replacement | What happens when the mechanism for a variable is replaced? | Effect of ordinary configuration toggles |
| Test execution | Does this frozen test oracle pass in this environment? | That the oracle is valid or covers the defect |
| Configuration perturbation | Does the behavior vary across this frozen feature/target/configuration factor? | That the factor is the sole cause when interactions remain |
| Tool allocation | Does spending compute/tool budget change what can be learned or decided? | A physical/software-system intervention |

Every action has an action ID, action kind, exact task/source/config snapshot, policy digest, parameter digest, intended discrimination, declared budget, risk, blast radius, reversibility, permission/authorization reference, and expected stopping behavior. An action proposal is not proof that the action executed.

A future Rust `ExperimentAction` enum should encode these variants so impossible category substitutions are rejected at compile time. This is a CEB adapter type, not a replacement for #8286's general probe-selection work.

### 3.4 Prospective forecast and realized outcome

Every candidate experiment that is being evaluated must have a separately sealed forecast before action or result visibility.

Forecast fields:

- forecast ID, task/source snapshot digest, hypothesis-set digest and policy/profile digest;
- action ID and action-kind identity;
- predicted observation distribution (or explicit unknown / not-calibrated state);
- predicted information gain and expected decision-value change as separate quantities;
- predicted task-error reduction;
- resource cost, wall-clock/latency estimate, deadline risk;
- safety risk, expected disturbance, reversibility and blast radius;
- declared alternatives and forecast horizon;
- append-order sequence and digest commitment.

Outcome fields:

- forecast ID and action ID;
- actual execution receipt reference or explicit not-executed status;
- observed result digest, measurement/provenance reference and authority label;
- measured cost, latency and side effects;
- whether the result was missing, censored, invalidated or not observable;
- model update and affected hypothesis IDs.

**Forecast immutability:** the forecast must be a separate immutable record. New evidence may correct the current diagnosis but may not rewrite what was predicted before the experiment. The append order establishes ledger order only; it does not prove trustworthy wall-clock time or an isolated execution. Those require qualified signing and runner evidence in the existing attestation layer.

**Missingness:** an unselected experiment or unobserved counterfactual is `not_observed`, not zero utility. Distinguish `not_run`, `not_measurable`, `censored`, `invalidated`, and `not_applicable`.

### 3.5 Causal ablation identity

An ablation record binds treatment, task/source snapshot, backend/model, harness, prompt, tools, resource budget, seed, policy digest and evaluator digest. It names the mechanism being tested and one of:

- intact;
- lesion (mechanism computed but prevented from affecting the action);
- sham (matched overhead without the causal mechanism);
- rescue (the specific disabled consumption edge restored).

Compare treatment arms on matched task instances and budgets. Use the existing #5478 protocol; do not build a second scheduler. Include random matched-cost and static/simple-policy controls. A null result must be allowed to retire unnecessary complexity.

### 3.6 Repair and grade are separate from diagnosis

A repair proposal binds the task/source/config digests, patch digest, hypothesis IDs, experiment ledger digest, intended invariant changes, affected symbols/features/targets and declared regression risks.

An independent grade binds the exact candidate patch and immutable grader/evaluator revision. Primary grade and pristine regrade remain separate. A passing repair proves only the declared behavioral contract under the qualified grader and environment; it does not retroactively prove that the nominated root cause was uniquely identified.

## 4. Append-only event model and invariants

The future adapter should expose typed events roughly equivalent to:

- `TaskBound`
- `EvidenceReferenced`
- `HypothesisProposed`
- `ForecastSealed`
- `ActionProposed`
- `ActionAuthorized`
- `ActionObserved`
- `OutcomeRecorded`
- `HypothesisUpdated`
- `RepairProposed`
- `RepairGraded`
- `Abstained`

Every event binds the ledger/task identity, monotonic sequence, previous-event digest, event digest, source/task revision and its parent evidence/action/forecast references. Event IDs are unique. The verifier recomputes digests and all derived counts from records rather than trusting summary counters.

Required invariants:

1. **No post-hoc forecast:** an outcome must reference an existing sealed forecast whose ledger sequence precedes the action and outcome.
2. **Exact snapshot binding:** forecast/action/outcome/repair/regrade must agree on task and source/configuration identity where the protocol requires equality.
3. **No evidence laundering:** authority labels cannot be raised by copying, summarizing or re-hashing lower-authority evidence.
4. **No duplicated independence:** repeated or descendant evidence sharing one origin is grouped and cannot be counted as independent confirmation.
5. **No unobserved-zero substitution:** missing counterfactuals remain explicitly missing through scoring and calibration.
6. **No causal overclaim:** source text, syntax, compiler facts, dream coherence, correlations and test pass are not intervention effects.
7. **No false posterior certainty:** probabilities have declared meanings and are evaluated on held-out data; arbitrary scalar interpolation is not a calibrated Bayesian posterior.
8. **No silent OOD promotion:** an out-of-domain task/profile cannot inherit a prior calibration claim without a transfer evaluation.
9. **No action-type collapse:** observation, intervention, mechanism replacement, tests, config perturbations and tool allocation remain distinct.
10. **No diagnosis/repair conflation:** a successful repair does not automatically upgrade causal identification status.
11. **No receipt substitution:** the ledger does not replace the execution receipt, signer policy, independent regrade or runner qualification.
12. **No runner bypass:** no task-controlled build script, test, candidate executable or grader executes before #10064 is qualified.

A local hash chain can detect accidental or post-commit modification relative to a trusted anchor. It does not authenticate the producer, prove independent custody, establish real-world chronology, or prove the truth of recorded observations.

## 5. Scoring: measure the decision process, not just the final patch

Keep each measure separate:

- root-cause top-1/top-3 accuracy, with a null and unresolved class;
- hypothesis calibration (Brier score, log loss and reliability plots) on held-out tasks;
- causal-edge precision/recall stratified by evidence authority;
- forecast outcome calibration and decision-value error;
- discriminating-experiment yield, information gain, and decision change per unit cost;
- risk-weighted experiment utility and unsafe/prohibited action rate;
- false confident-singleton and premature-commitment rates;
- independently accepted repair rate, false-fix rate and regression rate;
- controlled-task bug recall and edit precision; natural-task admissible-change and collateral-change review;
- cost per independently accepted repair, experiments per accepted repair, latency and human review burden;
- correct abstention / escalation, coverage and selective risk;
- held-out repository-family, fault-family, configuration-world and environment transfer.

Do not use ECE as the only calibration measure, and do not set a numerical calibration gate until there is a sufficiently large held-out set. Report task/family clustered uncertainty, paired differences and complete dispositions (pass, fail, abstain, unsupported, tool failure, timeout, ungraded).

## 6. No-runner implementation order

This order is deliberately executable without running task-controlled code:

**N0 — Architecture and vocabulary**
- Review this contract against #5476, #5478, #8286, #8721, #10115, #10314 and #10370.
- Freeze the authority, action-kind, status and missingness vocabularies.
- Record every missing API explicitly; do not invent a dependency that does not exist.

**N1 — Pure typed ledger model (prototype authored; compilation pending)**
- A standalone, dependency-free Rust crate now exists at `tools/ceb-causal-ledger/` with typed task bindings, immutable hypothesis revisions, evidence provenance, typed action/authorization, forecasts, outcomes, ablations, repairs and grade records.
- Its deterministic validator checks identity/order/reference constraints, evidence-origin double counting, causal-authority ceilings, missingness, exact repair/grade patch linkage, and primary-to-regrade ordering.
- It is outside the production workspace and is not a production dependency. Seventeen unit tests are authored, but have not been compiled or executed in this environment.
- Digests are opaque caller-supplied values; this crate does not hash, sign, authenticate or prove runner isolation. Keep DSSE verification and trusted-root policy in the existing verifier work.

**N2 — Synthetic record tests (authored; execution pending)**
- The current crate includes inert-struct unit tests for valid records, accepted repair with unresolved diagnosis, missing assumptions, action authorization, post-hoc forecasts, task/source mismatch, dream-only causal promotion, duplicate evidence origin, missing-versus-zero outcomes, denied actions, exact patch linkage, action-type authority, and evidence-parent ordering.
- Additional version-chain and primary/regrade-ordering controls are included.
- These tests have not run because no Rust compiler/toolchain is available in this environment. They must be executed with the pinned toolchain before the prototype is treated as compiled or tested.
- The validator does not yet implement JSON serialization, hash-chain recomputation, DSSE, trusted-root verification, grader isolation or runner attestations.

**N3 — Contract integration**
- Bind task identity to the existing CEB task profile.
- Reconcile the ledger artifact reference with #10314 / #10370 and existing DSSE verifier work; no fourth receipt schema.
- Add explicit adapter tests proving that no second scheduler or probe engine has been introduced.

**N4 — Execution qualification (blocked)**
- Only after #10064: execute CEB fixtures in qualified isolation, use the independent grader and pristine regrade, and emit execution-grade receipts.
- Treat signer trust, runner identity/independence and key custody separately under #10369.
- Do not publish CEB performance or SOTA claims until task validity, grader integrity, isolation and independent regrade are established.

## 7. Immediate acceptance for #10387

- This architecture note is committed and linked from #10387.
- Every proposed field maps to an existing contract or is explicitly marked CEB-local / missing.
- The current dream integration is treated as a candidate generator, not a causal evidence source.
- #5476/#5478/#8286 semantics are reused without claiming their research specifications are already integrated Rust APIs.
- No second scheduler, probe engine or benchmark receipt is introduced.
- A later Rust adapter can be validated entirely with synthetic records before any candidate-code execution.
- #10064 remains a hard gate for execution.

## 8. External benchmark alignment

The causal ledger does not replace external benchmarks. CEB should remain a separate, auditable engineering benchmark and report contract-matched comparisons against:

- SWE-Bench Pro V2 Full and HARD for issue resolution;
- SWE Atlas Codebase QnA, Test Writing and Refactoring for comprehension and engineering quality;
- PDB for atomic bug recall and edit precision;
- HiL-Bench and AgentAbstain for selective information acquisition and safe abstention;
- SWE-fficiency for correctness-preserving performance optimization;
- SRE-Bench for source-free reverse engineering.

These benchmark scores must remain separate because their task populations, graders, environments and resource budgets differ. Record exact dataset/grader revisions and the date checked. The CEB causal claim is narrower: whether prospectively forecast, risk-aware experiments improve independently accepted repairs over matched simpler baselines on held-out engineering fault families.
