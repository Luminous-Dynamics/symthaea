# Computational Engineering Bench: Causal Debugging V0

**Status:** research protocol / design-only  
**Issue:** [#10366 — CEB-CAUSE-001](https://github.com/Luminous-Dynamics/symthaea/issues/10366)  
**Parent benchmark:** [#10115 — Computational Engineering Bench V0](https://github.com/Luminous-Dynamics/symthaea/pull/10115)  
**Claim ceiling:** this document specifies a falsifiable experiment. It does not establish that Symthaea performs causal debugging, that any fixture is executable or qualified, or that any model has improved.

## 1. Research objective

Test whether an agent that maintains competing causal explanations and selects discriminating interventions can deliver more independently accepted engineering repairs at lower investigation cost than retry-, retrieval- or single-diagnosis systems.

The capability under test is a closed loop:

```text
failure + repository snapshot
          |
          v
typed system model + evidence ledger
          |
          v
competing causal hypotheses
          |
          v
predicted outcomes for candidate experiments
          |
          v
risk / cost / information-aware experiment choice
          |
          v
bounded intervention and observation
          |
          v
hypothesis update or falsification
          |
          v
repair proposal -> independent verification -> regrade
```

The loop must be allowed to conclude `unresolved`, `non_identifiable`, `abstained` or `no_safe_experiment`. A plausible explanation is not a confirmed cause. A test pass is not proof that the hypothesized cause was correct. A patch is not an accepted repair until the independent task contract and pristine regrade agree.

## 2. Why this is distinct from existing CEB work

- **Repository twin (#10173):** measures graph construction, typed relationships, invalidation and change-impact prediction. Causal debugging consumes that model but asks which intervention distinguishes competing explanations.
- **Initial CEB task qualification (#10116):** supplies the eventual task/grader separation and execution prerequisites.
- **Experiment allocation (#8721):** supplies scientific tool-allocation lessons. This is related, but tool allocation is not identical to choosing an intervention that identifies a software fault.
- **Prospective value-of-computation (#5476):** supplies the reusable forecast-before-outcome and separate realized-outcome contract. CEB must bind predictions before an experiment and must not turn unobserved counterfactual utility into zero.
- **Causal metareasoning ablations (#5478):** supplies matched-budget treatment arms and intact/lesion/sham/rescue discipline for determining whether the causal policy itself produced a gain.
- **Active probe design (#8286):** has an existing 100-seed Boolean mechanism pilot and a planned extension to noisy multi-node SCMs. Reuse its typed probe/intervention semantics where applicable; do not create a duplicate probe engine.
- **Causal experience transfer (#9516):** supplies qualified experience lineage for later cross-task transfer tests.
- **Receipt verification (#10167 / #10212):** supplies the CEB-specific evidence direction. Reconcile it with the general benchmark-run receipt work (#10314) before introducing any shared receipt contract; this programme must not create a third receipt format.
- **Execution isolation (#10064):** remains a hard prerequisite before running repository-controlled build scripts, tests or candidate code.

### 2.1 Existing Symthaea structures are not yet an engineering diagnosis loop

A source review found two existing structures that are useful context but must not be treated as already-qualified capabilities:

- `CausalHypothesis` in `crates/core/symthaea-core/src/hdc/consciousness_feedback_dynamics.rs` is currently a cause/effect/prior/HV/priority record used by `CausalDreamIntegrator`. Its extraction from `CausalMind` is explicitly a placeholder; hypotheses are manually queued. The "what if" operation returns a textual counterfactual question, not a controlled software intervention.
- `InterventionPlan` in `src/consciousness/causal_byzantine.rs` recommends parameter adjustments for Byzantine-defense behavior. Its semantics are domain-specific and should not be repurposed as a generic repository repair experiment.

The CEB integration should be a task-scoped evidence adapter, not a replacement for these modules and not a new generic scheduler. Its conceptual record should bind:

1. exact task/source/configuration snapshot and evidence authority;
2. competing fault hypotheses and unresolved alternatives;
3. an immutable forecast linked to #5476 before the intervention;
4. a typed experiment action and risk/reversibility contract linked to #8286;
5. the predeclared predicted observation and actual outcome;
6. the hypothesis update, including falsified explanations and missing counterfactuals;
7. causal-ablation identity linked to #5478;
8. any repair proposal and its independent outcome receipt.

The adapter must distinguish observation, hard intervention, mechanism replacement, test execution, configuration perturbation and tool-allocation actions. They answer different questions and must not share an untyped "experiment" label. The explicit integration work is tracked by [#10387 — CEB-CAUSE-INTEGRATION-001](https://github.com/Luminous-Dynamics/symthaea/issues/10387).

## 3. Task and fault model

### 3.1 Initial design set

Design 24–30 task specifications across at least five fault families:

1. **Rust semantics and API evolution:** changed trait bounds, blanket implementations, public API migrations, generic constraints and cross-crate references.
2. **Cargo configuration:** default/optional features, target-specific dependencies, mutually exclusive feature assumptions and build-script effects.
3. **Behavioral regressions:** incorrect state transitions, boundary behavior, stale assumptions, test-oracle gaps and interacting defects.
4. **Distributed behavior:** duplicate delivery, retries, idempotency, ordering, crash/restart, partial commit and recovery.
5. **Performance regressions:** algorithmic complexity, lock contention, allocation churn, cache behavior and resource contention.
6. **Security/configuration:** path handling, privilege boundaries, secret exposure, policy interactions and fixes that improve one property while weakening another.

The initial count is a design target, not a sample-size justification. Final statistical power and task counts must be based on pilot variance and the smallest effect considered practically important.

### 3.2 Required difficulty factors

Each family should contain controlled variation in:

- one fault versus interacting faults;
- direct versus indirect symptoms;
- noisy, incomplete or contradictory observations;
- deterministic versus intermittent failures;
- a real root cause versus observationally equivalent explanations;
- reversible, expensive, risky and prohibited interventions;
- faults that cannot be distinguished within the available budget;
- correct abstention where no safe or identifying experiment exists.

Use at least two independently designed repository templates per family. Hold out complete root-cause families and repository templates, not merely random rows from the same template.

### 3.3 Ground-truth authority

The evaluator retains the injected cause, causal graph, intervention effects, expected outcomes and hidden acceptance tests. These are not agent-visible.

Label evidence by authority:

- `source_declared`: directly present in source or configuration;
- `syntax_derived`: derived from parsing, without name/type resolution;
- `compiler_resolved`: established by the pinned compiler or language server;
- `runtime_observed`: measured by a named execution and instrumentation setup;
- `intervention_observed`: measured after a controlled intervention;
- `inferred`: a candidate explanation that has not been established;
- `unresolved`: insufficient or conflicting evidence.

Do not promote one authority into another. A tree-sitter edge is not a type-resolved edge. A correlation in a trace is not an intervention effect. A model-generated hypothesis is not an observed fact.

## 4. Experiment protocol

For every investigation, require the agent to emit a structured, timestamped hypothesis ledger before selecting an intervention:

- hypothesis identifier and precise claim;
- supporting and contradicting evidence references;
- confidence or probability, with a declared interpretation;
- predicted observations under each candidate experiment;
- expected diagnostic value and uncertainty;
- execution cost, risk, reversibility and blast radius;
- evidence or assumption required for identification;
- explicit stop / abstain condition.

The selected experiment must have a declared purpose. Where safe and feasible, vary one causal factor at a time. When factorial or adaptive interventions are necessary, preserve the assignment and analysis plan so interactions can be distinguished from confounding.

Before an experiment executes, freeze its predicted outcome. After execution, record the actual output and update all affected hypotheses, including explanations that were weakened or falsified. Do not permit the agent to rewrite its prediction after seeing the result.

### 4.1 Candidate experiment selection

Reuse the existing prospective value-of-computation machinery where semantics fit. Compare at least:

- random selection at matched cost;
- fixed sequence / fixed threshold;
- retrieval-ranked experiment;
- uncertainty-only selection;
- disagreement-only selection;
- prospective expected information gain or value of computation;
- Symthaea's full hypothesis-and-risk-aware selection.

Information gain is not itself engineering utility. A highly informative experiment can be unsafe, irrelevant to the repair, too expensive, or unable to change the final decision. Report diagnostic information, repair utility, safety and cost separately.

### 4.2 Stopping

Stop when the task contract is satisfied with sufficient independent evidence, when the remaining experiments cannot change the decision within the declared budget, or when no safe identifying experiment remains. Also stop when evidence contradicts the causal model strongly enough that further repair attempts would be speculative.

A fixed retry budget is a baseline, not the proposed stopping policy.

## 5. Matched treatment arms

Run all arms on identical snapshots, initial observations, tool availability and total budgets:

1. **Retry-only:** conventional repair/retry loop without an explicit hypothesis ledger.
2. **Error retrieval:** retrieval-ranked prior fixes and error similarity.
3. **Single-shot diagnosis:** one diagnosis followed by a repair attempt.
4. **Random matched-cost:** random permitted experiment with the same budget.
5. **Static heuristic:** frozen uncertainty/disagreement thresholds or a fixed experiment order.
6. **Prospective VoC:** existing #8721 machinery, calibrated before the sealed evaluation.
7. **Symthaea causal loop:** typed repository twin, competing hypotheses, predicted outcomes, risk-aware experiments and evidence-gated repair.

Where model comparison is intended, include a minimal model-only harness and a strong coding-agent harness using the same model. Also compare the same model plus Symthaea. Do not attribute a foundation-model capability to Symthaea, or vice versa.

Freeze the model version, harness commit, system prompt, tools, task version, network policy, runner image, toolchain, resource budget and verifier revision for each matched comparison. Use paired task/seed assignments and repeated trials where stochasticity is material.

## 6. Metrics

Do not compress these into a single score.

### Diagnosis

- root-cause top-1 and top-3 accuracy;
- causal-edge precision, recall and F1, stratified by evidence authority;
- localization precision/recall at file, symbol, subsystem and configuration levels;
- hypothesis calibration using Brier score and log score;
- calibration plots and ECE on held-out data.

### Experiment quality

- predicted-outcome calibration before execution;
- discriminating-experiment yield;
- information gained per unit cost;
- fraction of experiments that change the decision;
- unnecessary, redundant, unsafe and non-identifying experiment rates;
- prediction falsification rate and correct hypothesis revision rate.

### Engineering outcome

- independently accepted repair rate;
- false-fix rate and regression rate;
- preserved-invariant rate;
- pristine regrade pass rate;
- change-impact recall/precision, connected to #10173;
- stale-fact rate after invalidation and rebuild.

### Efficiency and autonomy

- experiments, tool calls, wall time, CPU/GPU/memory, tokens and provider cost;
- cost and time per independently accepted repair;
- human review minutes and intervention burden;
- abstention coverage, abstention precision and selective risk;
- destructive-action / privilege-escalation proposals blocked or escalated correctly.

### Transfer

- held-out root-cause family;
- unseen repository template;
- unseen language or build configuration where supported;
- changed dependency/toolchain version;
- shifted runtime regime and partial observability.

A gain on a random row split is not transfer. The holdout must separate causal mechanisms and repository families.

## 7. Preregistered hypotheses and prospective gates

The primary hypothesis is:

> Under matched model, harness and resource budgets, the Symthaea causal loop increases independently accepted repairs on sealed tasks compared with the strongest non-Symthaea baseline, without increasing false fixes or regressions.

Proposed gates, not achieved results or industry standards:

- at least +10 percentage points in independently accepted repair rate over the strongest matched baseline on sealed tasks, with paired uncertainty intervals and a preregistered practical-effect threshold;
- at least 20% fewer experiments **or** 20% lower cost per independently accepted repair, with no more than a 1 pp correctness regression;
- at least 0.90 precision and recall for supported causal relations on the qualified fixture gold set;
- zero semantic-authority inflation;
- 0/1,000 false accepts on varied negative controls and 0/100 false rejects on independently adjudicated positive controls, with exact denominators and attack-family dependence disclosed;
- ECE <= 0.05 only after adequate held-out calibration data; always report Brier/log score and confidence intervals too.

The zero-error control targets do not prove zero underlying error probability. Report exact binomial confidence bounds and cluster related attacks by root cause. Do not interpret repeated variants of one exploit as independent evidence.

Before running confirmatory tasks, publish the primary outcome, practical effect threshold, exclusions, randomization, sample-size reasoning, baseline configurations, analysis code and stop rules. Preserve an untouched confirmation set.

## 8. Benchmark-integrity and grader requirements

The task author, hidden answer key, agent workspace, execution broker and grader are distinct trust domains.

- Hidden labels and reference patches never enter the agent workspace.
- Reference patch passes and empty patch fails before a task is admitted.
- Known-bad, semantically wrong and regression-inducing patches are included as negative controls.
- Grader files and secrets are outside candidate write access.
- The primary result is regraded from a fresh immutable source snapshot and independently prepared environment.
- The grader recomputes outcomes from raw per-task checks, not agent-reported counters.
- Incomplete, refused, unsupported, failed-tool, timed-out and abstained outcomes remain visible in the task denominator and are not silently dropped.
- Evidence records bind task, source, patch, environment, tools, grader, experiment predictions, intervention outputs and regrade identities.

The receipt representation must be reconciled with #10115/#10212 and #10314. Do not publish an aggregate score until the exact accepted receipt contract and independent verifier are qualified.

## 9. No-runner tranche

Until #10064 has a qualified OS-level boundary, work is limited to non-executable artefacts:

1. task and evidence contracts;
2. static synthetic evidence bundles;
3. evaluator-side gold-label authoring guidance kept outside public agent-visible files;
4. deterministic metric/scoring logic over recorded evidence;
5. baseline preregistration and statistical analysis plan;
6. static schema validation and adversarial negative controls.

Do not execute repository-controlled build scripts, procedural macros, tests, generated binaries or agent patches on the host-opt-in runner. The later execution tranche requires isolation, restricted filesystem and network, dropped privileges, resource limits, process-tree cleanup and a pristine external grader.

## 10. External comparison portfolio

The external benchmarks test different constructs and must retain separate scorecards:

| Benchmark | What it tests | How CEB should use it |
|---|---|---|
| SWE-bench Pro V2 / Hard | Long-horizon real-repository issue resolution | Public comparison plus private/held-out transfer; do not infer generalization from the near-ceiling public split |
| SWE-rebench | Fresh issue resolution with cost/token measures | Pin the selected date window and dataset revision; the displayed May 15–July 1, 2026 window is not a November-wide sample |
| Terminal-Bench 4.0.0 | Broad frontier terminal work | Diagnostic comparison only; Epoch AI's September 4, 2026 review labels the release flawed and reports defects in 30/66 tasks |
| CursorBench 4.0 | Ambiguous, multi-file tasks from real Cursor sessions | Measure realism, cost, tokens and steps alongside accepted changes |
| SRE-Bench (software reverse engineering) | Source-free analysis of protected binaries | Separate system-comprehension track; not a code-patch leaderboard |

Benchmark snapshot at protocol authorship (2026-11-04):

- Scale's September 22 V2 refresh reduced the public set from 731 to 642 tasks after removing invalid tasks. Its public results include 638/642 for Opus 5; the reported private split is materially lower (222/272 for Opus 5). This public/private gap is a reason to prioritize sealed repository families and genuinely private tasks.
- Vals AI's October 29 Terminal-Bench 4.0 table lists Opus 5.5 + Mini-SWE-agent at 65.15%, Sonnet 5.5 at 64.14%, Kimi K3.1 Pro at 63.64% and GPT-6 Astra at 59.60%. These are third-party Mini-SWE-agent results, not official maintainer-board scores. Epoch AI's review still classifies Terminal-Bench 4.0.0 as flawed.
- SWE-rebench's currently selected dashboard window is May 15–July 1, 2026. Its displayed leaders include Fable 5 at 64.5% ($4.40/problem), Grok 4.5 at 63.8% ($1.47) and GPT-5.6 Sol at 62.3% ($0.85). Treat it as a pinned snapshot, not a continuously current November sample.
- CursorBench 4.0 currently displays Opus 5.5 Max at 57.8%, with $13.43, 218,363 tokens and 185 steps per task. Keep model configuration and harness distinct from SWE-bench and Terminal-Bench.
- Columbia DAPLab's SRE-Bench (October 2026 update) contains 262 binary instances and 1,572 deterministic task grades across 19 private programs. Its public-facing GPT-6 Astra result is 5.55/6 tasks per graded instance and 149/181 fully solved graded instances; only 181/262 instances were graded, so coverage/refusals must remain explicit. This measures software reverse engineering, not repair.

These snapshots are dated references. Before any public leaderboard claim, re-fetch the named source, record its retrieval date, selected task window, configuration, harness, sample count and uncertainty.

## 11. Decision rule

The programme succeeds only if the causal loop beats the strongest matched baseline on independently accepted engineering outcomes, survives sealed root-cause/repository transfer, and does not buy apparent gains by increasing regressions, unsafe experiments, false accepts or human review burden.

If simpler retrieval or static heuristics reach the same frontier, prefer them. If an experiment cannot identify the cause, report non-identifiability. If the verifier cannot establish the outcome, report unresolved. A more elaborate cognitive architecture is not itself evidence of a better engineer.
