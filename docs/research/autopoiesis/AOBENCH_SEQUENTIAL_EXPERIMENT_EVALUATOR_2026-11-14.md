# AOBench — Sequential Experiment Design Evaluator

**Status:** Initial deterministic sequential evaluator committed; exact-head Rust qualification pending  
**Date:** 2026-11-14  
**Tracking:** [AOBENCH-011 #12173](https://github.com/Luminous-Dynamics/symthaea/issues/12173)  
**Implementation:** `crates/domains/symthaea-organizational-closure/src/experiment_design.rs`

## 1. What this measures

The existing finite assay evaluator asks whether a *fixed panel* distinguishes every non-equivalent pair in a frozen finite model class.

This new evaluator asks a different question:

> Can an experimenter adaptively choose the next probe, use only independently qualified outcomes to update its live hypotheses, obey intervention/cost/risk budgets, and stop when the registered target-equivalence class is identified—or correctly abstain when no useful safe probe remains?

It turns experimental design into an executable capability rather than treating it as an invisible detail of reasoning over supplied evidence.

## 2. Public scenario versus evaluator-only oracle

The agent receives:
- candidate mechanism hypotheses and positive prior weights;
- the target-equivalence class for each hypothesis;
- candidate probe IDs, cost/risk units and a deterministic response table;
- total cost, risk and trial budgets.

The evaluation harness separately receives the hidden actual-hypothesis ID. The oracle is not part of the agent-facing scenario. In a sealed benchmark, public task data and private oracle data must be materialized through separate channels and bound to a frozen manifest.

A target-equivalence class can contain multiple distinct mechanisms. The system should identify the class required by the task, not waste interventions distinguishing mechanisms that are equivalent under the declared target contract.

## 3. Sequential protocol

For every trial:
1. The agent selects a declared probe.
2. The evaluator measures expected information gain against the current active hypothesis set and prior weights.
3. The probe must fit remaining cost, intervention-risk and trial budgets.
4. Delivery, isolation and observation evidence are assessed independently.
5. The outcome can update the hypothesis set only when all three evidence gates are verified.
6. A qualified outcome is checked against the evaluator-only deterministic world. An outcome outside the frozen model class remains `Inconclusive`; the system must not force a false causal attribution.
7. The evaluator records first target-class identification, cumulative expected information gain and regret against the best safe/affordable probe available at that decision point.

The agent then either stops with a target class or abstains with a reason.

## 4. Stop and abstain rules

- **Stop:** accepted only when exactly one target-equivalence class remains and the declared class matches the evaluator-only truth.
- **Premature stop:** rejected when multiple target classes remain.
- **Correct abstention:** accepted when multiple target classes remain but no informative probe fits all remaining budgets.
- **Unnecessary abstention:** rejected when an informative probe remains within the frozen budgets.
- **Unnecessary continuation:** rejected when a target class was already identified before another probe.
- **Model mismatch:** if a qualified observation contradicts every registered response expectation, preserve `Inconclusive` and permit a model-revision/abstention path rather than inventing an explanation.

Unavailable or rejected delivery, isolation or observation evidence never updates the hypothesis set.

## 5. Metrics currently implemented

- selected-probe expected information gain (bits);
- best available information gain (bits) under current budgets;
- selected and best information gain per cost unit;
- cumulative expected information gain;
- cumulative oracle-efficiency regret;
- informative-probe count;
- target-class count before each probe;
- first probe count at which one target class remains;
- cost, risk and trial-budget use;
- qualified versus unqualified trial count;
- correct target-class stop / correct abstention status.

The oracle-efficiency baseline uses expected information gain divided by `max(cost_units, 1)`; risk is a hard feasibility constraint, not a compensable reward. A future protocol may add an independently preregistered risk-aversion utility, but must report the raw cost, risk and information quantities as well.

## 6. Deterministic pilot boundaries

This implementation uses a finite deterministic response table. It does not estimate empirical power, confidence intervals, stochastic equivalence margins, multiplicity, optional-stopping-valid p-values or clustered-trial uncertainty. It is an executable foundation for sequential task control and decision integrity, not a substitute for #12048's stochastic assay qualification.

A response outside the frozen candidate model is `Inconclusive`, signalling that the model class or environment contract may be incomplete. The module does not infer unrestricted causal structure from arbitrary physical systems.

## 7. Deterministic tests added

- Sequentially identify a hidden mechanism through two adaptive probes, then stop.
- Reject premature stopping.
- Ensure unavailable delivery evidence does not update hypotheses.
- Keep qualified observations outside the frozen response model inconclusive.
- Enforce cost and risk budgets independently.
- Accept target-class stopping without distinguishing mechanisms inside the same class.
- Accept abstention when no informative safe/affordable probe remains.
- Detect an uninformative no-op probe.
- Measure positive oracle-efficiency regret for a less efficient probe choice.

## 8. Next qualification work

1. Exact-head format, compile, tests and strict Clippy.
2. Mutation-test the scenario validator, information-gain computation, evidence gates, budget guards and stop rule under AOBENCH-009.
3. Compare random, fixed schedule, one-factor-at-a-time, greedy information gain, active orientation, PC/FCI/GES and LLM-based planners on identical scenarios.
4. Add stochastic outcomes only with preregistered power, uncertainty, multiplicity and stopping rules.
5. Use mechanism-family-disjoint held-out scenarios and sealed oracle data.
6. Couple successful mechanism discovery to #12118 structural admission, #12167 semantic continuity, #12067 intervention integrity and the qualified #6970 environment.

## Claim boundary

The initial evaluator measures sequential decisions over a finite, deterministic, declared model class. It does not establish general scientific discovery competence, stochastic causal identification, physical-world truth, or organizational safety. No benchmark result or SOTA comparison is claimed until exact-head execution and matched external baselines are completed.
