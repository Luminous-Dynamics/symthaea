# Physical Serviceability Trial Protocol for Owner-Maintainable Aircraft

**Date:** 2026-10-17  
**Status:** research protocol and evaluator; not approved maintenance data, certification evidence, or an airworthiness claim  
**Tracking:** [AIRMAINT-002 #7771](https://github.com/Luminous-Dynamics/symthaea/issues/7771)  
**Related:** [aircraft family requirements, PR #7761](https://github.com/Luminous-Dynamics/symthaea/pull/7761), [maintenance task-card validator, PR #7750](https://github.com/Luminous-Dynamics/symthaea/pull/7750), [ownership-cost model, PR #7763](https://github.com/Luminous-Dynamics/symthaea/pull/7763)

## 1. The paradigm shift

A design is not “easy to maintain” because a CAD review says the parts look accessible. Treat maintainability as an experimentally measured product property, alongside mission performance and lifecycle cost.

The unit of evidence is an assigned task trial. Successful, failed, blocked, aborted, and not-started tasks must all remain visible. Report active labor, total elapsed time, waits, completion, first-pass success, rework, unrelated disassembly, evidence coverage, and safety-critical errors as separate dimensions. Never combine them into one weighted score that lets a faster average offset a dangerous error or missing evidence.

NASA's Human Integration Design Handbook describes maintainability in terms of design effects on complexity, duration, frequency, safety, skill, tools, access, and workload. The aircraft-family study adapts that systems-engineering principle to controlled, non-flight mock-up trials; NASA spaceflight guidance is not aircraft certification data.

## 2. Test scope and safety boundary

This protocol is for non-flight mock-ups, representative access rigs, and qualified supervised task trials. It does not provide repair steps, torque values, inspections, task authorization, or return-to-service decisions.

Before each trial, the responsible test lead must verify the selected task is suitable for the mock-up and permitted for the people assigned to it. Task-class labels in this dataset are experimental strata, not legal determinations. Any real-aircraft work must follow the applicable jurisdiction, approved/current source data, applicable maintenance-record requirements, and appropriately qualified personnel.

The evaluator checks record structure and computes descriptive metrics. It does not authenticate external credentials, read or verify a maintenance manual, prove that the procedure is correct, certify a person, approve a repair, or establish airworthiness. A boolean evidence-complete field is a recorded assessment, not an independent cryptographic proof.

## 3. Pre-register the comparison

Freeze the benchmark plan before looking at candidate results:

1. **Task set:** define the service intent, task boundary, acceptance criteria, task class, and a stable task-definition ID. Keep those IDs equal across designs. Design-specific procedure/manual revisions may differ and must be recorded separately.
2. **Cohort:** list pseudonymous maintainer IDs and relevant experience/qualification strata. Do not store names or personal identifiers in the benchmark.
3. **Repetitions:** set repetitions per task and maintainer. Select the minimum paired-completion threshold before seeing the data. That threshold is only a guardrail, not a power calculation.
4. **Conditions:** record mock-up configuration, tools, consumables/parts, environment, lighting, clothing/PPE, instructions, manual revisions, observer, and reset procedure. Use equivalent starting states.
5. **Order:** randomize or counterbalance baseline/candidate and task order to reduce learning, fatigue, and order effects. Preserve the schedule/sequence and deviations with the trial evidence. Do not claim causal improvement from an uncontrolled before/after demonstration.
6. **Coding:** predefine what counts as a completion, first-pass completion, rework, blocked/aborted attempt, unrelated assembly removal, and safety-critical error. Train observers on examples before the recorded run.
7. **Stop rules:** any safety-critical error triggers the pre-registered stop, containment, review, and corrective-action process. It cannot be waived because other trial timings look good.

The plan must schedule every listed maintainer for every listed task and repetition in each configuration. If a task cannot start because a tool, part, procedure, or qualified person is missing, record the assigned trial as not-started or blocked, with the reason in its evidence record. Do not drop it from the denominator.

## 4. Recorded trial fields

The Rust evaluator in `crates/domains/symthaea-operations-research/src/maintainability.rs` accepts typed records. Each trial binds:

- trial, design, and configuration identifiers;
- common task ID/definition ID, task class, and the design-specific procedure revision;
- pseudonymous maintainer ID and repetition index;
- outcome and declared required/completed step counts;
- active seconds and total elapsed seconds;
- mutually exclusive elapsed-time categories waiting for parts, tools, documentation, and authorization;
- rework count, unrelated assemblies removed, and safety-critical error count;
- a recorded evidence-completeness assertion.

The evaluator requires active time plus the declared wait categories not to exceed elapsed time. Remaining elapsed time can include setup, inspection/verification, pauses, or other non-categorized time; explain it in the test record rather than silently relabeling it as active work.

The plan and task IDs are comparison keys, not hashes or proof of source identity. Keep source files, observation evidence, and the actual counterbalancing schedule in controlled, revision-bound artifacts outside this module.

## 5. Metrics and denominators

For (N) assigned trials:

- **Completion rate:** (first-pass completions + completions after rework) / (N).
- **First-pass rate:** first-pass completions / (N).
- **Evidence coverage:** trials with a complete evidence record / (N).
- **Safety-critical errors:** total recorded events and number of trials with at least one such error.
- **Rework:** total rework operations and number of trials needing rework.
- **Disassembly burden:** count of unrelated assemblies removed, both total and mean per assigned trial.
- **Active/elapsed time:** report totals across all assigned trials. Report median and nearest-rank p90 for completed trials only, with the completed-trial sample count beside every distribution.
- **Wait burden:** report each recorded wait category separately to distinguish design access from supply/tool/documentation bottlenecks.

Successful-only timing is deliberately not the whole result. It must be read beside all-attempt completion/first-pass rates and the full list of outcomes. A design that is quick when it succeeds but often cannot complete the task must not win on median time.

## 6. Baseline/candidate comparison rules

The comparison function runs each design against the same plan. It fails closed if a run is missing any scheduled task/maintainer/repetition slot; has duplicate trial IDs or duplicate slots; mixes design/configuration identity; changes a task definition or class; or contains impossible timing/step counts.

For each task separately, the report pairs records on task ID, maintainer ID, and repetition. Time deltas are candidate minus baseline; negative means faster. Different tasks are not pooled into one speed score, so a difficult task cannot disappear inside easy-task averages.

A task's time-improvement claim is eligible only if all the following hold:

- neither design has any recorded safety-critical error;
- every assigned trial in both runs has complete evidence;
- candidate completion and first-pass counts are not lower than baseline for that task;
- the pre-registered minimum number of pairs completed successfully in both designs is met;
- median paired active-time and elapsed-time deltas are both below zero;
- p90 paired active-time and elapsed-time deltas are both no greater than zero.

These are conservative descriptive gates, not a hypothesis test. The evaluator does not calculate confidence intervals, infer causation, or claim statistical significance. Publish raw per-trial records (with personal information removed), denominators, per-task outcomes, distributions, uncertainty analysis, protocol deviations, and exclusion reasons. Any later inferential analysis must state its method and pre-registered decision rule.

## 7. Proposed engineering targets — hypotheses, not results

The family-level study already proposes a target that 80% of eligible routine non-structural tasks should be completed within 60 minutes including required verification, and recommends a serviceability mock-up before aircraft-level performance optimization. This protocol makes that target testable only after the precise task set and task boundaries are frozen.

A proposed next experiment:

- zero safety-critical errors for a candidate to remain eligible for a time-improvement claim;
- 100% trial-record evidence coverage for comparison eligibility;
- no regression in completion or first-pass success;
- target at least 20% lower median active time on preselected high-frequency routine tasks, while also improving or holding the paired p90;
- reduce unrelated assembly removals and total rework without simply shifting time into parts/tool/documentation waits;
- present task-level results, not just an aggregate;
- use baseline measurements and a pre-registered sample-size/precision rationale before making a general claim.

Missing these goals is useful information. Revise packaging, interfaces, task documentation, parts planning, or the target itself only through a documented engineering change. Do not quietly change task boundaries after seeing the result.

## 8. Research references

- NASA, *Human Integration Design Handbook*, section 9.3, Maintainability: [PDF](https://www.nasa.gov/wp-content/uploads/2023/03/human-integration-design-handbook-revision-1.pdf).
- NASA, *System Engineering Handbook: Appendix*, including its maintainability definition and lifecycle specialty-engineering context: [NASA reference](https://www.nasa.gov/reference/system-engineering-handbook-appendix/).
- FAA, AC 43-9D, *Maintenance Records and FAA Form 8130-3 Return to Service*, active, issued 2025-09-22: [FAA advisory circular page](https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.information/documentID/1044416).
- Current product reference examples for service access and removable components may be taken from manufacturer publications such as [Cessna Skyhawk specifications](https://cessna.txtav.com/en/piston/cessna-skyhawk) and [Van's RV-12iS features](https://www.vansaircraft.com/rv-12iS/). Manufacturer claims are reference points, not independent measurements or proof of a new design's performance.

## 9. Acceptance boundary

A successful evaluator run demonstrates only that the recorded dataset is structurally complete under the supplied plan and that the descriptive calculations follow this contract. It is not physical validation, proof of procedure correctness, maintenance authorization, regulatory compliance, certification, or airworthiness.

The module is intended to make the first maintenance mock-up experiment harder to game and easier to reproduce—not to replace qualified engineers, mechanics, human-factors specialists, or the competent authority.
