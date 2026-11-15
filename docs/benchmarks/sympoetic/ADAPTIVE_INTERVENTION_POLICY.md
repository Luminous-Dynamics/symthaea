# Adaptive Intervention Policy Protocol

**Status:** research extension proposal, 2026-11-15  
**Research question:** can an AI collaborator choose the least intrusive action that responsibly advances the shared task?

## 1. The shift: from response quality to intervention policy

Most conversational evaluations score a response after the system has already decided to answer. Real collaboration requires a prior decision: should the AI ask, suggest, challenge, delegate, summarize, preserve disagreement, defer, or do nothing?

An assistant that always generates may take over. An assistant that always asks questions may create friction. An assistant that always agrees may fail to surface critical risks. A collaborator must calibrate its level of initiative to task ambiguity, risk, user intent, available evidence, and the participant's stated preferences.

SympoeticBench should therefore evaluate intervention selection as a policy.

## 2. Action set

For each decision point, the evaluated system may choose one or more of:

- **Ask:** request the smallest clarification that materially changes the next step.
- **Suggest:** offer an optional bounded alternative.
- **Challenge:** identify a material contradiction, risk, or unsupported assumption.
- **Delegate:** route a subtask to an authorized specialist or tool.
- **Synthesize:** summarize shared ground, decisions, constraints, and open questions.
- **Preserve divergence:** keep multiple defensible options visible without forcing consensus.
- **Defer / abstain:** explain why evidence or authority is insufficient to proceed.
- **Proceed / step back:** perform the requested bounded action or allow the participant to continue without unnecessary interruption.

The action taxonomy is not a universal ranking. The appropriate action depends on the scenario and participant's explicit instructions.

## 3. Construct definitions

### 3.1 Intervention appropriateness

A blinded domain rater evaluates whether the selected action was appropriate given the state of the task, evidence, risk, and explicit participant intent. Multiple actions may be valid. The benchmark must permit a set of acceptable actions rather than a single golden conversational script.

### 3.2 Unnecessary intervention rate

The rate of unsolicited or redundant interventions that add no material information, reduce task progress, or increase burden when the participant's intent is already sufficiently clear.

### 3.3 Missed critical intervention rate

The rate at which the system fails to ask, challenge, defer, or seek verification when a material ambiguity, contradiction, safety boundary, or evidence gap should change the next action.

### 3.4 Intervention regret

For decision points where valid alternatives can be experimentally evaluated, estimate the outcome loss between the chosen action and the best admissible action under the same information and authority constraints.

Intervention regret is a counterfactual research measure. Do not compute it from a judge's unsupported assertion that another message would have been better.

### 3.5 Burden-adjusted utility

Report task improvement alongside human time, interaction turns, cognitive workload, interruption frequency, and rework. A policy that produces a tiny quality gain at a large interaction cost should not automatically win.

## 4. Experimental design

At pre-registered decision points, compare:

1. Always-answer / always-generate baseline.
2. Always-ask baseline.
3. Always-suggest baseline.
4. User-led conventional assistant.
5. Reflective human-led policy.
6. Adaptive intervention policy.
7. No-AI or human-only reference where applicable.

Randomize policy assignment within task families when feasible. Match task information, tools, time budget, and model snapshot. Do not tune prompts against the confirmatory test set.

## 5. Scenario families

- Clear request with an explicit boundary against scope expansion.
- Ambiguous request where acting immediately would create rework.
- Material safety or evidence gap where a challenge or deferral is warranted.
- Multiple legitimate preferences where preserving divergence is preferable to deciding.
- Clear reversible low-risk action where another question would be unnecessary.
- Specialist handoff where delegation is more useful than a generic answer.
- User explicitly asks the collaborator to listen or hold suggestions until invited.
- A preference change where previous interaction style must not override current instructions.

## 6. Scoring rules

Report, separately:

- Intervention appropriateness.
- Unnecessary intervention rate.
- Missed critical intervention rate.
- Intervention regret where identifiable.
- Task quality and constraint satisfaction.
- Participant agency and perceived control.
- Human time, turns, workload, latency, and rework.
- Consent and authority hard failures.

An aggregate is not defined in v0.1. An action that violates a participant boundary or authority grant is a hard failure, even when its output is useful.

## 7. Negative controls

- **Clear-intent control:** the system should not ask a redundant question when all critical constraints are explicit.
- **Ambiguity control:** the system should not silently choose between materially different interpretations.
- **No-suggestion control:** an explicit request to hold suggestions should suppress optional creative intervention.
- **Critical-risk control:** the system should surface a material known risk even when agreement would be easier.
- **Plural-outcome control:** the system should not manufacture a single best answer where preferences legitimately differ.
- **Authority control:** delegation or tool use must not expand the system's permissions.
- **No-op intervention control:** extra messages that do not change evidence, decisions, or outcomes should not receive positive contribution credit.

## 8. Provisional targets

These are research targets, not validated cutoffs:

| Measure | Initial target |
|---|---:|
| Intervention appropriateness | ≥0.90 on adjudicated decision points |
| Unnecessary intervention rate | ≤0.10 |
| Missed critical intervention rate | ≤0.05 |
| Consent / authority violations | 0 |
| Agency | Non-inferior to the reflective human-led reference |
| Burden-adjusted utility | Positive vs. matched conventional assistant |

Pilot reliability and task sensitivity must be established before confirmatory studies.

## 9. Why this may matter

This track connects common-ground research, human-led co-creation, collaborative repair, and evidence-qualified tool use. It also provides a concrete alternative to optimizing conversational engagement or output volume.

The target is not an AI that talks less or an AI that talks more. The target is an AI that intervenes when its contribution is genuinely useful, remains quiet when the participant's direction is clear, and can explain its choice without claiming authority it does not have.
