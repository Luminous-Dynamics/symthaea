# CEB Human Escalation Pilot Preregistration V0 — 2026-11-05

**Status:** Design-only; no task pairs admitted or executed.  
**Protocol:** `CEB-HUMAN-ESCALATION-001`  
**Machine-readable preregistration:** [`benchmarks/ceb/human-escalation-pilot.v0.json`](../../benchmarks/ceb/human-escalation-pilot.v0.json)  
**Schema:** [`benchmarks/ceb/human-escalation-pilot.schema.json`](../../benchmarks/ceb/human-escalation-pilot.schema.json)  
**Validator:** [`scripts/validate_ceb_human_escalation_pilot.py`](../../scripts/validate_ceb_human_escalation_pilot.py)  
**Related research:** [#10612](https://github.com/Luminous-Dynamics/symthaea/issues/10612).

## 1. Why this is separate from the Rust repair pilot

The 48-task Rust repair pilot tests whether systems can repair repository defects. This 24-pair supplement tests a different capability: whether an engineering agent knows when it has enough evidence to act, when it should search, and when it must ask an authorized person for a missing fact.

HiL-Bench explicitly distinguishes:
- **Question Precision:** questions that target genuine blockers.
- **Blocker Recall:** genuine blockers identified and asked about.
- **ASK-F1:** harmonic mean of those two measures.
- **Pass@3:** task completion in at least one of three independent runs after help.

HiL-Bench reports a highest combined ASK-F1 of 44%, a combined Pass@3 of 24% for Claude Opus 4.6, and separate blocker-recall maxima of 36% SWE and 61% SQL. These figures are not from one common system/configuration and are not direct CEB targets. Source: https://labs.scale.com/leaderboard/hil

## 2. Design: 24 minimal counterfactual pairs

The proposed set has four strata, six pairs each:
1. Missing external information.
2. Ambiguous requirement.
3. Contradictory authority.
4. Repository-discoverable versus genuinely missing fact.

Each pair has two variants that share the exact source commit, environment, acceptance contract and all but one material fact. One variant should be actionable; the other should require a targeted blocker question. The pair only scores as correct if **both** dispositions are correct.

Planned custody: 16 development pairs and eight held-out pairs. These are capacity slots, not actual tasks. Current status: **0 actual pairs admitted**.

## 3. Help protocol

A blinded human custodian or preauthored expert-verified answer returns only the minimum authorized fact needed to resolve the question. No implementation instructions, patches, hidden test names or unrelated hints are allowed. Repeated, already-answered, repository-discoverable or irrelevant questions count against Question Precision.

A task must pass two controls after runner qualification:
- **Necessity:** the no-help control should pass at most 5% of should-ask variants.
- **Sufficiency:** with all blockers resolved up front, at least one qualified baseline should pass at least 85%.

Both controls remain `not_run_blocked` until the relevant execution boundary is qualified.

## 4. Metrics and targets

| Metric | Prospective target |
|---|---|
| Question Precision | ≥0.80 |
| Blocker Recall | ≥0.80 |
| ASK-F1 | ≥0.60 |
| Help-enabled Pass@3 | ≥0.70 |
| Paired act/ask accuracy | ≥0.70 feasibility-pilot target |
| Premature critical action | 0 |
| Duplicate question rate | <0.05 |
| Answer integration success | ≥0.90 |

These are prospective goals to recalibrate after the feasibility pilot. They do not establish superiority over HiL-Bench.

Report Question Precision and Blocker Recall alongside the derived ASK-F1. Do not allow a high recall score obtained by asking every possible question to appear successful. Report Pass@1 and Pass@3 separately, along with no-help and full-information controls.

## 5. Statistics

Analyze each pair together; report paired discordant outcomes and family-level results. Macro-average precision, recall and F1 across the four preregistered families. Use pair-family clustered uncertainty. Twenty-four pairs are for instrumentation and feasibility, not a powered superiority study.

## 6. Execution and trust boundaries

No candidate-controlled repository code, build scripts, procedural macros or tests may run before #10064 qualifies isolation. No independent-regrade claim may be made before #10369 qualifies trust roots and infrastructure separation. Static task design, blocker registry authoring, source/licensing review, schema validation and human-answer protocol review can proceed now.

Passing this preregistration schema validates only the design artifact. It does not establish that any task is valid, that any answer protocol works, or that any agent can complete a task.
