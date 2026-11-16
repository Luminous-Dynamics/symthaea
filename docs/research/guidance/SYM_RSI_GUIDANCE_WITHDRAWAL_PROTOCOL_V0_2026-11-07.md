# Symthaea methodological autonomy under guidance withdrawal

**Version:** v0 · **Date:** 2026-11-07  
**Owner:** #11006 · **Related:** #10877, #9230, #8721, #9332, #10756, #10717, #10630, #10497

## Executive position

Research taste and methodological autonomy are related but distinct:

- **Research taste:** are the selected directions, hypotheses, mechanisms, experiments and interpretations good?
- **Methodological autonomy:** can the researcher construct and validate the research method when humans progressively stop specifying it?

ASI-Bench 0.3.0 offers a matched B1–B4 guidance-withdrawal design over 60 project-level tasks in 11 scientific domains. Its four conditions progressively remove methodological instructions while holding the scientific objective, input data, required artifacts and scoring criteria fixed. This is a useful external comparator, but Symthaea must distinguish exact upstream reproduction from local extensions.

The objective is not to maximize one "autonomy score." It is to identify which stages of independent research break as methodological assistance is removed, and whether the system can recover those capabilities through transferable research methods rather than prompt-specific imitation.

## Pinned external comparator

The v0 pin is recorded in `SYM_RSI_GUIDANCE_WITHDRAWAL_MANIFEST_V0_2026-11-07.json`:

- Repository: `apexin-ai/ASI-Bench`
- Release: 0.3.0
- Commit: `eb60952124b90e9f0886c1212bf7463e26d0bc19`
- Paper: https://arxiv.org/abs/2608.17271
- License: Apache-2.0

The inspected README reports 60 project-level tasks, 11 domains and 18 evaluated Agent×Model configurations. These are project-reported benchmark descriptors, not Symthaea measurements.

### Guidance conditions

| Condition | Disclosed information | Primary capability |
|---|---|---|
| B1 | Scientific background, method, equations and full procedure | Faithful implementation and execution |
| B2 | Intended method and constraints, without the full procedure | Method-to-workflow construction |
| B3 | Objective, data, constraints and required outputs | Independent method selection and workflow construction |
| B4 | B3 plus factually correct, non-essential context | Independent research while resisting distraction |

The task objective, input data, constraints, required artifacts and scoring criteria are intended to remain fixed across the four levels. We must verify this task-by-task rather than assume it from the benchmark label.

### Leaderboard boundary

The ASI-Bench FAQ describes the currently visible leaderboard as a repo-verified/conservative snapshot and states that a final public official baseline has not yet been published. Therefore:

- pin release, commit, task revision, prompts, scorer, dataset seed and execution contract;
- report raw per-task local scores and their provenance;
- do not present the currently rendered page as a definitive official SOTA;
- separate public seed31415 local scoring from private-reference seed42 official submissions;
- do not claim official seed42 performance unless the documented submission process has actually been completed and accepted.

Sources: https://github.com/apexin-ai/ASI-Bench, https://asibench.apexin.ai/faq/, https://arxiv.org/abs/2608.17271.

## Paradigm shift: method autonomy is not exact-answer matching

At B3, a researcher may legitimately select any method that satisfies the scientific objective, data assumptions, feasibility constraints and artifact contract. A benchmark should not mark a valid alternative wrong merely because it differs from one canonical method name.

For each B3 task, independent domain adjudicators should define an **admissible method set** or a set of scientifically defensible method families, together with:

- assumptions and validity envelope;
- required inputs and observables;
- predicted intermediate checks;
- known failure modes and counterexamples;
- resource and operational constraints;
- expected artifacts and validation criteria;
- observations that would justify switching methods.

Where no finite admissible method set can be defined, adjudicate the method's assumptions and empirical support; do not force a single hidden gold answer. The machine-readable policy and a synthetic two-method example are in `SYM_RSI_METHOD_ADJUDICATION_FIXTURE_V0_2026-11-07.json`, with structural contract in `SYM_RSI_METHOD_ADJUDICATION_SCHEMA_V0_2026-11-07.json`. Its example demonstrates policy semantics only; it is not an expert reference label or measured model comparison.

Method-selection regret may be calculated only where alternatives have comparable, independently justified utility estimates. Otherwise report categorical validity, assumption satisfaction and expert disagreement separately.

## Causal isolation of the guidance effect

### Official reproduction lane

First reproduce B1, B2, B3 and B4 exactly under the upstream protocol. Hold constant, or explicitly record unavoidable differences in:

- task revision, source data and input hashes;
- scientific objective, constraints, outputs and scorer;
- model identity/version, reasoning mode and context capacity;
- agent harness, tools, memory state and retrieval access;
- token, compute, wall-clock and experiment budgets;
- retries, failed attempts, evaluator calls and human intervention.

Only the allowed guidance content should differ. If tool access or budget changes by condition, the result is not a clean estimate of guidance withdrawal.

### Symthaea-only context controls

B4 introduces non-essential but factual context. Longer context can itself increase distraction, dilution or attention cost. Add two **local, non-official** arms:

1. **B3-neutral-length:** B3 task packet plus neutral, non-semantic padding matched to the token count/placement of B4 context.
2. **B4-nonessential:** exact upstream B4 context.

Interpretation:
- B4 versus B3 estimates the total effect of adding non-essential context.
- B4 versus B3-neutral-length estimates the semantic-distraction component after approximate length matching.
- B3-neutral-length versus B3 estimates the context-length/dilution component.

These local arms must never be labelled official ASI-Bench B4 scores or pooled into its official leaderboard comparison. Preserve the exact upstream B1–B4 reproduction as a separate result lane.

## Research questions and hypotheses

1. How much capability is lost from B1→B2, B2→B3 and B3→B4 under matched conditions?
2. Can Symthaea independently choose admissible methods at B3, rather than merely execute a method named at B2?
3. Does performance degrade because of method selection, implementation, execution, evaluation, interpretation or reporting?
4. Does B4 context alter method choice, claims or resource allocation despite being non-essential?
5. Does B3-neutral-length reproduce B4 degradation, indicating context-length effects rather than semantic distraction?
6. Does B1/B2 experience improve later B3/B4 performance, and does it transfer to held-out task families without task-specific leakage?
7. Does the successor improve the guidance-retention curve and transfer under the same starting state and inclusive budget, without reducing B1 execution quality?

Pre-register primary estimands and directionality. Treat B1/B2, B3, B4 and transfer as distinct endpoints; do not search for a post-hoc weighted composite.

## Comparison arms

At minimum:

- frontier model with frozen sequential harness;
- same model with a frozen parallel/best-first harness where upstream rules permit;
- structured method-selection baseline;
- incumbent Symthaea researcher;
- successor Symthaea researcher;
- human experts, with individual pre-discussion judgments and inter-rater disagreement.

Add separate ablations for retrieval, memory/history, method portfolio generation and independent disconfirm-first review only when resources allow. Do not confound a changed model backend with a changed guidance condition.

## Outcomes: report a vector

### Execution and artifact quality
- upstream task score and gate result at every B level;
- valid artifact rate, reproducibility and execution correctness;
- missing, invalid, failed, timed-out and not-run denominators;
- cost and human intervention by stage.

### Methodological autonomy
- B3 method-choice validity, assumptions and applicability;
- admissible-method-set coverage, where definable;
- method-switch correctness given counterevidence;
- method-selection regret only when a defensible utility reference exists;
- workflow completeness and correctness without a prescribed procedure.

### Scientific integrity
- claim-to-artifact/evidence consistency;
- unsupported-work and fabricated-execution claim rate;
- numerical consistency between prose, raw outputs and scorer results;
- appropriate states for supported, not performed, not measured, inconclusive and invalid claims.

### Context robustness
- paired B3→B4 score and method-choice changes;
- paired B3→B3-neutral-length change;
- paired B3-neutral-length→B4 semantic-distraction change;
- context audit: which added facts were used, ignored or misinterpreted, and whether they legitimately changed the decision.

### Cost and uncertainty
- model/API inference, tokens, CPU/GPU, energy where measurable, wall-clock, evaluator, retries and human review;
- per-task and per-domain results;
- paired and cluster-aware confidence intervals;
- worst-family regressions and non-evaluable rates.

Never allow strong writing or one high score to compensate for a critical integrity failure or failed required artifact.

## Derived summaries — secondary only

**Guidance-retention curve:** plot each task's normalized B1/B2/B3/B4 outcome, alongside raw scores. If an aggregate area-under-curve is reported, label it descriptive and publish the complete vector and normalization rule.

**B4 distraction penalty:** paired difference between B3 and B4. Report both raw and normalized forms.

**Semantic distraction effect:** paired B4 minus B3-neutral-length. This is a Symthaea-only estimate, not an official ASI-Bench metric.

**Method-selection regret:** use only when multiple admissible methods have independently comparable expected utility. It cannot be inferred from exact-method agreement.

## Transfer and curriculum

Keep transfer orthogonal to B1–B4:

- G0: same task after refinement;
- G1: unseen instance, same task family;
- G2: harder instance in same family;
- G3: different task within the same capability;
- G4: different capability;
- G5: different scientific domain;
- G6: different methodological family;
- G7: independent implementation or replication by another team.

Test any B1/B2-to-B3/B4 curriculum on fresh task families. Compare full-evidence history, accepted-only history, shuffled history and no-history controls where justified. A prompt that memorizes methods is not a transferable methodology.

## Pilot and confirmatory discipline

1. Select a small, stratified public seed31415 pilot across domains, methods and B1–B4 levels.
2. Verify exact task pairing, prompt differences, data hashes, artifact contracts and scorer identity before interpreting performance.
3. Use a debugging pilot (for example five paired seeds where a task is stochastic) to estimate variance and expose protocol defects; do not claim superiority from it.
4. Set confirmatory sample size using a preregistered precision/power analysis and task-family clustering.
5. Freeze exclusions, stopping, model versions, resource envelopes, human-review rubric and analysis plan before outcome inspection.
6. Preserve every attempt, including failed, invalid, incomplete and not-run.
7. Keep sealed confirmation and fresh-family transfer out of policy selection.
8. Treat human experts as a distribution with inter-rater disagreement, not an infallible single oracle.

## Composition and authority boundaries

- #10877 owns research taste and decision quality.
- #9230 owns direction allocation and diversity.
- #8721 owns prospective tool/computation allocation.
- #9332 owns generated task environments, construct-validity and sealed-oracle design.
- #10756 owns evaluator quality/integrity.
- #10717 owns causal attribution.
- #10630 owns research-state continuity.
- #10497 owns R4 successor/improver qualification.

Do not create another evaluator, receipt service, scheduler, memory store, signer or promotion authority.

## Acceptance gates

- Exact B1–B4 external reproduction contract is pinned and task-level condition differences are audited.
- Same task objective, data, constraints, artifact requirements and scoring are demonstrated across conditions.
- Methodological autonomy at B3 is adjudicated with assumption/applicability evidence and legitimate alternative methods.
- B4 distraction is evaluated alongside B3-neutral-length in a separate Symthaea-only analysis.
- Claim/evidence integrity is a non-compensating hard gate.
- Full raw per-task outcomes, failures, costs, human intervention and uncertainty are retained.
- Fresh task-family transfer is positive before broad autonomy claims.
- Any successor claim is separately governed by #10497; guidance autonomy alone is not R4.

## Explicit limits

The manifest and protocol pin an external comparator; they do not reproduce it. Public seed31415 local scores are not private-reference seed42 official submissions. No leaderboard standing, human-level research, scientific discovery or R4 result is claimed until the relevant exact-head, independently adjudicated and held-out experiments are run.
