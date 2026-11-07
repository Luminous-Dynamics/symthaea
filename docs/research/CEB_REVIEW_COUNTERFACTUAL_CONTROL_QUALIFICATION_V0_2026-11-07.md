# CEB Review Counterfactual-Control Qualification V0 — 2026-11-07

**Status:** Public methodological protocol and annotator worksheet. No task is admitted, no source control is qualified by this document, no benchmark result is claimed, and no candidate code is executed.  
**Owner:** Existing CEB-REVIEW / CEB-CAUSAL / CEB-EVAL; [#10723](https://github.com/Luminous-Dynamics/symthaea/issues/10723), [#10367](https://github.com/Luminous-Dynamics/symthaea/pull/10367), [#10419](https://github.com/Luminous-Dynamics/symthaea/issues/10419).  
**Composes with:** [Causal-diagnosis preregistration](CEB_REVIEW_CAUSAL_DIAGNOSIS_PREREGISTRATION_V0_2026-11-07.md), [gold-completeness protocol](CEB_REVIEW_GOLD_COMPLETENESS_AND_OPEN_SET_DISCOVERY_V0_2026-11-07.md), [sequential evidence-acquisition policy](CEB_REVIEW_SEQUENTIAL_EVIDENCE_ACQUISITION_V0_2026-11-07.md), [finding-adjudication card](CEB_REVIEW_FINDING_ADJUDICATION_CARD_V0_2026-11-07.md).  
**Unchanged primary CEB-REVIEW measures:** `defect_recall_at_5`, `actionable_precision`, `reviewer_burden`.  
**Execution/regrade boundaries:** #10064 / #10369; task admission/custody remains under #10607/#10609.

## 1. The unresolved validity problem

The P/I/F/N/M/IRR design is useful only when its variants actually isolate the claimed causal mechanism.

A historical parent commit, bug-introducing commit and later fix are not automatically a controlled counterfactual experiment:

- A bug-fix commit may bundle the actual repair with refactoring, compatibility changes, test updates, documentation, dependency changes or unrelated fixes.
- A parent revision can differ in compiler, dependencies, configuration, caller set or environmental assumptions beyond the alleged introduction.
- A fix may remove one symptom or one caller path while leaving the same causal mechanism reachable elsewhere.
- A nearby mutation can be textually small but semantically alter multiple causal conditions.
- An unchanged or reverted line is not proof that an intervention is causally irrelevant.
- A test that passes on a repaired revision does not establish that the intended mechanism alone was removed.
- A source-only prediction cannot be promoted into a runtime-observed effect.

Empirical software-engineering evidence makes this a first-order risk. Herbold et al.'s large fine-grained study reports that 17–32% of all changed lines in sampled bug-fixing commits directly modify source to fix the underlying problem (66–87% when restricted to production-code files). Approximately 11% of lines were difficult to label; depending on the use case, estimated noise without manual untangling ranged from 3–47%. These estimates are specific to their data and labeling procedure, not universal constants, but they demonstrate why commit-level provenance is insufficient. [Herbold et al., *A fine-grained data set and analysis of tangling in bug fixing commits*](https://link.springer.com/article/10.1007/s10664-021-10083-5).

Schulte et al.'s 2026 exploratory study of bug-introducing changes documents relations among process variables and explicitly cautions that its observational relations are not causal estimates; confounding must be controlled before causal conclusions are made. [Schulte et al., *An exploratory study of bug-introducing changes*](https://link.springer.com/article/10.1007/s10664-026-10822-6).

Ma and Eisty's June 2026 rapid review analyzes 27 applications of causal reasoning in software testing. It finds research concentrated on identification and effect estimation, with causal representation and discovery underexplored; model misspecification, untested assumptions and limited empirical validation are recurring cross-layer problems. [Ma & Eisty, *Applications of Causality in Software Testing: A Rapid Review*](https://arxiv.org/abs/2606.15683).

**Design rule:** a Git revision is a source-state observation. It becomes a usable control for a particular claim only after its causal relevance, scope, confounders and evidence limitations have been independently reviewed.

## 2. Paradigm shift: qualify the control, not the commit

Replace the assumption:

> “The parent is clean; the fix is fixed; therefore the two revisions prove the defect cause.”

with:

> “For this exact claim, these source states provide this bounded level of evidence, under these assumptions and unresolved alternatives.”

A control is not globally “valid” or “invalid.” Historical P/I/F source-state observations are not randomized assignments; their qualification is **claim-specific**. A fixed companion may establish that a target branch was removed while remaining insufficient to establish whole-program safety. A parent may show that a line was introduced without proving that its behavior was reachable in the deployment configuration. A source-level intervention can support a structural claim without establishing runtime behavior.

Keep the existing evidence authorities:
- `source_declared`
- `syntax_derived`
- `compiler_resolved`
- `runtime_observed`
- `intervention_observed`
- `inferred / unresolved`

The qualification dimensions below are review annotations and stop/go conditions. They are not a new authority class, metric lane, receipt schema, task schema or registry.

## 3. Six independent control-qualification dimensions

Every P/I/F/N/M/IRR control receives a separate disposition for each relevant dimension. Do not average the dimensions into one confidence score.

| Dimension | Question | Minimum evidence |
|---|---|---|
| Q1 — Identity and lineage | Are the exact source states and their ancestry known? | Commit/tree/blob IDs, parent edge, diff identity, repository identity, submodule and generated-source scope |
| Q2 — Change disentanglement | Which changes implement the target mechanism, tests, specification updates, refactors, environment changes and unrelated work? | Independent change-contribution labels, changed-path inventory, disputed-line record and rationale |
| Q3 — Causal mechanism and reachability | Is the alleged trigger/path/operation/invariant actually the mechanism in the declared configuration? | Caller/data/control path, entry conditions, feature/target/configuration, alternate paths, relevant invariant |
| Q4 — Intervention fidelity | Did the variant change the intended causal factor without silently changing other causal parents or protected invariants? | Explicit treatment, intended causal edge/condition, affected symbols, confounder inventory, negative/irrelevant intervention rationale |
| Q5 — Environment and applicability | Are source, dependencies, toolchain, feature flags, target, inputs and runtime assumptions comparable, or are differences explicitly modeled? | Environment/lock/configuration digests and scope statement, or explicit unresolved differences |
| Q6 — Outcome and oracle validity | Does the available evidence establish the predicted outcome at the claimed authority? | Frozen requirement, target-specific evidence, test/PoC semantics if authorized, positive and negative controls, result binding and adjudication |

For each dimension use one of:
- **supported** — required evidence is present and independently reviewed for the stated scope;
- **limited** — some evidence exists, but the claim must be narrowed;
- **unresolved** — evidence is absent, conflicting or insufficient;
- **not applicable** — justified by a recorded reason.

A missing dimension is never silently treated as supported. The final control disposition is the conjunction of the dimensions required for the claim, not a mean or majority vote.

## 4. Required control records by state

### P — Parent / pre-introduction

P is a historical source state, not a randomized assignment.

Required questions:
1. Is the claimed introducing change an actual ancestor/descendant transition from P to I?
2. Does the target mechanism, including its required precondition and path, appear in P?
3. Did a dependency, feature flag, caller, configuration, generated file, platform target or external contract change between P and I?
4. Is the behavior being compared under a common applicability scope?
5. Could the target mechanism already exist through another path in P?

Acceptable conclusion: “The identified source mechanism is absent in the reviewed P tree under this static scope.”  
Not acceptable without further evidence: “P is behaviorally safe” or “the introducing commit alone caused the production incident.”

### I — Introducing revision

I is the historical revision where the target is alleged to become live.

Required questions:
1. Which exact changed statements or semantic edges create the trigger/path/operation?
2. Does the change introduce one defect or several independent defects?
3. Is the alleged behavior reachable under the frozen task's input/configuration assumptions?
4. Does the source contract establish expected behavior independently of the defective implementation/comment?
5. Are the tests and requirements consistent, or does the test encode the same mistaken assumption as the implementation?

The target mechanism must be described independently of the candidate finding. A candidate's explanation is not its own oracle.

### F — Fixed companion

F is a historical repair state and is **not presumed to be a minimal intervention**.

Required questions:
1. Does the fix remove the target's actual causal condition or only the observed symptom?
2. Did the fix change multiple callers, APIs, feature flags, dependencies or state transitions?
3. Does a residual route still reach the same unsafe operation?
4. Did the fix add a guard, change an invariant, change the caller, change data representation, or merely catch the resulting error?
5. Are newly added tests specific to the target contract or overfit to the repair's exact shape?
6. Does the patch contain unrelated behavior changes that can alter the diagnosis?
7. Was the fixing commit itself later reverted, superseded or partially cherry-picked?

Required dispositions:
- target mechanism removed for the declared path/scope;
- target mechanism partially removed; residual path remains;
- symptom masked but causal mechanism remains;
- target status unresolved due to tangled change or incomplete tree.

A commit message, merged status, test-green badge or maintainer approval is not enough to choose the first disposition.

### N — Neighboring-mechanism control

N must test a plausible alternative explanation, not merely look similar.

Required questions:
1. Does N share the same file/function/symptom while differing in the proposed causal mechanism?
2. Is the target-specific precondition absent or blocked?
3. Is there a different valid or invalid mechanism that could mislead a reviewer?
4. Can an adjudicator explain why the candidate target diagnosis should not transfer to N?
5. Is the case genuinely distinct, or merely a duplicate symptom with a different line number?

N can be a positive example of a different defect, a valid no-defect example, or a boundary/unresolved case. Its role must be stated.

### M — Minimal causal intervention

“Minimal” means minimal in the **causal model**, not necessarily fewest lines changed.

Required questions:
1. Which one causal parent, precondition or transition is being intervened on?
2. Which non-target variables and protected invariants should remain unchanged?
3. Could this intervention introduce an alternative cause or suppress the symptom downstream?
4. Does the intervention remove every in-scope path to the target operation?
5. Is the intervention semantically equivalent to the baseline despite textual changes?
6. Which outcome distinguishes target removal from simple symptom suppression?

Before qualified execution is possible, M can be reviewed as a source-level structural intervention only. It cannot establish a runtime effect. After #10064, the execution protocol may validate the target-specific outcome with a pinned environment. An equivalent mutant or a non-discriminating intervention is unresolved, not a successful causal control.

### IRR — Irrelevant intervention

“Irrelevant” is relative to the target causal claim, not globally irrelevant to the program.

Required questions:
1. Which target causal parents, path and operation should remain invariant?
2. Why is the changed source outside the target's causal cone?
3. Could the change indirectly alter scheduling, ordering, resource use, feature selection, error handling or concurrency?
4. Is the expected persistence of the target mechanism justified by source evidence or behavior evidence?
5. Is the chosen change a genuine sham/control rather than a hidden partial repair?

An edit being far away, stylistic, or in a different file does not prove irrelevance.

## 5. Change disentanglement and historical-fix audit

For every historical introducing/fixing change used to support causal diagnosis, an independent reviewer classifies each materially changed hunk (or smaller semantic unit where line-level labels are ambiguous):

- **TARGET_CAUSAL:** necessary to introduce/remove the nominated target mechanism;
- **TARGET_VALIDATION:** test, fixture, assertion or oracle change specifically validating the target requirement;
- **CONTRACT_CONTEXT:** requirement, API, documentation or configuration that clarifies the target contract;
- **ENABLING_OR_FOLLOW_ON:** necessary dependency, caller, migration or supporting change, but not the target's direct causal operation;
- **UNRELATED:** unrelated refactor, formatting, documentation, dependency or other behavior;
- **DISPUTED:** reviewers cannot determine the contribution confidently.

Do not infer categories from commit messages or file names alone. Preserve line/hunk identities, evidence and disagreements.

Minimum process:
1. Two reviewers independently classify changed hunks without seeing the model's diagnosis.
2. Record agreement and disagreements. Do not collapse disputed changes into the target cause.
3. A third reviewer resolves only when evidence is sufficient; otherwise retain `DISPUTED`.
4. Keep the original change and any untangled reconstruction linked by exact source/tree IDs.
5. If the target mechanism cannot be isolated from unrelated effects, use the historical state only as contextual evidence or exclude it from the causal-control role.

A high percentage of target-related lines does not automatically make the patch causal-isolated; a small number of non-target lines can alter behavior substantially. Line counts are descriptive diagnostics, not causal weights.


### 5.1 Historical states, semantic minimality and claim ceiling

Historical parent/introducing/fix revisions are **historical source-state observations**, not randomized treatments or interventions. A commit that happens to remove the observed target is not automatically a causal-isolated control.

Minimality is judged **minimal in the causal model**: no proper subset of the changed causal conditions achieves the same qualified effect under the declared scope. A one-line edit is not necessarily minimal, and a multi-line semantic change can be.

When only source text and lineage have been qualified, report **source-only structural claims** (for example, “this path is reachable under condition X”). Do not turn them into runtime outcome, incident attribution or intervention-effect claims without the corresponding evidence and authority.


## 6. Causal identification assumptions and confounder register

Before interpreting P/I/F/N/M/IRR as a causal comparison, explicitly state:

- **Treatment:** the exact source/configuration change intended to alter the target mechanism.
- **Outcome:** the observable target behavior and the test/oracle contract.
- **Causal path:** trigger → path/state → operation → violated invariant → consequence.
- **Applicability:** input, user/tenant, platform, feature, target and environment conditions.
- **Confounders:** concurrent source changes, dependencies, configuration, caller population, timing/order, data, environment and external services that differ between states.
- **Mediators:** intermediate checks/state changes that the intervention is meant to affect.
- **Alternative paths:** other callers or mechanisms that can produce the same symptom or reach the same unsafe operation.
- **Assumptions:** consistency of intervention, no unmodeled interference between paired variants, adequate control of confounding, and correct mechanism representation.

Do not treat a static source DAG as proof that all causal parents have been found. State what is known, inferred and unknown. If a critical confounder or alternate path cannot be excluded, narrow the estimand or leave the comparison unresolved.

Schulte et al. (2026) emphasize that observed process relations may be confounded and explicitly do not claim their observational study establishes causal effects. The CEB protocol must preserve the same distinction. [Paper](https://link.springer.com/article/10.1007/s10664-026-10822-6).

## 7. Mutation testing is useful, but not a substitute for control validity

A mutation can provide a precise counterexample only if the mutated program is semantically non-equivalent and the chosen observation can detect the target behavior.

Required mutation dispositions:
- **non-equivalent, target-removing** — changes the claimed causal mechanism;
- **non-equivalent, target-preserving** — changes code but leaves target mechanism live;
- **equivalent or likely equivalent** — no relevant behavior change established;
- **collateral** — changes target and other mechanisms/invariants;
- **unresolved** — semantic effect cannot be established.

A surviving mutant can mean an inadequate test, an unreachable mutant, an equivalent mutant or a mismatch between test semantics and the target contract. Do not automatically interpret survival as a test deficiency. A killed mutant may be killed by an unrelated assertion or setup failure; inspect the failure cause.

The 2026 review of predictive mutation testing identifies threats to validity, class imbalance, project heterogeneity and reproducibility. CEB should therefore report the mutant/operator population and class-wise outcomes; never present a mutation score as a causal-validity score. [Delgado-Pérez et al., *Methodological pitfalls in predictive mutation testing*](https://link.springer.com/article/10.1007/s10515-026-00626-9).

## 8. Environment and outcome equivalence

A causal comparison must pin or explicitly model:
- repository, commit, tree, submodules and generated sources;
- dependency lockfiles and external dependency identities;
- compiler/runtime/toolchain versions;
- feature flags, target triples, build profiles and platform;
- input fixtures, data and external-service assumptions;
- test/oracle version and command semantics;
- environmental state, nondeterministic seeds and relevant clocks/order.

A historical P/I/F comparison may legitimately have different environments. In that case, state the change and whether it threatens identification. Do not silently rebuild old source on a new dependency graph and call it a historical outcome.

When execution is not qualified, use static source evidence and record that behavioral equivalence is unresolved. Do not run candidate-controlled code, build scripts, macros, tests, PoCs or graders before #10064.

## 9. Qualification disposition and stop/go rules

This is a claim-support decision, not a numeric tier or score:

| Intended claim | Minimum conditions |
|---|---|
| Source lineage statement | Q1 supported |
| Change-contribution statement | Q1 and Q2 supported; disputed hunks retained |
| Static causal-structure statement | Q1–Q4 sufficiently supported, scope limited to inspected source |
| Runtime target-behavior statement | Q1–Q6 supported with authorized, exact-bound execution after #10064 |
| Independent causal-control result | Above plus independent pristine regrade/custody after #10369 and the applicable admission/oracle rules |

Fail closed:
- **Use as a contextual historical example** if lineage is verified but causal isolation is not.
- **Use as a static structural control** only when the causal path and intervention are independently reviewed, while explicitly withholding runtime claims.
- **Use as a behavioral counterfactual** only after an authorized and reproducible outcome establishes the target-specific effect under a pinned environment.
- **Exclude or mark unresolved** if target identity, patch disentanglement, residual paths, environment or outcome cannot be adjudicated.

Control variants from one original defect/PR are paired siblings, not independent statistical samples. Cluster by originating defect and repository. Do not count P/I/F/N/M/IRR as six independent bugs.

## 10. Human annotation card for a counterfactual control

- Control ID / source-state label:
- Original target defect / source issue:
- Repository identity:
- Commit/tree/blob IDs:
- Exact source/configuration scope:
- Intended causal intervention:
- Target trigger and mechanism:
- Target path and alternate paths:
- Changed hunk inventory:
- Independent reviewer 1 classifications:
- Independent reviewer 2 classifications:
- Disputed hunks and resolution:
- Confounders and mediators:
- Protected invariants:
- Environment differences:
- Positive and negative oracle controls:
- Strongest evidence authority actually established:
- Q1 identity/lineage: supported / limited / unresolved / N/A
- Q2 disentanglement: supported / limited / unresolved / N/A
- Q3 mechanism/reachability: supported / limited / unresolved / N/A
- Q4 intervention fidelity: supported / limited / unresolved / N/A
- Q5 environment/applicability: supported / limited / unresolved / N/A
- Q6 outcome/oracle: supported / limited / unresolved / N/A
- Claim that this control may support:
- Claims this control must not support:
- Final disposition: contextual / static structural / behavioral (gated) / unresolved / exclude
- Reviewer disagreement and rationale:

Do not include candidate-model conclusions in the human gold/control record before the source and oracle review is frozen.

## 11. Integration and measurement boundaries

- The six P/I/F/N/M/IRR states remain in the existing causal-diagnosis preregistration.
- The evidence-action crosswalk continues to specify the smallest discriminating action; this document qualifies the control used to interpret that action.
- The gold-completeness protocol continues to distinguish frozen targets, independently adjudicated open-set findings and the unobserved remainder.
- The sequential evidence policy continues to select and stop evidence actions under a budget.
- The causal integration architecture's causal-ablation identity and evidence-authority rules remain authoritative.
- The existing CEB integrity catalog, task schema, run receipt, measurement matrix and task-admission registry are unchanged.

No new primary metric, task schema, receipt, grader, oracle authority, runner, mutation engine or hidden-test store is introduced. The Q1–Q6 labels are checklist dimensions only, not a new authority taxonomy.

## 12. Runner-independent work sequence

1. Apply the hunk-contribution labels to the three public synthetic calibration cases and one inert historical-diff vignette; keep them public development only.
2. Have two independent reviewers annotate the same control and preserve disagreements.
3. Run a source-only confounder/path audit for each P/I/F/N/M/IRR variant.
4. Classify all interventions as target-removing, target-preserving, collateral, equivalent/likely-equivalent or unresolved.
5. Update the experiment's task eligibility table before any benchmark task admission.
6. After #10064 and #10369 are separately qualified, decide which controls warrant behavioral validation and independent regrade.

## 13. Claim boundary

This protocol defines how to qualify counterfactual controls. It does not qualify any current CEB control, assert that historical fixes are causal experiments, establish the accuracy of the 26 scenarios or three calibration cases, or report benchmark performance. All examples remain public development material and permanently ineligible as sealed holdout. No candidate-controlled code or benchmark task is executed.
