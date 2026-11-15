# EGDDM ShiftBench D6.4 — benchmark task admission, blind reconstruction and custody

**Version:** 0.1  
**Date:** 2026-11-15  
**Status:** task-admission contract, Rust structural checker, schema and one synthetic fixture authored. No real benchmark tasks have been admitted, no reference patches have been run, and no blind adjudication or agent trial has occurred.

## 1. Why task admission is part of the benchmark

An engineering benchmark is only as valid as its task, oracle, isolation and custody. A model score is uninterpretable when the reference patch is wrong, the empty patch passes, the acceptance tests contradict the request, the candidate can discover the gold patch, or the grader evaluates a modified workspace rather than the submitted artifact.

This is not an EGDDM-only problem. SWE-Bench Pro V2 reports that 89 of 731 public tasks were removed after audit, including instruction/test contradictions; corrected instructions were blind-solved by expert engineers, the agent network was restricted, and every patch was regraded in a pristine image. SWE-Gate separately demonstrates that functional test success can fail to satisfy review constraints. D6.4 adopts these lessons as explicit machine-checkable admission gates rather than informal benchmark curation.

References:
- [SWE-Bench Pro V2 paper](https://arxiv.org/abs/2610.24206) and [benchmark protocol](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full)
- [SWE-Gate](https://arxiv.org/abs/2609.04167)
- [SWE-Shift](https://arxiv.org/abs/2610.32196)
- [ChainSWE, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.371/)

## 2. Paradigm shift: benchmark the state transition, not only the patch

Traditional issue benchmarks ask whether a patch resolves one static issue. EGDDM models a task as a counterfactual state-transition experiment:

1. Freeze an initial repository, configuration world, requirements and evidence ledger.
2. Save the same starting state for a clean continuation and a shifted continuation.
3. Introduce one declared intervention at a preregistered work position.
4. Keep affected-obligation gold labels and acceptance oracles outside the candidate workspace.
5. Observe the agent's impact prediction, evidence reuse/invalidation, patch, revalidation choices and recovery.
6. Regrade the exact submitted patch in a pristine, independently controlled environment.
7. Score the full trajectory, including time spent with critical obligations unverified—not just the last checkpoint.

This creates three separate questions:
- **Task validity:** is there a real, solvable requirement with a discriminating oracle?
- **Agent capability:** can the agent preserve valid work, detect the shift, adapt and recover?
- **Evidence validity:** are the claims about the patch and its execution current, correctly scoped and independently supported?

Do not collapse these into a single "verified" score.

## 3. Admission contract

A task may be admitted only when all gates below hold.

### 3.1 Two-sided oracle test

The reference patch must pass the acceptance contract. The empty/no-op patch must fail. Both controls must be evaluated against the same pinned acceptance contract and grader-environment digest, with clean-room regrading enabled.

A task is inadmissible when:
- the reference patch fails;
- the empty patch passes;
- the two controls use different grader environments;
- one result is produced from a mutable developer workspace;
- the acceptance contract changes between the two control runs; or
- the exact patch and grader evidence cannot be pinned.

This gate tests whether the task is actually discriminating. It does not, by itself, prove that the task is useful, non-ambiguous, representative or difficult.

### 3.2 Blind independent reconstruction

At least two adjudicators who did not author the task must independently reconstruct a solution from the task instructions, without seeing the reference patch or hidden impact labels. Their identities must be unique and pseudonymous in public artifacts. Disagreements must be resolved and retained in the adjudication ledger; a disagreement must not simply be erased.

A successful blind reconstruction is a task-validity signal, not a guarantee that every reasonable solution is captured by the hidden tests. Publish exclusion reasons and oracle limitations.

### 3.3 Candidate/evaluator separation

The candidate receives only the task bundle and admissible history. It must not be able to read:
- reference patches;
- gold affected-obligation labels;
- hidden tests or grader credentials;
- gold-label manifests;
- post-cutoff evidence; or
- other trial outputs that reveal the answer.

Candidate and evaluator workspaces must be separated. The candidate execution environment should permit only the declared model endpoint and explicitly approved dependencies. Regrade the submitted patch in a fresh pristine image; never trust an in-place pass alone.

### 3.4 Gold-label custody

Keep the gold impact manifest outside the candidate bundle. Pin it by digest and record a distinct custodian. Record evaluator access and candidate denial separately. The admission checker validates declarations only; custody must later be demonstrated with access logs, immutable artifacts and independent review.

## 4. Intervention families and positions

D6 uses twelve intervention families:

1. Requirement addition or reinterpretation.
2. Requirement removal or narrowing.
3. Dependency/API breaking change.
4. Toolchain or build-system upgrade.
5. Configuration-world drift.
6. Runtime or external-service change.
7. Session, tool or execution-context change.
8. Authority or policy revocation.
9. Evidence invalidation or contradiction.
10. Test or acceptance-contract revision.
11. Common-cause environment change.
12. Cross-repository contract or schema change.

The exact intervention position is assigned by planned work completed, not wall-clock elapsed time: p10, p20, p30, p40, p50, p60, p70, p80 or p90. Store both the exact percentile and its planned-work denominator in every trial record. The manifest's `position_percentile` is the exact position; `position` is only its broad reporting aggregate: early (10–30%), middle (40–60%) or late (70–90%). The admission checker rejects percentiles outside the nine-point schedule and mismatches between the exact position and aggregate. A nominal "late" intervention is not late if the agent has done little work.

## 5. Controls and causal identification

Every intervention family should be represented by target, sham and common-cause controls where applicable.

- **Target intervention:** the declared target change occurs and has a preregistered impact set.
- **Sham intervention:** the event is operationally similar but should not invalidate the target obligations. This detects agents that overreact to any change notification.
- **Positive control:** a deliberately affected obligation must be invalidated and rechecked.
- **Negative control:** an explicitly unaffected obligation should retain reusable evidence.
- **Common-cause control:** a shared dependency, service, toolchain or policy change affects multiple apparently separate checks. This detects false independence assumptions.

Positive and negative controls must be balanced and hidden from the candidate as labels. A task that does not expose a measurable difference between the intervention and control condition cannot support the corresponding causal claim.

## 6. Rust admission checker

The branch adds:
- \`tools/development-map-validator/src/shiftbench_admission.rs\`
- CLI command \`development-map-validator shiftbench-admission <manifest.json>\`
- JSON Schema \`egddm-shiftbench-task-admission-v0.1.schema.json\`
- a synthetic fixture \`fixtures/egddm-shiftbench-task-admission-v0.1.synthetic.json\`

The checker fails closed on missing or malformed task/snapshot digests, unsupported intervention families or positions/percentiles, phase/percentile mismatches, reference-patch failure, empty-patch success, non-matching grader environments, incomplete blind reconstruction, author/adjudicator overlap, gold-label exposure, shared candidate/evaluator environment, unrestricted network, or missing pristine regrade/custody declarations.

The sample fixture is intentionally marked \`fixture_kind: synthetic\` and contains placeholder digests. It demonstrates the expected contract; it is not a real or admitted benchmark task.

The checker does not fetch or hash patch files, execute the two controls, validate hidden tests, inspect network policy, authenticate adjudicators/custodians, or prove workspace separation. Its output is a structural admission result only. The evidence for each gate must be independently collected before a production task is admitted.

## 7. Preregistered scorecard and benchmark comparisons

Do not optimize only for terminal pass rate. Report a vector:

| Dimension | Primary measurement |
|---|---|
| Task validity | admitted / attempted tasks; exclusion reasons |
| Shift recovery | shifted pass@1; paired clean/shifted success |
| Retained work | valid functionality preserved after intervention |
| Evidence freshness | stale receipt reuse; missing current evidence |
| Critical exposure | critical unverified obligation-seconds; maximum critical gap |
| Causal selectivity | impact precision/recall; sham false-positive invalidation |
| Evidence reuse | unaffected evidence reuse at matched critical-control coverage |
| Revalidation efficiency | runner compute, wall time, retries and test cost |
| End-to-end efficiency | agent cost + runner cost + reviewer/adjudicator cost |
| Reliability | repeated-trial pass-all-N and confidence intervals |
| Security | planted leakage, replay, scope, role and custody control acceptance |

Use full rerun, reset-and-restart, conversation-only memory, static dependency traversal, D2 temporal replay, D2+D3 evidence acquisition, complete EGDDM and expert-maintainer decisions as baselines/ablations. Match model version, harness, context, tools, intervention position, permissions, repository snapshots and resource budget.

The primary efficiency comparison is a Pareto frontier: lower total cost at equal or better shifted recovery and critical-control coverage. Do not allow a weighted scalar to trade a missed critical obligation for faster execution. Critical unauthorized action, hidden-label leakage, post-cutoff evidence and evaluator tampering are hard failures.

## 8. Targets—not achieved results

Admission targets:
- 100% of admitted tasks pass both sides of the oracle gate.
- At least two independent blind adjudicators per task.
- 0 candidate access to gold labels, hidden tests or evaluator credentials.
- 0 external code-host/proxy access during the agent phase except explicitly approved model/dependency endpoints.
- 100% of submitted patches regraded in pristine environments.
- 100% of task snapshots, acceptance contracts, runner images and network policies pinned by digest.
- 0 false admissions across the negative-control manifest suite.

Capability targets must be set after the pilot reveals the baseline distribution. The first preregistration should specify minimum paired sample sizes, task exclusions, confidence intervals, best-of-N/pass-all-N reporting and a stopping rule. Do not pre-claim a numerical improvement over SWE-Shift or SWE-Bench Pro V2.

A reasonable first operational pilot is one independently authored task per family/position cell (108 candidate tasks), then add seeds only after the admission pipeline and scorer survive red-team controls. This 108-task pilot is a validation stage, not a replacement for the D6 target of 540 matched scenario pairs (five seeds per cell).

## 9. Acceptance tests and limits

The admission manifest's negative-control suite must reject at least:
- reference patch fails;
- empty patch passes;
- grader environments differ;
- author is an adjudicator;
- fewer than two adjudicators;
- duplicated adjudicators;
- unresolved disagreement;
- candidate can read gold;
- candidate and evaluator share an environment;
- network is not restricted;
- regrade is not pristine; and
- invalid family, position, object ID or digest.

The Rust unit tests are authored but not compiled or executed in this environment. JSON/schema fixture parsing and structural review do not substitute for Rust compilation, schema-validator execution, artifact inspection, blind human adjudication or agent evaluation.

**Claim ceiling:** D6.4 implements an admission-contract checker and documents the experimental protocol. No real task has been admitted and no result demonstrates improved agent performance, a causal effect, SOTA superiority or production assurance.
