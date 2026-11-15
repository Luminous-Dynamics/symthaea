# EGDDM ShiftBench D6 — blinded corpus construction plan

**Status:** proposed corpus plan. No task corpus has yet been collected, blinded, independently adjudicated or evaluated.

## 1. Target scale

The proposed benchmark uses 12 intervention families and three intervention positions. Each family/position cell targets ten independently authored task seeds.

| Component | Target |
|---|---:|
| Intervention families | 12 |
| Positions per family | 3 |
| Family/position cells | 36 |
| Seeds per cell | 10 |
| Matched clean/shifted scenario pairs | 360 |
| Trajectories per pair | 2 |
| Trajectories per model per trial | 720 |
| Independent trials | 3 |
| Model/harness families | 3 |

The target is 360 matched scenario pairs and 720 trajectories per model per trial, not 360 already-existing tasks. Three independent trials across three model/harness families would imply up to 6,480 trajectories before baseline and expert runs. Stage the study so task validity and evaluator quality are established before incurring that cost.

## 2. Intervention families

The machine-readable blueprint lists the families, gold-impact design and negative controls. The twelve families cover:

1. Requirement addition or reinterpretation.
2. Requirement removal, narrowing or deprecation.
3. Dependency/API breaking change.
4. Toolchain/compiler/build-system upgrade.
5. Configuration-world drift.
6. Runtime or external-service state change.
7. Session, tool identity or execution-context change.
8. Authority or policy revocation.
9. Evidence invalidation, supersession or contradiction.
10. Test/oracle/acceptance-contract revision.
11. Common-cause environment change.
12. Cross-repository contract/schema change.

The three positions are early (15% of planned work), middle (50%) and late (85%). These are scheduling strata, not wall-clock times; define work units before randomization.

## 3. Two-sided task admission

Every task must satisfy both sides of the validity gate:
- the reference solution passes the acceptance contract; and
- the empty/no-op solution fails it.

A task that cannot distinguish an actual repair from doing nothing must not enter the benchmark. When instructions or acceptance criteria change, an independent engineer should solve against the corrected instructions without seeing the reference patch or hidden gold labels.

SWE-Bench Pro V2 provides a concrete benchmark-integrity reference: its September 2026 refresh removed invalid tasks, corrected instruction/test contradictions with blind expert solves, used network restrictions, and regraded changes in pristine images. D6 should meet at least that bar before publishing leaderboard claims. See [the V2 protocol](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full).

## 4. Candidate/evaluator separation

Candidate-visible task bundles contain only the allowed repository snapshot, initial request, admissible history, and intervention information appropriate to the current trial. Gold affected-obligation labels, hidden tests, reference changes, grader credentials and post-cutoff evidence must be outside the candidate-readable environment.

Pin and record:
- T0 repository commit, tree and configuration world;
- task, intervention and allowed evidence bundle digests;
- model, harness, tools and system prompt versions;
- runner image and network policy;
- grader and hidden-test identities;
- every receipt's source, observation and availability time;
- clean-regrade artifacts and exact result digests.

Use isolated disposable execution environments. The parser/scorer must remain read-only and must not execute repository-controlled code. Candidate patches must be regraded in a pristine copy. Gold labels should be held and adjudicated independently from the agent authoring the task.

## 5. Scorecards

Do not collapse performance into a single weighted score. Keep separate reporting for:
- terminal retained functionality and active-requirement conformance;
- time-weighted requirement and critical-only conformance;
- critical unverified obligation-seconds and maximum critical evidence gap;
- stale receipt reuse and future-evidence leakage;
- impact precision/recall and abstention quality;
- paired shifted success and pass-all repeated-trial reliability;
- revalidation cost at matched critical-control coverage;
- model/tool tokens, compute, wall latency, retries, external service use and reviewer time.

Critical unauthorized actions, evaluator tampering and future-evidence leakage remain hard failures and must not be offset by success on unrelated tasks.

## 6. Baselines and study design

Compare full rerun, reset-and-restart, conversation-only memory, static dependency traversal, D2 temporal replay alone, D2+D3 evidence acquisition, full EGDDM and expert-maintainer decisions. Match model versions, harnesses, context budgets, permissions, task snapshots, intervention positions and resource budgets.

Use repository-family and forward-time holdouts. Report paired differences, discordant pairs, exact denominators, confidence intervals, task exclusion reasons and repeated-trial outcomes. Report both best-of-N and pass-all-N reliability.

## 7. Phased execution

**Phase A — validity pilot:** one task per family/position cell, plus positive, negative, sham and common-cause controls. Validate scoring and blind custody before expanding.

**Phase B — corpus expansion:** ten seeds per family/position cell. Independently curate gold affected-obligation labels and acceptance oracles.

**Phase C — matched evaluations:** three trials where feasible across at least three model/harness families; execute baseline and ablation arms at matched budgets.

**Phase D — independent replication:** publish the task version, evaluator, artifacts, excluded-task ledger and result receipts; invite external replication before claiming SOTA improvement.

## 8. Falsifiable EGDDM claim

The claim to test is:

> At matched recovery success and critical-control coverage, evidence-scoped invalidation reduces unnecessary revalidation while preserving previously passing functionality and meeting newly active requirements after a workspace shift.

The claim fails if selective reuse increases stale evidence, misses critical obligations, lowers shifted recovery success, or does not reduce measured end-to-end cost after reviewer effort is included.

**Claim ceiling:** design proposal only. No benchmark result, SOTA lead, causal effect, real-world safety guarantee or independent validation is established by this blueprint.
