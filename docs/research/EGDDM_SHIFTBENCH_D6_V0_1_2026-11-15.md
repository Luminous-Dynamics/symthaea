# EGDDM D6 — ShiftBench: evidence-qualified recovery across changing workspaces

**Version:** 0.1  
**Date:** 2026-11-15  
**Status:** research protocol plus read-only Rust trace evaluator, temporal exposure scoring and synthetic known-answer fixture authored. No compilation, test execution, model rollout, external adjudication or benchmark result is claimed.

## 1. Why this track exists

Static repository repair asks whether an agent can make a change in a fixed workspace. Long-horizon maintenance asks whether a project remains correct as later changes accumulate. A further question is under-measured:

> When the world changes during a task, can an agent identify precisely which earlier conclusions are invalid, preserve the conclusions that remain valid, satisfy the new requirements, and resume safely?

Recent work makes this a practical research problem, not a speculative one:

- [SWE-Shift (October 2026)](https://arxiv.org/abs/2610.32196) introduces matched clean and shifted continuations. Its 120 scenarios cover 12 change families and nine intervention positions. All 17 evaluated systems performed worse under shift; the paper reports GPT-6 Astra falling from 67.35% clean success to 24.79% shifted success, with 20.34% paired success. Tool/session changes and stale evidence produced zero paired success across systems.
- [ChainSWE (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.371/) evaluates sequential dependent bug fixes across 304 chronological issues in 54 Python projects. The paper reports performance dropping by up to 70% as chain length increases.
- The [2026 Long-Horizon Software Engineering Agents survey](https://longhorizonswe.github.io/) argues that length alone is not continuity. It identifies retained functionality and conformance to active requirements as useful measures, and reports a ChainSWE comparison where per-issue resolution falls from 69.1% with reference fixes to 49.0% with agent-produced patches, and 47.4% when prior dialogue is also retained.
- [SWE-Gate](https://arxiv.org/abs/2609.04167) finds that functional success is not complete acceptance: 221 of 644 repairs that passed functional tests failed review-derived constraints.
- [SWE-Time](https://arxiv.org/abs/2611.12083) treats repository evolution as the reasoning target, including retrospective and prospective temporal understanding.
- [SWE-Atlas (May 2026)](https://scale.com/blog/swe-atlas-complete) evaluates 284 codebase Q&A, test-writing and refactoring tasks. Its report finds large reliability gaps across repeated trials: systems were reported to be 2–3× more likely to succeed on at least one of three trials than on all three; it also highlights investigation and native harness tooling as important confounders.
- [SWE-fficiency (ICML 2026)](https://proceedings.mlr.press/v306/ma26l.html) evaluates 498 performance-optimization tasks across nine repositories and reports that top agents averaged less than 0.23× expert speedup. This is not a D6 task family, but it reinforces that engineering quality must include measured cost and expert-relative efficiency, not only patch acceptance.

D6 must complement these benchmarks. It does not claim that EGDDM invented workspace-shift evaluation, temporal repository reasoning, sequential maintenance, or review-constraint testing.

## 2. Paradigm shift: from "re-run everything" to evidence-scoped continuity

A conventional workflow often makes one of two mistakes after a change:

1. **Under-invalidation:** it reuses a passing result even though the affected code, contract, configuration or toolchain changed.
2. **Over-invalidation:** it discards every prior result and repeats all checks, even when the change did not affect their scope.

D6 measures the middle path: a change creates an **invalidation frontier** over obligations and evidence. Evidence for affected obligations must be refreshed. Evidence for unaffected obligations may remain usable only when its scope is explicit and the benchmark's impact oracle says it was not affected.

This is not a claim that a dependency graph proves causality. The scope oracle is an evaluation label established independently for benchmark tasks. Production use requires source provenance, configuration identity, authority and independently verifiable execution receipts.

The resulting project state is not merely a repository snapshot. It is a time-indexed set of obligations, evidence receipts, invalidation events and decision constraints.

## 3. Matched task protocol

Every scenario begins from one frozen initial snapshot and branches into two matched continuations. The evaluator requires identical initial active obligations and equivalent initial evidence outcomes, source scope and availability in both arms; otherwise the pair is invalid:

- **Clean arm:** continuation proceeds without a material workspace intervention.
- **Shifted arm:** an intervention is introduced at a preregistered point while preserving the same saved prefix.

Change families should include requirements, dependency/API contracts, toolchains, configuration, runtime state, session/tool identity, authority/policy, and contradictory or stale evidence. Intervention positions should be balanced across early, middle and late work.

At each checkpoint, the agent reports:
- obligations it believes are active;
- which obligations the current change affects;
- check outcomes and the exact evidence scope behind them;
- which previous receipts it retained or revalidated;
- resource and review costs; and
- whether the available evidence is sufficient to resume, or whether it must abstain.

The candidate receives task-visible change information and active requirements. Gold affected-obligation labels, hidden tests, evaluator credentials and post-cutoff artifacts remain outside candidate-readable files. The public synthetic fixture includes gold labels solely to make the local known-answer example inspectable; it is not a deployable blinded benchmark task.

Each task is frozen at T0. Only evidence available at its permitted cutoff may be used. A continuation must not gain future commits, issue comments, PR discussion, hidden test output or labels by accident. Clean and shifted arms must begin from the same state and agent prefix. At each stage, earlier still-active obligations are checked again.

## 4. First evaluator increment

The standalone read-only module is:

    tools/development-map-validator/src/workspace_shift.rs

CLI entry after integration:

    development-map-validator workspace-shift <scenario.json>

The v0.1 trace has:
- a shared initial repository/tree/configuration snapshot;
- explicit obligation IDs and criticality;
- matched clean and shifted arms;
- monotonically ordered state epochs;
- evaluator-side affected-obligation labels;
- active requirements at each epoch;
- candidate impact predictions;
- check receipts pinned to source epochs and exact repository/tree/configuration identities;
- exact checkpoint cutoffs, receipt-observation times and actor-availability times;
- first-visible-checkpoint rules for receipts; and
- declared revalidation cost units.

The evaluator derives freshness from the last gold impact epoch for each obligation. A receipt from an earlier epoch remains reusable for an unaffected obligation, but becomes stale when a later intervention affects that obligation. A source receipt cannot claim a future epoch or mismatch its own epoch's repository/tree/configuration pins. Receipt observation must not predate its source epoch; observation must precede availability; and availability must be no later than the checkpoint cutoff. A receipt must also first appear at the earliest checkpoint where it claims to be available. This blocks within-epoch, late-disclosure and post-checkpoint evidence leakage, not only future-epoch leakage.

The scorer reports separate measures:
- **Retained functionality:** proportion of baseline-passing obligations that remain active and still have fresh passing evidence.
- **Active-requirement conformance:** proportion of currently active obligations with fresh passing evidence, including newly introduced obligations.
- **Impact precision/recall:** agreement between predicted and evaluator-held affected obligation sets.
- **Stale active receipts:** latest evidence that predates a change affecting its obligation.
- **Critical unverified obligations:** critical active requirements without fresh passing evidence.
- **Revalidation cost and savings:** declared post-baseline cost compared with a full-rerun reference that rechecks every active obligation after each later epoch. Baseline acquisition is excluded from both sides.
- **Paired success:** both clean and shifted arms preserve prior passing functionality and satisfy all active requirements.

### Longitudinal evidence exposure

Terminal correctness is insufficient: an agent may eventually recover while leaving a critical requirement unverified for a long interval. D6 therefore scores each obligation over time.

For every active obligation interval, the evaluator:
1. starts an interval at initial activation, reactivation or a gold invalidation event;
2. treats the interval as unverified until a current-scope passing receipt becomes available;
3. closes the gap when a passing receipt becomes available;
4. reopens the gap when a later fail/unknown receipt becomes available; and
5. ends or restarts the interval at removal or the next invalidation.

The time-weighted requirement conformance score is:

\[
C_{time}=1-\frac{\sum_i U_i}{\sum_i A_i}
\]

where \(U_i\) is the unverified duration for obligation \(i\), and \(A_i\) is its active duration. A separate critical-only score uses only critical obligations. The report also includes total and critical unverified obligation-seconds, recovery-event count, maximum unverified gap and per-obligation exposure.

The synthetic fixture yields the following hand-derived values:

| Measure | Clean arm | Shifted arm |
|---|---:|---:|
| Active obligation-seconds | 1,800 | 2,100 |
| Unverified obligation-seconds | 120 | 480 |
| Critical unverified obligation-seconds | 60 | 420 |
| Time-weighted conformance | 93.33% | 77.14% |
| Critical time-weighted conformance | 93.33% | 65.00% |
| Maximum critical evidence gap | 60 s | 180 s |

The shifted arm still reaches 100% terminal conformance. The longitudinal metric reveals that its critical API and audit requirements spent 420 combined seconds unverified during recovery. This prevents a final green snapshot from erasing an earlier evidence gap.

The metric is a benchmark proxy, not a field-validated operational hazard measure. It relies on the scenario's event and receipt timestamps, assumes the gold affected-obligation labels are correct, and does not authenticate execution or actor identity.

A missing receipt, failed or unknown check, stale critical receipt, or incomplete active requirement cannot be converted into terminal success by a high score elsewhere. The v0.1 digest is a deterministic integrity field, not a signature or proof that a test actually ran.

## 5. Synthetic known-answer case

The fixture models three obligations:
- API compatibility (critical);
- reproducible build (non-critical); and
- a new audit-log requirement (critical).

The shifted arm changes the dependency/API world, invalidating API compatibility evidence. The build obligation is deliberately not affected, so its original receipt remains valid. The new audit obligation must be checked at the shifted epoch.

The hand-derived expected result is:
- clean arm: 100% retention, 100% active conformance, zero revalidation cost in the continuation;
- shifted arm: 100% retention, 100% active conformance, 100% impact precision and recall;
- shifted revalidation: 2 obligation checks versus 3 checks in the full-rerun reference, for one-third fewer check units; and
- paired success: true for the synthetic expected trace.

These are fixture expectations, not executed Rust results or empirical agent outcomes.

## 6. Baselines and ablations

Use identical model versions, agent harnesses, context budgets, tool permissions, initial snapshots, task text, intervention times and compute budgets. Report at least three independent trials per task when feasible; publish single-run and pass-all reliability separately. Agent harness and native scaffolding identity must be pinned because tool orchestration materially affects performance.

Compare at least:

1. **Reset-and-restart:** discard earlier agent state and reconstruct from the shifted workspace.
2. **Conversation-only memory:** retain transcript but no explicit evidence/obligation graph.
3. **Static dependency traversal:** invalidate descendants of changed nodes with no temporal evidence model.
4. **Full-suite rerun:** rerun every eligible check after every change.
5. **No-history baseline:** act only on current repository and current request.
6. **D2-only temporal replay:** use time-scoped evidence history without D3 acquisition planning or D6 selective-continuity scoring.
7. **D2+D3:** temporal replay plus decision-sufficient evidence acquisition.
8. **Full EGDDM:** temporal ledger, exact source scope, evidence acquisition, impact prediction, policy gates and independent receipts.
9. **Expert-maintainer baseline:** independent human impact and recovery decisions.

Run ablations for each component. Report both model-only and system-level outcomes. An observed gain from the complete system does not establish which component caused it.

## 7. Adversarial controls

At minimum, include:

- stale pass receipt for an affected critical obligation;
- old receipt from an unaffected obligation that should remain reusable;
- incorrect configuration-world digest;
- wrong repository, commit or tree identity;
- future-dated evidence and actor-availability leakage;
- hidden/new requirement omitted from the candidate's active set;
- tool/session identity change with an old tool receipt;
- policy/authority change that blocks a previously permitted action;
- contradictory positive/negative observations;
- a change that affects no declared obligation (negative control);
- a sham intervention with no underlying state change; and
- a common-cause configuration change that affects several otherwise unrelated components.

Every task must pass a two-sided validity gate: the reference solution satisfies the acceptance contract and an empty/no-op solution fails. Regrade candidate outcomes in pristine environments. Evaluator and candidate must not share implementation code when that would create common-mode scoring risk. Gold labels and scorer administration must be independently controlled.

## 8. Benchmark targets (prospective, not achieved)

Initial targets for a suitably sized, independently adjudicated held-out set:

| Measure | Initial target |
|---|---:|
| Affected-obligation recall | ≥95% |
| Affected-obligation precision | ≥85% |
| Unaffected-obligation retention | ≥90% |
| Critical stale-evidence acceptance | 0 in planted controls |
| Unauthorized irreversible actions | 0 in planted controls |
| Active-requirement conformance | ≥95% |
| Prediction interval coverage | 90–95%, reported with interval width |
| Revalidation cost | ≥20% below full rerun at matched critical-control coverage |
| Shifted recovery success | statistically significant paired improvement over strongest non-EGDDM baseline |
| Cost reporting | 100% of runs include token, compute, latency, tool, reviewer and retry cost |

A zero observed error on a small control set is not a zero-risk guarantee. Publish denominators, confidence intervals, invalid/excluded task counts and adjudication method. A target that cannot be measured reliably should be revised rather than presented as success.

For paired binary outcomes, report discordant pairs and a paired confidence interval/test; do not compare unrelated aggregate pass rates. For cost, report median and tail latency, total attempts and the cost of failed recoveries. Use repository-family and forward-time holdouts to avoid memorizing project-specific structure.

## 9. Positioning against industry and SOTA

These benchmarks are complementary, not interchangeable:

| Benchmark | What it measures | Design lesson for EGDDM |
|---|---|---|
| [SWE-Bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full) | Large-scale repair with aggressive task/evaluator integrity, including reference-pass and empty-patch-fail controls | Treat task validity and evaluator integrity as first-class; no single score without denominator, clean regrade and invalid-task accounting |
| [SWE-Shift](https://arxiv.org/abs/2610.32196) | Mid-task workspace shifts across 12 families and nine intervention positions, evaluated across 17 systems | Do not claim novelty for shift testing; add explicit evidence identity, temporal admissibility and selective invalidation |
| [ChainSWE](https://aclanthology.org/2026.emnlp-main.371/) | 304 chronological issue chains across 54 projects | Test multi-stage maintenance and preservation of previous work, not only isolated fixes |
| [SWE-Time](https://arxiv.org/abs/2611.12083) | 1,661 temporal repository tasks across 65 repositories | Pin T0/T1 information cutoffs and distinguish retrospective from prospective knowledge |
| [SWE-Gate](https://arxiv.org/abs/2609.04167) | Review-derived constraints beyond functional tests; 221 of 644 functionally passing repairs failed those constraints | Score active requirements, review constraints and critical misses separately from functional pass |
| [SWE-Atlas](https://scale.com/blog/swe-atlas-complete) | 284 codebase Q&A, test-writing and refactoring tasks, including repeated-trial reliability | Report pass-all reliability, investigation precision and harness/tooling identity, not best-of-N only |
| [SWE-fficiency](https://proceedings.mlr.press/v306/ma26l.html) | 498 performance-optimisation tasks across nine repositories, benchmarked against expert speedup | Measure end-to-end cost and reviewer time; successful patching is not equivalent to expert-level engineering efficiency |

A single leaderboard rank cannot fairly compare EGDDM with these suites. D6 should publish adapters and matched task subsets rather than assert equivalence between unlike datasets.

### Proposed differentiation to test

The strongest near-term claim is narrower and falsifiable:

> At matched recovery success and critical-control coverage, EGDDM reduces unnecessary revalidation while preserving more previously validated behavior and meeting new active requirements after a workspace shift.

This is a proposed contribution, not a demonstrated lead. It must beat full-rerun, static dependency, conversation-memory and reset baselines on held-out tasks. If it does not, the selective-invalidation thesis is not supported.

To exceed current SOTA in a meaningful way, aim for:
- better paired shifted-continuation success, not a higher score on a different dataset;
- better retention and active-requirement conformance across agent-produced histories;
- fewer stale-evidence and authority-gate failures;
- lower end-to-end recovery cost at equal or better critical-check coverage;
- better pass-all reliability over repeated trials; and
- calibrated abstention where impact cannot be distinguished.

No cross-paper headline score should be described as a direct apples-to-apples comparison without a common harness, model version, task split and cost accounting.

## 10. Next experiments

1. Expand from the synthetic case to 12 intervention families and at least three intervention positions per scenario.
2. Add multiple change events so the invalidation frontier can grow, shrink or be partially superseded.
3. Add uncertainty intervals for impact and evidence freshness, with an explicit unknown state.
4. Integrate D3 EVSI so the system chooses which authorized checks to acquire under multidimensional cost budgets.
5. Evaluate D2 temporal replay under event delivery reordering and multi-actor partial-order history.
6. Build blind task and label custody outside the candidate-readable repository.
7. Run matched agent rollouts with at least three model/harness families and expert adjudication.

## 11. Limitations

- The supplied receipt digest can be recomputed by anyone with the trace. It proves neither signer identity nor test execution.
- The local fixture's affected-obligation labels are visible; this is not a blind benchmark.
- Cost units are declared inputs, not measured wall-clock or infrastructure costs.
- The scorer is a prototype and has not been compiled or executed in this environment.
- The longitudinal metric measures time between declared invalidation/activation and evidence-availability events. It does not establish real-world exposure without independently authenticated timestamps and independently witnessed check execution.
- Epoch-level and timestamp-level availability checks improve leakage controls but do not replace authenticated actor identities, signed execution receipts or independently administered benchmark custody.
- A dependency graph, even when time-indexed, is not causal evidence. D5 requires intervention design and valid controls.
- A high continuity score does not authorize deployment or irreversible changes. Authority, safety and execution remain independent gates.
