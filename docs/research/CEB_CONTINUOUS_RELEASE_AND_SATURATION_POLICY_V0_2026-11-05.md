# CEB Continuous Benchmark Release and Saturation Policy V0 — 2026-11-05

**Status:** Design-only initial release policy. No task resources have been calibrated and no task has been declared saturated.  
**Machine-readable release record:** [`benchmarks/ceb/release-lifecycle.v0.json`](../../benchmarks/ceb/release-lifecycle.v0.json)  
**Schema:** [`benchmarks/ceb/release-lifecycle.schema.json`](../../benchmarks/ceb/release-lifecycle.schema.json)  
**Validator:** [`scripts/validate_ceb_release_lifecycle.py`](../../scripts/validate_ceb_release_lifecycle.py)

## 1. Why benchmark releases need their own contract

A benchmark is executable software with tasks, prompts, environments, tools, resource budgets, graders and metadata. A new task can invalidate old system scores; a verifier fix can change grading without changing what the agent saw; a documentation correction may leave the experiment reusable. Treating all three as a single version bump destroys interpretability.

Terminal-Bench 4.0 provides a useful release precedent:
- it calibrated CPU/memory/time resources to reduce infrastructure noise;
- removed eight saturated, publicly solved, refused or unresolved tasks and fixed 19;
- defined changes requiring rerun, regrade or reuse under semantic versioning;
- treated a task as saturated when every model class in every family of the latest generation solved it 5/5.

Source: https://www.tbench.ai/news/terminal-bench-4-0

The integrity caveat matters: Epoch AI still rated Terminal-Bench 4.0.0 flawed after identifying publicly documented scoring defects in 30 of 66 tasks. Maintainer task updates do not substitute for independent review. Source: https://epoch.ai/benchmarks/terminal-bench-4/review

## 2. CEB version semantics

| Change class | Examples | Version action | Required evaluation action |
|---|---|---|---|
| Agent-visible | Task set, task prompt, source revision, environment, dependency, tool policy, network policy, model budget or resource budget | Major | Full rerun for a comparable new release; preserve old results |
| Grader-only | Verifier, rubric, oracle, grader code or judge configuration, with agent-visible inputs unchanged | Minor | Regrade retained raw artifacts where complete; otherwise rerun affected tasks |
| Metadata-only | Documentation, labels or metadata that do not alter any evaluation input | Patch | Reuse only after evaluation-input digests are proven identical |

If a release includes more than one class, the highest-impact change controls the action. A grader-only change must never be called metadata-only. If raw trajectories, patches, tool events or required grader inputs are missing, regrade is not possible and affected tasks must be rerun.

Version policy is not permission to change a benchmark silently. Each release is immutable and carries exact input, grader and metadata digests.

## 3. Task saturation

Saturation is not a judgment that a task is easy, expensive or inconvenient. It is evidence that the task no longer differentiates eligible systems.

V0 proposes:
- at least five independent trials per registered model-family/configuration;
- a preregistered current-generation model-family cohort;
- all eligible family/configuration cohorts must solve all five trials for a task to be considered saturated;
- the observation window and model generation must be recorded.

A saturated task leaves the active discriminative aggregate but remains in the longitudinal panel. The historical task denominator and past scores are never rewritten. New tasks replace capacity in a new major release, not by silently editing old denominators.

A task that is invalid, flaky, refused or incompatible is also retained with an explicit disposition and decision record. The reason for removal and the affected denominators are published.

## 4. Resource calibration and noise separation

Before a matched experiment, measure infrastructure behavior separately from agent behavior.

The proposed resource-calibration slice uses at least three representative reference tasks per environment and at least ten independent repeats per reference task. It must report the number of tasks and the minimum repeated-trial count separately; ten aggregate runs across an environment are not sufficient. Record:
- worker/image identity and resource policy;
- CPU, memory, disk and network saturation;
- wall-clock distribution and p95 runtime;
- infrastructure-induced failures and timeouts;
- agent-caused timeouts, refusals and output-token exhaustion separately;
- repeated-measurement noise and the rule used to set time and resource headroom.

V0's prospective infrastructure failure-rate target is at most 1%. This is a design target, not a measured result. A task must not be labeled agent-failed when the runner itself is unqualified or exhausted its declared resources.

Do not infer isolation from a Nix shell, container label or benchmark receipt; the #10064 execution-isolation gate remains mandatory.

## 5. Initial state

The V0 record intentionally has:
- no previous qualified release;
- no task/environment/grader digests yet;
- no task lifecycle entries;
- no saturation observations;
- no resource calibration runs.

It is a release-policy design artifact, not a benchmark run.

## 6. Validation contract

The offline validator checks:
1. Initial design state contains no fabricated qualified artifacts or observations.
2. Agent-visible changes cannot be under-bumped to a minor/patch release.
3. Grader-only changes cannot be treated as metadata reuse.
4. Reuse requires identical evaluation-input digests.
5. Regrade requires the needed raw artifacts; otherwise affected tasks must rerun.
6. Saturation requires every preregistered current-generation family/configuration to achieve 5/5.
7. A saturated/invalid/retired task stays in the longitudinal panel and preserves its historical denominator.
8. Infrastructure and agent failure outcomes are explicitly separated.

The self-tests mutate synthetic manifests in memory. They do not execute benchmark tasks, candidate code, verifiers, graders or attack payloads.
