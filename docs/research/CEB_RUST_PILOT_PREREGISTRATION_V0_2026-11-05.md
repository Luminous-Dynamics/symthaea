# CEB Rust Repository-Repair Pilot Preregistration V0 — 2026-11-05

**Protocol:** `CEB-RUST-PILOT-001`  
**Status:** Design-only; not executed.  
**Machine-readable preregistration:** [`benchmarks/ceb/rust-pilot-preregistration.v0.json`](../../benchmarks/ceb/rust-pilot-preregistration.v0.json)  
**Schema:** [`benchmarks/ceb/rust-pilot-preregistration.schema.json`](../../benchmarks/ceb/rust-pilot-preregistration.schema.json)  
**Execution dependencies:** [#10064 OS-level isolation](https://github.com/Luminous-Dynamics/symthaea/issues/10064) and [#10369 trust roots/independent runner attestations](https://github.com/Luminous-Dynamics/symthaea/issues/10369).

## 1. Purpose and claim ceiling

The first Rust pilot should validate the experimental instrument before attempting a leaderboard result. It is a feasibility and variance-estimation pilot, not a confirmatory state-of-the-art study.

The proposed 48-task slice is stratified across six fault families: Rust types/traits, multi-crate API changes, configuration/target behavior, issue reproduction/debugging, security correctness and performance correctness. The target is six repositories with eight tasks per family. The 32 development / 16 held-out split is a pilot custody design; 16 holdout tasks are not enough to support a broad superiority claim.

## 2. Task admission

Each candidate task must have:
- A licensed public repository, exact source commit and recorded provenance.
- Objective acceptance conditions independent of the reference patch's textual form.
- Two independent gold reviewers, with a third adjudicator for disagreement.
- A reference patch expected to pass and an empty patch expected to fail, once the runner is qualified.
- Existing pass-to-pass contracts and security-sensitive behavior identified.
- No dependence on secrets, production systems, mutable services or undeclared network access.

A task with an ambiguous oracle, missing rights, unpinned environment, exposed fix commit, unstable dependency or unresolved reviewer disagreement is not admitted. Invalid tasks remain in the admission report; they are not silently removed after scores are seen.

## 3. Baseline arms

The minimum comparison is deterministic tooling, the same foundation model with a minimal harness, the same model with a strong conventional harness, and the same model+harness with exactly one Symthaea component added. A Symthaea-native arm is separate and only used where it can attempt the task without a frontier model.

Human-only and human-plus-agent are separate arms. Record expertise, elapsed time, interventions, review burden and accepted quality. Never merge human observations into agent seed counts.

Hold model identifier, toolchain, source commit, harness, context budget, turns, tokens, tool calls, compute, wall time and cost constant except for the single preregistered treatment difference.

## 4. Primary endpoint and outcomes

Primary endpoint: **accepted regression-safe repair rate**. A repair counts only when issue-specific fail-to-pass checks succeed, qualified pass-to-pass behavior is preserved, independent review accepts the patch, and there is no critical security or authorization failure.

The task ledger must preserve at least these outcomes separately: accepted, candidate incorrect, regression, security failure, invalid task, oracle unresolved, execution unqualified, abstained, escalated, timeout and budget exhausted.

Secondary measurements include edit precision, bug recall, critical-contract preservation, security false accepts, context sufficiency, mutation kill rate, collateral-change ratio, cost per accepted repair, wall time, human-review minutes, abstention/escalation precision and evidence completeness.

There is no single weighted score. A critical security failure cannot be compensated by cheaper execution or a higher ordinary resolve rate.

## 5. Statistical plan

Use at least three independent seeds per stochastic arm; use five where pilot variance warrants it. Pair all arms on the same task. Report discordant outcomes, task-level denominators, paired effect sizes and confidence intervals. Also report repository/root-cause cluster bootstrap sensitivity.

The 48-task pilot estimates task difficulty, baseline success, paired discordance, cluster variance, task invalidity and run cost. Use those observations to calculate the confirmatory study's sample size. Do not claim 80% power, statistical superiority or causal component value from the pilot without the appropriate calculation.

Preregister one primary arm contrast and endpoint before holdout access. Treat other contrasts as exploratory. Do not silently remove timeouts, refusals, escalations, abstentions, ungraded tasks or budget-exhausted cases.

## 6. Safety and execution gates

No candidate-controlled build scripts, procedural macros, repository tests, task commands or attack payloads should run until #10064 is independently qualified. A signed receipt is not proof of isolation. Independent regrade must not be claimed until #10369 establishes signer custody, trust-policy provenance and infrastructure/failure-domain separation.

Before those gates are met, proceed with task sourcing, license review, task manifest authoring, acceptance-contract review, gold-label adjudication on static evidence, schema validation and statistical preregistration only. Do not represent static review as task execution.

## 7. Reporting

Publish exact source, task, grader, toolchain, runner-policy and evidence digests; all task dispositions; denominators; per-repository and per-fault-family outcomes; costs; latency; review burden; abstentions; escalations; security failures; unresolved cases; and deviations.

The pilot can establish whether the measurement design is workable. It cannot establish CEB qualification, independent regrading, or a direct win over SWE-Bench Pro, Rust-SWE-bench or any other external benchmark.
