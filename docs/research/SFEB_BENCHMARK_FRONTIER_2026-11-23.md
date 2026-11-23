# SFEB benchmark frontier and competitive research — 2026-11-23

**Parent:** [PR #13162](https://github.com/Luminous-Dynamics/symthaea/pull/13162)  
**Comparator ledger:** `benchmarks/frontend/external-comparator-ledger-v1.json`  
**Status:** external snapshots refreshed; no SFEB competitive score or SOTA claim is made

## Executive conclusion

SFEB should not compete as another screenshot-to-code leaderboard or another browser-agent success-rate table. The defensible research position is **independently verified engineering change under realistic state, security and maintenance constraints**.

The opportunity is to combine capabilities that current programmes measure separately:
1. Vision2Web's visual, interactive and full-stack website breadth.
2. FrontierCode 2.0's maintainer-defined mergeability and thoroughness.
3. OSWorld 2.0's long-horizon, dynamic-state, conflict and clarification stress.
4. WebdriverIO's controlled browser-tool comparison and cost telemetry.
5. Interaction-as-a-Judge's interaction-grounded diagnosis.
6. Benchmark task qualification and independent evidence replay, which SFEB should make an explicit precondition for publishing a score.

Do not pool their scores. They differ in tasks, scoring, judges, agents, budgets and execution environments.

## 1. Current external evidence

### Vision2Web — nearest broad frontend-development comparator

Vision2Web describes 193 tasks, 918 prototype images, 1,255 test cases and 16 categories. It spans static responsive webpages, interactive multi-page frontend tasks and full-stack applications; verification combines workflow-driven GUI testing with VLM visual judging.

A historical S4-2026 indexed row showed Claude-Fable-5 + Claude Code at 82.3 overall (87.4 static, 83.2 interactive, 76.3 full-stack). However, the direct page render on 2026-11-23 displayed no results for the selected S4-2026 season. The 82.3 row must remain a historical indexed snapshot, not a verified live leader.

**SFEB lesson:** report a three-level matrix, separate visual from functional acceptance, preserve official task/harness context and qualify the oracle before scoring.

Sources: https://proceedings.mlr.press/v306/he26n.html ; https://vision2web-bench.github.io/index.html

### FrontierCode 2.0 — strongest mergeability comparator

The official benchmark contains 132 tasks, authored by 20+ experienced maintainers/developers who reportedly spent more than 40 hours per task. Grading combines unit tests, repository-specific rubrics and additional verifiers; failing a blocking criterion yields a score of zero. It explicitly evaluates whether a maintainer would merge the patch.

The 2026-11-23 official leaderboard row for Claude Fable 5.2 / Claude Code xhigh is 50.2% weighted score, 54.7% pass rate, $17.73 mean cost per rollout and 156k output tokens. These are separate measurements. The official changelog notes a scoring rescore on Nov 11 after the cheating scanner was revised to remove false positives.

**SFEB lesson:** measure thoroughness, related-input coverage, scope discipline, compatibility and regression preservation. Keep a maintainer rubric and deterministic gates separate; a weighted score must not hide hard blockers.

Source: https://cognition.com/frontiercode

### OSWorld 2.0 — long-horizon operational stress

OSWorld 2.0 describes 108 workflows across 31 self-hosted websites. 69.6% take a skilled user more than one hour, the median human task time is about 1.6 hours, and the paper reports an average of 318 tool calls for Claude Opus 4.7 at maximum effort. Its primary paper-reported 500-step result is 20.6% binary completion and 54.8% partial score for Claude Opus 4.8 with maximum thinking and batched tool calls.

The site also exposes selectable live leaderboard views. A search-indexed 2026-11-23 row showed Claude Opus 5 / Max Batch tool v2.1 at 44.33% binary and 77.67% partial. The exact selected step-budget/release filter was not captured, so that row is recorded only as a provisional live snapshot and is not a replacement for the paper's 500-step result.

OSWorld's challenge analysis highlights implicit-state inference, multi-item state tracking, conflict disambiguation, dynamic environments, streaming interaction and proactive clarification.

**SFEB lesson:** use a deterministic event-sourced work board with explicit latency, duplicate delivery, late events, permission revocation, conflict and reconnect controls. Measure convergence and unsafe side effects, not only final completion.

Source: https://osworld-v2.xlang.ai/ ; https://arxiv.org/abs/2606.29537

### WebdriverIO Browser Agent Benchmark — browser-tool comparison

The benchmark fixes model, agent harness and prompt while changing the browser tool, using a fixed random sample of 100 Online-Mind2Web tasks across live websites. It reports success with a Wilson interval and also cost, tokens, elapsed time and tool calls.

Its indexed October 9, 2026 results included a WebdriverIO session row of 62% over 103/165 observations, with a 55–69% Wilson interval, median $0.013, 174k tokens, 156 seconds and 26 tool calls. The current page render on Nov 23 says no published benchmark runs yet, so the October row is a historical observation only. The project warns about live-site variability, roughly ±10-point intervals for 100 tasks and WebJudge disagreement with human reviewers on about one run in seven.

**SFEB lesson:** report browser-tool effectiveness separately from frontend engineering quality. Keep a deterministic offline lane for reproducibility and an explicitly separate live-site lane for external robustness.

Source: https://benchmark.webdriver.io/

### Interaction-as-a-Judge — useful diagnostic evaluator

Arena's Nov 12 research page reports broken-requirement detection rising from 63.0% (code-only judge) to 71.1% (interaction-grounded judge), and iterative repair rising from 55.4% to 75.5% under its own protocol. It reports released dataset cards of 6,390 preference pairs and 189 app versions with 362 human-verified requirements.

**SFEB lesson:** interaction-grounded diagnosis should be a secondary evaluator arm. Pin model, prompt, tool and calibration sets; keep deterministic security, identity, evidence and explicit-contract rules authoritative. Human calibration and malicious-page-instruction tests are mandatory.

Source: https://arena.ai/blog/interaction-as-a-judge

### Benchmark validity is itself a measured risk

OpenAI's 2026-07 audit reported widespread SWE-Bench Pro task-quality issues and estimated roughly 30% of tasks were broken; the Epoch AI review independently labeled the benchmark flawed after audits reported at least 30% problematic tasks. That is a meta-evaluation signal, not an agent score.

**SFEB lesson:** publish two-sided task validity, prompt-to-oracle traceability, placebo rejection, positive-control acceptance and independent regrading as release gates.

Sources: https://openai.com/index/separating-signal-from-noise-coding-evaluations/ ; https://epoch.ai/benchmarks/swe-bench-pro/review

## 2. SFEB's differentiating benchmark programme

### A. Verified Change Efficiency

Use the proposed metric as a family rather than a single leaderboard scalar:

[
VCE_{\$}=\frac{\text{independently accepted changes}}{\text{total compute + operating cost}}
]

[
VCE_h=\frac{\text{independently accepted changes}}{\text{agent + human engineering hours}}
]

[
VCE_r=\frac{\text{independently accepted changes}}{\text{review + rework cost}}
]

A change enters the numerator only after the independent verifier accepts every required oracle. Report cost, human intervention, regressions, accessibility, security and evidence completeness separately. Do not let the scalar hide a security failure.

### B. State reconciliation and recovery

Add the following first-class measurements to INT-004 and later long-horizon tasks:
- **Convergence correctness:** exact authoritative final-state match after replay/reconnect.
- **Divergence duration:** time from first inconsistency to restored convergence.
- **Recovery latency:** time from reconnect or conflict revelation to verified consistency.
- **Unsafe mutation count:** unauthorized, stale or duplicate side effects; target zero.
- **Idempotency coverage:** event/action replay cases with no duplicate effects.
- **Clarification precision/recall:** ask when the evidence is ambiguous; proceed when it is not.
- **Audit completeness:** every accepted transition appears once, and rejected/pending actions do not appear as accepted.

### C. Evaluator integrity

Before task scoring:
- a reviewed witness/reference patch must pass;
- no-op and seeded-defect baselines must fail the intended oracle;
- positive and semantics-preserving controls must pass;
- a second reviewer must verify brief solvability and requirement-to-oracle mapping;
- the evaluator re-grades on a pristine image;
- worker oracle IDs/disposition are diagnostic only;
- missing evidence or verifier errors yield inconclusive, never accepted.

### D. Longitudinal and maintenance evaluation

Score a task over a sequence of change requests, dependency/framework updates, regressions, permission changes and changing product requirements. Hold out entire task families and interaction patterns, not just different fixture IDs. Use a controller-only/hidden partition of at least 30% for publication-grade evaluation.

## 3. Prospective targets

These are proposed research targets, not achieved results:

| Dimension | Target |
|---|---:|
| Critical/security/evidence integrity mutation detection | 100% |
| High-severity mutation detection | ≥95% |
| Overall mutation detection | ≥90% |
| Positive-control false rejection | 0 |
| Metamorphic relation failures | 0 |
| Unauthorized or duplicate authoritative side effects | 0 |
| Independent verifier false acceptance in seeded adversarial tests | 0 |
| A4 versus A0 paired acceptance uplift | ≥5 percentage points, paired CI excludes zero |
| Hidden/controller-only task share for headline results | ≥30% |
| Cost efficiency | Report VCE frontier; require improvement without weaker quality gates |

Do not use external headline percentages as SFEB's numeric targets. Reproduce a native comparator protocol before making direct SOTA statements.

## 4. Experimental design to beat the field honestly

Keep the existing matched A0–A4 comparison and H0 human reference. Add a second experiment that isolates the cost of verification:
1. same agent/model/tool budget, no independent replay;
2. same agent/model/tool budget, independent replay and repair;
3. same agent/model/tool budget, independent replay plus frozen policy and isolated browser execution.

Use task-block randomization, task-clustered bootstrap intervals, paired tests, multiple-comparison correction and preregistered primary outcomes. Report all runs, blocked/infra errors, timeouts and unknowns. Use native task splits for external comparisons and do not train on hidden test tasks.

## 5. Immediate engineering sequence

1. Fix and compile the standalone INT-004 Rust model; add strict event shape and bounded replay-window tests.
2. Implement the fixture-local event service and a minimal Leptos work board with keyboard alternative actions.
3. Implement #13247's verifier-derived event/state oracles: parse raw traces, replay from frozen initial state and policy, derive results, bind evidence digests and fail closed.
4. Finish #13228 signed execution statements and isolated Chromium, with #7435 enforced network egress as a prerequisite.
5. Complete L2-005 on the same trust path, then run the 30-task pilot and matched study.

## 6. Evidence boundary

The Nov 23 comparator refresh is a research snapshot. SFEB has not run the external benchmark suites, has no comparable SOTA score, and has not demonstrated superiority. INT-004's independent in-memory V8 reference check is not Rust execution, candidate-browser qualification or trusted verification. The branch still records `runnable_task_count=0`.