# Symthaea Scientific-Agent Benchmark Strategy — 2026-11-27

## Executive thesis

The field is moving from chat-based research assistance toward project-level scientific agents. Symthaea should not compete only on a single resolution-rate leaderboard. The defensible opportunity is **evidence-qualified scientific autonomy**: a system that can select experiments, execute them, preserve provenance, detect invalid evidence, calibrate its uncertainty, abstain when warranted, and transfer improvements to new tasks without silently regressing.

The benchmark programme should therefore optimize a vector:

1. scientific task success and quality;
2. independent evidence integrity;
3. calibration and selective risk;
4. reproducibility and artifact completeness;
5. transfer/generalization;
6. cost, wall-clock time, and tool efficiency;
7. robustness to missing, delayed, contradictory, and adversarial observations.

No single scalar should conceal a severe failure in another dimension. Publish the Pareto frontier and per-domain/per-task results.

## Current external comparison (snapshot 2026-11-27)

All results below are snapshots from public benchmark pages, not Symthaea measurements. Harness, model, effort, trial count, task version, and scoring rules differ. They are not interchangeable.

| Benchmark | Public reference | What it measures | Interpretation for Symthaea |
|---|---|---|---|
| Terminal-Bench-Science | Official Snorkel page reports 70 tasks across five domains; GPT-6 Astra/Codex 68.1% ±3.2 and Opus 5.5/Claude Code 63.3% ±3.3. [Leaderboard](https://snorkel.ai/leaderboard/terminal-bench-science/) | Long-horizon executable research workflows with task verifiers | Primary external task-completion target; match the exact harness/version and report uncertainty, cost and time |
| Terminal-Bench-Science cost frontier | Vals AI's Nov 19 snapshot reports GPT-6.2 Astra 71.43%, Gemini 4 Neon 62.86%, GPT-6 Astra 62.86%, with per-test cost/tokens/duration. [Results](https://vals-ai.com/benchmarks/terminal-bench-science) | Completion plus resource consumption | Optimize cost-quality-time Pareto frontier, not raw success alone |
| ASI-Bench | 60 tasks, 11 domains, progressive methodological-guidance removal (B1–B4); leaderboard lists GPT-6.1 Sol/Codex at 63.90 and GPT-6 Astra/Codex at 62.23. [Leaderboard](https://asibench.apexin.ai/) | Project-level research, method reconstruction, robustness to missing guidance | Essential test of whether the system can reconstruct executable methods rather than follow a supplied recipe |
| Autonomous Science Bench 0.1 | Six pilot tasks; ten configurations × three trials per task. GPT-6 Astra/Codex 72.2%, Opus 5.5/Claude Code 66.7%. Authors explicitly caution that small differences are not meaningful at three trials. [Leaderboard](https://asb.open-ssi.org/leaderboard) · [Methodology](https://asb.open-ssi.org/about) | Real laboratory workflow planning, observation interpretation, iterative improvement | Use as a laboratory campaign target, but require a larger confirmatory evaluation before superiority claims |
| SCI-AGENT-EXAM | 120 tasks, 11 fields, seven workflow task types; paper reports GPT-6 Astra 80.9 overall, with only 26% of runs above 0.95 success threshold. [Paper](https://arxiv.org/abs/2610.32441) | Executable and textual scientific outputs; progress and stability | Track run-level reliability and threshold success, not only average score |
| SciAgentArena | Approximately 200 tasks with stepwise verification across scientific contexts. [Project and benchmark](https://sciagentarena.github.io/) | Agent-agnostic real-world scientific analysis and open-ended work | Use to measure domain unevenness and novel-insight limitations; do not substitute narrow coding success |
| AstaBench | Stanford AI Index summarizes over 2,400 problems and a best agent score around 0.53 at roughly $3.40/problem in its cited 2025 evaluation. [AI Index science chapter](https://hai.stanford.edu/assets/files/ai_index_report_2026_chapter_5_science.pdf) | Broad end-to-end scientific workflow capability | Broad coverage and cost-per-problem anchor; align to the exact benchmark release before comparing |
| OpenScience claims | OpenScience reports 75.7% Terminal-Bench-Science and other leading results, but its page also notes evaluation issues and planned reruns after comparison/scoring problems. [Benchmark page](https://www.openscience.sh/benchmark) | Vendor-reported system results | Treat as a hypothesis to reproduce, not a settled leaderboard fact |

### Comparison discipline

- Never compare scores unless task set, task version, harness, model access, tool access, time budget, attempt count, grader, and aggregation method match.
- Keep official benchmark submissions separate from local reproductions and vendor-reported numbers.
- Report confidence intervals and per-task outcomes. ASB 0.1 is a useful warning: three trials per task are not enough to interpret small leaderboard gaps.
- For cost claims, include all inference, tool, orchestration, verification, and human-review costs. Report both cost per attempt and cost per successful task.
- Preserve failed runs and incomplete artifacts. Do not silently remove timeouts, invalid outputs, or verifier failures.

## Paradigm shifts to pursue

### 1. From answer generation to evidence-carrying decisions

Every consequential research action should bind to:
- a versioned question and objective;
- the current evidence-prefix digest;
- a complete permitted-action offer and payload digest;
- a prospective prediction;
- explicit resource bounds and stopping conditions;
- an execution receipt;
- observations with event/availability time and provenance;
- an independently adjudicated outcome.

This turns a research trace into a replayable decision object. The scientific-decision replay work in PR #13933 is a prototype contract, not yet a production-isolated execution system.

### 2. From confidence to calibrated, abstention-aware judgment

Accuracy alone does not tell us whether a system knows when it is wrong. Track Brier score, calibration error, valid adjudication sample size, and risk-versus-coverage. Missing labels must remain missing; model confidence must never be promoted to correctness. PR #13969 implements the first bounded step: adjudication-only Brier and ten-bin ECE.

### 3. From one-shot experiments to causal, budget-aware active learning

Select experiments by expected information gain or expected decision value under explicit resource constraints. Compare against random, greedy uncertainty sampling, fixed design, and domain-expert baselines. Use matched initial states and equal realized budgets. Record failed experiments and negative results. Do not optimize a proxy such as novelty or confidence in isolation.

### 4. From static benchmark wins to transfer under distribution shift

Use held-out task families, domains, instruments, and data-generating processes. Measure both average transfer and worst-group regressions. RecurSciBench-style repeated starts and ASI-Bench's guidance gradient are complementary: one tests transfer, the other tests method reconstruction.

### 5. From a monolithic agent to independently checkable roles

Separate proposer, executor, evidence custodian, evaluator, and policy updater. The proposer must not see sealed outcomes; the evaluator must not inherit the proposer’s confidence as truth; the policy updater consumes only adjudicated evidence. Process/OS isolation and sealed evaluation remain future work, not properties established by typed Rust structures alone.

## Proposed Symthaea target ladder

These are **research targets**, not achieved results. Freeze benchmark versions and baselines before claiming progress.

| Workstream | Near-term target | Stretch target | Required controls |
|---|---:|---:|---|
| Terminal-Bench-Science | ≥70% on a matched 70-task evaluation | ≥75% | Same harness/model/tool budget as comparator; ≥5 trials/task where stochastic; cost/time and failure taxonomy |
| ASI-Bench | ≥65 overall, no B-level collapse >10 points | ≥70 overall; B4 within 8 points of B1 | Report B1–B4 separately; distractor and method-reconstruction controls |
| SCI-AGENT-EXAM | ≥83 overall and ≥35% runs >0.95 | ≥86 and ≥50% runs >0.95 | Preserve task-type/domain slices and trajectory stability |
| SciAgentArena | +5 absolute points over matched baseline on ≥3/5 selected task groups | +10 points on ≥3/5 with no >5-point regression elsewhere | Frozen task subsets, independent grading, domain-level regressions |
| ASB-style laboratory campaigns | ≥15/18 resolved on a defined 6-task × 3-trial pilot | ≥90% on a larger confirmatory suite | The 18-run pilot is plumbing only; expand to ≥10 trials/task before superiority claims |
| Calibration | Brier and ECE reported with n; improve over uncalibrated baseline | Brier skill >0 on held-out domains; ECE ≤0.05 at adequate n | Reliability diagram, 5/10/15/20-bin sensitivity, confidence intervals, source/time stratification |
| Selective prediction | At 80% coverage, reduce error relative to always-answer baseline | 50% lower risk at 80% coverage without pathological abstention | Risk-coverage curve, AURC, coverage, abstention reasons |
| Reproducibility | 100% successful claims linked to immutable artifacts and replay inputs | Independent clean-room replay of ≥99% of eligible runs | Negative controls, artifact hashes, environment lock, no hidden mutable state |
| Transfer | Non-negative mean transfer; ≤20% held-out tasks regress | ≥5% mean gain; ≤10% regressions | Matched starts, equal budgets, held-out task families, report worst group |
| Cost efficiency | Report cost per verified success | ≥25% lower cost at matched quality | Include model, tools, compute, retries, evaluator, and human review |

Targets should be revised only after a baseline run and preregistration. A high target is not evidence that the architecture can achieve it.

## Immediate execution sequence

1. **Close the calibration loop:** PR #13969 now has a passing focused Rust 1.96 gate (strict rustfmt, 20/20 standalone production-source tests, strict Clippy) at exact head `37e67653b22001dd4b6a97302c1527a1337ae3cb`. This does not yet validate the entire web-research runtime or prove independent adjudicator quality.
2. **Complete decision custody:** integrate the proposed offer/context digest contracts with the upstream AgentView owner; then add prefix-only candidate replay and process-level separation.
3. **Make time semantics explicit:** asynchronous experiments need event time, observation/availability time, and decision time. Simulation tick alone is not enough for delayed or censored evidence.
4. **Add selective-risk reporting:** calibration is not sufficient if the agent cannot abstain safely. Evaluate risk-coverage and decision utility.
5. **Run a small matched AS-Bench campaign:** pin API/schema, fixtures, exact design IDs, cost/resource ledger and independent outcome adjudication; do not call a simulated replay a laboratory campaign.
6. **Run R4 matched-start evaluation:** same starting states and realized budgets, independent evaluator, held-out transfer tasks, preserve all failures.
7. **Only then make SOTA claims:** public reproducibility package, frozen benchmark commit, exact run manifests, and independent rerun.

## What would constitute a genuine field-level advance?

A credible claim would be: *at equal or lower total cost, Symthaea produces more independently validated scientific progress, with lower unsupported-claim rate, better calibrated uncertainty, more reliable abstention, and stronger out-of-domain transfer than matched frontier-agent baselines.*

A leaderboard score alone cannot establish that. The evidence package must include the complete action-to-observation chain, negative controls, independent labels, and clean-room replay.

## References

- [Terminal-Bench-Science official leaderboard](https://snorkel.ai/leaderboard/terminal-bench-science/)
- [Vals AI Terminal-Bench-Science results](https://vals-ai.com/benchmarks/terminal-bench-science)
- [ASI-Bench](https://asibench.apexin.ai/)
- [Autonomous Science Bench leaderboard](https://asb.open-ssi.org/leaderboard)
- [Autonomous Science Bench methodology](https://asb.open-ssi.org/about)
- [SCI-AGENT-EXAM](https://arxiv.org/abs/2610.32441)
- [SciAgentArena](https://sciagentarena.github.io/)
- [Stanford AI Index 2026, science agents](https://hai.stanford.edu/assets/files/ai_index_report_2026_chapter_5_science.pdf)
- [OpenScience benchmark and its evaluation caveat](https://www.openscience.sh/benchmark)
- [Uncertainty-calibration metrics and risk-coverage discussion](https://link.springer.com/article/10.1007/s44163-026-02240-w)
