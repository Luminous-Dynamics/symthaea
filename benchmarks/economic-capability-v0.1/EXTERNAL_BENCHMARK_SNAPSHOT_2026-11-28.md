# External Benchmark Snapshot — 2026-11-28

**Status:** literature/leaderboard crosswalk only. These are external results, not Luminous reproductions. They are not directly comparable to each other because task sets, budgets, tool stacks, evaluation methods and score definitions differ.

## Professional work: GDPval

OpenAI describes GDPval as an evaluation of economically valuable professional work covering 44 occupations across nine sectors contributing to US GDP. Tasks were constructed from representative work by industry professionals averaging 14 years of experience. The public evaluation write-up states that frontier models are approaching expert deliverable quality in this task set, while additional context, reasoning effort and scaffolding improve results.

- Source: https://evals.openai.com/
- Benchmark discussion: https://openai.com/index/gdpval/
- Interpretation: use GDPval-style deliverable tasks as one professional-work comparator, but extend the measurement with accepted outputs, full human review time, costs, jurisdiction and deployment constraints. GDPval is not a physical-work, labour-market or net-employment benchmark.
- Result boundary: this snapshot records the official task scope, not an independent head-to-head score or an asserted universal model accuracy.

## Computer use: OSWorld 2.0

The official OSWorld 2.0 site describes 108 long-horizon tasks, 31 self-hosted websites, 69.6% of tasks taking more than one hour, more than 250 average agent steps and 27.25 average scoring checkpoints.

At the time of this snapshot, its official leaderboard displayed:
- Claude Opus 5 (Max Batch tool v2.1): 44.33% binary end-to-end accuracy;
- Muse Spark 1.3 (Max Batch tool v2.1): 39.25%;
- Claude Opus 5 (High Batch tool v2.1): 36.89%.

The leaderboard includes additional entries and modes. Any replication must pin the exact release, model, reasoning setting, tool harness, offline/full subset and step budget.

- Official research/leaderboard: https://osworld-v2.xlang.ai/
- Snapshot: accessed 2026-11-28.
- Interpretation: full workflow completion is still a difficult measure, and partial credit should not be represented as full task completion. The economic capability benchmark should add human intervention, total cost, task quality and postcondition truth.
- Claim boundary: the figures are official leaderboard entries for this benchmark configuration, not an estimate of the general computer-use capability of the named models.

## Long-horizon robotics: LADDER

LADDER's research page describes 145 tasks across three suites, one to ten subgoals, eight task categories, two embodiments and 2,024 simulated episodes per system. It highlights the degradation of un-orchestrated low-level policies over longer horizons and the benefits/limits of orchestration and verification.

The live LADDER leaderboard at this snapshot displayed:
- Gemini 4 Carbon Lite via GPT-Policy: 71.5% tasks completed, $1.73 per completed task;
- Claude Fable 5.2 Lite via GPT-Policy: 66.0%, $7.85 per completed task;
- GPT-6 Astra via GPT-Policy: 63.3%, $6.49 per completed task.

- Research and detailed methodology: https://ladderrobotics.ai/research
- Current leaderboard: https://ladderrobotics.ai/
- Snapshot: accessed 2026-11-28.
- Interpretation: cost per completed task is important alongside success. Long-horizon failures and physical execution remain central. Match suite, embodiment, task horizon, verifier and simulation/real setting before comparison.
- Claim boundary: these are published benchmark results, not Luminous results and not a guarantee of performance on different embodiments or industrial tasks.

## Financial research agents

The Finance Agent Benchmark v2.1 leaderboard crosswalk lists a top reported result of 67.8 partial-credit accuracy for Claude Opus 5.5, dated 2026-10-29. It separately identifies v1.1 and v2.1 as different question sets whose scores are not comparable.

- Leaderboard crosswalk: https://benchmarks.finance/leaderboards/vals-finance-agent/
- Snapshot: accessed 2026-11-28; displayed result date 2026-10-29.
- Interpretation: use this only as a version-specific research-agent comparator. It is not a backtest, alpha, investment return, suitability assessment or live execution benchmark.

## What these comparisons imply for Luminous

1. **Do not compete on one blended score.** GDPval deliverable quality, OSWorld binary workflow completion, LADDER task success/cost and financial-agent partial-credit accuracy are different quantities.
2. **Compare on matched cases and budgets.** Use the same base model, task, context, tools, step/token/time budget and acceptance oracle when measuring the incremental contribution of Symthaea or Symtropy.
3. **Track the complete economic envelope.** Include human setup, supervision, correction, exception handling, integration, maintenance, compute, energy, capital and failed work.
4. **Keep independent safety and evidence gates.** A score increase cannot compensate for unauthorized effects, unsafe physical behavior, unsupported claims or incomplete provenance.
5. **Target cost-adjusted accepted work.** A plausible research target is a >=5 percentage-point improvement in end-to-end acceptance on a matched cohort, or a >=20% reduction in cost per accepted task at non-inferior quality/safety. These are experimental hypotheses, not industry standards or promised outcomes. They require preregistration, adequate sample size, uncertainty intervals and independent evaluation.
6. **Robotics should optimize accepted productive work per available robot-hour**, including first-pass yield, human rescue, downtime, energy, wear, maintenance and inspection—not just action success or simulator reward.
7. **Financial strategy results require their own point-in-time and statistical controls.** Never infer investment skill from professional finance QA performance.

## Source and version discipline

Before any public benchmark claim, archive the exact leaderboard page/source snapshot, release identifier, model/provider/harness version, date, score definition, evaluation budget, task split and any known limitations. Preserve provider-reported, benchmark-owner-reported and independently reproduced results as distinct evidence classes.
