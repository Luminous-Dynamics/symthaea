# External Benchmark Snapshot — 2026-11-28

**Status:** dated literature/leaderboard crosswalk. These are external results, not Luminous reproductions. The scores measure different task sets, budgets, embodiments, tools and judging rules; they must not be averaged or presented as one ranking. Live leaderboards can change, so archive the exact page and configuration used for any public comparison.

## Professional work: GDPval

GDPval evaluates economically valuable professional work across 44 occupations and nine sectors. The published dataset comprises 1,320 specialized tasks, with a 220-task open subset; task authors were experienced professionals, averaging 14 years of domain experience. The benchmark's core measure is human-expert preference on work products, not a fully deployed workflow's economics. Its initial format is substantially one-shot, so it does not fully measure iterative review, source updates, application postconditions, implementation, or long-running collaboration.

- Official overview: https://evals.openai.com/
- Technical paper/proceedings: https://openreview.net/forum?id=GDPval
- OpenAI benchmark description: https://openai.com/index/gdpval/
- Snapshot: accessed 2026-11-28.
- Interpretation: use GDPval-style deliverable tasks to compare research, engineering, legal/financial analysis and other professional artifacts. Add source/claim traceability, revision handling, independent acceptance, human review/rework time, cost and regional context.
- Claim boundary: a GDPval preference result is not proof of production deployment, engineering correctness, physical work, labour-market impact, or net employment change.

## Computer use and long-horizon application workflows: OSWorld 2.1

The OSWorld v2 project describes 108 long-horizon tasks across 31 self-hosted websites; 69.6% of tasks take more than one hour, with more than 250 average agent steps and 27.25 average scoring checkpoints. The project emphasizes authentic input artifacts and cross-application state. Its challenge analyses include cross-source reasoning, visual-spatial precision, implicit-state inference, multi-item state tracking, conflict disambiguation, multimodal editing, dynamic environments, streaming and proactive interaction.

The official leaderboard snapshot for the v2.1 harness displayed:
- Claude Opus 5, Max Batch tool v2.1: **44.33% binary end-to-end accuracy** (77.67% partial score);
- Muse Spark 1.3, Max Batch tool v2.1: **39.25% binary accuracy** (69.56% partial score);
- Claude Opus 5, High Batch tool v2.1: **36.89% binary accuracy** (68.92% partial score).

- Official benchmark and leaderboard: https://osworld-v2.xlang.ai/
- Official repository/releases: https://github.com/xlang-ai/OSWorld
- Snapshot: accessed 2026-11-28; leaderboard configuration identified as v2.1.
- Interpretation: a task is not complete merely because an agent reports success. OSWorld-like evaluations should be paired with independently checked terminal application state, safety/side-effect auditing, recovery, operator intervention and total cost. The task set's long horizons make durable state, conflict resolution and postcondition verification high-value engineering targets.
- Claim boundary: binary accuracy, partial score and task completion are distinct. Pin OSWorld release, task subset, computer/tool stack, model settings, budget and grader before comparing. The figures are not general-purpose capability estimates.

## Long-horizon robotics: LADDER

LADDER describes 145 tasks in three suites (54 SimplerEnv-Long, 47 RobotArena-∞-Long and 44 RoboLab-Long), with one to ten subgoals, eight categories, two embodiments and 2,024 simulated episodes per system. Its published analysis shows that direct low-level policy execution degrades sharply as instruction horizon increases; orchestration improves results, but physical execution and verifiable completion remain important bottlenecks.

The live official leaderboard at this snapshot displayed:
- Gemini 4 Carbon Lite via GPT-Policy: **71.5% completed**, **$1.73 per completed task**;
- Claude Fable 5.2 Lite via GPT-Policy: **66.0%**, **$7.85 per completed task**;
- GPT-6 Astra via GPT-Policy: **63.3%**, **$6.49 per completed task**.

- Research/methodology: https://ladderrobotics.ai/research
- Current leaderboard: https://ladderrobotics.ai/
- Snapshot: accessed 2026-11-28.
- Interpretation: report both success and cost. For Luminous, separate planner/orchestrator quality from controller reliability, perception/localization, grasp/manipulation success, recovery and independent inspection. Measure accepted units per available robot-hour, intervention minutes, downtime, wear, energy, maintenance and transfer from simulation to real operating conditions.
- Claim boundary: per-task leaderboard cost is not full industrial total cost of ownership. Do not equate simulated task completion with physical qualification or industrial reliability.

## Financial research agents: Finance Agent Benchmark v2.1

The official Vals AI v2.1 leaderboard snapshot (accessed 2026-11-28; displayed evaluation update 2026-11-19) listed:
- Claude Fable 5.2: **69.47% partial-credit accuracy**, **$4.83/test**;
- Gemini 4 Neon: **68.56%**, **$5.85/test**;
- Claude Opus 5.5: **67.85%**, **$8.91/test**;
- Inkling 2 Small: **67.54%**, **$0.18/test**.

The reported harness allows up to two hours per test, uses six finance/research tools and scores through a three-judge LLM jury with deal-breaker gates. v2.1 is a distinct question set from v2 and v1.1; scores across editions are not comparable.

- Official leaderboard/method notes: https://vals.ai/benchmarks/finance-agent
- Benchmark crosswalk: https://benchmarks.finance/leaderboards/vals-finance-agent/
- Snapshot: accessed 2026-11-28.
- Interpretation: the narrow accuracy gap between the leading system and a much lower-cost entry suggests an efficiency-frontier question, not simply a race to maximize score. Track all-pass rate, unsupported-claim/deal-breaker rate, source correctness, latency, token/tool use and cost per accepted task alongside partial-credit accuracy.
- Claim boundary: financial research QA is not point-in-time backtesting, alpha, investment return, suitability, accounting correctness or permission to execute trades. Quant strategy evaluation must remain separately gated by #11234, the shared temporal kernel #11253 and canonical portfolio oracle #11298.

## Engineering simulation: OpenSees, ASME V&V and NAFEMS

Engineering software should be compared on verification/validation evidence, not only model-generated explanations or successful process exit.

- OpenSees official examples and verification material: https://opensees.berkeley.edu/wiki/index.php?title=Examples
- ASME V&V 10-2019 (reaffirmed 2025): computational solid mechanics verification and validation; overview: https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-solid-mechanics
- ASME VVUQ standards overview, including V&V 20 for CFD/heat transfer and uncertainty quantification: https://www.asme.org/codes-standards/verification-validation-uncertainty
- NAFEMS standard finite-element benchmarks: https://www.nafems.org/publications/resource_center/r0032/
- Snapshot: accessed 2026-11-28.
- Recommended qualification ladder:
  1. parser/format tests and byte-level provenance;
  2. analytic known-answer problems (bar, truss, beam, frame);
  3. OpenSees official examples with independently checked response quantities;
  4. NAFEMS standard benchmark cases and declared numerical tolerances;
  5. solution verification (mesh/time-step/convergence and residuals);
  6. validation against appropriate physical observations, including measurement uncertainty;
  7. independent design-code/safety review where the intended claim requires it.
- Distinguish code verification (is the implementation solving its stated equations?), solution verification (is this numerical solution sufficiently resolved?), validation (does the model adequately represent reality for the intended use?), and uncertainty quantification. A parsed recorder file is only an early evidence step; it does not establish convergence, model validity, structural safety or design adequacy.
- Claim boundary: a bridge parser is not an engineering qualification. The current OpenSees recorder parser slice (#14248) is parser-only groundwork; real-solver output, independent golden models, provenance and V&V remain pending.

## What these comparisons imply for Luminous

1. **Do not compete on a single blended score.** Professional deliverable preference, desktop binary task completion, robotics subgoal completion, finance partial-credit accuracy and engineering V&V are different constructs.
2. **Measure incremental system contribution.** Compare the same base model and tool environment with and without Symthaea orchestration, memory, evidence verification or planning. Match task, context, data, tool permissions, reasoning budget, action budget, retry policy and grader.
3. **Make hidden state explicit.** Use an event-sourced task ledger, versioned evidence graph and typed postconditions so the workflow can reconstruct what it knew, what changed, which sources conflicted, and why a task was accepted, held, corrected or rolled back.
4. **Treat abstention and clarification as first-class outcomes.** When evidence is stale, contradictory, incomplete or outside operating support, measure whether the system asks a useful question, narrows scope, safely holds, or escalates rather than hallucinating completion.
5. **Report cost per independently accepted work unit.** Include setup, prompting, integration, review, supervision, rescue, rework, compute, energy, capital, downtime, maintenance, material, connectivity and failed attempts. Unknown cost remains unknown.
6. **Keep hard gates non-compensable.** Unauthorized effects, false acceptance, safety failure, privacy/rights violations or missing lineage cannot be offset by higher throughput or a better model score.
7. **Separate financial research from financial action.** Research quality and strategy profitability need different datasets, oracles, holdouts and authority boundaries; Mycelix remains the execution/authority owner.
8. **Use local transfer tests.** A global benchmark does not prove South African operating economics, language coverage, infrastructure reliability, labour outcomes, or site-level safety.

## Prospective Luminous benchmark goals (not achieved results)

These are hypotheses to preregister and test, not industry standards or claims of existing capability:

- **Digital workflows:** beat a matched base-model/tool baseline by at least **5 percentage points** in binary end-to-end completion, or reduce fully burdened cost per accepted task by at least **20%** at non-inferior quality; independently verify **100%** of credited terminal postconditions and lineage.
- **Long-horizon reliability:** reduce state-tracking, source-conflict and wrong-postcondition errors by **20%** versus the matched baseline; report failure/recovery curves by task horizon and never drop timeouts or escalations.
- **Economic value:** target at least **10%** improvement in a preregistered primary business endpoint against a matched incumbent workflow, with no critical safety, quality, privacy or rights regression; treat human intervention reduction as secondary and preserve meaningful oversight.
- **Robotics:** target **10%** more independently accepted output per available robot-hour, **20%** fewer human intervention/recovery minutes per accepted unit, and **20%** less recoverable unplanned downtime, with no increase in critical safety events and complete lineage.
- **Finance research:** target **70%+** v2.1-equivalent partial-credit accuracy only if the exact benchmark access, dataset and judging setup are available; separately aim to reduce cost per accepted research task, improve all-pass rate and drive material unsupported claims to zero. Never infer alpha from this score.
- **Engineering:** **100%** expected parser disposition on frozen valid/invalid fixtures; exact provenance for binary, environment, model, input, raw output and parser; all analytic and selected NAFEMS known-answer cases within preregistered tolerances before any validation or design-qualification claim.
- **Quant strategy:** require **zero** temporal leakage and exact portfolio-transition reconciliation on the frozen synthetic corpus before any historical strategy result; use #11241/#11253 temporal semantics and #11298 accounting oracle. No return target is meaningful until lawful point-in-time data, full execution costs, capacity, PBO/CSCV, DSR and untouched confirmation periods exist.

## Source and version discipline

For every public comparison, retain the exact benchmark release, URL/archive, access date, dataset/task split, model/provider/harness version, tool stack, reasoning/action/time/token budget, grader, score definition, cost boundary and reported limitations. Keep provider-reported, benchmark-owner-reported and independently reproduced results as distinct evidence classes.
