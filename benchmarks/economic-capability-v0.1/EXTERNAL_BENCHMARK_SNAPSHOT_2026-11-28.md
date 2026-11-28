# External Benchmark Snapshot — 2026-11-28

**Status:** dated literature/leaderboard and evaluator-quality crosswalk. These are external results, not Luminous reproductions. The measures below are not a common scale. Task set, release, harness, model configuration, action/time/token budget, grader and cost boundary must be pinned before comparison.

## Professional knowledge work: GDPval

GDPval v1 covers 44 occupations across nine major sectors and includes 1,320 specialized work-product tasks; its open gold subset contains 220 tasks. The tasks were created with experienced professionals (average 14 years of domain experience). The initial evaluation uses expert pairwise preference as its primary measure, with an experimental automated grader on the public gold subset.

- Official benchmark overview: https://evals.openai.com/
- Paper and ICLR 2026 proceedings: https://proceedings.iclr.cc/paper_files/paper/2026/hash/290c2430f91912204f30bbcc990fff1d-Abstract-Conference.html
- OpenAI methodology/limitations: https://openai.com/index/gdpval/
- Snapshot: accessed 2026-11-28.
- Interpretation: GDPval is a useful professional-deliverable comparator. Its first version is primarily self-contained knowledge work, not physical or manual work, deep tacit-context work, extensive interpersonal workflows, or a full production deployment. It does not measure all human time spent gathering context, revising, integrating, reviewing and accepting work.
- Claim boundary: expert preference is not proof of objective correctness, field deployment, net employment effect, or end-to-end operating economics.

## Computer use: OSWorld 2.0 and 2.1

The OSWorld 2.0 benchmark covers 108 long-horizon workflows on 31 self-hosted websites. The project reports a median human task time of about 1.6 hours and an average 318 tool calls for one evaluated Claude Opus 4.7 configuration. Its paper's primary binary-completion metric at a 500-step budget reports Claude Opus 4.8 at **20.6% full task completion** and **54.8% partial score**. The official site reports a steep decline as task duration increases, with severe hidden-state, multi-item tracking, source-conflict and dynamic-environment failures.

- Official benchmark and research: https://osworld-v2.xlang.ai/
- Paper: https://arxiv.org/abs/2606.29537
- Snapshot: accessed 2026-11-28.
- Interpretation: full task completion, partial reward, number of checkpoints, and time/action budgets are different metrics. Do not substitute partial score for binary end-to-end acceptance. Measure postcondition truth, intervention, recovery and cost alongside both.

**OSWorld 2.1 evaluator-quality warning.** Epoch AI reviewed a stratified sample of 50 of the 108 tasks in the September 16, 2026 v2.1 release. It identified grading/authoring errors in 36 tasks, including 11 with whole-task score impact and 25 with partial-score impact. Epoch estimates that the errors could affect up to 50.4% of the maximum score in that audited sample; this is a potential grading-impact upper bound, not an observed correction to every model score. Its review found both false negatives (correct work not credited) and false positives (incorrect work credited), and rated the release Flawed.

- Independent evaluator audit: https://epoch.ai/benchmarks/osworld-2-1/review
- Interpretation: do not use v2.1 leaderboard movement as unquestioned ground truth. Pin the release and task IDs, independently audit graders, retain task-level traces and perform blind regrades before making precise model-ranking claims.

## Software engineering agents: SWE-Bench Pro V2

Scale/Reflection's September 22, 2026 SWE-Bench Pro V2 update has 642 public tasks across 11 repositories, reduced from 731 after 89 invalid tasks were removed. It corrected instructions for 69 tasks that contradicted their tests, using blind expert resolution; disabled web access during the agent phase; regraded diffs on pristine images; and added a two-sided gate requiring the reference patch to pass and the empty patch to fail. Its review caught concrete benchmark-gaming attempts and even a regression in its own reference patch.

- Official release and methodology: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2
- Snapshot: accessed 2026-11-28; update dated 2026-09-22.
- The official description says leading systems score around 23% on its public set versus 70%+ on SWE-Bench Verified. These figures are context from different benchmark suites, not a direct matched comparison.
- Interpretation for CEB: copy the strongest methods—expert task repair, blind solve checks, reference/empty controls, restricted network, pristine regrade, shipped trajectories and task-level grade evidence—rather than trying to maximize task count alone.
- Claim boundary: its own authors disclose residual trust/execution risks in verifier operation; a two-sided gate reduces, but does not eliminate, the need for independent task and grader assurance.

## Long-horizon robotics: LADDER

LADDER comprises 145 tasks across three suites, eight categories, two embodiments, one to ten subgoals and 2,024 simulated episodes per system. Its research page reports that an un-orchestrated low-level policy drops from 85.4% success on one-subgoal tasks to 0% on five-subgoal tasks. Orchestration improves results, but the study still identifies physical execution as a limiting factor.

The current project leaderboard at this snapshot displays:
- Gemini 4 Carbon Lite via GPT-Policy: **71.5% tasks completed**, **$1.73 per completed task**;
- Claude Fable 5.2 Lite via GPT-Policy: **66.0%**, **$7.85 per completed task**;
- GPT-6 Astra via GPT-Policy: **63.3%**, **$6.49 per completed task**.

- Research/methodology: https://ladderrobotics.ai/research
- Current leaderboard: https://ladderrobotics.ai/
- Snapshot: accessed 2026-11-28.
- Interpretation: compare by suite, embodiment, policy, horizon and simulation/real setting. Report task completion, subgoal completion, cost, execution failures, intervention, recovery and transfer. API cost per benchmark task is not industrial total cost of ownership.
- Claim boundary: simulated success is not field reliability or safety qualification.

## Financial research agents: Finance Agent v2.1

Vals AI's official page, updated November 19, 2026, reports:
- Claude Fable 5.2: **69.47% partial-credit accuracy**, **$4.83 per test**;
- Gemini 4 Neon: **68.56%**, **$5.85 per test**;
- Claude Opus 5.5: **67.85%**, **$8.91 per test**;
- Inkling 2 Small: **67.54%**, **$0.18 per test**.

Finance Agent v2.1 is a proprietary question set and a different version from v1.1/v2; its official page says edition scores are not directly comparable. The score is a research-agent accuracy measure, not a trading or portfolio-return result.

- Official benchmark and leaderboard/methodology: https://www.vals.ai/benchmarks/fabv2-1
- Snapshot: accessed 2026-11-28; board states updated November 19, 2026.
- Interpretation: the narrow reported accuracy gap between the leading and much lower-cost entry makes cost/quality frontier analysis important. Also track all-pass rate, material false claims, source admissibility, latency, token/tool use and cost per independently accepted research task.
- Claim boundary: research QA does not qualify point-in-time backtesting, accounting reconstruction, investable alpha, investment suitability or financial authority.

## Engineering computation: verification, validation and uncertainty

Engineering-tool comparisons should be based on known-answer and physical validation evidence—not model-generated explanations, a successful process exit, or parser success alone.

- ASME V&V 10-2019 (computational solid mechanics): https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-solid-mechanics
- ASME VVUQ overview: https://www.asme.org/codes-standards/verification-validation-uncertainty
- NAFEMS finite-element benchmarks: https://www.nafems.org/publications/resource_center/r0032/
- OpenSees examples: https://opensees.berkeley.edu/wiki/index.php?title=Examples
- Snapshot: accessed 2026-11-28.

Recommended evidence ladder:
1. parser/format tests and byte-level provenance;
2. analytic known-answer examples (bar, truss, beam, frame);
3. OpenSees reference examples and independently checked response quantities;
4. NAFEMS benchmark cases under declared tolerances;
5. solution verification (mesh/time-step/convergence/residuals);
6. validation against suitable physical observations with measurement uncertainty;
7. independent design-code and safety review where required.

Keep code verification, solution verification, validation and uncertainty quantification distinct. The parser slice in PR #14252 is only parser groundwork: it does not establish convergence, model validity, structural safety or design adequacy.

## What these comparisons imply for Luminous

1. **Benchmark evaluators, not only agents.** A task score is conditional on task validity, reference quality, grader reliability, environment integrity and exact accounting. CEB's #10138/#10370/#10167/#10212 should be composed into every Luminous benchmark programme.
2. **Compare the incremental architecture.** Hold base model, tools, task, data, context, permissions, action budget and grader constant; ablate Symthaea's durable task state, source reconciliation, causal experiment selection and postcondition verification separately.
3. **Target failure modes that dominate long-horizon tasks.** Hidden state, late information, multi-item completeness, conflict resolution, partial execution, recovery and postcondition verification are more promising than optimizing a single-turn answer score.
4. **Count safe holds as a distinct successful disposition.** Do not reward a system for executing a plausible action when the correct outcome is clarification, no action or human authorization.
5. **Use cost per independently accepted work unit.** Include human setup/review/recovery, compute, energy, connectivity, capital, maintenance, rework, downtime and failed work. Unknown cost stays unknown.
6. **Keep independent hard gates.** Unauthorized effects, false acceptance, unsafe physical behavior, privacy/rights violations or missing lineage cannot be offset by throughput or partial-credit score.
7. **Run local transfer studies.** Global benchmarks do not establish South African operating economics, regional service reliability, local labour outcomes, language coverage or public-service sufficiency.

## Prospective Luminous benchmark goals (not achieved results)

These are hypotheses to preregister, not industry standards or existing Luminous results:

- **Digital workflows:** at least +5 percentage points in binary end-to-end completion versus a matched base-model/tool baseline, or at least 20% lower fully burdened cost per independently accepted task at non-inferior quality; independently verify every credited postcondition.
- **Long-horizon reliability:** at least 20% fewer stale-state, source-conflict, wrong-postcondition and duplicate-effect errors than the matched baseline; publish performance by sequence depth and recovery event count.
- **Economic value:** at least 10% improvement in one preregistered business endpoint versus matched incumbent workflow, with no critical quality, safety, privacy, rights or authority regression.
- **Robotics:** 10% more independently accepted output per available robot-hour, 20% fewer human intervention/recovery minutes per accepted unit, and 20% less recoverable unplanned downtime, with no increased critical safety events.
- **Finance research:** use Finance Agent v2.1 as a version-specific comparator where access permits; target improved cost per accepted task and material-claim reliability, not accuracy alone. Do not infer alpha.
- **Engineering:** 100% expected dispositions on frozen parser fixtures; then analytic and selected NAFEMS cases within preregistered tolerances, followed by separate validation/uncertainty work.
- **Quant strategy:** zero temporal leakage and exact portfolio reconciliation on the synthetic oracle before any historical return result; then lawful point-in-time data, full costs, capacity, PBO/CSCV, DSR and untouched confirmation periods.

## Source and version discipline

For every public comparison retain the exact benchmark release, URL/archive, access date, task universe/split, model/provider/harness, tool permissions, reasoning/action/time/token budget, grader version, cost boundary and known limitations. Distinguish provider-reported, benchmark-owner-reported, independently audited and independently reproduced evidence. Preserve every task-level result and every exclusion.
