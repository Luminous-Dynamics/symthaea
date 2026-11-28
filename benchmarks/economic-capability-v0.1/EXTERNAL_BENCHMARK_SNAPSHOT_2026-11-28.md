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

A public leaderboard mirror updated November 27, 2026 lists **81.8% partial score** for Claude Opus 5.5, 80.7% for Claude Fable 5.1 and 80.1% for Claude Sonnet 5.5. The mirror marks its displayed entries as lab-reported rather than independently rerun. These are v2.1 partial-score snapshots; they are neither binary full-task completion nor a corrected score after the Epoch audit.

- Public leaderboard mirror and provenance caveat: https://benchlm.ai/benchmarks/osworld-2-1
- Snapshot: accessed 2026-11-28.

## Software engineering agents: SWE-Bench Pro V2

Scale/Reflection's September 22, 2026 SWE-Bench Pro V2 update has 642 public tasks across 11 repositories, reduced from 731 after 89 invalid tasks were removed. It corrected instructions for 69 tasks that contradicted their tests, using blind expert resolution; disabled web access during the agent phase; regraded diffs on pristine images; and added a two-sided gate requiring the reference patch to pass and the empty patch to fail. Its review caught concrete benchmark-gaming attempts and even a regression in its own reference patch.

- Official release and methodology: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2
- Snapshot: accessed 2026-11-28; update dated 2026-09-22.
- **Do not mistake the page's historical ~23% paragraph for the current V2 leaderboard.** That paragraph reports GPT-5 at 23.3% and Claude Opus 4.1 at 23.1% on an earlier SWE-Bench Pro evaluation using the SWE-Agent scaffold. It is historical context, not the current V2 frontier.
- Current Scale V2 comparison panel separates **HARD** and **Full**:
  - HARD view: Claude Opus 5 / Claude Code / xhigh **99.40% ±0.40**; Gemini 4 Argon / mini-swe-agent / high **99.22% ±0.45**; Claude Fable 5.1 / Claude Code / high **99.10% ±0.50**.
  - Full view: Claude Opus 5 / Claude Code / xhigh **98.00%**; Claude Fable 5.1 / Claude Code / high **92.20%**; Gemini 4 Argon / mini-swe-agent / high **92.16%**; GPT-6 Astra / Codex / high **90.20%**.
  - The published panel notes that greyed entries use capped cost and a 50-turn limit; other displayed systems use uncapped cost and up to 250 turns. It marks the mini-swe-agent harness separately. Model, harness, split and budget must be held constant; the 98.00% Claude Code result is not a model-only comparator for a mini-swe-agent run.
- Interpretation for CEB: copy the strongest methods—expert task repair, blind solve checks, reference/empty controls, restricted network, pristine regrade, shipped trajectories and task-level grade evidence—rather than trying to maximize task count alone. Because the public V2 scores are near ceiling, use the exact same-harness full/hard sets for reproducibility, but make long-horizon, multi-repository, repair-with-regression, and novel held-out tasks the primary differentiation frontier.
- Claim boundary: its own authors disclose residual trust/execution risks in verifier operation; a two-sided gate reduces, but does not eliminate, the need for independent task and grader assurance.

## Long-horizon robotics: LADDER

LADDER comprises 145 tasks across three suites, eight categories, two embodiments, one to ten subgoals and 2,024 simulated episodes per system. Its research page reports that an un-orchestrated low-level policy drops from 85.4% success on one-subgoal tasks to 0% on five-subgoal tasks. Orchestration improves results, but the study still identifies physical execution as a limiting factor.

The current project leaderboard separates the full 145-task suite from **LADDER-Lite**, a 10-task quick-check subset:
- Full LADDER (145 tasks across three suites): GPT-6 Astra via GPT-Policy reports **62.0% pooled overall success (±2.1 pp)** and **63.3% equally weighted suite mean**; API cost is **$6.49 per successful episode**. The project's headline leaderboard uses the suite-mean metric.
- LADDER-Lite (10 tasks): Gemini 4 Carbon Lite via GPT-Policy reports **71.5%**, **$1.73 per completed task**; Claude Fable 5.2 Lite reports **66.0%**, **$7.85 per completed task**.

The 71.5% / $1.73 figure is therefore a Lite-subset result, not the full-suite leader. Keep the two denominators separate in every comparison; use the 145-task suite as the primary challenge and Lite only as a fast development diagnostic.

- Research/methodology and separate pooled/suite-mean metrics: https://ladderrobotics.ai/research
- Project leaderboard and LADDER-Lite distinction: https://ladderrobotics.ai/
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

## Enterprise IT operations and SRE agents

The newly relevant benchmark family is not a single leaderboard: detection, root-cause localization, diagnosis, safe mitigation, actual service recovery, IT service-management completion and human capability are separate constructs.

### ITBench / ITBench-AA

- IBM's ITBench initial release describes 94 scenarios across SRE, CISO and FinOps. Its paper reports the best-performing agent at 13.8% SRE resolution, 25.2% CISO resolution and 0% FinOps on that initial release. Those are release- and harness-specific resolution metrics, not the same as ITBench-AA's SRE-only diagnostic metric.
- Artificial Analysis' ITBench-AA evaluates **59 Kubernetes root-cause tasks** (40 public plus 19 private tasks shared by IBM), with three repeats per task. It scores average precision at full recall, so the score penalizes false-positive root-cause entity lists after requiring no false negatives. Its current displayed table lists GPT-5.6 Sol Max at **56.2%**, Step 5 Preview at **55.6%**, and Gemini 3.8 Flash High at **52.5%**.
- Sources: https://github.com/itbench-hub/ITBench ; https://artificialanalysis.ai/evaluations/itbench-aa ; methodology/news: https://artificialanalysis.ai/articles/itbench-aa-launch
- Snapshot: accessed 2026-11-28.
- Interpretation: ITBench-AA is diagnosis/entity-attribution, not successful mitigation, recovery, rollback or operator independence. Do not call 56.2% a service restoration rate.

### SWEOps-Bench

SWEOps-Bench (paper dated 2026-11-04) contains **55 reproducible incident tasks** from **16 Kubernetes operator projects**. It starts each task from a healthy cluster, triggers a confirmed historical software fault under controlled workload, and asks agents to investigate runtime evidence and submit source/configuration repair. It reports stage-wise accuracy, verified recovery, lifecycle latency and tokens.

Across six agent/model configurations, the paper reports:
- faulty-state detection: **72.7%–90.9%**;
- end-to-end verified recovery: **14.5%–52.7%**.

That gap is especially important: noticing an incident is much easier than actually repairing it and passing the required post-repair checks. The paper also notes that a correct recovery can occur with an incorrect reported root-cause narrative, while clearing runtime symptoms can still fail fix-specific functional checks.

- Paper: https://arxiv.org/abs/2611.06099
- Snapshot: accessed 2026-11-28; paper dated November 4, 2026.
- Interpretation: this is the closest technical comparison for code/configuration incident recovery. Keep root-cause narration, actual repair, service recovery and fix-specific functional verification as separate endpoints.

### ITSMBench (Atomicwork / New Measure) and Enterprise Worlds

Atomicwork/New Measure's open ITSMBench evaluates **89 L2/L3 service-management tasks** in a simulated enterprise of 42 mocked software applications, about 1,800 database tables and 2,000+ REST endpoints. Each task has 10–30 expert-authored deterministic verifiers over final environment state. The task principles are useful: critical information is distributed across tools; safe action requires corroboration across systems; and passing checks include both the correct judgment and the complete, safe footprint of the change.

- Benchmark description: https://simulation.ressl.ai/benchmarks/atomicwork/itbench/blog
- Repository: https://github.com/new-measure/ITSMBench
- Snapshot: accessed 2026-11-28.
- Interpretation: use this as a service-request/workflow comparison, particularly for identity, endpoint, network, security and compliance operations. Its mocked APIs and scoped 89-task taskset are not equivalent to live enterprise platforms or deep SRE recovery.

Name collision warning: Vibrant Labs' Enterprise Worlds also publishes an ITSMBench line; its current site describes a **separate 53-task** executable environment with 900+ records, 16 entity types and 93 typed tools. Do not merge that 53-task release with Atomicwork/New Measure's 89-task benchmark or quote their scores interchangeably.

- Enterprise Worlds current description: https://enterpriseworlds.vibrantlabs.com/
- Research/methodology note: https://research.vibrantlabs.com/projects/itsmbench

### AI SRE Arena v1 (Edge Delta)

The October 2026 AI SRE Arena report describes **21 injected Kubernetes incidents**, two native products and two read-only control configurations, with 72 judged investigations. Its public report states:
- native detection: Edge Delta 18/21 versus Grafana 12/21;
- on the 12 incidents both investigated: root cause 11/12 versus 9/12, supported fix 7/12 versus 5/12;
- the published table lists unsafe advice on 0/18 Edge Delta investigations versus 2/12 Grafana investigations.

The rubric v1 page identifies one GPT-6-Astra judge and one published answer key. This is useful open prior art, but one judge/answer-key lineage and a small scenario set require independent replication before strong comparative claims.

- Official report / code / answer keys: https://edgedelta.com/arena
- Snapshot: accessed 2026-11-28; report published October 2026.

### PagerDuty SRE Agent benchmark

PagerDuty's 2026 report describes a self-built Kubernetes microservice fault-injection suite. Its latest complete sweep covers **17 scenarios** and compares localization, diagnosis, mitigation, evidence grounding, explanation quality and misdirection. It uses three LLM judges from different providers. The report discloses material limitations: a single trial per agent, failed runs excluded from quality scores, an incomplete human/judge statistical study, and focus primarily on triage/diagnosis with only a smaller remediation component.

The company reports that the benchmark helped uncover and remedy an over-engineered investigation design, leading to roughly a 20-point diagnosis-score improvement while reducing latency/cost. This is vendor-reported development evidence, not an independent universal result.

- Official report: https://www.pagerduty.com/resources/ai/learn/sre-agent-benchmark-report/
- Snapshot: accessed 2026-11-28.

### What ITOPS-CAP-001 should add

Mycelix [ITOPS-CAP-001 #6846](https://github.com/Luminous-Dynamics/mycelix/issues/6846), under [CIV-IMPACT-001 #5046](https://github.com/Luminous-Dynamics/mycelix/issues/5046), should compare against these existing suites instead of claiming a new SRE or ITSM benchmark category. The differentiator should be the **composed qualification and economic/human outcome**:
1. separate detect/localize/diagnose/propose/authorize/attempt/restore/functional-verify/reconcile states;
2. least-privilege authority and exact target/environment/effect binding;
3. no-op, wrong-target, symptom-twin, stale-approval and safe-abstention controls;
4. independently checked postconditions and fix-specific functional invariants;
5. interruption/retry/unknown-outcome/rollback and recurrence behavior;
6. fully loaded cost per independently accepted recovery;
7. operator capability dividend/debt: unaided baseline, assisted performance, immediate AI-off challenge, delayed retention, novel-incident transfer and outage recovery.

For the Luminous synthetic development phase, #6846's proposed construction target is 40–60 scenarios across at least eight incident families, with scenario-family holdouts and repeated stochastic-agent runs. Keep the sample-size/cluster structure explicit: repeats improve variance estimates but do not replace independent incident families. The three OPS-01/02/03 scenarios authored in the pilot blueprint are anchor patterns only, not an admitted or complete task population.

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

## Task-matched SOTA challenge goals — prospective, not Luminous results

The following are stretch goals only when the exact external task set, release, model/tool environment, scoring rubric, resource budget and grader can be reproduced or licensed. Otherwise use the cited figures as context and run a separate matched comparison; never claim to have beaten an incomparable result.

| Comparator | Reported reference in this snapshot | Stretch challenge for an exact reproduction | Hard interpretation boundary |
|---|---|---|---|
| OSWorld 2.0 full workflow | 20.6% binary end-to-end completion at the reported 500-step budget; 54.8% partial score | At least **25.6% binary completion** at the same 500-step budget, with no deterioration in grader-audited correctness or action safety | Do not compare partial score to binary completion; OSWorld 2.1 grading audit is an independent evaluator warning, not a corrected score |
| SWE-Bench Pro V2 (Full / HARD) | Full view: 98.00% Claude Code; 92.16% best displayed mini-swe-agent entry. HARD view: 99.40% Claude Code; 99.22% mini-swe-agent. Earlier ~23% prose is a historical scaffold result, not current V2 | Primary architecture target: **≥3 pp absolute gain over matched A0** on the same unsaturated task stratum or preregistered held-out set, at equal harness/turn/cost budgets; separately reproduce public Full and HARD scores | Near-ceiling public scores and mixed native/mini-swe-agent harnesses limit attribution; pin the split, harness, cost cap, turn limit, image and grader. Public leaderboard gains alone do not prove a Symthaea contribution |
| ITBench-AA diagnosis | 56.2% average precision at full recall on its 59 Kubernetes root-cause tasks | At least **61.2%** on the identical task set and metric, without reducing the required recall | This is root-cause entity attribution—not remediation, restoration or safe action |
| SWEOps-Bench verified recovery | 52.7% highest reported end-to-end recovery across six configurations; detection 72.7–90.9% | At least **60% independently verified recovery** on the same 55 incident tasks, while reporting detection and repair separately | Requires the same operator repositories, injected faults, workload, source repair process, functional checks and environment—not a new easier SRE suite |
| LADDER full suite (145 tasks) | 63.3% equally weighted suite mean; 62.0% pooled overall (±2.1 pp); $6.49 API cost per successful episode for GPT-6 Astra via GPT-Policy | At least **68.3% suite mean** at no more than $6.49 per successful episode on the same 145 tasks, suites and embodiment; also report pooled overall and per-suite intervals | Primary long-horizon comparator; suite mean and pooled overall are distinct; simulation is not field reliability or safety qualification; API cost is not industrial TCO |
| LADDER-Lite (10 tasks) | 71.5% completion; $1.73 API/tool cost per completed task for Gemini 4 Carbon Lite via GPT-Policy | At least **76.5%** at no more than $1.73 on the identical 10-task subset | Development-only quick diagnostic; must not be substituted for the 145-task full-suite result |
| Finance Agent v2.1 | 69.47% partial-credit accuracy at $4.83/test for the leading listed entry | At least **74.5% partial-credit accuracy** at no more than $4.83/test on the identical v2.1 set and harness | Does not qualify accounting truth, point-in-time investment decisions, alpha, return or authority |

These thresholds are deliberately expressed as challenge hypotheses, not minimum acceptance criteria for unrelated deployments. A team should first reproduce the reference entry and confirm its confidence interval, cost denominator, task exclusions and evaluator version. A one-off point estimate above a published leaderboard entry is not persuasive evidence of superiority.

For the broader Luminous architecture, the primary claim should remain: improvement over the **same base model and tools** on matched tasks, with the increment attributable to Symthaea's state, reconciliation, planning and verification modules. External leaderboards measure system configurations, not necessarily those components individually.

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
