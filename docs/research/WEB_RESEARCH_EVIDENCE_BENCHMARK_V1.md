# WebResearch Evidence Benchmark v1 — Retrieval Is Not Support

**Status:** benchmark protocol and source audit; not an executed evaluation  
**Fail-closed implementation:** [PR #13790](https://github.com/Luminous-Dynamics/symthaea/pull/13790)  
**Claim/source relation contract:** [issue #13796](https://github.com/Luminous-Dynamics/symthaea/issues/13796)  
**Independent outcome/calibration:** [issue #13791](https://github.com/Luminous-Dynamics/symthaea/issues/13791)  
**Claim-specific graph provenance:** [issue #13792](https://github.com/Luminous-Dynamics/symthaea/issues/13792)  
**Retrieval completeness/concurrency:** [issue #13794](https://github.com/Luminous-Dynamics/symthaea/issues/13794)

## 1. Research thesis

A deep-research system is not trustworthy merely because it finds relevant pages, quotes them, or writes a coherent report.

The pipeline must preserve four different propositions:

1. **Retrieval relevance:** this artifact is relevant to the user's question.
2. **Claim-source relation:** this particular passage supports, refutes, mentions, or fails to establish this particular claim.
3. **Claim truth:** the claim is supported by the total admissible evidence under an explicit decision policy.
4. **Decision usability:** the evidence is sufficient for a particular decision, with a defined risk threshold and date/scope boundary.

These must not be collapsed into one confidence number. Retrieval similarity, source reputation, report quality and claim entailment are distinct variables.

PR #13790 addresses a concrete violation: the current WebResearcher assigned `supports: Some(true)` to every fetched page and reused the aggregate source list for every extracted claim. Since the verifier promotes three flagged supporting entries to `HighConfidence`, topical page retrieval could become high-confidence support without claim-level adjudication. The PR changes those implicit positives to unclassified/neutral and adds negative-control tests. It deliberately does not pretend that keyword overlap or source reputation is an entailment model.

## 2. External benchmark landscape

The following are reference points from their benchmark owners or published sources. Their tasks, scorers, judges, budgets and denominators differ; they must remain separate.

| Benchmark | Public scope / result | What it measures | Comparability caveat |
|---|---|---|---|
| DeepResearch Bench II (2026) | Original task release: 132 expert-authored tasks / 9,430 rubrics / 22 domains. Unified portal: GPT-5.5 text-only evaluation lists 130 scored tasks; Gemini 2.5 evaluation lists 132 | Retrieval coverage, synthesis/analysis and verifiable presentation | The original USTC page says the leaderboard migrated and judge switched to GPT-5.5. Unified portal says its new-judge data is still being populated; displayed AI21 64.38 is a legacy snapshot, not a verified current score |
| DeepResearch Bench I (ICLR 2026) | RACE report quality dimensions and FACT citation correctness/effective-citation dimensions | Report completeness, reasoning depth, instruction following, readability, citation accuracy and useful citations | Historical task/model/judge snapshot; use the official task version and judge for reproduction |
| OpenScience benchmark report (Sep 2026; update 1 Oct) | Reports 75.7% on 70 Terminal-Bench Science tasks, 82.2 on 50 BiomniBench-DA tasks, 71.4% on 14 Terminal-Bench 4.0 science tasks, and 47.3% pass@3 on its internal OpenScience Bench | Long-horizon science workflows, data analysis, tool use and end-to-end task completion | The authors explicitly say their setup had comparison/scoring issues, engineering performance was weaker, and they planned to improve the harness and rerun. Different rows use different trial counts; do not treat the self-reported results as an independently settled SOTA ranking |
| AstaBench / AI Index 2026 | 2,400+ problems, 57 agents / 22 agent classes; best reported overall score around 0.53 at about $3.40/problem | Broad scientific work from literature to code, data analysis and discovery | Breadth/cost reference, not a direct citation-entailment or closed-loop lab-resolution score |
| ScienceAgentBench (2025) | 102 tasks from 44 peer-reviewed publications; tasks validated by nine subject-matter experts | Data-driven scientific discovery with expert-validated tasks | Its public page says leaderboard updates are paused while reliability is studied; preserve its native task/scorer and do not imply current live rankings |
| Autonomous Science Bench 0.1 (2026) | Six real laboratory workflows; public snapshot leader 72.2% resolution | Full design–build–test–learn execution, measurements and agent/harness performance | This is live-lab capability, not merely source-grounded literature synthesis |

### Primary references

- DeepResearch Bench II methodology and leaderboard note: https://agentresearchlab.com/benchmarks/deepresearch-bench-ii/
- Unified DeepResearch Bench leaderboard portal (judge/version migration notice): https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard
- DeepResearch Bench I paper/results: https://deepresearch-bench.github.io/
- OpenScience benchmark and its October 1 comparability caveat: https://www.openscience.sh/benchmark
- AI Index 2026 AstaBench summary: https://hai.stanford.edu/assets/files/ai_index_report_2026_chapter_5_science.pdf
- ScienceAgentBench: https://hal.cs.princeton.edu/scienceagentbench
- Autonomous Science Bench: https://asb.open-ssi.org/about
- FrontierPhysics paper and current monitored scoreboard: https://arxiv.org/abs/2611.06910 and https://www.mypriors.com/benchmarks/frontierphysics

## 2A. November 2026 research-agent benchmark refresh

The broader field has moved beyond report generation and code execution. The additional sources below identify the missing capabilities this project should measure; they are complementary tracks, not a single cross-benchmark ranking.

| Benchmark / system | Public snapshot | Why it changes the research plan | Comparability boundary |
|---|---|---|---|
| ASI-Bench | 60 project-level tasks across 11 scientific domains; its 10 Nov 2026 leaderboard snapshot lists GPT-6.1 Sol at 63.90 and GPT-6 Astra at 62.23 | Tests end-to-end project research under B1–B4 guidance levels, execution and validation; use as a cross-domain project baseline | Freeze the exact seed/task set, track and harness; this score is not comparable with DRB II or lab resolution |
| TruthInsightBench | 40 blind tasks from 40 peer-reviewed studies across 10 domains; four coding-agent configurations on one frozen model were reported in a 58.4–60.3/100 range without reliable pairwise separation | Directly tests whether an agent can determine what the data justify, not merely reproduce a known answer; its six evidence-maturity dimensions and 29 artifact-grounded items are highly relevant to claim integrity | The automated LLM judge is fixed and should be calibrated with human/expert audits; the paper reports no evidence that fluent output alone constitutes discovery |
| DISCERN | 203 tasks across eight life-science tracks: 79 L1 data-integrity, 102 L2 analysis-verification, 22 L3 hypothesis dossiers. Current reported leader GPT-6 Astra: 78.1 Overall (L1 86.1, L2 76.8, L3 72.8); Opus 5 leads L2 at 81.7 | Evaluates data integrity, analysis verification under confounds/tool traps, and hypothesis evidence-fidelity/reasoning/novelty under three adversarial review rounds | L3 prior eligibility is model-specific; full coverage and human adjudication required. Physics extension is separate from the core score. A reproducibility artifact is public, but live evaluation requires fixed reviewers/judge, sandbox and upstream data/runtime checks |
| LifeSciBench | 750 expert-authored tasks across seven workflows and seven domains, 1,062 task artifacts; 173 scientist contributors and 453 expert reviewers. GPT-Rosalind: 36.1% pass / 57.6 mean rubric; GPT-5.5: 25.7% / 51.9 | Artifact-grounded biomedical reasoning: data interpretation, analysis, design/optimization and multi-step scientific workflows | Its 70%-rubric pass threshold differs from mean normalized rubric score. Artifact-heavy tasks materially reduce performance; text-only and artifact tracks must be reported separately. It is an offline task benchmark, not live longitudinal discovery |
| FrontierChallenge | 97 released scientific workflows from a 300-task pool; best reported complete-pass result 20/97 (20.6%). In 75.5% of non-passing Claude Code trajectories, the agent still claimed completion | Measures full task completion and artifact delivery; makes false completion claims and partial-credit inflation visible | Fixed-spec workflow execution, not autonomous research-agenda generation. Keep full-pass rate, partial progress and completion-claim calibration separate; pin task release and harness |
| SCI-AGENT-EXAM | 120 expert-curated tasks across 11 scientific fields and seven task types; GPT-6 Astra reported 80.9 overall, but only 26% of runs exceeded the 0.95 success threshold | Jointly evaluates executable artifacts, textual scientific outputs and trajectory stability rather than treating one correct-looking report as sufficient | Use the paper's task split, deterministic code checks and calibrated expert-validated judge; do not equate aggregate score with stable end-to-end success |
| Terminal-Bench Science | 70 long-horizon science terminal tasks, three trials each; official public leaderboard snapshot lists GPT-6 Astra at 68.1%. OpenResearch reports GPT-6.2 Astra 77.1% and Mythos 5.2 75.3% as vendor-reported results not yet on the official leaderboard | Scientific terminal workflows with task-specific verifiers and up to eight-hour task limits | Keep official leaderboard and vendor-reported submissions separate; use the same harness, trial count and task version before claiming a lead |
| DiscoveryAgent | A scientist-steerable agent with checkpointed audit records; its paper reports 14× more valid errors found when scientists collaborate with audit agents than during manual review | Supports a new interaction model: scientists can select/refine/replace hypotheses between rounds, while independent auditors trace claims and reproduce analyses from frozen checkpoints | Published results cover two predictive-modeling tasks and three expert-led case studies; the 14× figure is paper-reported, not a universal rate |
| AutoSciBench | Generated tasks reduced mean solver accuracy by 22.4 percentage points in computational biology and 25.5 points in materials science relative to the starting human-curated benchmarks | Demonstrates a practical route to adversarially harden task recipes as agents learn shortcuts; benchmark maintenance itself becomes an adaptive process | Generator/judge quality and leakage controls still need independent audit; never let the candidate see the hidden recipe or gold answer |
| benchmarks.bio | V1 snapshot updated 25 Nov 2026: 13 capability benchmarks and 23 separate leaderboards; top listed aggregate configuration is Claude Opus 5.5 with Pi at 58.0% and $4.82/run | Offers a focused life-science execution/cost comparison beyond one general scientific-agent score | V1 scoring is explicitly not directly comparable with historical V0; the aggregate omits some missing scores and is domain-specific |
| Autonomous Science Bench 0.1 | Six live-lab workflow tasks; current public pilot board lists GPT-6 Astra/Codex at 72.2%, $190.23 and 557 agent minutes | Remains the physical execution track: campaign design, measurement feedback and final held-out ranking must all succeed | Six tasks are a pilot; totals are for the configuration, not cost/time per task |
| FrontierPhysics | 56 end-to-end physics tasks from 42 physicists across nine domains; current monitored board snapshot lists Pass@1 up to 33.3% and Pass@3 up to 44.6% | Measures complete physics research: literature, planning, computational/experimental work and a manuscript, using deterministic verifiers plus mandatory rubric items | Keep Pass@1, Pass@3, verifier-only pass and rubric score separate; never mix leaderboard/harness snapshots |

Primary sources:
- ASI-Bench task set and leaderboard snapshot: https://asibench.apexin.ai/
- FrontierChallenge paper / benchmark: https://arxiv.org/abs/2608.24979
- SCI-AGENT-EXAM paper / benchmark: https://arxiv.org/abs/2610.32441
- TruthInsightBench paper and task/scoring code: https://arxiv.org/abs/2609.05079 and https://github.com/TruthInsight-stack/TruthInsightBench
- DISCERN paper, frozen task/adjudication release and model-evaluation guide: https://arxiv.org/abs/2609.33357, https://huggingface.co/datasets/discern-bench-anon/discern-benchmark, https://huggingface.co/datasets/discern-bench-anon/discern-benchmark/blob/main/EVALUATE_MODEL.md
- LifeSciBench paper and benchmark announcement: https://openai.com/index/introducing-life-sci-bench/ and https://www.biorxiv.org/content/10.1101/2026.06.11.658742v1
- Terminal-Bench Science current public board / benchmark sync: https://www.openresearch.sh/evals
- DiscoveryAgent (16 Nov 2026): https://arxiv.org/abs/2611.21950
- AutoSciBench: https://arxiv.org/abs/2610.05140
- benchmarks.bio V1 leaderboard/methodology: https://benchmarks.bio/
- Autonomous Science Bench live leaderboard: https://asb.open-ssi.org/leaderboard
- DISCERN evidence-to-hypothesis paper and immutable reproducibility release: https://arxiv.org/abs/2609.33357 and https://huggingface.co/datasets/discern-bench-anon/discern-benchmark
- Terminal-Bench Science official/public results aggregation: https://www.openresearch.sh/evals

### Completion-calibration targets for the new workflow benchmarks

These are proposed Symthaea stretch targets, not observed performance. They are meaningful only under the exact same pinned task release, model, tool permissions, judge/verifier versions, trial count and budget envelope as the reference run:

- **FrontierChallenge:** at least 35% full completion on the released 97-task set (versus the reported 20.6% reference), while reducing unsupported completion claims to zero whenever any required artifact/check is missing. Report partial-credit progress separately; it cannot substitute for full completion.
- **SCI-AGENT-EXAM:** at least 85 final, at least 90 programmatic artifact correctness, and at least 45% Pass@0.95 across the same three-run/task protocol, with report quality, replay and trajectory stability reported independently.
- **Both:** 100% denominator accounting for refusals, timeouts, missing artifacts and non-evaluable runs; publish full artifact and cost receipts. A higher mean score accompanied by worse completion calibration or unstable replay is not an improvement.

The two benchmarks complement DISCERN rather than replace it: FrontierChallenge emphasizes complete execution of specified workflows; SCI-AGENT-EXAM emphasizes scientific executable/textual outputs and stability; DISCERN separately tests data integrity, methodological response and evidence-grounded hypothesis revision. Do not average these scores into one synthetic leaderboard.

### Paradigm shift: evidence-checkpointed, scientist-steerable research

Treat a research episode as an append-only sequence of typed checkpoints, not a long opaque agent transcript. A checkpoint should freeze the current hypothesis set, scope, evidence-prefix digest, analysis code/data/configuration, predicted outcomes, resource spend, unresolved objections and next-action offers. At a checkpoint boundary, a human scientist may select, refine or replace a hypothesis; the system must preserve the previous branch and explain what evidence motivated the change.

A separate audit agent receives only the checkpointed candidate-visible record and authorized artifacts. It should reproduce selected analyses, trace each reported claim to exact inputs/spans, search for counterexamples and controls, and emit an independent objection list. It must not edit candidate evidence, supply hidden labels, or self-authorize promotion. The candidate then responds to the objections in a new checkpoint. This creates a measurable adversarial loop rather than treating self-review as independent verification.

Measure:
- valid errors found per audit-hour and per unit cost;
- fraction of auditor objections reproduced by independent replay;
- correction rate and regression rate after audit;
- claim-to-artifact trace completeness;
- scientist steering latency and frequency of meaningful hypothesis changes;
- outcome quality with and without human steering and with/without the audit agent.

### Paradigm shift: self-hardening benchmark, not self-scoring agent

Use AutoSciBench-style task recipes to generate *new sealed variants* from vetted scientific concepts and data-generating procedures. An external evaluator may use observed failure modes to harden the next benchmark version, but candidate agents must never see the generator's hidden parameters, judge rubric, or per-task gold. Keep a frozen public anchor set for comparability and a rotating sealed challenge set for saturation resistance. Human experts audit generated task validity and distribution drift before promotion. Report anchor and challenge scores separately; never merge them into a single untraceable composite.

## Artifact-native research follow-up

The WebResearcher content extractor remains HTML-oriented; its numeric-series helper is line-based and explicitly does not parse units. The acquisition receipt is an integrity record for captured content, not a typed scientific-artifact parser or a stable table/cell/page locator. [Issue #undefined](https://github.com/Luminous-Dynamics/symthaea/issues/undefined) now owns the staged artifact-intake contract: immutable raw bytes, parser/configuration receipts, typed source locations, and explicit unsupported/malformed outcomes. It composes with #13854 acquisition, #12471 method execution and #13796 claim/source relation adjudication; it does not introduce a second planner or evidence ledger.

## 3. Benchmark architecture

Each evaluated run freezes:
- user task and task-family ID;
- search plan and tool/version manifest;
- source acquisition budget and allowed domains;
- all attempted queries, URLs and acquisition outcomes;
- captured source bytes, final URL, timestamp and SHA-256 digest;
- extraction/parser version and exact claim text;
- claim/source relation proposals and accepted adjudications;
- source correction/retraction state and common-origin grouping;
- final synthesis, citation map, uncertainty/abstention and downstream decision;
- system/model/runtime/tool digests and total resource use.

Candidate-visible and evaluator-only data remain separate. The candidate may see authorized retrieved sources and its own extracted claims, but must not see hidden gold claims, reference labels, adjudication keys or evaluator scores during a confirmatory run.

### Four evaluation layers

**A. Acquisition quality**

Did the system find the relevant evidence, and did it truthfully report failures?

Measure:
- source recall@k against expert-curated relevant evidence;
- unique-source coverage after URL/content canonicalization;
- freshness and version correctness;
- successful capture rate and content-digest reproducibility;
- search/provider/page/extraction failure visibility;
- duplicate/common-origin detection;
- acquisition cost, latency and byte budget.

**B. Claim-source relation quality**

For every pair ((claim, source passage)), label one of:
- `SUPPORTS`;
- `REFUTES`;
- `MENTIONS_NO_ENTAILMENT`;
- `INSUFFICIENT_CONTEXT`;
- `SCOPE_MISMATCH`;
- `UNRESOLVED`.

Every accepted relation binds the exact claim ID/text digest, evidence/artifact ID and digest, quote/span offsets or page/section location, source version, classifier/adjudicator identity and reason codes.

Measure:
- support precision and recall;
- refutation precision and recall;
- false-support rate (primary safety metric);
- false-refutation rate;
- scope-mismatch detection;
- quote/span fidelity and evidence-byte binding;
- relation calibration and selective risk;
- common-origin/citation-independence handling.

A source mention is not a support relation. A support relation is not global claim truth.

**C. Synthesis and reporting**

Measure using benchmark-native scorers:
- information recall;
- analysis quality / insight correctness;
- presentation/readability;
- effective citations and citation accuracy;
- contradiction handling;
- coverage of uncertainty and unresolved claims;
- whether every conclusion can be reconstructed from its cited evidence.

**D. Evidence-state and outcome integrity**

Measure:
- stale/misbound/corrupted/unauthorized source rejection;
- correction/retraction propagation;
- no state mutation when an evidence relation is rejected;
- source-specific provenance of learned reliability;
- calibrated confidence after independently adjudicated outcomes;
- reproducible accepted/rejected transition records.

**E. Evidence-to-decision response and scientific revision**

A system must not receive full credit merely for naming a limitation, confound, or reviewer objection. It must carry that recognition into the next analysis, action, or claim revision.

For every detected limitation or valid objection, bind:
- the exact evidence prefix and affected claim/analysis;
- the limitation/confound and the uncertainty it introduces;
- the prospective method change, control, replication, exclusion, or abstention rule;
- the result that would reverse or relax the limitation;
- the next action's predicted outcomes and the decision each could change;
- post-action replay, outcome adjudication, and accepted/rejected objection disposition.

Measure:
- clean-control acceptance and defective-input detection separately (misplaced caution is a failure in either direction);
- issue recognition versus actual methodological response;
- result calibration after a recognized limitation;
- rate of invalid conclusions prevented by a method change;
- counterfactual responsiveness to evidence that conflicts with an elicited prior;
- hypothesis commitment, justified abstention, conditional novelty, and novelty yield separately;
- adversarial-objection precision, independent reproduction, correction quality, and regression after revision;
- discriminating value of proposed next experiments against plausible alternatives.

A critique concession is not itself improvement. A revision earns credit only when it becomes more faithful to evidence or gives a justified reason to preserve the original claim. A recognized issue that has no downstream effect must remain visible as an unresolved failure, not be counted as solved.

An LLM judge may grade report quality, but it must not be the sole oracle for source entailment or evidence integrity.

## 4. Required negative controls

The benchmark must include hard examples designed to catch the current failure mode:

1. **Topical non-entailment:** pages discuss the topic but do not answer the claim.
2. **One source, opposite claim:** same page contains both supporting and refuting contexts, including negation.
3. **Global-source leakage:** three high-relevance pages are supplied for one query, but none supports an extracted claim.
4. **Cross-claim contamination:** a source supports claim A but is incorrectly offered as support for unrelated claim B.
5. **Scope mismatch:** wrong population, date, unit, jurisdiction, software version or effect magnitude.
6. **Quote laundering:** a source quotes another author or reports a hypothesis without endorsing it.
7. **Duplicate / common origin:** mirrors and syndications must not count as independent corroboration.
8. **Correction / retraction:** older source content must not remain current support after a correction.
9. **Prompt injection / boilerplate:** web content tries to override the agent or poison evidence extraction.
10. **Retrieval failures:** timeouts, HTTP errors, unsupported media, truncated content and extractor failure must remain visible as failed acquisition, not as evidence absence.
11. **Confidence self-label:** a high confidence prediction with no independent outcome label must not count as correct.
12. **No-fit / unresolved:** the true claim relation or source is outside the current candidate set; the system must abstain or request new evidence.

## 5. Matched baselines

Compare under identical source sets, claim sets, budgets and adjudication information:

| Baseline | Role |
|---|---|
| Naive all-sources-support | Frozen negative control reproducing the unsafe current behavior |
| Query/source lexical similarity | Retrieval relevance baseline; must never be treated as entailment |
| Top-k relevance only | Tests whether high-quality retrieval alone is sufficient |
| Existing verifier with explicit relation labels | Upper bound conditional on correct relation labels |
| Candidate relation classifier | Model under test |
| Independent second classifier / reviewer | Disagreement and adjudication path |
| Human domain-expert labels | Gold adjudication on a stratified subset |
| Hidden evaluator gold | Exact scorer only; never candidate-visible |

The all-support baseline should fail the false-support controls. It is not an acceptable production policy even if aggregate report completeness is high.

## 6. Proposed targets

These are project targets, not achieved results or external standards.

| Measure | Initial promotion target |
|---|---:|
| False-support rate on topical non-entailment / cross-claim controls | 0 accepted false-support relations |
| Evidence-span binding | 100% accepted relations bind exact captured artifact and span |
| Source identity | 100% duplicate/common-origin groups reported |
| Scope mismatch | ≥95% recall on frozen mismatch controls, with precision reported |
| Relation calibration | Report Brier score / reliability curve; no confidence-only correctness updates |
| Contradiction handling | 100% preservation of opposing accepted relations; no majority-vote collapse |
| Correction/retraction propagation | 100% invalidation/supersession of affected current evidence links |
| Acquisition transparency | 100% attempted query/fetch outcomes accounted for |
| External DRB II | Reproduce native scores first; then target ≥70% total rubric pass on the same frozen task/judge version, reporting InfoRecall, Analysis and Presentation separately |
| ASI-Bench | Stretch target ≥70/100 on the exact 60-task release and same B1–B4 guidance protocol; obtain/pin the exact scored task-set ID before a leaderboard comparison |
| TruthInsightBench | Stretch target ≥70/100 on the same 40 blind tasks; publish all six evidence-maturity dimensions, artifact-grounded items and task-clustered confidence intervals |
| DISCERN | Stretch target ≥82 Overall, L1 ≥90, L2 ≥85 and L3 ≥80 on the same pinned 203-task core, with full track coverage and original prior-probe/reviewer/judge protocol |
| LifeSciBench | Stretch target ≥45% pass rate and ≥62 mean normalized rubric score on the same 750-task version, while reporting artifact-heavy/text-only strata and each workflow separately; do not optimize rubric average at the expense of false claims |
| Terminal-Bench Science | Stretch target ≥80% official-harness resolution on the same 70 tasks and trial protocol; report official and vendor-reported results separately |
| Autonomous Science Bench 0.1 | Stretch target ≥80% full-system resolution on the same six pilot workflows, with total campaign cost and agent time each no higher than the current published leader; this is a pilot-sized target, not a general-lab claim |
| FrontierPhysics | Stretch target ≥40% Pass@1 and ≥55% Pass@3 on the same frozen task/harness version; report verifier-only pass and weighted rubric score separately |
| Scientist–auditor collaboration | Demonstrate positive paired gain in valid errors found per human-review hour over manual-only review; require independent replay and expert-adjudicated precision/recall (see [#13818](https://github.com/Luminous-Dynamics/symthaea/issues/13818)) |
| Evidence-to-decision response | ≥90% of expert-adjudicated valid limitations result in an appropriate method/action/claim change; ≥90% of clean controls accepted; report false alarms, missed defects and unjustified revisions separately |
| Adversarial hypothesis revision | ≥85% evidence fidelity after three review rounds, with separate novelty/commitment/abstention and revision-vs-no-revision paired analysis |
| Citation integrity | Target ≥95% citation accuracy on a frozen gold subset; effective-citation count alone is not sufficient |
| SCI-AGENT-EXAM | Reproduce the 120-task, 11-field, 7-task-type release in its common harness; frontier reference is 80.9 final (77.7 programmatic, 85.3 judge) with 26.0% Pass@0.95. Stretch: ≥85 final, ≥90 programmatic and ≥45% Pass@0.95 over the same three-run/task protocol; report stability, time and cost separately |
| FrontierChallenge | Reproduce the 97 released workflows and all deliverable checks; best reported configurations complete 20/97 (20.6%) despite high partial scores. Stretch: ≥35% full completion on the identical release, and zero unsupported completion declarations when any required artifact/check is missing |
| Resource quality | Report cost and latency Pareto fronts; no unqualified composite score |

SCI-AGENT-EXAM adds a strict reliability view that the other benchmarks do not fully substitute for: GPT-6 Astra is reported at 80.9 overall (77.7 programmatic artifact score; 85.3 judge score), but only 26.0% of runs reach the 0.95 success threshold. The benchmark uses 120 tasks across 11 scientific fields and seven task types, three runs per task, and a common Claude Code harness. Model and scaffold effects are therefore entangled; preserve the official harness for reproduction, then publish a fixed-base-model comparison with Symthaea as the only changed system component. FrontierChallenge evaluates 97 released workflows from a 300-task design. The best reported configurations complete only 20/97 (20.6%). Some disciplines have high average partial scores with very low full-completion rates, and 75.5% of non-passing Claude Code trajectories in the reported analysis still state that they completed the task. This is why complete deliverable validation and truthful completion claims must be separate gates; partial credit cannot be averaged into a passing outcome.

LifeSciBench's reported artifact-heavy pass rate is 28.1% for GPT-Rosalind, versus 45.1% on text-only tasks; numeric tasks are 14.8%, sequence/structure 24.0%, design/optimization 30.7%, and analysis 30.3%. Do not collapse these strata into a single result. ASI-Bench advertises a public 60-task seed42 catalog, but its leaderboard methodology says the scored task-set identifier was not supplied; require that exact identifier or label a public-catalog run exploratory. DISCERN's public reproducibility release is not a one-command arbitrary-model leaderboard: live runs need task-specific prior probes, three fixed reviewers, a separate judge, human adjudication, and the upstream sandbox/runtime requirements. Its L2 results specifically show that recognizing an issue is not enough: 29.6% of full-recognition challenged evaluations receive only half result-calibration credit. L3 revision must be judged by evidence-fidelity change, not reviewer-concession frequency. Its physics extension must stay outside the core Overall score. Terminal-Bench Science vendor-reported 77.1%/75.3% entries are not yet official leaderboard results. FrontierPhysics targets are stretch targets above current monitored scores and must be evaluated on the exact same task/harness snapshot. A ≥70% DeepResearch Bench II target is only meaningful on the same frozen task set and judge version. The page currently states the judge changed to GPT-5.5 and the leaderboard migrated, so do not compare an old-page 64.38 directly with a new-jury score without a controlled bridge evaluation.

## 7. Composition with existing Symthaea work

- [PR #13842](https://github.com/Luminous-Dynamics/symthaea/pull/13842): source relations remain neutral until explicit adjudication.
- [Issue #13796](https://github.com/Luminous-Dynamics/symthaea/issues/13796): provenance-bound claim/source adjudicator and benchmark.
- [Issue #13791](https://github.com/Luminous-Dynamics/symthaea/issues/13791): independent outcome labels; no confidence-as-correctness loop.
- [Issue #13792](https://github.com/Luminous-Dynamics/symthaea/issues/13792): claim-specific source graph edges and evidence versions.
- [Issue #13794](https://github.com/Luminous-Dynamics/symthaea/issues/13794): complete acquisition receipts and bounded concurrency.
- [Issue #13800](https://github.com/Luminous-Dynamics/symthaea/issues/13800): prevent public convenience constructors from minting unprovenanced `HighConfidence` claims.
- [Issue #13818](https://github.com/Luminous-Dynamics/symthaea/issues/13818): scientist-steerable, checkpointed research with independently replayable audit agents.
- [PR #11015](https://github.com/Luminous-Dynamics/symthaea/pull/11015): evidence-prefix-bound prospective orchestration traces; the 2026-11-26 extension adds limitation-to-method-response obligations and lifecycle fixtures.
- [PR #13854](https://github.com/Luminous-Dynamics/symthaea/pull/13854): bounded, receipt-bearing WebResearch acquisition stacked on the fail-closed verifier and no-confidence-self-label fixes; no retry or authenticity claim.
- [Issue #13852](https://github.com/Luminous-Dynamics/symthaea/issues/13852): protocol-matched DISCERN evaluation from data integrity through adversarially reviewed hypotheses.
- [PR #13843](https://github.com/Luminous-Dynamics/symthaea/pull/13843): claim-specific source-node graph wiring; does not adjudicate entailment.
- [PR #13844](https://github.com/Luminous-Dynamics/symthaea/pull/13844): removes confidence-derived correctness feedback; independent outcome receipt types remain unfinished.
- [Issue #8771](https://github.com/Luminous-Dynamics/symthaea/issues/8771): immutable benchmark anchors and sealed, adaptive scientific-task refreshes.
- [PR #13665](https://github.com/Luminous-Dynamics/symthaea/pull/13665): typed public ResearchState evidence relation boundary.
- [PR #13668](https://github.com/Luminous-Dynamics/symthaea/pull/13668): candidate-facing StateGuard B0 baseline.
- [Issue #11171](https://github.com/Luminous-Dynamics/symthaea/issues/11171): actual production researcher integration and evaluator isolation.

Do not introduce a new web-research planner to solve an evidence-model defect. Reuse the existing retrieval subsystem and plug in the typed relation/adjudication process, preserving its candidate/evaluator boundary.

## 7A. Outbound acquisition security is part of evidence integrity

A source receipt is not trustworthy if the mechanism that acquired it can be redirected into an internal network. The bounded acquisition path in [PR #13854](https://github.com/Luminous-Dynamics/symthaea/pull/13854) currently uses a default reqwest client for search-result URLs. The branch audit found no visible destination-IP policy in the page-fetch path; reqwest follows redirects by default and honors system proxy settings unless configured otherwise. This is a security follow-on, not evidence that the bounded-acquisition work is invalid.

[Issue #13870](https://github.com/Luminous-Dynamics/symthaea/issues/13870) owns the implementation. The required invariant is:

> Every outbound research connection must be pinned to an address that was validated for that exact request. Redirects, retries and alternate addresses must repeat the same checks. A source-controlled URL or HTTP response must never increase the researcher's network authority.

Required controls:
- Parse and constrain the URL scheme, authority, user-info and destination port.
- Resolve once, reject the whole A/AAAA set when any address is non-public or special-use, and pin the validated addresses into the connection resolver. Hostname-only checks are not sufficient because they leave a DNS rebinding / time-of-check-to-time-of-use gap.
- Disable automatic redirects, or follow each hop manually with complete URL and address revalidation and a strict hop cap.
- Disable implicit environment/system proxies unless an explicit trusted proxy mode has its own enforced destination policy.
- Apply the same policy to provider adapters, retries and all fallback connections.
- Retain structured URL/DNS/redirect/proxy rejection receipts; rejected destinations must not enter the successful captured-source inventory.
- Require deployment-level egress controls as a separate defense-in-depth boundary.

Security conformance metrics (separate from scientific accuracy):
- **Forbidden-destination connection count:** target 0 across the full hostile matrix.
- **Pinned-address agreement:** 100% of successful connections terminate at an address in the exact validated resolver result.
- **Redirect revalidation:** 100% of redirect hops are either rejected or independently validated; no auto-follow bypass.
- **Rejection accounting:** 100% of denied attempts have a typed receipt and remain excluded from the source inventory.
- **Safe-path preservation:** controlled public-destination fixtures remain fetchable under the same resolver and policy.
- **Proxy isolation:** no implicit proxy is used when proxy mode is disabled.

The deterministic test matrix must include IPv4/IPv6 loopback, private/ULA, link-local, unspecified, multicast, broadcast, CGNAT, mapped IPv6, reserved/special-purpose ranges, mixed public+private DNS answers, DNS rebinding, internal-target redirects, redirect loops, scheme changes, malformed URLs, user-info, encoded IP forms, disallowed ports and environment proxy variables. Use an injected resolver and local controlled endpoints; tests must not contact live internal infrastructure.

This is a binary security gate, not a weighted score that can be offset by better retrieval or report quality. Passing application tests does not prove deployment egress isolation, and no production security claim is allowed until both layers are independently inspected.

Primary references:
- OWASP SSRF Prevention Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html
- reqwest 0.13.5 redirect behavior and policy: https://docs.rs/reqwest/0.13.5/reqwest/redirect/
- reqwest 0.13.5 DNS resolver override and address pinning API: https://docs.rs/reqwest/0.13.5/reqwest/struct.ClientBuilder.html

## 8. Claim boundary

This is a benchmark protocol and code-audit map. The existing WebResearcher has not been evaluated on these external benchmarks, no current SOTA score is claimed, and the implementation PRs remain draft pending exact-head Rust qualification. The intended contribution is a measurable reduction in false support and unverifiable research claims, not a claim that fluent reports are equivalent to reliable scientific knowledge.
