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
| DeepResearch Bench II (2026) | 132 expert-authored research tasks, 9,430 rubrics, 22 domains; InfoRecall, Analysis, Presentation | Retrieval coverage, synthesis/analysis and verifiable presentation | The original USTC page says the official leaderboard migrated and its judge switched to GPT-5.5. Its displayed AI21 total 64.38 is a legacy page snapshot, not a verified current leaderboard value |
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
| External DRB II | Reproduce native scores first; then target ≥70% total rubric pass on the same task/judge snapshot, reporting InfoRecall, Analysis and Presentation separately |
| Citation integrity | Target ≥95% citation accuracy on a frozen gold subset; effective-citation count alone is not sufficient |
| Resource quality | Report cost and latency Pareto fronts; no unqualified composite score |

A ≥70% DeepResearch Bench II target is only meaningful on the same frozen task set and judge version. The page currently states the judge changed to GPT-5.5 and the leaderboard migrated, so do not compare an old-page 64.38 directly with a new-jury score without a controlled bridge evaluation.

## 7. Composition with existing Symthaea work

- [PR #13790](https://github.com/Luminous-Dynamics/symthaea/pull/13790): source relations remain neutral until explicit adjudication.
- [Issue #13796](https://github.com/Luminous-Dynamics/symthaea/issues/13796): provenance-bound claim/source adjudicator and benchmark.
- [Issue #13791](https://github.com/Luminous-Dynamics/symthaea/issues/13791): independent outcome labels; no confidence-as-correctness loop.
- [Issue #13792](https://github.com/Luminous-Dynamics/symthaea/issues/13792): claim-specific source graph edges and evidence versions.
- [Issue #13794](https://github.com/Luminous-Dynamics/symthaea/issues/13794): complete acquisition receipts and bounded concurrency.
- [PR #13665](https://github.com/Luminous-Dynamics/symthaea/pull/13665): typed public ResearchState evidence relation boundary.
- [PR #13668](https://github.com/Luminous-Dynamics/symthaea/pull/13668): candidate-facing StateGuard B0 baseline.
- [Issue #11171](https://github.com/Luminous-Dynamics/symthaea/issues/11171): actual production researcher integration and evaluator isolation.

Do not introduce a new web-research planner to solve an evidence-model defect. Reuse the existing retrieval subsystem and plug in the typed relation/adjudication process, preserving its candidate/evaluator boundary.

## 8. Claim boundary

This is a benchmark protocol and code-audit map. The existing WebResearcher has not been evaluated on these external benchmarks, no current SOTA score is claimed, and PR #13790 is not yet Rust-qualified. The intended contribution is a measurable reduction in false support and unverifiable research claims, not a claim that fluent reports are equivalent to reliable scientific knowledge.
