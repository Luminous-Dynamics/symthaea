# CEB industry benchmark snapshot — 2026-11-04

**Snapshot date:** 2026-11-04.  
**Use:** External baseline map for CEB research and preregistration.  
**Claim ceiling:** These are externally reported model+harness+protocol results, not Symthaea measurements and not universal model capability constants. Leaderboard rows may have different run budgets, step counts, refusal handling and confidence intervals. Recheck the source page and pin the exact task/evaluator revision before any formal comparison.

---

## 1. Current benchmark map

| Capability | Benchmark / scope | Current source-reported reference | What it measures |
|---|---|---:|---|
| Public issue resolution | SWE-Bench Pro V2 Full, 642 tasks / 11 repositories | Opus 5 + Claude Code: 99.40 ± 0.40 | Resolve public long-horizon issue tasks under a refreshed protocol |
| Hard public issue resolution | SWE-Bench Pro V2 HARD, 51-task subset | Opus 5 + Claude Code: 98.00 | A harder, smaller subset; not the same denominator as Full |
| Repository comprehension | SWE Atlas Codebase QnA, 124 tasks / 11 repositories / 4 languages | Opus 5 + Claude Code: 63.17 ± 5.01 | Runtime analysis, multi-file tracing and technical repository questions |
| Test writing | SWE Atlas Test Writing | Fable 5.1 + Claude Code: 67.04 ± 5.33 | Meaningful test additions, mutation checks and rubric compliance |
| Architectural refactoring | SWE Atlas Refactoring, 70 tasks / 10 repositories / 6 languages | GPT-6 Astra + Codex: 59.05 ± 6.43 | Multi-file restructure, behaviour preservation, maintainability and cleanup |
| Precision-aware debugging | PDB-Single Hard, 5,751 tasks; PDB-Wild, 484 mixed tasks | PDB-Single: Opus 4.7 P 83.3 ± 0.8 / R 89.3 ± 0.6; PDB-Wild: Opus 4.7 P 77.8 ± 3.1 / R 83.4 ± 2.8 | Newer per-task result table; sets differ and use epsilon-relaxed edit matching |
| Expert-relative optimization | SWE-fficiency, 498 tasks / 9 repositories | Top evaluated agents average below 0.23× expert speedup | Performance improvements against expert reference workloads while preserving correctness |
| Selective escalation | HiL-Bench, 300 tasks / 1,131 blockers | Historical paper/page takeaway: 24% combined Pass@3 and 44% highest ASK-F1 | Question precision/recall and task completion after help; metrics differ |
| Paired act/abstain | AgentAbstain, 263 pairs / 42 environments / 8 scenarios | Gemini 3.1 Pro + Google ADK: 59.5% paired accuracy | Correct on both should-act and should-abstain members of a pair |
| Context retrieval | ContextBench, 1,136 tasks / 66 repos / 8 languages | CodeNib Recall 0.779, Pass@1 77.6%, F1 0.281, $0.68/task; different system holds F1 0.389 | Evidence/context retrieval quality and efficiency; maxima are not one system |

| Source-free reverse engineering | SRE-Bench, 19 private programs / 262 binary instances / 1,572 task grades | GPT-6-Astra: 5.55/6 on 181 graded instances; 149 fully solved | Semantic recovery and task completion from protected binaries without source |
| Code review bug detection | CRBugBench, 1,000 OSS tasks / 98 repositories + 800 private tasks | Opus 5 + Copilot CLI: 71.9% of human-verified bugs surfaced in top 5 comments | Review-agent bug detection and reviewer-facing comment ranking |
| Secure code generation | Agent Security League, 200 tasks / 108 Python projects / 77 CWE classes | Highest Secure score: 36.9%; highest Functional score: 91.6% (different system) | Security correctness in addition to functional correctness |
| Rust repository issue repair | Rust-SWE-bench, 500 tasks / 34 Rust repositories | RUSTFORGER: 28.6% using Claude Sonnet 3.7 | Rust-specific issue reproduction, repository structure and type/trait reasoning |
| Context retrieval | ContextBench, 1,136 tasks / 66 repositories / 8 languages (63 Rust tasks) | CodeNib: Recall 0.779, Pass@1 77.6%, Context F1 0.281, $0.68/task; another system reports F1 0.389 | Evidence/context retrieval quality and cost before task resolution |

The percentages and scores above are not mutually comparable. In particular, issue resolution, QnA rubric scores, mutation-validated test writing, binary RE task grades, ASK-F1 and Pass@3 have different units and evaluation pipelines.

## 2. Source and methodology notes

### SWE-Bench Pro V2

Scale reports a refreshed public split of 642 tasks across 11 repositories (89 tasks removed after review), correction of 69 instruction/test contradictions, pristine-image regrading of every agent diff, and a two-sided reference-patch / empty-patch admission gate. The Full and 51-task HARD results must be reported separately. The current leaderboard lists Opus 5 at 99.40 ± 0.40 on Full and 98.00 on HARD.

Sources:
- https://labs.scale.com/leaderboard/swe_bench_pro_public_v2
- https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full
- https://labs.scale.com/blog/swe-bench-pro-v2

**CEB implication:** public repair pass rate is a necessary cross-check, not the central competitive claim. Use private, sealed repository templates for confirmation and preserve exact task coverage and pristine regrades.

### SWE Atlas: QnA, Test Writing and Refactoring

Scale describes three separate capabilities. QnA uses 124 tasks drawn from 11 production repositories and four languages; its page notes that some newer Mini-SWE-Agent evaluations use 500 steps while earlier ones used 250. The current result table lists Opus 5 at 63.17 ± 5.01, Fable 5.1 at 59.95 ± 4.85 and GPT-6 Astra at 59.14 ± 4.88.

Test Writing evaluates manifests, mutation behaviour and rubrics; the published table lists Fable 5.1 at 67.04 ± 5.33, Opus 5 at 62.22 ± 5.58 and Fable 5 at 55.60 ± 5.80. The page explicitly notes that adding more tests does not necessarily improve performance; extra tests can be irrelevant or fail to test the specified behaviour.

Refactoring contains 70 tasks over 10 repositories and six languages. It expects substantially larger edits than SWE-Bench Pro (about twice the changed lines and 1.7 times the files) and checks test preservation plus mandatory maintainability, artifact-cleanup and anti-pattern rubrics. The current results table lists GPT-6 Astra at 59.05 ± 6.43, Fable 5.1 at 56.67 ± 6.52 and Fable 5 at 54.76 ± 6.76. The page's older TL;DR still cites Opus 4.7 at 48.57; use the current table with a date and do not copy the stale TL;DR as the current leader.

Sources:
- https://labs.scale.com/leaderboard/sweatlas-qna
- https://labs.scale.com/leaderboard/sweatlas-tw
- https://labs.scale.com/leaderboard/sweatlas-refactoring

**CEB implication:** #10173, #10400, #8311 and #10385 address upstream comprehension, contract recovery, oracle quality and repair precision. Keep the QnA, test-writing and refactoring slices distinct. Match agent step budgets and harness where possible; do not interpret a larger step budget as a free model gain.

### Precise Debugging Benchmark

The April 2026 PDB paper introduces edit-level precision and bug-level recall to expose over-editing that unit-test pass rates hide. A later public PDB-Results release now publishes per-task scores and confidence intervals: on PDB-Single Hard (5,751 single-line tasks), Claude Opus 4.7 reports 83.3 ± 0.8 precision and 89.3 ± 0.6 recall; Gemini 3.1 Pro reports 82.5 ± 0.8 and 89.4 ± 0.7. On PDB-Wild (484 mixed tasks including repository-level SWE-smith bugs), Opus 4.7 reports 77.8 ± 3.1 and 83.4 ± 2.8; Gemini 3.1 Pro reports 77.8 ± 3.0 and 85.6 ± 2.8. The earlier paper's below-45% precision finding describes its initial evaluated systems and should not be presented as today's best result. PDB edit matching is epsilon-relaxed (ε=2 for single-line sets, ε=1 for PDB-Wild).

Source: https://arxiv.org/abs/2604.17338

**CEB implication:** #10385 should retain at least bug-level recall, admissible-edit precision, unnecessary semantic change, regressions and pristine regrade as separate outputs. A small diff that leaves a bug unresolved must not score well; semantically equivalent repairs must not be punished merely for differing from a reference patch.

### SWE-fficiency

The ICML 2026 benchmark uses 498 tasks across nine data-science, ML and HPC repositories. Agents must optimize real workloads while passing the same correctness tests. The paper reports that top evaluated agents achieve less than 0.23× expert speedup on average.

Source: https://proceedings.mlr.press/v306/ma26l.html

**CEB implication:** #10385 must compare to a validated expert implementation on the same hardware, toolchain, input distribution and correctness contract. Candidate-vs-original speedup is not expert-relative speedup. Use repeated paired runs, randomized order, warm-up, thermal control, memory and tail-latency outcomes.

### HiL-Bench

The Scale page describes 300 tasks split evenly between software engineering and text-to-SQL, with 1,131 blockers, 200 public tasks and 100 private held-out tasks. Its explicit metric definitions say ASK-F1 is the harmonic mean of question precision and blocker recall, while Pass@3 asks whether one of three independent runs finishes the task after access to `ask_human()`.

The page's explicit takeaway still reports the earlier benchmark result of 24% combined Pass@3 and highest ASK-F1 of 44%. The live comparison table now displays Muse Wonder 1.0 at 79.67±4.17, Muse Spark 1.3 at 66.89±5.12 and GPT-6 Astra at 65.32±5.19, but the numeric column is not labelled in the table text. Do **not** relabel 79.67 as ASK-F1 or Pass@3 without clarification from the benchmark owner. Preserve the two defined metrics separately and treat 24% / 44% as the older explicitly identified takeaway, not the current unqualified leaderboard.

Source: https://labs.scale.com/leaderboard/hil

**CEB implication:** ask quality, blocker recall, task completion after help and human burden all need separate reporting. Do not score generic question volume as helpfulness.

### AgentAbstain

The NeurIPS 2026 oral benchmark contains 263 paired tasks across 42 executable MCP environments and eight abstention scenarios. Each pair contains a should-act task and a minimally perturbed should-abstain twin. The reported best paired accuracy is 59.5% for Gemini 3.1 Pro with Google ADK. The public table also provides act-side, abstain-side and conditioned-abstention metrics.

Sources:
- https://arxiv.org/abs/2607.10059
- https://github.com/AntiQuality/agentabstain
- https://agentabstain.github.io/

**CEB implication:** use paired tasks and score both sides. Include pre-execution and runtime-discovered blockers; acting and then apologizing is not abstention.

### SRE-Bench

The Columbia DAPLab page describes 19 private programs, 262 binary instances, 1,572 deterministic task grades, five RE domains and four source languages. It reports GPT-6-Astra at 5.55/6 on 181 graded instances, with 149 fully solved. Its page explicitly states that metrics are computed over graded instances only, and the model leaves 81/262 instances ungraded in the public-facing setting (approximately 30.9%). The fully solved 149/181 value must not be presented as 149/262.

Sources:
- https://daplab.cs.columbia.edu/benchmarks/sre-bench/
- https://daplab.cs.columbia.edu/projects/sre-bench/

**CEB implication:** every leaderboard needs coverage, graded denominator and refusal/unsupported/timeout reporting alongside conditional success. This is directly relevant to #10314 and #10370.

### CRBugBench — code review and change-level defect detection

Microsoft Research's October 2026 CRBugBench pairs human-verified bugs with the PRs that introduced them. It contains 1,000 open-source review tasks across 98 repositories and another 800 tasks from proprietary industry repositories. The best reported system, Claude Opus 5 with Copilot CLI, surfaced 71.9% of bugs within its top five comments. The study reports weaker detection on complex changes and bugs requiring non-local reasoning, with a sharp decline on proprietary repositories. An exploration-oriented prompt increased performance by up to 5.8% at 10% higher cost.

Source: https://www.microsoft.com/en-us/research/publication/when-should-a-software-factory-be-lights-off-benchmarking-code-review-agents-on-bug-detection/ (October 2026); paper: https://arxiv.org/abs/2610.13668

**CEB implication:** add review-agent tasks that predict defects before editing, with Recall@1/3/5, actionable precision, severity-weighted recall, false-positive review burden and non-local dependency reasoning. The system must cite exact changed lines and supporting repository paths. A plausible warning is not a detected bug; reviewers must adjudicate the bug claim.

### Agent Security League — functional success is not secure success

Endor Labs' Agent Security League reports 34 independently evaluated agent/model combinations. Its benchmark uses 200 tasks from 108 open-source Python projects spanning 77 CWE classes, with prompt hardening, workspace sanitization and automated cheating detection. The live page currently shows Claude Code / Fable 5.1 at 86.6% Functional and 36.9% Secure; Codex / GPT-6 Astra at 82.1% and 34.6%; and Codex / GPT-6.1 Sol at 77.7% and 34.1%. The highest Functional score is 91.6% for Muse Code / Muse Wonder, whose Secure score is 25.1%. Secure requires passing security checks as well as functional tests. The maxima are different configurations and should never be combined into a synthetic system.

Source: https://www.endorlabs.com/research/ai-code-security-benchmark (live page accessed November 4, 2026)

**CEB implication:** every engineering task that can introduce security regressions needs separate functional and security outcomes. Report CWE-stratified outcomes, high-severity false accepts, unsafe data flows, authorization boundary violations, dependency/supply-chain behavior and remediation correctness. A functionally passing but vulnerable patch is a failure, not a partial success disguised in an aggregate.

### Rust-SWE-bench — Rust deserves a dedicated lane

Rust-SWE-bench contains 500 repository-level tasks from 34 Rust repositories. Its February 2026 study reports ReAct-style agents resolving up to 21.2% of issues and RUSTFORGER with Claude Sonnet 3.7 resolving 28.6%, a 34.9% improvement over the strongest baseline used in that study. The paper highlights repository structure, Rust type/trait semantics and issue reproduction as major limitations. This is an important Rust-specific baseline, but the reported model/harness is not a November 2026 frontier comparison.

Sources: https://arxiv.org/abs/2602.22764 and https://conf.researchr.org/details/icse-2026/icse-2026-research-track/67/Evaluating-and-Improving-Automated-Repository-Level-Rust-Issue-Resolution-with-LLM-ba

**CEB implication:** use Rust-SWE-bench as an external replication target after license and task-access review. Measure issue reproduction, trait/type resolution, feature/target correctness, compiler diagnostics, repair precision and accepted patch rate separately. Compare the same model and harness against RUSTFORGER-style reproduction/tracing and the Symthaea repository/specification components.

### ContextBench — context retrieval quality is a first-class engineering variable

ContextBench's live leaderboard was updated October 23, 2026. The dataset contains 1,136 tasks across 66 repositories and eight languages, including 63 Rust tasks. CodeNib (Haiku 5.5 search pair + Opus 5.5 mini-SWE-agent) leads Recall at 0.779, with Pass@1 77.6%, Context F1 0.281 and average cost $0.68. A different system, openJiuwen-JiuwenSwarm + DeepSeek-V4-Pro, has the best Context F1 at 0.389 (Recall 0.411, Pass@1 60.8%). This is a strong example of why best recall, task success and context F1 must not be merged into one row.

Source: https://contextbench.github.io/ (accessed November 4, 2026); paper: https://arxiv.org/abs/2602.05892

**CEB implication:** extend repository-twin evaluation with gold-context Recall@k, Precision@k, Context F1, context sufficiency, irrelevant-token fraction, latency, retrieval cost and downstream task success. Symthaea should test whether typed graph traversal improves context quality over lexical search, AST retrieval and a matched agent scaffold—not merely whether it retrieves more files.

---

## 3. The actual competitive gap

The benchmark landscape now exposes nine separable gaps between coding success and engineering competence:

1. **Understanding before action:** SWE Atlas QnA and ContextBench make repository comprehension and context retrieval separate measurable capabilities.
2. **Knowing what should be true:** test writing, mutation checks and CEB-SPEC measure whether the agent recovers the right contract rather than inventing one.
3. **Changing only what is necessary:** PDB separates resolved bugs from necessary edits and collateral changes.
4. **Improving systems against a real reference:** SWE-fficiency exposes weak expert-relative optimization.
5. **Reviewing changes before merge:** CRBugBench measures whether agents detect human-verified defects in incoming changes, including non-local bugs.
6. **Producing secure software:** Agent Security League shows that functional correctness and security correctness can diverge sharply.
7. **Handling language-specific semantics:** Rust-SWE-bench identifies Rust type/trait reasoning and issue reproduction as a distinct bottleneck.
8. **Knowing when to ask, stop or abstain:** HiL-Bench and AgentAbstain measure different aspects of selective autonomy.
9. **Accounting for assurance and coverage:** SRE-Bench's graded denominator and CEB's evidence receipts show why conditional score and assurance cannot be collapsed.

SRE-Bench remains a separate optional source-free reverse-engineering capability, not a substitute for source-based repository comprehension.

## 4. CEB strategy

### Primary research claims to pursue

- **CEB-REPO:** typed repository model with source anchors, compiler-authority labels, feature/target worlds, impact prediction and stale-fact invalidation.
- **CEB-SPEC:** bounded behavioural-contract discovery, contradiction detection, falsification and independent-oracle qualification.
- **CEB-CAUSE:** competing causal hypotheses, prospective experiment forecasts, typed safe interventions and independently observed outcomes.
- **CEB-DEBUG-QUALITY:** bug recall, edit precision, regression control and pristine regrade.
- **CEB-OPT:** expert-relative performance improvement under fixed workloads and correctness gates.
- **CEB-AUTONOMY:** paired act/abstain and targeted escalation under progressive blocker discovery.
- **CEB-INTEGRITY:** full task accounting, hidden answer custody, exact-head evidence, trust qualification and isolated execution.
- **CEB-REVIEW:** pre-merge defect discovery with top-k recall, actionable precision, severity and reviewer burden.
- **CEB-SECURE:** secure code-change correctness, CWE coverage, false-accept rate and functional/security non-inferiority.
- **CEB-RUST:** Rust-specific issue reproduction, trait/type correctness, configuration scope and repository comprehension.
- **CEB-CONTEXT:** context retrieval precision/recall/F1, sufficiency, token cost and downstream task success.
- **CEB-EVAL / CEB-VERIFY:** evaluator integrity, benchmark admission, answer-leakage resistance, receipt binding, independent regrading and false-accept/false-reject behavior.
- **CEB-DOCS:** static evidence that documented Cargo manifest/package/binary commands refer to tracked sources in the selected workspace; this is a documentation-integrity control, not proof that commands build or pass.

### Matched baseline design

For every capability that CEB claims to improve, compare:

1. deterministic/tool baseline;
2. model-only minimal harness;
3. same model with a strong industry-style coding scaffold;
4. same model plus the Symthaea component under test;
5. Symthaea-native path where the task does not depend on a frontier foundation model;
6. lesion/sham/rescue variants where a causal mechanism claim is being made.

Hold model, harness, prompt, tools, step/token/time budget, environment, task revision and grader fixed. When direct matching is impossible, label the comparison contextual, not causal.

### Evidence and statistics

- Keep primary and pristine-regrade outcomes separate.
- Report per-task dispositions and full eligible denominators.
- Use paired tasks and matched resources.
- Cluster by repository template and root-cause family.
- Run at least three independent seeds, five where stochastic variance warrants it.
- Report paired confidence intervals, effect sizes, calibration and seed variance.
- Preserve timeouts, refusals, unsupported items and ungraded tasks.
- Do not use LLM-as-judge as the sole authority.
- Preregister the primary outcome and stop rules before holdout results are observed.

## 5. Target-setting discipline

Aim to exceed strong baselines on **each named metric under a directly matched protocol**, not to claim that one aggregate engineering score beats all industry benchmarks.

Suggested first gates already tracked in #10385 / #10400:
- controlled bug recall >= 0.90 and edit precision >= 0.80, then >= 0.95 / >= 0.90 on sealed holdout;
- expert-normalized speedup >= 0.50× initially, then >= 0.80× on holdout;
- specification supported-claim precision >= 0.95 / recall >= 0.90, with zero fabricated source/compiler evidence;
- paired act/abstain correctness >= 0.75 initially, then >= 0.85;
- ASK-F1 >= 0.65 initially, then >= 0.75;
- 100% evidence/provenance coverage for claims used to authorize a repair;
- high-impact false-action rate <= 1% initially, <= 0.5% holdout;
- code-review defect recall@5 >= 0.80 initially, >= 0.85 on sealed holdout, with actionable precision and reviewer burden reported separately;
- secure-task pass >= 0.55 initially, >= 0.65 on sealed holdout, while functional correctness remains within a preregistered 2 pp non-inferiority margin;
- context retrieval Recall@10 >= 0.85 initially, >= 0.90 sealed; Context F1 >= 0.45 initially, >= 0.50 sealed, with downstream task success and retrieval cost reported;
- Rust-SWE-bench issue resolution >30% in a matched replication before pursuing >35%; do not compare this directly with the February 2026 RUSTFORGER/Claude Sonnet 3.7 result without matched harness and model controls.

These are prospective CEB goals, not current Symthaea results. Final sample sizes and confidence bounds must be set from pilot variance and repository/fault-family clustering.

## 6. What must not be claimed yet

- No public leaderboard result establishes that Symthaea is competitive today.
- The research PR and causal ledger prototype are not qualified merely because source and authored tests exist.
- A skipped CI workflow is not a passing test.
- A compiler-resolved fact is not established by a static graph fixture.
- A signed receipt does not prove that the run assertion is true.
- Distinct signer labels do not establish operational independence.
- A host runner or Nix development shell is not an isolation boundary.
- A score on graded tasks is not a score on the whole task universe.

The immediate route is runner-independent authoring, static source/manifest checks, schema and gold-label review, benchmark preregistration and adversarial test design. Dynamic tasks remain blocked by #10064 until the OS isolation boundary is qualified.

---

## References

- SWE-Bench Pro V2: Scale Labs leaderboard and methodology.
- SWE Atlas: Scale Labs Codebase QnA, Test Writing and Refactoring.
- Precise Debugging Benchmark: Zhu et al., April 2026, [paper](https://arxiv.org/abs/2604.17338), [versioned model outputs and result tables](https://huggingface.co/datasets/Precise-Debugging-Benchmarking/PDB-Results).
- SWE-fficiency: Ma et al., ICML 2026.
- HiL-Bench: Scale Labs benchmark page and paper.
- AgentAbstain: Liu et al., 2026; official code and leaderboard.
- SRE-Bench: Columbia University DAPLab, October 2026 update.
- CRBugBench: Microsoft Research / Yang et al., October 2026: https://arxiv.org/abs/2610.13668
- Agent Security League: Endor Labs, live benchmark page: https://www.endorlabs.com/research/ai-code-security-benchmark
- Rust-SWE-bench / RUSTFORGER: Xiang et al., ICSE 2026: https://arxiv.org/abs/2602.22764
- ContextBench: Li et al., 2026, leaderboard updated October 23: https://contextbench.github.io/
- Terminal-Bench 4.0 maintainer release notes and 4.1 verifier roadmap: https://www.tbench.ai/news/terminal-bench-4-0
- Epoch AI independent review of Terminal-Bench 4.0.0 (2026-09-04): https://epoch.ai/benchmarks/terminal-bench-4/review

**Snapshot limitation:** this is a dated source audit. Re-read the cited pages before future claims; live score tables, model identities, harness settings and benchmark revisions can change.
