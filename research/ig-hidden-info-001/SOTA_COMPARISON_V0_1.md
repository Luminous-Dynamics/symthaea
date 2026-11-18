# IG-HIDDEN-INFO-001 — SOTA and industry comparison ledger v0.1

**Status:** literature/architecture crosswalk, not a leaderboard result.  
**Owner:** [#12762](https://github.com/Luminous-Dynamics/symthaea/issues/12762)  
**Pinned benchmark:** [HiddenBench source manifest](external/hiddenbench/SOURCE_MANIFEST.json)

## 0. Published result payloads

The original MIT-licensed [HiddenBench-results dataset](https://huggingface.co/datasets/YuxuanLi1225/HiddenBench-results) publishes the model-run JSON payloads and a file manifest. The file manifest says the result contents are copied without modification and identifies three GPT-4.1 files needed to score the full 65-task HiddenBench comparison:

- 80 hidden runs across eight manual/adapted tasks;
- 570 hidden runs across 57 generated tasks;
- 650 full-profile runs across all 65 tasks.

The exact source paths, run counts, byte sizes, dataset revision and intended CLI arrangement are pinned in [PUBLIC_RESULT_SOURCE_MANIFEST_V0_1.json](external/hiddenbench/PUBLIC_RESULT_SOURCE_MANIFEST_V0_1.json). The result payloads have not yet been downloaded or independently re-scored in this branch. A publication-reproduction stage should fetch and hash these exact files, score them with the official Python scorer and the new Rust scorer, then explain any discrepancy. Only after that should model API re-runs begin.

## 0a. The baseline ladder: coordination must earn its compute

Mohammadabadi, *More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs* (arXiv:2611.17610, submitted 13 November 2026), synthesizes nine empirical evaluations, 516 unique configurations and a six-dimensional resource vector. It reports that comparisons to a single greedy pass can confound coordination with sample multiplicity and test-time compute; against sample-count- or resource-matched baselines, the synthesis finds multi-agent accuracy differences averaging -0.042 and 74% of comparable contrasts at or below baseline. It argues that positive returns require partitioned evidence, external verification or genuine error diversity. This is a synthesis/preprint, not a replacement for task-specific reproductions. https://arxiv.org/abs/2611.17610

Every Symthaea campaign therefore uses this baseline ladder, with all comparisons paired on the same tasks/seeds/backbone:
1. **Single greedy pass:** reference point only; not sufficient to claim coordination gain.
2. **Single-agent best-of-N / self-consistency:** same number of generated samples, same total token budget and same answer-selection/evaluation budget.
3. **Independent parallel agents with no communication:** same sample count and independent private outputs, then the same aggregation/selection rule.
4. **Communication with identical evidence:** isolates coordination overhead from genuinely new information; positive gains are not assumed.
5. **Communication with partitioned evidence:** tests discovery of information unavailable to any one agent initially.
6. **External verification:** a separately budgeted tool/program/ground-truth check; compare with and without communication.
7. **Cross-model / error-diverse agents:** count all provider calls and test whether error diversity, rather than team size, explains gains.
8. **Full resource-matched architecture:** equalize total input/output tokens, model calls, tool calls, context windows, wall-clock/deadline and measured money; report remaining non-matched dimensions explicitly.

The primary causal question is not "does a team beat one greedy pass?" It is **does communication improve verified outcomes beyond an equally resourced single-agent sampling/verification strategy, and under which information-structure conditions?**

## 1. Direct scientific comparator

| Work | What it establishes in its own setup | Required use in this programme |
|---|---|---|
| [HiddenBench / Li, Naito & Shirado, ICML 2026](https://proceedings.mlr.press/v306/li26ej.html) | 65 Hidden Profile tasks; reports 30.1% multi-agent accuracy with distributed information vs 80.7% for a single agent with all information; identifies failures to elicit undisclosed evidence. | **Primary direct benchmark.** Reproduce hidden/full profiles, original prompt templates, 15 rounds, seeded assignment, sequential speaker order and official scorer before introducing new treatments. |
| [Toxtli & Delaflor, CSCW Companion '26](https://doi.org/10.1145/3785651.3831525) | Three-agent hiring decision study crossing synchronous/asynchronous and shared/hidden information; shows information surfacing and integration can diverge; asynchronous benefit appears for one model family in a small exploratory cell. | Separate replication cohort. Add async, length-matched sync, dedicated synthesizer, candidate/attribute matrix and integration audit. Do not pool with HiddenBench. |
| [Misalignment Contagion, COLM 2026](https://proceedings.colm.cc/paper_files/paper/2026/hash/871ed095b734818cfba48db6aeb25a62-Abstract-Conference.html) | Private/public/private stance shifts and topology effects across 35,290 trials in its tested setup. | Compose with #11320 for evidence-supported revision vs unsupported influence and topology counterfactuals. |
| [Believed but Not Said, 2026 preprint](https://arxiv.org/abs/2610.22138) | Private adoption of false claims may exceed public endorsement; surface agreement is not a complete view of agent state. | Compose private re-elicitation and public/private divergence with #11320; model the hidden-state claim only as an elicited response, not access to internal beliefs. |
| [PHANTOM-Bench, Nov 2026](https://arxiv.org/abs/2611.08971) | Ambiguous public signals can distort estimates of peers' private judgments; authentic private transparency can help recovery. | Compose with #11682/#11683; distinguish belief-about-peer from own stance revision and evidence integration. |

**Published reference numbers are not Symthaea results.** Different tasks, group sizes, models, prompts, budget, judges and metrics prevent transferring a reported score from one paper into another benchmark's leaderboard.

## 2. Strong method comparators for debate control

| Method | Main idea | How to compare fairly |
|---|---|---|
| [Free-MAD, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1600/) | Consensus-free one-round debate; score the full trajectory instead of last-round majority; anti-conformity. Reports results on eight reasoning datasets. | Implement/reproduce as a decision-layer treatment under the same backbone, task, token and tool budget. Measure whether it preserves correct minority evidence and whether its trajectory scores are calibrated. |
| [Meta-Moderator, Findings EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.1250/) | Separately trained meta-cognitive controller monitors debate utility, controls further deliberation and adjudicates. Reports improvements across five benchmarks and transfer. | Use as the strongest adaptive-moderation comparator where weights/data/harness are available. Match compute spent on the moderator and count its training/data and inference cost. |
| [MultiAgentBench, ACL 2025](https://aclanthology.org/2025.acl-long.421/) | Task completion plus collaboration/competition metrics across star, chain, tree and graph coordination protocols. | Use as a transfer benchmark after the Hidden Profile tasks; it is not a replacement or direct score equivalent for hidden-information discovery. |
| [MAST / MAST-Data, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b1041e52d3be19f0a9bc491657488e4a-Abstract-Datasets_and_Benchmarks_Track.html) | 1,600+ annotated traces from seven multi-agent frameworks; 14 failure modes grouped into system design, inter-agent misalignment and task verification. | Use its failure taxonomy for post-hoc trace annotation and compare whether Symthaea reduces failure rates on held-out task families. It is a reliability taxonomy, not a single accuracy baseline. |

## 2a. Communication-efficiency and evaluator reliability comparators

| Work | Published result in its own setup | Required Symthaea treatment |
|---|---|---|
| [MessageSieve / EMNLP 2026](https://aclanthology.org/2026.emnlp-main.2179/) | Training-free, per-message pruning before prompt assembly; reports 19–49% fewer evaluation tokens and +2.59 points on Gemini 3.1 Flash-Lite / +4.10 on Qwen3-8B versus dense Full-MAS. | Direct comparator for #12805. Reproduce its state-card/message-instance decisions; measure evidence-root and negative/minority evidence recall, not token reduction alone. |
| [AgentJudgeBench / EMNLP 2026](https://aclanthology.org/2026.emnlp-main.659/) | 3,808 workflow-DAG instances, six topologies, three difficulty tiers; judge alignment worsens with task difficulty, and hard no-ground-truth cases cluster around 77–82% across tested judges. Rubrics help up to 6.5 pp but not uniformly. | Use as an evaluator-reliability stress test. Prefer programmatic ground truth for structured outcomes and report human adjudication separately. |
| [Nine Judges, Two Effective Votes](https://arxiv.org/abs/2605.29800) | Nine frontier judges from seven model families provided about two independent votes' worth of information in tested settings because errors were correlated. | Compose with #9431. Measure error correlation/effective sample size; preserve strongest-single-judge and independent-family controls. Never certify by panel size alone. |

These results mean the benchmark itself needs an evaluation-integrity layer. A large judge panel cannot repair a benchmark with weak references, shared model failure modes or task-dependent over-anchoring.

## 3. Industry orchestration baselines

The major platforms already provide mature multi-agent orchestration and tracing. Symthaea should not claim novelty for basic delegation, group chat, concurrent execution, handoffs or workflow graphs.

| Platform | Publicly documented capability | Baseline implementation |
|---|---|---|
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | Agents, tools, handoffs/agents-as-tools, guardrails, sessions, human-in-the-loop, and built-in traces/spans for generations, tools and handoffs. | Manager-with-specialists and specialist-handoff baselines; preserve traces and exact SDK/model versions. |
| [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview) | Typed/graph workflows, sessions, middleware, integrations, observability; stable orchestration patterns include sequential, concurrent, handoff, group chat and Magentic. | Group-chat, concurrent and Magentic-manager baselines. Pin SDK/package versions and workflow definitions. |
| [Google ADK](https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system) | Graph-based and dynamic workflows, plus collaborative coordinator/specialist teams with generated delegation tools. | Coordinator/specialist and graph-workflow baselines with equivalent agent roster and budget. |

These products primarily solve **how to build and operate agent workflows**. This programme evaluates an orthogonal quality layer: can the workflow discover distributed evidence, preserve provenance, integrate evidence correctly, maintain independent judgments and stop when further communication has low value?

Tracing and observability are a baseline requirement, not a Symthaea differentiator. The differentiator must be measured evidence semantics and verified decision quality.

## 4. Foundation-model backbones

A reproducible multi-agent comparison needs exact API model IDs, not brand names alone. The provider documentation available on 18 November 2026 lists these practical candidate backbones:

- **OpenAI:** GPT-6.2 Astra (`gpt-6.2-astra`), GPT-6.2 Sol (`gpt-6.2-sol`), GPT-6 Luna (`gpt-6-luna`). Sources: https://developers.openai.com/api/docs/models and https://developers.openai.com/api/docs/changelog
- **Anthropic:** Claude Fable 5.2 (`claude-fable-5-2`), Claude Opus 5.5 (`claude-opus-5-5`), Claude Sonnet 5.5 (`claude-sonnet-5-5`), Claude Haiku 5.5 (`claude-haiku-5-5`). Source: https://platform.claude.com/docs/en/models/overview
- **Google:** Gemini 4 Hydrogen (`gemini-4-hydrogen`), Gemini 4 Carbon (`gemini-4-carbon`), Gemini 4 Argon (`gemini-4-argon`). Source: https://firebase.google.com/docs/ai-logic/models

The exact IDs and proposed evaluation roles are pinned in [MODEL_ROSTER_V0_1.json](MODEL_ROSTER_V0_1.json). Availability is account-, region-, endpoint- and provider-policy-dependent; re-check the live model catalogue at execution time and record the resolved model/revision and API endpoint in each run receipt. Do not silently substitute a retired model or a rolling alias.

Run coordination architectures within one exact backbone first, matching task, prompts, tools, reasoning/effort settings, total tokens, deadlines and call budgets. Then repeat the same comparison across at least two model families. A frontier-model single-agent score is not interchangeable with multi-agent HiddenBench performance. Published vendor benchmark claims are not comparable unless the underlying task, scoring, tools, budgets and evaluation protocol match.

## 5. Matched experimental arms

The confirmatory comparison should include, where exact implementations/artifacts are available:

1. Single agent, full information — individual-reasoning upper bound, not a fair collective budget.
2. Independent agents, private information — no-communication baseline.
3. Original HiddenBench 15-round sequential discussion — historical reproduction.
4. Published HiddenBench structured communication protocol — direct method baseline.
5. Single-agent best-of-N / self-consistency and independent parallel sampling at the same total sample/token budget.
6. Round-robin evidence sharing — simple transparent control.
7. Free-MAD — consensus-free trajectory decision.
8. Meta-Moderator — learned adaptive moderator.
9. MessageSieve — message-instance communication pruning.
10. Microsoft Agent Framework Magentic / group chat or equivalent orchestrator.
11. OpenAI Agents SDK manager-as-tools / handoff pattern.
12. Symthaea evidence-directed protocol — information inventory, targeted evidence request, verified delivery, typed evidence matrix, independent integration audit and bounded stop.

Every method must be compared at:
- same task, agent count, initial private evidence allocation and answer options;
- same backbone/model version and system/developer prompt scope;
- same total token/tool/call budget and deadline;
- same randomization block and reproducible seed;
- same scoring/evaluator protocol;
- counted controller/synthesizer/auditor tokens and tool time.

Keep a non-budget-matched oracle only as a labelled upper bound.

## 6. Outcome vector

Do not collapse these into a single collective-intelligence score.

| Dimension | Required measurements |
|---|---|
| Task outcome | Per-agent accuracy, strict-majority accuracy, group choice, correct abstention, error severity. |
| Information discovery | Unique/decisive evidence coverage, disclosure latency, request precision/recall, undisclosed decisive facts. |
| Integration | Matrix recall, source/entity/attribute/polarity accuracy, signed contribution error, omitted negative evidence, false evidence uptake. |
| Social influence | Private/public divergence, unsupported persistent stance shift, exposure/topology effect, consensus hallucination. |
| Decision control | Premature stop, redundant rounds, verified marginal value of next action, incomplete/indeterminate termination. |
| Reliability | MAST taxonomy failure rates, trace completeness, tool/authority boundary violations, evaluator disagreement. |
| Cost/latency | Total model tokens, calls, tool use, dollars, CPU/GPU/energy where available, p50/p95 end-to-end latency and cost per correct verified decision. |

## 7. Proposed Symthaea gates

These are project decision rules, not accepted industry standards or current results:

- reproduce the official HiddenBench reported baseline under the exact setup before claiming an improvement;
- at matched budgets, target +10 percentage points over reproduced standard debate on confirmatory HiddenBench tasks, with a paired task-level interval excluding zero;
- target +20 percentage points decisive-evidence coverage without increasing unsupported evidence uptake;
- show a statistically supported integration-fidelity gain when all decisive evidence is surfaced;
- reduce cost per correct verified decision by at least 15%, or improve quality at equal total cost;
- preserve evidence-positive revision sensitivity and correct abstention;
- transfer to a held-out task family, topology and model family;
- keep synthetic, model-executed, externally reproduced and human-reference evidence classes separate.

Freeze confirmatory splits and targets before opening the test results. Exploratory target calibration must use separate data.

## 8. Claims Symthaea can and cannot make

Potential differentiated claim after evidence:
> Under specified model, task, topology and budget profiles, the Symthaea evidence-directed protocol improves the discovery, provenance-preserving integration and correct use of distributed evidence over named baselines, with quantified quality/cost and uncertainty.

Not justified by these benchmarks alone:
- general intelligence or universal collective intelligence;
- human-group behavioural validity;
- consciousness, subjective experience or moral status;
- universal superiority to commercial agent frameworks;
- political legitimacy or authority from consensus;
- safety merely because agents agree, abstain, or preserve dissent.

The research goal is not more agents or more elaborate dialogue. It is **better decisions from distributed knowledge, with independently inspectable evidence for why the result is warranted.**
