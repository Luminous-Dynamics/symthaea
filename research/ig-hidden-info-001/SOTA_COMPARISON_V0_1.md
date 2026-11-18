# IG-HIDDEN-INFO-001 — SOTA and industry comparison ledger v0.1

**Status:** literature/architecture crosswalk, not a leaderboard result.  
**Owner:** [#12762](https://github.com/Luminous-Dynamics/symthaea/issues/12762)  
**Pinned benchmark:** [HiddenBench source manifest](external/hiddenbench/SOURCE_MANIFEST.json)

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

Use current model revisions available to the evaluation account, not brand-name-only references. The November 2026 Anthropic system-card index lists Fable 5.2 / Mythos 5.2, Haiku 5.5, Sonnet 5.5 and Opus 5.5. The Anthropic comparison page also reports Fable 5.2, Opus 5.5, GPT-6 Astra and Gemini 4 Carbon on several tool-using task suites.

- System cards: https://www.anthropic.com/system-cards
- Model capability comparison: https://www.anthropic.com/claude/mythos

Those numbers are model/system-level scores on different benchmarks, not multi-agent HiddenBench results. Do not compare them numerically with collective-decision scores. Choose at least two model families that can actually be invoked, pin revisions/endpoints/date/decoding, and run every treatment on the same backbone first. Cross-model generalization is a separate factor.

## 5. Matched experimental arms

The confirmatory comparison should include, where exact implementations/artifacts are available:

1. Single agent, full information — individual-reasoning upper bound, not a fair collective budget.
2. Independent agents, private information — no-communication baseline.
3. Original HiddenBench 15-round sequential discussion — historical reproduction.
4. Published HiddenBench structured communication protocol — direct method baseline.
5. Round-robin evidence sharing — simple transparent control.
6. Free-MAD — consensus-free trajectory decision.
7. Meta-Moderator — learned adaptive moderator.
8. Microsoft Agent Framework Magentic / group chat or equivalent orchestrator.
9. OpenAI Agents SDK manager-as-tools / handoff pattern.
10. Symthaea evidence-directed protocol — information inventory, targeted evidence request, verified delivery, typed evidence matrix, independent integration audit and bounded stop.

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
