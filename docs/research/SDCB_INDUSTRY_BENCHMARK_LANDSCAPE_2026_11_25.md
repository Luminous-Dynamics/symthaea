# SDCB competitive and benchmark landscape — 25 November 2026

**Purpose:** identify external comparators and explain what can and cannot be inferred from published scores. This is a research-planning snapshot, not a live leaderboard claim.

## 1. Neural-network spectral structure

| Reference | Reported result / scope | How SDCB should use it | What it does not prove |
|---|---|---|---|
| Staats, Thamm & Rosenow, NeurIPS 2025, *Small Singular Values Matter* | Pretrained transformer weight spectra show deviations from a random-matrix null at both high and low singular-value ends; the associated directions overlap activation covariance structure. Removing deviating directions increased perplexity more than removing bulk directions; the smallest decile became influential after fine-tuning. | Reproduce the stated analysis on an accessible pinned transformer before transferring the diagnostic to HDC/CfC. Test both spectral tails, matched random ablations, and restoration. | Does not show that HDC/CfC has the same spectral organization, or that spectrum similarity is a cognition mechanism. |
| Thamm et al., *Random matrix analysis of deep neural network weight matrices*, Physical Review E (2022) | Establishes RMT as a useful baseline for trained neural weight matrices. | Use architecture-appropriate nulls, not one universal Marchenko–Pastur fit. Record matrix shape, weight distribution, activations and finite-size corrections. | Agreement/deviation from an RMT null is not a complete information-theoretic interpretation. |

**Benchmark implication:** spectral description is cheap; causal mode deletion/restoration is the stronger test. The NeurIPS result is the current matched comparator for the SDCB tail-intervention track. It must not be presented as a Symthaea result.

## 2. Neural criticality and nonequilibrium dynamics

| Reference | Reported result / scope | How SDCB should use it | What it does not prove |
|---|---|---|---|
| *Is criticality a unified setpoint of brain function?*, Neuron (2025) | Reviews evidence and reports a meta-analysis of 140 datasets published 2003–2024; the authors argue that criticality may be a unifying computational setpoint. The topic has a long methodological controversy. | Treat criticality as a testable regime hypothesis. Sweep subcritical/near-critical/supercritical coupling; use several diagnostics and finite-size scaling; quantify task-dependent Pareto frontiers. | Does not establish that criticality is universally optimal for every task, or that a digital model at a similar point is biological or conscious. |

**Benchmark implication:** do not optimize one avalanche exponent. Require multiple diagnostics, finite-size analysis, perturbation response, recovery, memory and task performance. Compare to stable off-critical controls.

## 3. Nonlinear-system identification and operator learning

| Reference | Reported result / scope | How SDCB should use it | What it does not prove |
|---|---|---|---|
| CoDBench (2024) | Compares 12 data-driven models from four categories on 10 PDE datasets, including neural operators and transformer architectures; studies generalization, super-resolution, data efficiency, noise and compute. | Adopt its evaluation dimensions and use an existing PDE suite for any physical-dynamics claim. Do not create another tiny in-house PDE leaderboard. | PDE performance does not directly establish cognitive-memory or proof-reasoning performance. |
| Koopman-based control reviews (2025) | Survey EDMD/DMD, sparse regression, neural lifting and control integration; emphasize approximation error and closed-loop guarantees. | Report excitation, observability, noise, horizon, approximation error and stability. Evaluate model predictive control only with a safe, held-out environment. | A linear lifted approximation is not proof that the true nonlinear dynamics have a finite-dimensional exact linear representation. |
| Adversarial system-identification work (2026) | A recent paper reports convergence/certification conditions for Koopman spectral learning and matching impossibility results outside those conditions, with examples spanning oscillators, chaotic fluid flows and sea-ice forecasting. | Explicitly mark identifiability assumptions: excitation, sequence length, stationarity/regime conditions, measurement noise, observability and model class. Include adversarial non-identifiable trajectories. | Do not generalize a theorem beyond its assumptions or equate a single observed trajectory with complete system identification. |

**Benchmark implication:** SDCB should benchmark when a dynamical model is *identifiable*, when it is not, and whether the system can detect that uncertainty. This is a more important contribution than reporting a high average prediction score.

## 4. Formal mathematics and proof assistants

| Reference | Reported result / scope | How Symthaea should compare | What remains different |
|---|---|---|---|
| LeanGraphSearch (current project results page) | On an 810-query fair subset of MathlibQR, reports Recall@100 = 94.7%, compared with LeanSearch v2 Standard = 82.7%. On MathlibMPR, reports Covered@20 = 42.8% versus LeanSearch v2 Reasoning = 39.1%, and Covered@50 = 44.2% versus 41.3%. | Reproduce the exact fair subset and graph retrieval baseline before claiming retrieval superiority. Keep premise retrieval separate from proof completion and trust closure. | Retrieval recall is not theorem-proving accuracy, a complete proof, or human comprehension. |
| Goedel-Architect, preliminary ICML-submission PDF | Reports 99.2% pass@1 on MiniF2F-test and 75.6% pass@1 on PutnamBench using an open-weight 284B-A13B backbone. Optional Gemini 3.1 Pro natural-language proof seeding closes the remaining two MiniF2F problems and reports 100%. The paper labels itself preliminary and under review. | Treat as a high-capability prover comparator. Record backbone, compute, sample budget, NL guidance and exact task split. Run no-guidance and guidance conditions separately. | The 100% result is not the same condition as the 99.2% pass@1 baseline. Neither score measures proof explanation quality, scope correctness or human transfer by itself. |
| miniF2F / PutnamBench / MathlibQR / MathlibMPR | Established and evolving proof/statement/premise-retrieval evaluation families. | Benchmark distinct subtasks: autoformalization, retrieval, plan correctness, proof completion, kernel checking, checker diversity and explanation transfer. | Scores from different splits, theorem sets, proof budgets and model backbones are not directly comparable. |

**Benchmark implication:** Symthaea should not set a near-term claim of beating frontier theorem provers on pass@1. Our differentiated target is an evidence-carrying comprehension layer: exact theorem identity, complete obligation accounting, dependency-aware explanation, independent checker evidence, counterfactual robustness and delayed human transfer. Those outcomes must be measured alongside conventional proof success.

## 5. General-purpose agent systems

| Reference | Captured finding | How SDCB / Luminous should use it | Limit |
|---|---|---|---|
| IBM Research / Exgentic Open Agent Leaderboard (launched 2026) | Current page reports that model choice accounts for 28% of task variance while agent choice can shift results by up to 11 percentage points. The captured live ranking's leading row reported 72.7% average success at an average cost of $5.97; this is a dated site snapshot, not an enduring universal SOTA number. The live leaderboard reports system success and average cost. It evaluates 100 sampled tasks per benchmark and says its initial leaderboard includes only closed models. | Adopt full-system evaluation, transparent harness identity, per-domain success, cost, latency and reproducibility. Evaluate the agent scaffold separately from the backbone. | Snapshot changes; sampled subsets and closed-model coverage prevent general claims about the entire field. |
| 2026 review, *From benchmarks to deployment: a comprehensive review of agentic AI evaluation* | Audits 15 benchmarks and argues that common success-only metrics omit safety, cost, trajectory quality and deployment concerns. | Continue the Luminous non-compensatory vector: correct outcome, authority compliance, evidence quality, recovery, cost, latency, and maintainability. | A review's benchmark audit is a scoped analysis, not proof that every benchmark has identical omissions. |

**Benchmark implication:** publish both a capability curve and a deployment curve. For each task, show success, constraint violations, independent evidence, recovery, wall time, token/API cost, memory and energy (only when measured). Do not collapse these into one compensating scalar.

## 6. Competitive positioning

There are two different ways to compete:

1. **Capability competition:** solve more held-out tasks at matched budgets. For this, reproduce the exact external baselines (LeanGraphSearch; Goedel-Architect; CoDBench; Exgentic) before making a comparative claim.
2. **Evaluation and assurance innovation:** measure properties neglected by task success alone. Luminous can build on its existing proof-plan hypergraph, global obligation accounting, trust-closure capsule, checker diversity, trace evidence and independent sufficiency architecture.

The second is the more realistic near-term differentiator. It does not excuse weak task performance; it creates a second axis on which a research system can be evaluated rigorously.

## 7. Promotion gates (not achieved results)

- **Gate A — reference integrity:** dependency-free analytic controls; deterministic generators; mutation coverage; source and task digests.
- **Gate B — baseline reproduction:** reproduce one published baseline per domain on the identical task subset and report deviations.
- **Gate C — mechanism evidence:** spectral/Koopman/criticality features improve held-out prediction, and pre-registered interventions plus restoration support the stated mechanism.
- **Gate D — generalization:** replicate across seeds, task families, model sizes, and independent hardware.
- **Gate E — independent reproduction:** an external group reproduces both positive results and negative controls.
- **Gate F — SOTA claim:** only after a matched head-to-head with identical split, data, compute, model access, tool permissions, prompts, sampling, and scoring.

A missed gate stays visible. A skipped runner, a structural JSON parse, or a synthetic control is never a benchmark pass.


## 8. Internal Symthaea cross-domain integration

SDCB should reuse existing physics and formal-assurance work rather than building parallel physical solvers:

- **#13629 — positivity-, invariant- and response-constrained kinetic representation transitions.** This provides a domain-specific contract for safe state reduction in collisionless and collisional kinetic systems.
- **#13633 — structured response-constrained low-rank correction.** This explores matrix-free/low-rank KKT methods, feasibility, active sets and response constraints against published positivity-preserving dynamical low-rank baselines.
- **#13415 / #13628 — phase-resolved response measurements and mode-level response-loss estimation.** These are relevant to SDCB's intervention-and-restoration methodology.
- **#13485 — hard-memory refinement control.** This connects the memory/cost frontier to explicit resource refusal rather than silent degradation.
- **#13672 — SDCB parent research programme.** Owns the cross-domain protocol, analytic reference controls, identifiability/abstention track, benchmark governance and external comparison ledger.

The shared methodology is: exact state/source identity → declared observables → identifiability conditions → controlled intervention → independent response measurement → uncertainty/resource receipt. A successful physical invariant check does not validate a cognitive mechanism; a successful cognitive benchmark does not validate a physical solver. Reuse the evidence and measurement infrastructure, not domain-specific truth claims.


## 9. Scientific-discovery and realistic-workflow benchmarks (expanded 25 November 2026)

| Reference | Captured scope / reported results | SDCB use |
|---|---|---|
| Autonomous Science Bench 0.1 | Six real laboratory pilot tasks across three domains. Current public page lists GPT-6 Astra Codex at 72.2% and Opus 5.5 Claude Code at 66.7%. It explicitly evaluates experiment discovery and learning about designs never measured. | Highest-priority comparison for active experiment selection, exploration/exploitation, and unmeasured-design ranking. Keep the six-task pilot limitation visible. |
| FIRE-Bench, ICML 2026 | Agents rediscover documented findings from recent high-impact ML work through experiment design, execution, and evidence-backed conclusions. The paper reports the strongest evaluated agents below 50 F1, high run-to-run variance, and failures in experiment design/execution/evidence reasoning. | A verifiable scientific rediscovery track. This directly addresses the weakness of LLM-as-judge paper scoring. |
| SciAgentArena, June 2026 | Approximately 200 interactive, agent-agnostic scientific research tasks with stepwise verification; reports current agents are better on well-specified data analysis than novel insight and self-directed exploration. | Interactive full-loop adapter and held-out research-family evaluation. |
| AstaBench / Stanford AI Index 2026 | More than 2,400 scientific research problems and 57 evaluated agents across 22 agent classes; the AI Index summary reports a best score around 0.53 at about $3.40 per problem in the captured leaderboard snapshot. | Broad workflow coverage and cost-efficiency comparison. Preserve the precise AstaBench snapshot and scoring protocol. |
| SorryDB (ICML 2026) | Dynamic tasks from 78 real Lean formalization projects; the public January 2026 snapshot has 1,000 open proof obligations. Each task binds repository, commit, Lean version and source location; candidate proofs are independently compiled in the original project. Public snapshot currently reports Aleph Prover at 57.2% pass@1. | Real-world mathematical maintenance and fresh source-lineage evaluation; add SDCB proof-plan/scope/explanation metrics as a separate report. |
| Goedel-Prover-V2, ICLR 2026 | Open Lean prover: 8B reports 84.6% MiniF2F pass@32; 32B reports 88.1% standard and 90.4% self-correction pass@32; 86 PutnamBench problems at pass@184. | Strong formal-proving comparator. Keep pass@k, model scale and inference budget separate; this is not explanation-quality evidence. |
| Open Agent Leaderboard / Exgentic | Full-system open evaluation; reports both performance and cost. Its launch write-up reports unsuccessful runs cost 20–54% more than successful runs and tool shortlisting improved performance across tested models. | Report cost of failure, tool-selection ablations, system scaffold identity and trajectory quality, not only success rate. |

**Strategic conclusion:** SDCB should become an interoperability layer and a harder, narrowly targeted extension—not a replacement for these upstream benchmarks. The differentiating measurements are task-relative identifiability, active sensor/probe selection, surprise recovery, unmeasured-candidate learning, exact evidence provenance, and independent replay. See [SDCB scientific-discovery benchmark interoperability](SDCB_SCIENTIFIC_DISCOVERY_INTEROPERABILITY_V1.md).

## 10. Source links for the expanded benchmark snapshot

- Autonomous Science Bench 0.1, written 1 October 2026: https://asb.open-ssi.org/about
- FIRE-Bench, Proceedings of Machine Learning Research volume 306 (ICML 2026): https://proceedings.mlr.press/v306/wang26d.html
- SciAgentArena, Microsoft Research, June 2026: https://www.microsoft.com/en-us/research/publication/benchmarking-ai-agents-for-addressing-scientific-challenges-across-scales/
- Stanford AI Index Report 2026, Section 5.3 (AstaBench summary): https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf
- SorryDB public leaderboard and evaluation pipeline: https://sorrydb.org/
- Goedel-Prover-V2, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html
- Open Agent Leaderboard / Exgentic launch article: https://huggingface.co/blog/ibm-research/open-agent-leaderboard
