# MACE Decision-Sufficient Memory: benchmark frontier and competitive plan

**Snapshot:** 27 November 2026  
**Status:** research strategy and proposed targets; not an empirical MACE result or leaderboard claim.

## Executive thesis

The next generation of memory systems should be evaluated as **decision-support infrastructure with a lifecycle**, not merely as stores that improve recall. The hard question is whether a memory system changes a later decision for the better, with valid evidence, at an acceptable total cost—and whether it knows when not to act.

MACE's differentiator should be the ability to show, with reproducible paired evidence:

1. which source observations support a decision;
2. whether those observations are current, in scope, sufficiently independent and still authorised;
3. whether memory improved or harmed the downstream decision relative to no memory;
4. whether a new observation would be worth its cost;
5. what must be superseded, quarantined, forgotten or rebuilt after a change;
6. what the entire lifecycle costs, not only the final Search call.

This is a research thesis. It becomes a field contribution only after independent, matched external evaluation.

## 1. Current external baseline landscape

Do not combine the following scores into one ranking. Their tasks, denominators, system interfaces and judges differ.

### Agent Memory Leaderboard Cycle 2

The official Cycle 2 results were published on 19 November 2026. The table below reports the leading Open-source Methods and Commercial Products entries in each track.

| Track | Open-source leader | Commercial leader |
|---|---:|---:|
| Textual | Resurface — 61.22 | MemoraX — 69.26 |
| Multimodal | Khipu — 43.26 | MemoraX — 53.68 |
| Coding | InsightMem — 63.33% | MemoraX — 68.00% |

Source: Agent Memory Leaderboard, Cycle 2 results, https://agentmemoryleaderboard.ai/results/cycle-2/

The track score is not a complete picture of capability. For example, the official Textual leaderboard shows Resurface at 76.55 in Memory Governance and 68.59 in Epistemic Safety & Privacy, but 38.82 in Personalization & Care and 49.65 in Streaming. Commercial MemoraX leads the textual overall board at 69.26 and reports 78.10 Memory Governance, 81.98 Epistemic Safety & Privacy and 63.11 Streaming. These capability-level gaps suggest that memory quality is multi-dimensional, not a single retrieval number.

AML says Cycle 2 received 402 formal Full evaluation tasks, 333 completed successfully and 292 valid leaderboard records were published after review and correction. It used more than 6,000 Textual instances and 150 base Coding tasks in relevant/noisy conditions. The three tracks have different tasks and scores must not be compared across tracks. Source: https://agentmemoryleaderboard.ai/blog?post=phase2-results-20261119

### Long-horizon agentic memory

**AMA-Bench (ICML 2026).** The published AMA-Agent result is 57.22% average accuracy, 11.16 points above the strongest baseline in that paper's matched setup. The benchmark covers six agentic domains and four capabilities (recall, causal inference, state updating and state abstraction), with trajectories from real agent applications and scalable synthetic trajectories. AMA-Agent uses a causality graph and tool-augmented retrieval. This is the relevant reference for MACE's causal/trajectory line; 57.22% is not directly comparable to AML or DSM. Sources: https://proceedings.mlr.press/v306/zhao26bs.html and https://ama-bench.github.io/

**MemoryArena (ICML 2026).** MemoryArena evaluates interdependent Memory-Agent-Environment loops over multiple sessions, including web navigation, preference-constrained planning, progressive information search and sequential formal reasoning. Its central finding is that near-saturated long-context QA performance does not guarantee successful future agentic task execution. MACE should treat it as an action-transfer evaluation, not a retrieval leaderboard. Source: https://proceedings.mlr.press/v306/he26am.html

### Governed use, collaboration and forgetting

**TimlMemBench (Findings of EMNLP 2026).** The current package contains 208 reviewed Chinese/English samples, 1,566 sessions, 1,715 questions and 1,821 governance or activation cases. Its canonical-pool control and raw-native ingestion track distinguish answer quality with a common memory pool from actual system behaviour after ingesting source sessions. It exposes weaknesses in lifecycle updates, scope/boundary control, provenance recovery, activation and restraint. Source: https://aclanthology.org/2026.findings-emnlp.507/

**COLLABMEM (EMNLP 2026).** This benchmark targets multi-agent memory visibility, access, routing, provenance, conflict resolution and retention. It has 58 hand-authored scenarios (1,118 core probes), a scaled 174-scenario split (3,359 probes), and six transactive-memory competencies. Its TMS-GRAPH++ system reports a 69.6% highest non-Oracle average over five splits. MACE should compare against this result only on the same split, SDK, answerer, judge and harness. Source: https://aclanthology.org/2026.emnlp-main.1194/

**Memora / FAMA (Findings of ACL 2026).** Memora evaluates remembering, reasoning and recommending over weeks-to-months conversations. Its Forgetting-Aware Memory Accuracy (FAMA) penalizes reliance on obsolete or invalidated memories. This directly motivates MACE's stale-memory action and invalid-memory reuse measures. MACE should report FAMA on the official protocol when its executable artifacts and exact scoring are pinned; do not invent a crosswalk from DSM regret. Source: https://aclanthology.org/2026.findings-acl.1337/

## 2. Paradigm shifts that could move the field

### A. From retrieval relevance to decision-sufficient evidence

A high similarity score is not a reason to act. The memory subsystem should return a compact evidence packet that passes consumer-owned requirements for identity, scope, temporal validity, authority, lineage and independence. It should be allowed to return "insufficient", "contradicted", "expired" or "observation required".

Research question: can a system preserve the decision with a much smaller admissible evidence set than full context, while never using an invalid record to justify action?

### B. From facts to causal experience and state transitions

For agentic tasks, store the structure of experience: preconditions, action, observed outcome, environment/objective, delayed consequence, confounders and source lineage. Avoid converting correlations or tool success messages into causal rules. Ask whether the experience transfers to a new regime and whether it is still supported by current evidence.

This is the MACE complement to the AMA-Bench causal-trajectory direction—not a claim that a causal graph alone establishes causal truth.

### C. From passive memory to active observation policy

When evidence is missing, the system should choose among abstention, deferral, re-observation or an authorised probe. Score the expected or robust change in the final decision, not entropy reduction alone. Keep calibrated expected value of information separate from probability-free minimax-regret improvement.

The existing adaptive decision acquisition, robust policy oracle and decision-value retrieval work should remain the owners of query selection. DSM should consume their typed request/outcome receipts rather than create another planner.

### D. From append-only remembering to governed lifecycle

Correction, supersession, revocation, retention, selective repair, erasure and replay safety are core memory operations. A stale projection must not remain actionable because it is still retrievable. A successful revalidation should permit a new projection; it must not revive the old one.

The work in claim-relative invalidation, guarded projection rebuild and value-free metamemory summaries provides a foundation. The next capability is to connect these lifecycle receipts to externally observed decisions and measured downstream harm.

### E. From one universal memory substrate to a qualified substrate portfolio

Lexical search, dense embeddings, typed HDC, HDC-LTC, symbolic indexes, causal/event graphs, raw-source lookup and live observations have different cost/error profiles. A learned router should select the cheapest substrate or combination that meets the consumer's evidence and decision requirements.

Do not assume HDC or any single architecture wins every track. Demonstrate selective advantage on held-out workload families and compare against exact-scope, BM25, dense, graph and full-history controls.

### F. From single-user archives to role- and purpose-scoped shared memory

Collaborative memory needs explicit writer identity, role visibility, purpose limitation, independent provenance and conflict handling. A fact available to one role must not automatically be exposed to another. Evaluate leakage and unauthorized activation as hard failures, not as a small reduction in recall.

### G. From benchmark scores to evidence-carrying results

Every result should bind the exact source revision, model identifiers, prompt, data split, task-set digest, retrieval configuration, cost observations, grader, seed set, run receipts and independently checked outcome. Publish missingness and failure dispositions. A CI skip is not a validation pass; a synthetic fixture is not an external result.

## 3. Competitive target ladder

These are proposed research targets, not existing MACE results. Numerical goals should be preregistered and adjusted only from development-set variance and a power analysis—not tuned against held-out scores.

| Benchmark | Published reference | Proposed MACE target | Comparability gate |
|---|---|---|---|
| AML Textual | Resurface 61.22 open; MemoraX 69.26 commercial | ≥66.22 open (+5.00); stretch ≥71.26 commercial (+2.00) | Exact Cycle protocol, API, answerer/judge and current release |
| AML Multimodal | Khipu 43.26 open; MemoraX 53.68 commercial | ≥48.26 open; stretch ≥55.68 commercial | Exact multimodal tasks, payload and scoring version |
| AML Coding | InsightMem 63.33% open; MemoraX 68.00% commercial | ≥68.33% open; stretch ≥70.00% commercial | Relevant/noisy split, task and execution-budget match |
| AMA-Bench | AMA-Agent 57.22%, +11.16 pp over strongest paper baseline | First replicate; then ≥60.22 (+3 pp) matched target and ≥62.22 (+5 pp) stretch | Same backbone, trajectories, QA, judge and official split |
| MemoryArena | Multi-session agent/environment benchmark; no single directly portable score cited here | ≥3 pp task-progress gain at equal measured cost, then ≥5 pp stretch | Same task environment, policy, model, interaction limits and success definition |
| TimlMemBench | 208 samples, 1,566 sessions, 1,821 governed-use cases | ≥50% relative reduction in stale/revoked/scope-ineligible use vs strongest matched baseline; keep valid-evidence recall loss ≤1 pp | Exact raw-native protocol and reviewed artifact version |
| COLLABMEM | TMS-GRAPH++ 69.6% non-Oracle average | ≥74.6% (+5 pp) on same splits, with zero cross-role leakage in hard isolation tests | Fixed trace, role, answerer, judge, SDK and all five splits |
| Memora / FAMA | FAMA explicitly penalizes obsolete-memory reuse | ≥50% relative reduction in invalid-memory reuse while preserving current-memory recall | Official FAMA implementation, released dataset, same answerer and update schedule |

The AML commercial stretch is deliberately separate from the open-source goal. If MACE's model profile or hosting does not satisfy the relevant division's eligibility rules, do not claim that comparison.

For MemoryArena, the task-progress target is a proposed matched experimental objective, not a published leaderboard score. For TimlMemBench, the relative reduction must be computed over an identical case denominator and reported by failure type. For COLLABMEM, 69.6% is the published non-Oracle average; the target does not imply equivalent difficulty outside the official splits.

## 4. DSM's own primary research metrics

DSM must keep these dimensions separate:

1. **Decision regret:** realised evaluator loss minus the minimum loss over the declared action set.
2. **Memory gain:** paired no-memory loss minus memory-arm loss.
3. **Memory-induced harm:** fraction of paired episodes where memory increases loss versus no memory.
4. **Decision sufficiency:** minimum-loss decision with all required evidence and no invalid selected evidence.
5. **Evidence precision and required-evidence recall:** adjudicated against evaluator-private evidence requirements.
6. **Stale/revoked/forbidden/untrusted use:** raw count and per-decision rate, by family.
7. **Selective behaviour:** abstention and observation-request coverage, risk at each coverage level, and net decision value of observations.
8. **Lifecycle latency:** time or event count to propagate correction, revocation and erasure through each derived projection.
9. **Security:** cross-scope/cross-role disclosure, unauthorized activation, poisoning persistence, repair correctness and erased-memory resurrection.
10. **Full lifecycle cost:** ingestion, model calls, indexing, retrieval, verification, maintenance, storage, network, deletion, p50/p95 latency, CPU, RAM and energy.

A composite score may be a secondary visualisation only after the weights, units and non-compensatory failure gates are public. One critical privacy or revocation failure must not be offset by a better retrieval score.

## 5. Experimental design to beat strong baselines honestly

Use paired world/episode blocks with identical initial state, available observations, policy, deadlines and budgets. Randomize arm order; cluster-bootstrap at the independent world or source-work level. Keep separate development and held-out generator templates/seeds.

Minimum baseline family:

- no cross-session memory;
- bounded recent context;
- full-history diagnostic upper bound;
- BM25/lexical;
- dense embedding retrieval;
- hybrid lexical+dense;
- graph/temporal memory;
- qualified symbolic memory;
- static HDC;
- HDC-LTC;
- typed-content HDC;
- oracle evidence selector (evaluator-only upper bound).

Ablate evidence freshness, source authority, root independence, lineage, causal structure, typed fields, HDC dynamics, active observation, compression and local maintenance independently. Include negative controls for future-information access, evaluator leakage, wrong tenant, revoked source, stale summary, common-cause root duplication and erased-item resurrection.

Report per-family distributions and confidence intervals. A mean alone can hide a spectacular worst-group failure. Include worst-group regret, harm rate and coverage. All thresholds and grader versions must be frozen before confirmatory runs.

## 6. Near-term programme sequence

1. **Make the new DSM episode format executable.** Build candidate/private separation, strict response records, balanced-arm scoring and deterministic first-four-family generation in Rust.
2. **Connect the existing paired runner.** Add a versioned adapter from its candidate decision/evidence outputs and phase-attributed cost receipts; do not fork a second paired runner.
3. **Reuse dependency-aware knockouts.** Adapt the existing event/root/family knockout and cluster-level aggregation tools to DSM identifiers instead of creating a new ablation engine.
4. **Add hidden evaluator custody.** Candidate and evaluator must run under separate OS identities or services; the current preparation digest is a consistency binding, not a security boundary.
5. **Add multi-seed, held-out benchmark packs.** Use generated seeds for development and sealed templates/source lineages for confirmation.
6. **Submit AML separately.** Implement its exact Add/Search contract and run the official smoke/full evaluation. Native DSM scores cannot substitute for AML scores.
7. **Run external suites in their own lanes.** AMA-Bench, MemoryArena, TimlMemBench, COLLABMEM, Memora/FAMA and LongMemEval retain their native datasets, answerers, metrics and cost reporting.

## 7. Immediate engineering boundary

The DSM parent proposal (#13895) defines the episode protocol and synthetic seed fixtures. This child implementation adds a Rust reference generator, candidate/private split and response scorer. It deliberately does not claim:
- that a memory system has been benchmarked;
- that the six positive-control responses were generated by an agent;
- that the candidate process is isolated by the evaluator binary;
- that any response's resource counters or latency are independently measured;
- that MACE has beaten AMA-Agent, TMS-GRAPH++, AML leaders or any commercial product.

The relevant executable benchmark result will come only after exact-head Rust checks, independent candidate execution, sealed evaluator scoring, held-out runs and matched external comparison.

## Sources

- Agent Memory Leaderboard, Cycle 2 results (19 November 2026): https://agentmemoryleaderboard.ai/results/cycle-2/
- AML Cycle 2 results announcement and evaluation notes: https://agentmemoryleaderboard.ai/blog?post=phase2-results-20261119
- Zhao et al., AMA-Bench, ICML 2026: https://proceedings.mlr.press/v306/zhao26bs.html
- He et al., MemoryArena, ICML 2026: https://proceedings.mlr.press/v306/he26am.html
- Zhou et al., TimlMemBench, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.507/
- Wu et al., COLLABMEM, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.1194/
- Uddin et al., Memora / FAMA, Findings of ACL 2026: https://aclanthology.org/2026.findings-acl.1337/
