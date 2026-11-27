# MACE memory benchmark frontier, paradigm shifts and target ladder

**Snapshot:** 27 November 2026  
**Status:** literature and leaderboard synthesis; proposed targets are research goals, not measured MACE results.

## Executive thesis

The memory field is moving beyond "retrieve the right conversation snippet". Long-horizon agent systems need to maintain a governed, causally useful, temporally correct state that can support an action, detect when support is no longer valid, and know when to abstain or acquire new evidence.

MACE's opportunity is not to win every retrieval benchmark with another vector index. It is to establish an evidence execution contract across the full memory lifecycle:

1. source event and authority;
2. bitemporal observation and validity;
3. retrieval and candidate evidence;
4. independent admissibility and conflict resolution;
5. decision support and downstream action;
6. feedback and causal outcome;
7. correction, revocation, forgetting and projection rebuild;
8. measured end-to-end cost and reproducible qualification.

The decisive unit should be a **paired downstream decision under controlled evidence and lifecycle interventions**, not an isolated memory hit.

## 1. Competitive landscape and what each benchmark actually measures

These results are not one universal ranking. Do not compare scores across benchmarks unless the task, answerer, judge, context budget, interaction budget, release and cost accounting match.

### Agent Memory Leaderboard — Cycle 2

The published Cycle 2 leaders were:

| Track | Open-source leader | Commercial leader |
|---|---:|---:|
| Textual | Resurface — 61.22 | MemoraX — 69.26 |
| Multimodal | Khipu — 43.26 | MemoraX — 53.68 |
| Coding | InsightMem — 63.33% | MemoraX — 68.00% |

The leaderboard also exposes uneven capability profiles. The leading textual open-source entry has stronger governance than personalization and streaming; the commercial textual leader is stronger overall but still has non-uniform capability scores. This argues for reporting a capability vector, not only the headline score.

Cycle 2 reported 402 formal full-evaluation tasks, 333 successful completions and 292 valid leaderboard records after review/correction. Evaluation volume exceeded 6,000 Textual instances and included 150 base Coding tasks in relevant/noisy conditions. A submission must follow the exact division constraints and official scorer; native DSM metrics are not AML scores.

### Long-horizon and agentic memory

**AMA-Bench (ICML 2026).** AMA-Agent reports 57.22% average accuracy and an 11.16 percentage-point lead over the strongest baseline in that paper's matched experiment. Its focus includes causal inference, state updating and state abstraction across long interaction trajectories. This is an important comparator for MACE's causal-experience line, but its score is not directly comparable to AML.

**MemoryArena (ICML 2026).** It evaluates multi-session Memory-Agent-Environment interaction across interdependent tasks. Its central methodological lesson is that good long-context QA does not guarantee competent future action. Use it to test transfer into a live task environment rather than treating it as a static retrieval test.

**LongMemEval-V2.** The parent DSM programme's current review records 451 questions, up to 500 trajectories and approximately 115 million tokens. AgentRunbook-C is reported at 74.9% on the small setting and 70.1% on medium, with reported latency of 108.3s and 139.9s respectively. This suggests an accuracy-latency frontier target: match or beat accuracy without hiding latency, retrieval budget or reader cost.

### Governance, forgetting and shared memory

**TimlMemBench (Findings of EMNLP 2026).** The released benchmark reports 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases. It distinguishes a canonical-pool track from raw-native ingestion, which is important: a system that reads a supplied clean pool is not necessarily good at ingesting, updating and governing raw conversation history.

**COLLABMEM (EMNLP 2026).** It evaluates multi-agent memory visibility, access, provenance, routing, conflict resolution and retention. The scaled test has 174 scenarios and 3,359 probes; TMS-GRAPH++ reports the strongest non-Oracle mean of 69.6% across five splits. MACE must use the same split and harness before making a direct claim.

**Memora / FAMA (Findings of ACL 2026).** Forgetting-Aware Memory Accuracy penalizes reliance on information that has become obsolete or invalidated. This is aligned with MACE's focus on supersession, revocation, correction and deletion, and should be evaluated through the official metric rather than approximated by an unrelated DSM score.

**MemSecBench (2026 preprint).** This line adds adversarial memory poisoning, persistence, repair and post-attack recovery. It is a security evaluation, not simply a memory QA leaderboard. Treat unauthorized activation and persistence of retracted poison as hard safety outcomes.

## 2. Paradigm shifts that could move the field

### A. From relevance retrieval to decision-sufficient evidence

Similarity is a ranking signal, not a warrant for action. The memory system should return a compact evidence packet whose claims, scope, lineage, observation time, validity interval and authority can be checked independently.

The consumer should be able to distinguish:
- retrieved candidates;
- evidence admitted as current and in-scope;
- evidence actually selected to support the action;
- required evidence that was missing;
- conflict or insufficiency that forces abstention or a new observation.

A key research question: can MACE preserve the downstream decision with materially less admissible evidence than full history, without increasing regret, stale use or harm?

### B. From timeless facts to bitemporal epistemic state

A record has both a time when it was observed and a time interval in which it is valid. Future observations must not leak into past decisions. A later-observed backdated revocation may change what is considered valid now, but must not rewrite what the system could have known at the earlier decision point.

Corrections, supersessions and revocations must flow through every derived representation. A projection should become stale or quarantined immediately when its support no longer qualifies; revalidation should authorize a fresh projection rather than revive the old one.

### C. From stored snippets to causal experience and state transitions

Store preconditions, intervention, action, observed outcome, delayed consequence, confounders, source roots and identification status. Do not promote correlation, a successful tool call or a model-generated explanation into a causal rule.

Evaluate whether an experience transfers to a new regime, whether the claimed causal relation survives counterfactual ablation, and whether a correction invalidates downstream learned policies.

### D. From passive retrieval to active observation and value-of-information policy

When the memory packet is insufficient, the system should choose among abstention, a bounded query, an authorised probe, deferral or a human escalation. Measure net downstream value, not entropy reduction alone.

Use the existing exact expected-value / robust policy oracle and paired runner. Do not create a second query-planning or paired-intervention engine inside the benchmark harness.

### E. From a single substrate to a qualified memory portfolio

Lexical/BM25, dense embeddings, hybrid retrieval, symbolic records, temporal/event graphs, typed HDC, HDC-LTC and raw-source lookup will have different latency, cost, update and error envelopes.

Route by workload and evidence requirement. An HDC result is valuable where it wins on a held-out task family at a matched cost; it is not a universal win by architecture label. Maintain simple exact-scope, BM25, dense and full-history controls.

### F. From one private archive to purpose-, role- and tenant-scoped shared memory

A record visible to one agent role must not silently become available to all roles. Provenance, visibility, purpose, retention and deletion authority need explicit semantics. Leakage is a non-compensatory failure: a better QA score cannot offset a cross-tenant disclosure.

### G. From benchmark scores to evidence-carrying results

Every result should bind:
- exact dataset and split digest;
- task/world/cluster identifiers;
- model and artifact digests;
- policy, prompt and reader seed;
- run manifest and adapter versions;
- evaluator and grader versions;
- retrieved and selected evidence receipts;
- raw and phase-attributed costs;
- per-family results and failure dispositions.

Publish missingness. A CI skip is not a test pass; a synthetic fixture is not an agent run; a candidate-reported latency is not a host measurement.

## 3. Proposed competitive target ladder

These are goals to preregister after a development pilot and power analysis. They are not current MACE scores.

| Benchmark | Published reference | MACE target | Qualification condition |
|---|---:|---:|---|
| AML Textual | Resurface 61.22 open; MemoraX 69.26 commercial | ≥66.22 open; stretch ≥71.26 | Same Cycle protocol, division eligibility, official scorer and model/tool constraints |
| AML Multimodal | Khipu 43.26 open; MemoraX 53.68 commercial | ≥48.26 open; stretch ≥55.68 | Same payload, task set, context and official scoring |
| AML Coding | InsightMem 63.33% open; MemoraX 68.00 commercial | ≥68.33% open; stretch ≥70.00 | Same relevant/noisy split, harness and execution budget |
| AMA-Bench | AMA-Agent 57.22%; +11.16 pp over its strongest matched baseline | Replicate first; then ≥60.22%, stretch ≥62.22% | Same backbone, trajectory release, judge and official split |
| MemoryArena | Multi-session agent-environment task benchmark | ≥3 pp task-success gain at equal measured total cost; stretch ≥5 pp | Same environment, reader, action policy, interaction limit and success definition |
| LongMemEval-V2 | AgentRunbook-C 74.9% small / 70.1% medium | Beat by ≥3 pp while matching or improving p50/p95 latency and total cost | Same official release, track, answerer, scoring and context limits |
| TimlMemBench | 1,821 governance / activation cases in the reviewed release | ≥50% relative reduction in stale, revoked or scope-ineligible use with ≤1 pp loss in valid-evidence recall | Same raw-native track and case denominator |
| COLLABMEM | TMS-GRAPH++ 69.6% non-Oracle mean | ≥74.6% on matched five-split evaluation and zero cross-role leakage in hard isolation tests | Same scenarios, SDK, answerer, judge and splits |
| FAMA | Official forgetting-aware metric | ≥50% relative reduction in invalid-memory reuse while preserving current-memory recall | Official dataset, metric and update schedule |
| MemSecBench | Poisoning persistence / repair / recovery evaluation | Zero successful activation of revoked poison after the declared repair SLA; no more than 1 pp regression on valid-memory tasks | Same attack budget, threat model, evaluator and recovery window |

For AML, open-source and commercial targets are intentionally separate. Only make a commercial-track comparison when MACE meets that division's published eligibility rules. For any benchmark, beat a baseline only after reproducing it in the same environment and reporting confidence intervals.

## 4. DSM's primary metrics

Do not collapse these into a single weighted score:

1. **Decision regret:** chosen-action loss minus the minimum loss over the evaluator-defined action set.
2. **Paired memory gain:** no-memory loss minus memory-arm loss under the same frozen downstream policy and reader seed.
3. **Memory-induced harm:** fraction of paired episodes where memory increases loss.
4. **Decision sufficiency:** optimal action, all required support present, no inadmissible retrieved or selected evidence, and no unresolved retrieved contradiction.
5. **Retrieval precision and selected-support precision:** report separately.
6. **Required-evidence recall:** report separately from relevance ranking.
7. **Stale / revoked / untrusted / forbidden reuse:** count and per-decision rate by family.
8. **Selective risk:** regret and harm at each coverage / abstention level; observation requests and net value of information.
9. **Lifecycle propagation:** events and wall time until correction, revocation or erasure reaches every derived projection.
10. **Security:** unauthorized disclosure, role leakage, poisoning persistence, resurrection after erasure and repair success.
11. **Total lifecycle cost:** ingestion, model calls, indexing, retrieval, verification, maintenance, storage, network, p50/p95 latency, CPU, RSS, energy and money.

A composite score may be a secondary plot only. Privacy, authorization and revocation failures must remain hard gates and cannot be offset by higher recall.

## 5. Baselines and experiment design

Minimum architecture and policy controls:
- no cross-session memory;
- bounded recent-context window;
- full-history diagnostic upper bound;
- exact-scope deterministic lookup;
- BM25 / lexical retrieval;
- dense embedding retrieval;
- hybrid lexical+dense retrieval;
- graph / temporal memory;
- qualified symbolic memory;
- static HDC;
- HDC-LTC;
- typed-content HDC;
- evaluator-only oracle evidence selector.

Use identical model, prompt, downstream policy, decision seed, context budget, tool limits and action environment across paired arms. Randomize arm execution order. Resample independent world/source clusters, not individual questions, when episodes share a world. Report family-level distributions, worst-group regret, confidence intervals, failure dispositions and total cost.

Ablate one property at a time: temporal validity, authority, root independence, scope, lineage, causal identification, typed fields, HDC dynamics, active observation, compression and lifecycle maintenance. Include negative controls for future information, source poisoning, duplicate common-cause evidence, wrong tenant, revoked source, stale projection, missing metadata and erased-memory resurrection.

Keep development generator templates and seeds separate from sealed confirmatory templates and source lineages. Freeze the task set, grader, run manifest, thresholds and analysis plan before confirmatory runs.

## 6. Immediate implementation sequence

1. Reuse the parent DSM Rust library for paired action-loss, regret and retrieval scoring.
2. Finish the opt-in JSONL CLI and strict candidate-run / evaluator-private schemas.
3. Add the shared candidate-visible bitemporal source-event projection and evaluator-private pair/cluster manifest.
4. Reuse the paired intervention runner's phase cost receipts and #13836 cluster-aware inference. A pure adapter must return NotComparable if source authority/time/lineage metadata is absent.
5. Add held-out scenario generation, hidden evaluator custody and negative-control suites.
6. Add leave-one-out runs through the existing scorer; do not infer causal contribution from retrieval overlap.
7. Submit AML and external suites through their native protocols, each in a separate reproducible lane.

## 7. Current implementation boundary

The paired scoring core, candidate-only episode schema and run-manifest contract live in the parent DSM proposal. The new CLI is a thin adapter around that core. It adds exact candidate/artifact binding, bitemporal source-event checks, selected-versus-retrieved evidence accounting and paired JSONL orchestration.

The candidate/evaluator split does not by itself create process isolation. The evaluator-private file must be held outside the candidate's mounts, prompts, logs and telemetry. The candidate hash and artifact digest are consistency bindings, not authenticity or custody attestations.

The CLI has not been compiled or executed in this environment because the Rust toolchain is unavailable here. Eight source-level tests are authored; they are not passing tests until executed. The benchmark targets above remain proposals, not achieved results.

## Sources

- Agent Memory Leaderboard, Cycle 2 results (19 November 2026): https://agentmemoryleaderboard.ai/results/cycle-2/
- AML Cycle 2 evaluation volume and methodology: https://agentmemoryleaderboard.ai/blog?post=phase2-results-20261119
- Zhao et al., AMA-Bench, ICML 2026: https://proceedings.mlr.press/v306/zhao26bs.html
- He et al., MemoryArena, ICML 2026: https://proceedings.mlr.press/v306/he26am.html
- Zhou et al., TimlMemBench, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.507/
- Wu et al., COLLABMEM, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.1194/
- Uddin et al., Memora / FAMA, Findings of ACL 2026: https://aclanthology.org/2026.findings-acl.1337/
