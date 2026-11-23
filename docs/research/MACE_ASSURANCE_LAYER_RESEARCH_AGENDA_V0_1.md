# MACE: Memory Assurance and Counterfactual Evaluation
## Research agenda v0.1 — 2026-11-23

**Status:** research agenda and proposed qualification protocol. Targets are hypotheses, not results. No benchmark score, security qualification, production erasure guarantee, or SOTA claim is established by this document.

## 1. Thesis

MACE should be a backend-neutral assurance layer for systems that retain and reuse experience.

The field has become very good at building useful memory representations: lexical indexes, dense retrieval, extracted facts, event timelines, entity graphs, procedural notes, coding-agent memory, and multimodal trajectory stores. The next contribution should not be another representation that asks users to trust its own summaries.

**Research hypothesis:** a memory system should be evaluated by whether it preserves decision-relevant information, whether that information remains authorized and admissible throughout its lifecycle, and whether an independent evaluator can measure its causal influence on downstream decisions.

MACE separates three concerns:

1. **Representation and retrieval:** what experience can be found, in what form, at what cost and latency?
2. **Evidence qualification:** who or what produced it, under which authority, rights, scope, time interval, derivation chain and lifecycle state?
3. **Decision consequences:** what changes because this evidence is present, absent, stale, poisoned, corrected, compressed or erased?

A backend can be excellent at retrieval and still fail evidence qualification. Another backend can retrieve less while preserving better decision utility and stronger governance. MACE must make both visible.

## 2. Architectural principle

Keep retrieval replaceable and assurance independent.

```text
Raw experience
  ├─ conversations and documents
  ├─ web/enterprise trajectories and screenshots
  ├─ tool results and observations
  └─ scientific measurements
          |
          v
Replaceable memory backends
  ├─ BM25 / lexical
  ├─ dense / hybrid / reranking
  ├─ extracted facts and summaries
  ├─ event timelines and entity graphs
  ├─ procedural/documentary memory
  └─ HDC / typed / temporal representations
          |
          v
Candidate evidence envelope
          |
          v
Independent qualification
  ├─ byte integrity
  ├─ origin authenticity
  ├─ issuer authority and purpose
  ├─ rights and scope
  ├─ semantic support / contradictions
  ├─ lineage and common-cause independence
  └─ freshness, revocation and erasure state
          |
          v
Decision-sufficient context
  ├─ retrieve more / inspect source / ask / acquire
  └─ abstain when evidence is insufficient
          |
          v
Independent counterfactual evaluator
  ├─ accuracy, utility and regret
  ├─ memory influence and decision flips
  ├─ repair and benign-memory preservation
  └─ all-in cost / latency / replay receipts
```

The candidate-evidence envelope is an interchange format, not a trust certificate. Backend-supplied rank, source IDs, parent IDs and digests are untrusted claims until an independent qualification service verifies them.

## 3. Six paradigm shifts

### 3.1 From similarity retrieval to decision-sufficient memory

A memory is useful when it preserves enough evidence to make a good decision, not merely when it returns a semantically similar passage.

Let (a_M) be the action selected using memory representation (M), (a^*) the evaluator's optimal action, (y) the evaluator's outcome, and (L) a preregistered task loss:

[
R(M)=L(a_M,y)-L(a^*,y)
]

The optimization target is a cost/utility frontier, not maximum compression:

[
min_M left(
lambda_s S(M)+
lambda_i C_{mathrm{ingest}}(M)+
lambda_q C_{mathrm{query}}(M)+
lambda_v C_{mathrm{verify}}(M)+
lambda_r R(M)
ight)
]

subject to a preregistered regret bound and hard rights/authority constraints.

Measure:
- absolute regret against evaluator optimum;
- regret against raw/full-context and no-memory references;
- compression-induced regret and decision flips;
- expected and tail regret (including CVaR or worst-family reporting);
- abstention and calibration;
- full storage, update, verification, retrieval, reader and image costs.

Do not equate a low byte count with efficient memory if it silently excludes source retention, index overhead, replicas, embedding state, evidence verification, backups or reader tokens.

**Hypothesis:** the smallest memory that preserves a decision within a fixed regret tolerance is a more meaningful unit of memory quality than a summary judged in isolation.

### 3.2 From isolated records to a causal evidence graph

Every derived memory should preserve the complete set of parents and transformation receipts.

Useful edge semantics include:

- `asserted_by`
- `derived_from`
- `supports`
- `contradicts`
- `supersedes`
- `revokes`
- `valid_during`
- `erased_by`
- `shares_origin_with`

The graph must distinguish independent corroboration from repeated copies, aliases, model summaries and multiple records that share one root source.

Use controlled interventions:
1. Run a frozen task with the complete memory state.
2. Remove one evidence root from the candidate state.
3. Compute and remove/invalidate its dependent descendants and materialized views.
4. Replay the same task, query and reader configuration.
5. Measure the change in evidence exposure, action and regret.
6. Repeat for pairs, groups and trigger-conditioned combinations.

Distinguish record presence, retrieval exposure, decision influence and harmful consequence. A poisoned item that remains stored but is never retrieved is not equivalent to one that causes a harmful action.

### 3.3 From passive recall to active evidence acquisition

When evidence is missing, stale or contradictory, the system should select the next admissible information-gathering operation:

- retrieve another source;
- inspect the original document, turn or screenshot;
- query an independent source;
- check a superseding event or correction;
- ask the user;
- perform a measurement;
- abstain.

Use decision-relevant Expected Value of Sample Information (EVSI):

[
EVSI(q)=R_{mathrm{before}}-
mathbb{E}[R_{mathrm{after}mid q}]
-C_{mathrm{acquisition}}(q)
-C_{mathrm{delay}}(q)
-C_{mathrm{disclosure}}(q)
]

Authorization, source rights, scope, privacy approval and deadline feasibility remain hard gates. Predicted information value must never override them.

A lower-entropy observation can be useless if it cannot change the decision; a higher-entropy observation can be valuable if it resolves a costly action choice.

### 3.4 From deletion calls to verified lifecycle transitions

Correction, revocation and erasure must propagate through all derived representations:
- source records;
- extracted facts and summaries;
- embeddings and HDC projections;
- graph nodes and edges;
- retrieval indexes;
- caches and working sets;
- agent-specific memory;
- replicas and backups.

A repair receipt is issued only after affected views have been invalidated or rebuilt, unrelated benign information has been preserved, and the resulting state has been independently checked.

A monotonic counter or hash chain stored in the same rollback domain as the data is not sufficient protection against whole-store rollback. The trusted witness needs its own independently controlled persistence and trust anchor.

### 3.5 From average accuracy to reliability frontiers

Memory quality must be reported as a frontier over:
- answer accuracy;
- query latency;
- ingestion and update latency;
- context size;
- persistent bytes and actual RSS;
- verification and acquisition cost;
- decision regret;
- rights/authority violations;
- security and repair behavior.

For LongMemEval-V2, optimize the official LAFS frontier. For other datasets, publish complete accuracy/latency/cost curves rather than selecting a single favorable point.

### 3.6 From self-reported benchmarks to independently replayable evidence

Each run should bind:
- code commit, binary and container digests;
- exact dataset revision and split checksum;
- reader, judge, backbone, embedding and reranker;
- tokenizer, prompts, context packing and top-k;
- rank-fusion constants and candidate limits;
- domain, concurrency and resource limits;
- candidate output and evaluator output digests;
- raw timing distributions and complete resource accounting.

The candidate process must not receive evaluator labels, gold answers, hidden attack assignment or privileged question metadata. Evaluator custody must be independently controlled. A candidate/evaluator stream split in the same process is not independent custody.

### 3.7 From fact memory to retrieval-policy memory

The current LongMemEval-V2 frontier establishes an important second memory class: **memory about how to retrieve evidence**, not just memory containing evidence.

AgentRunbook-C V2 reports that repeated queries often share search strategies, useful file paths and exhausted search directions. Its online variant consolidates those retrieval traces into reusable strategy notes without using evaluation labels. Its published result is a concrete reminder that test-time learning about search policy can improve future queries, separately from storing more facts. See the [V2 method report](https://xiaowu0162.github.io/longmemeval-v2/agentrunbook-c-v2/) and [official leaderboard](https://xiaowu0162.github.io/longmemeval-v2/).

MACE should therefore maintain two explicitly separated memory planes:

1. **World/evidence memory:** source records, events, state transitions, facts, screenshots, documents and their rights/lineage/lifecycle qualification.
2. **Retrieval-policy memory:** reusable search plans, successful query decompositions, productive tool sequences, useful paths and already-exhausted directions, each bound to a domain, tool/API version, environment revision, observed success condition and freshness window.

A retrieval policy is not a fact about the world. “Search `orders/` before `history/` in environment revision X” can be reusable procedural evidence; it cannot establish that an order exists or that a retrieved file is authoritative. Strategy notes must never self-authorize access to sources or bypass evidence qualification.

**Safe consolidation rule:** the candidate process may learn only from its own permitted interaction trace: operations issued, files/paths actually inspected, evidence actually returned, source receipts, and whether the query process itself found sufficient evidence. It must not receive question IDs, hidden task families, gold answers, evaluator judgements or attack assignments. A retrieval attempt that returns no result is not automatically an exhausted direction; exhaustion requires a recorded, bounded search under a pinned environment and tool policy.

Proposed policy-memory protocol:
- Write a retrieval trace receipt for each opaque query invocation, binding the query digest, run/domain, backend/tool versions, executed actions, inspected source IDs, candidate-set digest, stop reason and actual cost.
- Consolidate only after the query is complete; emit a new strategy note with explicit parent trace IDs, an algorithm/version digest, applicability predicates, expiry/freshness and expected failure conditions.
- Treat promoted strategy notes as untrusted hypotheses until a later query validates them. Strategy memory may suggest where to search; it cannot alter source authority, rights or lifecycle state.
- Detect negative transfer: if an environment, site, API, permissions or schema revision changes, stale strategy notes are suspended or revalidated rather than blindly reused.
- Maintain per-domain/per-tool namespaces and prevent strategy leakage across tenants or benchmark domains unless explicit policy permits it.

**Required ablations:**

| Arm | Evidence memory | Retrieval-policy memory | Purpose |
|---|---|---|---|
| A | None | None | No-memory control |
| B | Raw BM25/dense | None | Retrieval-only baseline |
| C | None | Strategy only | Does policy help without factual stores? |
| D | Facts/events/notes | None | Structured memory baseline |
| E | Facts/events/notes | Strategy | Dual-memory hypothesis |
| F | Raw source + qualified evidence | Strategy with lineage/expiry | Full MACE design |
| G | AgentRunbook-C V2 Online | Native strategy notes | Strong external strategy-memory comparator |

Measure cold-start and warmed-up performance separately. Randomize question order; repeat with independent order permutations; use leave-one-domain-out and leave-one-tool-version-out transfer. Report accuracy, LAFS, latency, actions/tool calls, inspected bytes/tokens, consolidation cost, strategy-note growth, staleness detection, negative transfer, privacy leakage and downstream evidence admissibility. Include a “no consolidation” and a “random strategy-note” negative control. A warmed-up result must not be presented as a cold-start result.

The central MACE hypothesis is **dual memory with asymmetric authority**: strategy memory can optimize evidence acquisition, while only source-bound, independently qualified evidence can support a consequential decision.

## 4. Competitive landscape and comparison rules

### 4.1 Agent Memory Leaderboard Cycle 2

The official Cycle 2 results were published on 2026-11-19. There are nine separate leaderboards across textual, multimodal and coding tracks, with commercial, open-method and closed-source divisions.

| Track | Open-method leader | Commercial/closed leader |
|---|---:|---:|
| Textual | Resurface 61.22 | MemoraX 69.26 |
| Multimodal | Khipu 43.26 | MemoraX 53.68 |
| Coding, resolved | InsightMem 63.33% | MemoraX 68.00% |

These are separate track measures. Do not pool them into a single cross-track score.

The textual leaderboard reports capability-level dimensions including explicit fact recall, relational/multi-hop reasoning, temporal/event sequencing, memory governance, personalization/care, workflow execution, epistemic safety/privacy and streaming. MACE comparisons should publish each dimension, not just the aggregate. An aggregate retrieval improvement must not hide a governance or privacy regression.

### Capability-level gap diagnosis

The same Cycle 2 textual table gives a much more useful engineering map than the overall score alone:

| Capability | MemoraX commercial | Resurface open-method | Gap |
|---|---:|---:|---:|
| Explicit fact recall | 70.85 | 62.98 | 7.87 |
| Relational / multi-hop | 64.40 | 58.94 | 5.46 |
| Temporal / event sequencing | 67.05 | 57.76 | 9.29 |
| Memory governance | 78.10 | 76.55 | 1.55 |
| Personalization / care | 54.70 | 38.82 | 15.88 |
| Workflow execution | 61.20 | 59.67 | 1.53 |
| Epistemic safety / privacy | 81.98 | 68.59 | 13.39 |
| Streaming | 63.11 | 49.65 | 13.46 |

These are published scores in the same Cycle 2 textual table, not a matched internal experiment. The gaps should guide hypotheses, not be treated as proof that one architecture causes the difference.

### Proposed capability-level targets

The initial target closes at least half of each open-to-commercial gap; the stretch target reaches or slightly exceeds the commercial reference. These targets are intended for the same Cycle 2 textual protocol and scoring dimensions, not as cross-benchmark thresholds.

| Capability | Open reference | Commercial reference | Initial MACE target | Stretch target |
|---|---:|---:|---:|---:|
| Explicit fact recall | 62.98 | 70.85 | 66.92 | 71.85 |
| Relational / multi-hop | 58.94 | 64.40 | 61.67 | 65.40 |
| Temporal / event sequencing | 57.76 | 67.05 | 62.41 | 68.05 |
| Memory governance | 76.55 | 78.10 | 77.33 | 79.10 |
| Personalization / care | 38.82 | 54.70 | 46.76 | 55.70 |
| Workflow execution | 59.67 | 61.20 | 60.44 | 62.20 |
| Epistemic safety / privacy | 68.59 | 81.98 | 75.29 | 82.98 |
| Streaming | 49.65 | 63.11 | 56.38 | 65.11 |

The targets deliberately emphasize the largest gaps—personalization/care, epistemic safety/privacy, streaming and temporal reasoning—while retaining a non-regression expectation for stronger governance/workflow dimensions. They are goals, not measured MACE capabilities.

**Research implications:**

1. Governance and workflow are comparatively close among the current leaders. MACE should seek differentiation through verifiable lineage, correction, rights-aware retrieval and replay, not simply assume those dimensions are solved.
2. Personalization/care, epistemic safety/privacy and streaming have the largest open-to-commercial gaps. These should become explicit ablation and qualification families: user-controlled memory scope, purpose limitation, retention minimization, preference correction, sensitive-data refusal, streaming consistency and late-event reconciliation.
3. Temporal reasoning needs an explicit bitemporal model: distinguish when an event was true from when the system learned or corrected it. Replaying “what did the agent know at decision time?” should be a first-class benchmark, not a side effect of timestamps.
4. A system that improves personalization while weakening privacy is not a net improvement. Publish per-capability deltas and rights/authority violation counts alongside aggregate AML scores.


Sources:
- Official results: https://agentmemoryleaderboard.ai/results/cycle-2/
- Resurface: https://github.com/lilysttail/Resurface
- Khipu: https://github.com/Lv999APTITUDE/khipu
- InsightMem: https://github.com/Kline-song/InsightMem

### 4.2 LongMemEval and LoCoMo

Agent Zero Memory reports 95.60% LongMemEval and 93.60% LoCoMo in its August 2026 paper setup. It combines episodic timelines, associative entity-event graphs and citation-locked documentary memory.

Redis reports 86.5% task-averaged LongMemEval Small accuracy for Remis + Instruct on a common GPT-4o backbone in its September 2026 report. Its result illustrates complementarity between raw topic-segmented evidence and extracted facts. The report also warns that reproductions of published systems can differ substantially when reader/backbone/judge configurations are normalized.

These numbers are reference points, not matched measurements. LongMemEval Small, original LongMemEval, LongMemEval-V2 and AML Cycle 2 are different protocols. Reproduce the native setup first; then add a separate matched-backbone comparison.

Sources:
- Agent Zero Memory: https://arxiv.org/abs/2608.29606
- Redis Agent Memory benchmark: https://redis.io/resources/agent-memory-benchmark-report/
- LongMemEval: https://github.com/xiaowu0162/LongMemEval

### 4.3 LongMemEval-V2

The official benchmark contains 451 manually curated questions, five memory abilities, web and enterprise domains, up to 500 trajectories per haystack and up to 115M tokens in the largest haystacks. It assesses static state recall, dynamic state tracking, workflow knowledge, environment gotchas and premise awareness.

The current public Small-tier frontier reports:
| System | Family | Accuracy | Average query latency | LAFS gain |
|---|---|---:|---:|---:|
| AgentRunbook-C V2 Online | Coding-agent controller + online strategy consolidation | 75.6% | 130.5s | +2.8231 |
| Perpetual Recall V3 | Hybrid retrieval | 53.2% | 1.96s | +1.0955 |
| OPS-Agentic-Memory-v1 | Coding-agent controller | 81.4% | 153.6s | +0.3223 |

The highest peak accuracy does not imply the best frontier point. Perpetual Recall's lower accuracy but much shorter latency is why a full frontier is essential; OPS reports higher accuracy but a lower LAFS gain than AgentRunbook-C V2 Online.

A separate AgentRunbook-C V2 controller report gives additional operating points under named backbones/reasoning tiers:
- GPT-5.4-mini: low 65.41% at 26.11s, medium 72.73% at 56.68s, xhigh 75.61% at 130.54s.
- GPT-5.6 Luna: medium 66.52% at 24.72s and xhigh 75.17% at 48.94s.


Do not merge these rows with the leaderboard's single online result or compare different backbone/tier runs as if they were the same experiment. Their value is methodological: report multiple frontier points, identify the exact reader/backbone/reasoning budget, and measure online consolidation cost rather than only offline retrieval latency.

The highest peak accuracy is not necessarily the best frontier point. MACE should optimize LAFS and publish its operating points.

The official LAFS integral spans 1–200 seconds on a log-scaled latency axis. Each operating point combines paired web and enterprise runs using example-count-weighted averages. Report Small and Medium separately, and include task-family breakdowns, actual query wall time and resource cost.

MACE targets:
- Initial: achieve **at least +3.2 LAFS gain** on Small under the pinned official reader/judge and paired web/enterprise protocol.
- Stretch: **+4.0 or greater LAFS gain**.
- Report accuracy at 1, 5, 10, 30, 60, 120 and 200 second budgets, with p50/p95/p99 latency, memory size, reader tokens and full infrastructure costs.
- Do not claim multimodal retrieval until `query_image` is actually used in retrieval/ranking and matched image-query cases are evaluated.

Sources:
- Official repository and benchmark definition: https://github.com/xiaowu0162/LongMemEval-V2
- Official leaderboard and LAFS: https://xiaowu0162.github.io/longmemeval-v2/
- AgentRunbook-C V2: https://xiaowu0162.github.io/longmemeval-v2/agentrunbook-c-v2/

### 4.4 Memory security portfolio

Keep the protocols separate and reproduce their native tasks and scoring.

| Protocol | Coverage | MACE use |
|---|---|---|
| MemSecBench | 310 cases across 48 contexts; Write–Execute–Forget and selective repair | End-to-end persistence, consequence and repair |
| MemoryDojo | 155 multi-session trajectories, four domains, five memory systems | Long-horizon cross-session propagation |
| MemPoison | 1,227 cases: L1 353, L2 253, L3 621 | Direct, compositional and trigger-conditioned poisoning |
| Agent Memory Integrity at Rest Internet-Draft | Ten storage edits, three controls; includes T9 whole-store rollback and T10 erasure resurrection | Restart/replay, tampering, rollback and erasure durability |

The November 2026 at-rest document is an Internet-Draft, not an RFC or published standard. Its T10 case is distinct from T9: restoring a copy older than a write tests rollback; restoring a copy older than an erasure tests whether a supposedly erased item can reappear. The current draft reports that six of seven measured subsystems served erased data again under T10. It requires edit verification and no-op/restart controls; inapplicable cases are not passes.

Sources:
- MemSecBench: https://arxiv.org/abs/2607.27080
- MemoryDojo: https://www.microsoft.com/en-us/research/publication/memorydojo-benchmarking-poisoning-through-llm-agent-memory/
- MemPoison: https://arxiv.org/abs/2607.14651
- Agent Memory Integrity at Rest draft: https://datatracker.ietf.org/doc/html/draft-khandelwal-bpm-agent-memory-integrity-00

## 5. MACE qualification targets

These are proposed research gates, not measured results.

### 5.1 Retrieval and decision utility

| Track | Reproduction gate | Initial MACE target | Stretch |
|---|---|---:|---:|
| AML textual | Reproduce pinned open-method track within protocol tolerance | 64.22 | 70.26+ |
| AML multimodal | Reproduce pinned Khipu multimodal branch/config | 46.26 | 54.68+ |
| AML coding | Reproduce pinned InsightMem setup | 66.33% resolved | 69.00%+ resolved |
| LongMemEval Small / LoCoMo | Native reproduction, then same reader/backbone/judge | +1 pp over best matched baseline | +2 pp |
| LongMemEval-V2 Small | Official paired-domain run | +3.2 LAFS gain | +4.0 LAFS gain |
| Decision sufficiency | Full-context and no-memory controls | ≤1 pp compression-related regret at selected quality point | Lower all-in cost at equal or lower regret |

AML initial targets exceed their open-method leader but do not exceed all commercial leaders; the stretch targets explicitly aim beyond the current commercial leader. Tracks remain separate.

For LongMemEval/LoCoMo, use matched deltas rather than asserting an absolute score across incomparable protocols. A single percentage point is meaningful only when the same dataset revision, reader, backbone, judge, prompt and token budget are used.

### 5.2 Lifecycle and adversarial qualification

Deterministic fixtures:
- zero unauthorized consequential memory uses;
- 100% propagation of required parent revocation through descendants and materialized views;
- zero stale-backup resurrection of an erased protected record;
- zero residual causal influence after a repair that claims completion;
- 100% preservation of unrelated benign memories in deterministic selective-repair fixtures;
- zero cross-user/cross-tenant disclosure or unauthorized reuse.

Empirical tasks:
- clean-task utility loss ≤1 percentage point at the selected defense operating point, with paired confidence intervals and worst-family breakdown;
- report first detection point (write, retrieval, action or audit); audit-only detection after serving is not prevention;
- separately report persistence, retrieval exposure, harmful consequence, attribution, repair precision/recall, benign retention and recovery time.

### 5.3 Decision-sufficiency and active acquisition

- At the chosen quality operating point, compression-related regret ≤1 pp, with uncertainty and worst-family reporting.
- Track decision flips and abstentions, not only answer accuracy.
- Measure all-in cost and latency, including verification, source inspection, acquisition, updates, image assets and reader/judge.
- On synthetic calibration fixtures, compare decision-relevant EVSI against entropy-only and seeded-random controls.
- On real tasks, use calibrated observation models and source receipts; do not count an unexecuted acquisition proposal as evidence obtained.

## 6. Required baseline matrix

Under one frozen reader/backbone/judge and a fixed evidence/context budget, compare:

1. no memory;
2. full-context reference;
3. raw-session BM25;
4. dense raw chunks;
5. BM25 + dense RRF;
6. neighbor-expanded raw evidence;
7. extracted facts only;
8. raw evidence + extracted facts;
9. event timeline + procedural/documentary notes;
10. entity/temporal graph;
11. typed/HDC retrieval;
12. MACE-qualified variants of the above.

For V2, add the official AgentRunbook-R and AgentRunbook-C baselines, **including AgentRunbook-C V2 Online**. Add the seven-arm evidence/policy-memory ablation in §3.7. Preserve text and image items separately; do not turn screenshots into synthetic captions and then call the result image retrieval. The consolidation process must be label-blind, and cold/warm, question-order and cross-version transfer must be reported separately.

For lifecycle, add clean, L1, L2 and L3 paired tasks and the T1–T10 storage edit suite with its controls.

For decision sufficiency, compare memory representations on identical world/task clusters, with evaluator truth invisible to the candidate process.

## 7. Statistical and reproducibility rules

- Preregister primary endpoints, margins, exclusions and stopping rules.
- Pair methods on the same task/world/trajectory and resample independent clusters, not individual retrieval rows.
- Use at least 30 independent clusters for exploratory intervals; target 100+ for stable subgroup analysis where feasible. Report when a dataset cannot support that count.
- Report 95% intervals, worst-family results, missingness and all exclusions.
- Correct for multiple primary comparisons or clearly label secondary analyses as exploratory.
- Do not combine AML, LongMemEval, LongMemEval-V2, LoCoMo or security protocol scores into a synthetic aggregate.
- Freeze benchmark labels and attack assignments outside the candidate process.
- Publish source pins, lockfiles, configs, manifests, raw output digests and replay commands.

## 8. Current engineering map

| Work | Contribution | Claim boundary |
|---|---|---|
| #13310 | Diagnostic decision-sufficiency frontier and synthetic EVSI planner | Typed generated corpus; no authenticated lineage qualification |
| #13320 | Adversarial lifecycle protocol and attack taxonomy | Protocol documentation; no external attack run |
| #13327 | Lifecycle reference kernel and pluggable witness contract | In-memory witness; identity adapter not a crypto verifier |
| #13346 | Source-preserving raw-session BM25 baseline | Tests authored, not executed; no dense retriever |
| #13347 | Text-only LongMemEval-V2 trajectory adapter | Screenshots retained as references before #13354; no official plugin at that point |
| #13354 | Screenshot path resolution and fail-closed context projection | Resolver only; no pixel-aware ranking |
| #13370 | Backend-neutral candidate evidence envelope | Structural validation, not evidence qualification |
| #13371 | Official LongMemEval-V2 Memory API shim plus Rust sidecar | Text BM25 with original screenshot evidence; query_image not used for ranking |
| #13386 | V2 retrieval-to-candidate evidence DAG integration | Explicit state/text/image/neighbor lineage, cross-language candidate-set commitment verification and durable query receipts; no issuer/rights/lifecycle qualification |

No row above should be interpreted as a CI pass or official benchmark score unless a later exact-head run receipt establishes it.

## 9. Execution sequence

### Stage 1 — Stabilize the protocol boundaries
- Review #13354, #13370 and #13371 independently.
- Run formatter, unit tests, Clippy and schema meta-validation on exact PR heads once a Rust runner is available.
- Run the Python ingress tests under the pinned official V2 repository.
- Add sidecar process restart/replay and full official save/load integration tests.

### Stage 2 — Produce native benchmark measurements
- Build the release sidecar and install the Python shim using the pinned V2 revision.
- Run official no-retrieval, raw-slice and AgentRunbook baselines first.
- Run MACE BM25 text retrieval; publish the full first-query and steady-state latency distributions.
- Measure Small and Medium, paired web and enterprise.
- Record screenshot bytes, file verification time, RSS, index size, storage and total reader tokens.

### Stage 3 — Improve retrieval without sacrificing source fidelity
- Reproduce AgentRunbook-C V2 Online as a distinct retrieval-policy-memory arm before building a MACE consolidation controller.
- Add a label-blind retrieval-trace receipt and bounded strategy-note consolidation prototype; compare cold, warm, shuffled-order and cross-version transfer.
- Keep retrieval-policy notes separate from factual/source evidence and require applicability, expiry and source lineage for every strategy note.
- Add pinned dense retrieval and an explicit hybrid fusion profile.
- Add a pinned reranker and query expansion only as distinct ablation arms.
- Add fact extraction as a separate memory pool; preserve raw evidence for audit and fallback.
- Add genuine query-image retrieval as a separate arm and report image-query capability separately.

### Stage 4 — Connect evidence qualification
- Use the candidate evidence envelope across raw session, trajectory, graph, HDC and fact backends; use #13386 as the first closed-lineage V2 reference adapter.
- Add independently authenticated issuer profiles, rights/scope verification and signed derivation/correction receipts.
- Attach lifecycle checkpoint and lineage qualification receipts; backend fields cannot self-authorize.
- Freeze candidate output before evaluator truth is opened.

### Stage 5 — Prove lifecycle repair and rollback behavior
- Implement an independently administered durable witness service with a trust anchor outside the memory-store rollback domain.
- Require atomic compare-and-append, authenticated checkpoint chain, monotonic generations, crash/failover tests and explicit trust-anchor recovery.
- Exercise T9 and T10 separately, including restart/no-op controls.
- Run L1/L2/L3 clean/poisoned paired tasks, selective repair and counterfactual replay.

### Stage 6 — Qualify the full system
- Publish native benchmark results and matched comparisons.
- Publish a separate MACE assurance report with lifecycle, causal and decision-sufficiency metrics.
- Treat any unrun, skipped, cancelled or missing status as pending—not success.
- Keep SOTA claims limited to exact matched protocols and independent evidence.

## 10. Strategic conclusion

The defensible contribution is not that MACE has another universal memory data structure. It is that multiple memory architectures can be tested under common, explicit standards for decision utility, evidence admissibility, source independence, rights, lifecycle, causal influence and replay.

Let BM25, dense retrieval, HDC, graphs, timelines and agent-managed memory compete on usefulness. Require all of them to answer to the same assurance protocol.

The research success condition is a reproducible demonstration that the assurance layer improves decision reliability and lifecycle safety without imposing unacceptable accuracy, latency, privacy or cost penalties.
