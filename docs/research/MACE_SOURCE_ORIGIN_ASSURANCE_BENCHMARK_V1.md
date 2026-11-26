# MACE Source-Origin and Provenance Assurance Benchmark V1

**Snapshot:** 2026-11-26  
**Status:** Proposed benchmark protocol. All numeric thresholds are targets, not achieved results.  
**Relationship:** Complements [MACE 2026 Benchmark Parity and Decision-Assurance Programme](https://github.com/Luminous-Dynamics/symthaea/pull/13701) and the source-origin verifier in [MACE Source-Origin Attestation V1](MACE_SOURCE_ORIGIN_ATTESTATION_V1.md).

## 1. Research question

Can a memory system demonstrate that the evidence used for a decision is not merely signed and internally consistent, but that the declared source and derivation bytes were independently acquired, correctly bound to the evidence graph, and not falsely presented as independent?

This benchmark isolates four claims that must not be collapsed:

1. **Byte integrity:** the bytes acquired now match the digests committed by the replayed graph.
2. **Origin authenticity:** a separately trusted authority attests to source identity and acquisition context.
3. **Lineage completeness:** all material upstream sources and transformations relevant to the evidence are represented.
4. **Causal independence:** sources sharing an upstream cause are not counted as independent support.

A correct SHA-256 check proves only byte equality. A valid DSSE signature proves that a pinned signer made a statement. Neither alone proves publication authorship, factual accuracy, rights, freshness or completeness of ancestry. The benchmark scores these separately.

## 2. Benchmark task families

### A. Exact artifact integrity

Each task supplies a graph commitment, source bytes, payload bytes and—when derived—transformation-receipt bytes. Negative controls mutate one artifact at a time while preserving lengths, alter only a suffix, substitute a valid but different source revision, truncate a stream, or change the evidence identifier without changing the bytes.

The verifier must accept only the exact source, payload and derivation bytes that match the graph. Root nodes must have no derivation receipt; derived nodes must provide the committed receipt. The fixture includes multi-megabyte and boundary-sized artifacts to test limits and aggregate accounting.

### B. Ancestry and transformation completeness

Gold provenance DAGs contain root sources, intermediate snapshots, extracted passages, transformed summaries and final candidate evidence. The evaluator injects:
- omitted parents;
- hidden shared datasets or model-generation dependencies;
- missing or substituted derivation receipts;
- dangling references and cycles;
- parent-set changes after qualification;
- sources whose content changed between acquisition and use.

Use two separately reported strata:
- **closed-world synthetic fixtures**, where the benchmark knows the complete graph and can score exact ancestry recall;
- **open-world acquisition tasks**, where completeness is only scored against independently collected public metadata and cross-resolver evidence. Do not describe this latter reference as omniscient ground truth.

### C. Common-cause and independence reasoning

Include source topologies that look independent by URL, publisher or file hash but are not causally independent:
- mirrors and syndicated articles;
- many pages derived from one report;
- separate reports sharing an upstream dataset;
- multiple model-generated summaries from one model output;
- multiple sensors sharing a clock, calibration source, power supply or physical vantage point;
- different vendors repackaging the same registry or commercial feed.

Vary the hidden common-cause depth, number of apparent sources, and whether the common cause is disclosed by only one acquisition path. Unresolved ancestry contributes zero independent roots. The evaluator grades root-family partitions, not citation counts.

### D. Authority, lifecycle and replay attacks

Test role/profile/key reuse across qualification, sufficiency, lineage and source-origin authorities; stale and revoked keys; profile rollback; expired receipts; replay across run, invocation, subject, DecisionNeed and graph digest; clock rollback; DSSE payload mutation; envelope substitution; and trust-profile replacement.

Add correction, expiry, revocation, rights withdrawal and erasure events while evidence is in active retrieval caches. The system must distinguish reversible epistemic quarantine from rights-based deletion, and it must not silently revive an expired or revoked origin claim.

### E. Disagreement and selective abstention

Create cases where independent acquisition operators disagree about publisher identity, parent edges, source revision, or common-cause groups. The correct outcome is often **indeterminate / re-acquire / abstain**, not majority-vote truth. Measure whether the system preserves disagreement and requests the cheapest useful next observation rather than collapsing conflicting assertions into a fabricated consensus.

## 3. Reference corpus design

Use three independently versioned corpora:

| Corpus | Purpose | Ground-truth status |
|---|---|---|
| Protocol fixtures | Exhaustive byte, signature, binding, expiry, duplicate, cycle and resource-limit controls | Exact by construction |
| Synthetic provenance benchmark | Hidden-parent, common-cause, transformation and adversarial resolver experiments | Exact only for the declared synthetic world |
| External acquisition corpus | Real sources with independently collected publisher, archive, dataset and transformation metadata | Partial; uncertainty and missing metadata must remain explicit |

Each case must contain:
- a frozen task ID and task-family label;
- source and payload artifact digests;
- graph and derivation manifests;
- independent acquisition records and collection time;
- expected root-family partition where the reference is known;
- rights and use-scope constraints;
- attack labels held out from the candidate system;
- scenario cluster ID for statistical correction;
- exact model, tool, resolver, trust-profile and resource-budget manifests.

The candidate resolver must not see hidden graph edges, gold root families or attack labels. Use public development fixtures and an independently controlled held-out set. Record benchmark contamination and tuning exposure.

## 4. Metrics and proposed targets

| Metric | Meaning | Initial target |
|---|---|---|
| Artifact mismatch detection | Reject altered source, payload or derivation bytes | 100% on exhaustive protocol fixtures |
| False artifact acceptance | Accept any deliberately altered artifact | 0 on exhaustive protocol fixtures |
| Ancestor artifact coverage | Every graph node, including non-candidate ancestors, has exact acquired artifacts and a verified receipt | 100% on closed-world fixtures |
| Ancestry completeness recall | Material known parent/derivation edges recovered by independent acquisition | ≥99.5% on frozen synthetic holdout; report per attack family |
| False-complete rate | Declare ancestry complete when a known required edge is missing | 0 on protocol fixtures; 0/3,000 held-out target with cluster-adjusted interval |
| False independent-root rate | Count common-cause roots as separate independent support | 0 on frozen common-cause fixtures |
| Root-family partition quality | Pairwise precision/recall and adjusted Rand index against the known graph | ≥0.99 on closed-world synthetic fixtures |
| Unknown ancestry handling | Treat unresolved or unverified ancestry as independent | 0 tolerance in exhaustive fixtures |
| Authority separation | Accept profile or signer reuse across protected roles | 0 on all deterministic role-separation controls |
| Stale-origin use | Use expired, revoked, corrected or superseded origin evidence | 0 on deterministic lifecycle fixtures |
| Selective risk / coverage | Unsafe acceptance rate as the system answers more cases | ≤1% at ≥80% supported coverage; publish the complete curve |
| Calibration | Coverage of declared confidence / risk bounds | At least nominal coverage, including critical strata |
| Resource overhead | Added p50/p95 latency, bytes, CPU, memory and all-in cost | Report; first establish a reproducible baseline before fixing thresholds |

The 0/3,000 held-out goal is not proof of zero field risk. Under independent Bernoulli trials it corresponds to a one-sided 95% upper bound near 0.1%. Shared scenario roots, repeated source families and attack templates invalidate the independence assumption; report effective sample size or cluster-aware confidence intervals rather than treating near-duplicate cases as independent trials.

## 5. Required ablations and baselines

Run the same frozen corpus and decision policies against:
1. digest-only integrity checks;
2. signed-origin statements without byte reacquisition;
3. source-lineage graph replay without origin verification;
4. current four-authority MACE consumer verification;
5. naive URL/publisher identity matching;
6. citation-count or URL-count independence;
7. two independent acquisition operators with disagreement-preserving abstention;
8. a future threshold/quorum origin profile (for example, 2-of-3 operators) with the same trust and resource limits.

Ablate each of source bytes, payload bytes, derivation receipts, ancestor coverage, profile separation, use-time revalidation and common-cause edges. Report where safety improves, where valid evidence is lost, and the cost of each additional verification stage.

The quorum arm is a research baseline until a threshold or multi-signature protocol and independently governed operators are implemented. It must not be represented as current MACE functionality.

## 6. External benchmark positioning

The official [Agent Memory Leaderboard Cycle 2 results](https://agentmemoryleaderboard.ai/results/cycle-2/) (published November 19, 2026) provide the current parity reference:

| Track | Open-source leader | Commercial leader |
|---|---:|---:|
| Textual | Resurface 61.22 | MemoraX 69.26 |
| Multimodal | Khipu 43.26 | MemoraX 53.68 |
| Coding | InsightMem 63.33% | MemoraX 68.00% |

These are separate tracks with different task sets and scoring definitions. Do not average them, compare them as a single ranking, or treat them as assurance scores. The corresponding commercial/open-source gaps are 8.04 textual points, 10.42 multimodal points and 4.67 coding percentage points.

For the textual capability table, the commercial leader MemoraX scores 54.70 on Personalization & Care, 63.11 on Streaming, 81.98 on Epistemic Safety & Privacy, 67.05 on Temporal & Event Sequencing, 70.85 on Explicit Fact Recall, 64.40 on Relational & Multi-hop Reasoning, and 78.10 on Memory Governance. The open-source leader Resurface scores 38.82, 49.65, 68.59, 57.76, 62.98, 58.94 and 76.55 respectively. The resulting gaps are:

| Textual capability | Commercial advantage |
|---|---:|
| Personalization & care | 15.88 points |
| Streaming | 13.46 points |
| Epistemic safety & privacy | 13.39 points |
| Temporal sequencing | 9.29 points |
| Explicit fact recall | 7.87 points |
| Relational / multi-hop reasoning | 5.46 points |
| Memory governance | 1.55 points |

These gaps suggest the most valuable memory-parity work is not just similarity retrieval: prioritize changing-preference tracking, continuous update latency, evidence-grounded epistemic safety and temporal state reconstruction. They are separate from source-origin assurance, and MACE has no external AML result yet.

The external benchmark suite should be run as complementary instruments:

- [AMA-Bench](https://proceedings.mlr.press/v306/zhao26bs.html) covers recall, causal inference, state updating and state abstraction over 205 trajectory samples and 2,460 QA pairs.
- [MemoryArena](https://proceedings.mlr.press/v306/he26am.html) evaluates interdependent multi-session Memory-Agent-Environment tasks, where memory must guide later actions rather than merely answer recall questions.
- [Memora/FAMA](https://aclanthology.org/2026.findings-acl.1337/) evaluates remembering, reasoning and recommending over long conversations and penalizes use of obsolete or invalidated memories.
- [MEMORA-Bench](https://aclanthology.org/2026.findings-emnlp.1211/) tests embodied memory on 45 hours of egocentric video, including retrospective faithfulness and prospective planning.
- [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) contains 451 curated questions over web and enterprise trajectories, with haystacks up to 115 million tokens; it measures answer accuracy and query latency.

A further complementary embodied-memory track is **MEMOBench** from Findings of EMNLP 2026: 30 history-dependent manipulation tasks, 1,500 expert demonstrations and 4,200 executable checkpoint instances. It labels process-level Storage, Update and Compression operations and reports that its strongest memory-module baseline reaches only 31.9% average task success. This is particularly relevant to MACE's streaming ledger and eventual Symtropy/robotics integration: measure memory operation correctness at intermediate checkpoints, not only terminal task success. [MEMOBench paper](https://aclanthology.org/2026.findings-emnlp.333/).

These benchmarks probe different capabilities and must be run with their exact official harnesses, models, answerers, task splits and resource budgets. None alone establishes source-origin truth or causal independence.

MACE therefore needs two scorecards:

- **Memory capability scorecard:** established benchmark parity, measured on matched models, prompts, task sets, graders and budgets.
- **Evidence assurance scorecard:** source integrity, origin authenticity, lineage completeness, common-cause handling, lifecycle correctness, calibrated sufficiency, regret and abstention.

Do not add the assurance score to AML or average across tracks. The assurance benchmark is a distinct contribution that should be published with its task manifest, reference graph format, verifier interface, scorer, confidence-interval method and independent replication package.

## 7. Research frontier beyond the current four-authority draft

The present consumer verifier checks exact bytes and a distinct fourth authority's signed assertion. The next frontier is **multi-operator evidence acquisition**:

- acquire the same source/ancestor through different networks, archives or custodians;
- compare immutable content, revision metadata and transformation receipts;
- preserve disagreements as typed observations rather than resolving them through an unqualified vote;
- require independent corroboration for high-consequence origin claims;
- use 2-of-3 or threshold attestations only when operator independence, correlated compromise and fallback rules are explicit;
- make disagreement, unavailability and unresolved parents reduce supported coverage rather than inflate source independence.

This turns provenance from a self-reported graph into an adversarially testable acquisition process. It still cannot prove universal truth: every result remains scoped to its source, collection time, evidence class, rights, trust assumptions and declared decision.

## 8. Reporting and anti-gaming rules

- Freeze datasets, source snapshots, graph manifests, prompts, model/tool versions, trust profiles and task inclusion before evaluation.
- Score the entire denominator, including abstentions, unavailable sources, failed fetches, timeouts and resource-limit rejections.
- Publish false accepts and false rejects separately; do not hide valid-evidence loss behind a single safety metric.
- Report each task family, ancestry depth, root count, lifecycle event, risk level and attack class, plus the worst supported stratum.
- Use paired seeds, scenario-clustered confidence intervals and at least one independently operated evaluator.
- Keep source-origin signer credentials and gold provenance records outside the candidate system and its rollback domain.
- Never treat a resolver's own graph as the gold reference for grading that resolver.
- Preserve all exact source/derivation digests, receipt envelopes, run IDs, commit SHAs and verifier versions needed for replay.
- Distinguish authored tests, locally executed tests, hosted CI, matched benchmark runs, independent replication and field validation.

This protocol is designed to make false independence and false completeness measurable, rather than allowing a signed, self-consistent lineage graph to be mistaken for verified provenance.


## 9. Multi-operator origin corroboration

The follow-on [MACE Source-Origin Witness Quorum V1](MACE_SOURCE_ORIGIN_WITNESS_QUORUM_V1.md) adds a policy-configurable per-node operator and failure-domain quorum over the fourth-authority source-origin result.

Add these metrics:
- **Per-node quorum coverage:** fraction of graph nodes with at least the required independently pinned operators and acquisition failure domains.
- **Shared-observation acceptance:** cases where two operator receipts reuse the same acquisition-record digest.
- **Source metadata disagreement:** witness disagreements over publisher identity or source revision.
- **Ancestry disagreement detection:** any signed witness parent, derivation or common-cause set that differs from the graph.
- **Majority-laundering rate:** cases in which the verifier accepts despite one valid signed witness reporting a known graph omission.
- **Authority key/profile reuse:** reused keys across the qualifier, sufficiency evaluator, lineage resolver, source-origin authority and witness operators.

Proposed targets are zero accepted shared acquisition records, zero accepted key/profile reuse, zero majority-laundered ancestry mismatches, and 100% per-node quorum coverage on supported cases. The system must abstain or return insufficient evidence when the minimum operator or failure-domain count is not met. A valid signed disagreement must remain visible and cannot be discarded as an outlier without a separately governed adjudication process.

The 2-of-3 policy is not a universal truth oracle: operator identity, failure-domain independence, publisher metadata and source completeness remain explicit trust assumptions. The external acquisition operator registry and fetch adapters must be independently audited and tested.


## External provenance interoperability

The follow-on [MACE W3C PROV and SCITT Interoperability V1](MACE_PROV_SCITT_INTEROP_V1.md) defines a JSON-LD context and proposed mapping for interoperable graph export, plus a separate SCITT transparency integration path. This complements the source-origin assurance benchmark but does not change its ground-truth requirements: neither PROV representation nor transparency-log inclusion proves source truth or completeness.


## 10. Streaming and bitemporal memory

The follow-on [MACE Streaming Temporal Session Ledger V1](MACE_STREAMING_TEMPORAL_SESSION_LEDGER_V1.md) adds a typed event-sourced reference state machine with separate valid/effective time and trusted observed time, exact event idempotency, correction/supersession, quarantine/revalidation, revocation and an explicitly limited in-memory erasure request.

The streaming benchmark must now include:
- out-of-order arrivals where valid time differs from observed time;
- correction chains with historical and current-state questions;
- stale derived summaries/plans after a corrected parent;
- exact event replay and event-ID conflict;
- sequence gaps and trusted-clock rollback;
- quarantine/revalidation and revocation;
- rights erasure across the whole dependency/index/cache/backup closure;
- multi-writer concurrency, crash recovery and unknown-outcome reconciliation.

Proposed initial targets:
- 100% deterministic replay and transition-fixture correctness;
- zero stale/superseded/revoked evidence uses in exhaustive lifecycle fixtures;
- 100% correction visibility in deterministic current-state queries;
- 100% valid-time/observed-time fidelity in the bitemporal test set;
- zero accepted event-ID reuse with different content;
- report p50/p95/p99 append-ack, durable-commit and Search-visible latency separately.

The current Rust module is an in-memory reference and does not yet meet durability, secure-erasure or hosted streaming SLOs. The production acceptance gate must require durable commit before acknowledgement, immediate Search visibility, transactionally allocated sequence numbers, idempotent replay after crashes, and independent deletion receipts from all derived stores.


## 11. Derived projection invalidation

The [MACE Derived-Memory Invalidation V1](MACE_DERIVED_MEMORY_INVALIDATION_V1.md) follow-on adds a projection-dependency graph for summaries, embeddings, preferences, plans, fact indexes and answer caches. Extend the benchmark from raw event lifecycle to transitive derivation lifecycle:

- direct and transitive invalidation after correction/supersession;
- quarantine and later revalidation without automatic revival of old projections;
- revocation and rights-erasure escalation after evidence was already quarantined;
- unrelated-source isolation, including shared and diamond-shaped dependency graphs;
- rebuild receipts bound to a new projection ID, current source events and current parent projections;
- erasure acknowledgements across every derived store and external storage domain.

Proposed targets are 100% direct and transitive invalidation on exhaustive fixtures, zero stale projection acceptance, zero unrelated false invalidation, zero old-projection resurrection after source revalidation, and complete downstream erasure receipts before any erasure-complete claim. Report false invalidation and rebuild cost alongside stale-use; indiscriminately invalidating every memory is not an acceptable way to obtain zero stale-use.


## 12. Derived projection closure and invalidation

The [MACE Derived-Memory Invalidation V1](MACE_DERIVED_MEMORY_INVALIDATION_V1.md) reference models derived memory as a dependency graph. Extend the benchmark to test the **complete downstream closure** after source corrections, quarantine, revocation and rights erasure. A memory system must not simply update the raw conversation while leaving stale summaries, embeddings, preferences, plans or answer caches available.

Add pairwise and transitive graph fixtures, diamond dependencies and shared projection parents. Require all descendants to transition out of Current, preserve unrelated projections, and prevent automatic resurrection after source revalidation. New versions must bind current source-event IDs, current parent projection IDs and independent derivation receipts. Report false invalidation, rebuild latency, rebuild tokens and update cost as well as stale-use rate.


## 13. Decision-value retrieval and value of information

The follow-on [MACE Decision-Value Retrieval V1](MACE_DECISION_VALUE_RETRIEVAL_V1.md) adds two separate retrieval-selection modes:

- **EVSI:** calibrated expected value of sample information, where each query's prior/likelihood model is bound by an independently verified calibration receipt.
- **Robust minimax regret reduction:** probability-free scoring over conservative possible-state/observation support sets.

Both modes enforce consumer-owned hard action constraints and require independently verified resource-profile receipts for tool calls, latency, context tokens, inspected bytes and utility cost. They only select queries with positive net value; otherwise they stop. Neither mode authorizes action or promotes a reusable retrieval strategy.

Benchmark additions:
- matched comparison against no-retrieval, BM25, dense, hybrid RRF, graph/causal, fixed top-k and full-context baselines;
- decision-flipping evidence and missing-requirement cases;
- redundant/shared-root sources and common-cause scenarios;
- miscalibrated, stale and shifted probability models;
- probability-free robust support tasks;
- resource/cost/latency/token budget sweeps;
- hard action prohibitions and abstention;
- sequential marginal-value stopping;
- downstream independently measured outcome and counterfactual regret.

Proposed targets: at least 10% lower paired decision regret than the strongest matched baseline; ≤1% unsafe actions at ≥80% supported coverage; zero hard-constraint violations; nominal calibration coverage; improved regret-vs-cost Pareto frontier; and zero strategy promotion without independent outcome receipts and held-out counterfactual qualification.
