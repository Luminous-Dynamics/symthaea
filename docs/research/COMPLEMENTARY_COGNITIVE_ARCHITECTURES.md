# Complementary Runtime Architecture: Typed Evidence + Predictive HDC

**Status:** architecture proposal; not yet implemented or benchmarked.  
**Date:** 2026-10-15  
**Purpose:** define a genuinely complementary runtime architecture for Symthaea without duplicating the SYM-ARCH-002 experiment infrastructure.

## Executive recommendation

Keep Symthaea's HDC and continuous-time dynamics as useful mechanisms, but stop treating one shared hypervector space as the universal source of truth for every kind of information. Prototype a **Typed Evidence and Predictive Runtime (TEPR)** in which:

- HDC/VSA is the associative and semantic *index* for related concepts, approximate retrieval, role–filler composition, and noisy cues.
- A typed evidence store is the authority for exact entity identity, relations, quantities, units, source, time, and contradiction status.
- Existing CfC/LTC components maintain and predict appropriate temporal state; they are not assumed to replace episodic memory or causal models.
- A resource-bounded coordinator chooses the least expensive reasoning path likely to meet the task's required quality.
- Independent validators check exact claims, constraints, provenance, and proposed actions where suitable checkers exist.
- Episodic, semantic, and procedural memory remain distinct but linked through evidence identifiers.

The hypothesis is not that symbolic systems are inherently superior, nor that every task needs all of these layers. The hypothesis is that **separating fuzzy association from typed truth conditions and evidence** may improve exactness and recoverability while preserving the benefits of HDC. It must be tested against simpler controls.

## Critical context: do not duplicate SYM-ARCH-002

Symthaea already has a staged, separate experiment program for stronger continual-compositional validation:

- [Issue #55 — SYM-ARCH-002 plan](https://github.com/Luminous-Dynamics/symthaea/issues/55)
- [PR #57 — SYM-ARCH-002A experimental core](https://github.com/Luminous-Dynamics/symthaea/pull/57)
- [PR #58 — hierarchical statistics and power planning](https://github.com/Luminous-Dynamics/symthaea/pull/58)
- [PR #65 — online acquisition and resource measurement](https://github.com/Luminous-Dynamics/symthaea/pull/65)
- [PR #66 — multiplicity-safe analysis plans](https://github.com/Luminous-Dynamics/symthaea/pull/66)

**Status snapshot checked 2026-10-15:** #57 was open and not a draft; #58, #65, and #66 were open drafts. This document neither assumes they have merged nor depends on their not-yet-landed code APIs.

SYM-ARCH-002 already proposes an evaluation core, stronger continual-learning controls, factorial mechanism ablations, task-free drift, deeper compositionality, and irregular-time stress tests. We should reuse that program's frozen data/seed identities, uncertainty methods, prequential measurements, resource accounting, and multiplicity control when eligible. Do not create a second bootstrap stack, change SYM-ARCH-001's frozen result, or call a mechanism-level result an end-to-end cognitive-architecture result.

### What SYM-ARCH-001 has already taught us

The frozen [SYM-ARCH-001 result](https://github.com/Luminous-Dynamics/symthaea/blob/main/docs/research/SYM_ARCH_001_RESULT_2026-08-25.md) was **NEGATIVE** for the tested 512-dimensional HDC-LTC-Hebbian candidate under its preregistered joint retention/composition/reversal criterion. It lost to the strongest controls by -0.2969 retention, -0.2539 held-out composition, and -0.4863 reversal-final accuracy. The simple online linear-SGD control had the best retention and compositional scores in that experiment; vanilla HDC led the reversal-final score. Low forgetting alone was not sufficient evidence of continual-learning strength.

The narrow conclusion is that the tested bundle did not outperform its controls on that task and rule. It does not condemn HDC or CfC generally. The post-hoc diagnosis points toward factorizing mechanisms and designing complementary, task-appropriate systems—not increasing complexity or tuning a losing bundle after the fact.

## Existing code inventory: extend these components, do not create a second graph

Source audit checked 2026-10-15. The public repo already has several of the architecture's building blocks, but the source-derived fields also show where a complementary integration could add value.

- [Knowledge extraction](https://github.com/Luminous-Dynamics/symthaea/blob/main/src/knowledge/extraction.rs) defines `ExtractedFact` with entities, predicate relations, causal/negation flags, source text, and extraction confidence. Its documented extractor is pattern-based and explicitly not full NLP.
- [HDC knowledge encoding](https://github.com/Luminous-Dynamics/symthaea/blob/main/src/knowledge/encoding.rs) defines `FactEncoding` with a composite `BinaryHV`, role vectors, source text, and encoding confidence. In this type, the vector is an index-like representation; it does not itself carry a stable external source locator, explicit measurement unit, valid-time interval, or derivation lineage.
- [Enhanced knowledge graph](https://github.com/Luminous-Dynamics/symthaea/blob/main/src/knowledge/graph.rs) already has `FactId`, cycle-based insertion/access metadata, confidence decay, corroboration/contradiction counters, domain tags, HDC similarity search, and contradiction alerts. This is the first graph/storage component to inspect and extend. Its cycle timestamps should not be silently treated as event-time or source publication time; its similarity ranking is not a calibrated probability.
- [Knowledge persistence](https://github.com/Luminous-Dynamics/symthaea/blob/main/src/knowledge/persistence.rs) already persists the graph to SQLite. Its current `FactRecord` stores the vector bytes, source text, confidence, domain, insertion cycle, and causal flag; it is not yet the full typed event/evidence contract proposed here. A concrete fail-closed defect was found in `load_facts()` and `load_causal_edges()`: `.filter_map(|r| r.ok())` could silently omit rows that fail typed decoding. Companion [PR #7472](https://github.com/Luminous-Dynamics/symthaea/pull/7472) removes this silent omission and adds strict `KnowledgeManager::try_new` initialization; the PR is open and CI is queued as of 2026-10-15, so its tests are not yet validated.
- [Reasoning context](https://github.com/Luminous-Dynamics/symthaea/blob/main/src/knowledge/reasoning_context.rs) already assembles HDC matches, causal chains, contradictions/uncertainty, and context for downstream reasoning. In the inspected `from_manager` path, a search hit is presently converted into text shaped like `fact:<id>`; the integration should trace whether and where the original source text is resolved before calling the result fully grounded.
- [Epistemic types](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-epistemic-types/src/global_ledger.rs) already define empirical (E), normative (N), and materiality (M) axes plus provenance variants. Check those types before proposing overlapping confidence/provenance enums; do not assume the levels alone establish evidence validity.
- [Evidence plane](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/core/symthaea-evidence-plane/src/lib.rs) means a *research integrity contract*: declared experiment mechanisms versus measured counters, expectation checks, run IDs, and a non-cryptographic config identity helper. It is not a factual provenance database. Reuse it for experiment-integrity checks where appropriate; keep its identity hash away from security-sensitive use.
- [Cognitive observation contract](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/core/symthaea-core/src/observation.rs) is intentionally narrow and currently exposes state dimensionality only. Its own documentation says to add methods only alongside a real consumer, not speculatively. Do not expand this as a generic catch-all API.

### Concrete gap to investigate first

Do **not** create a new `EvidenceStore`, `KnowledgeGraph`, `FactEncoding`, or global cognitive protocol yet. First trace one concrete path:

`ExtractedFact → FactEncoding → EnhancedKnowledgeGraph::insert/search → KnowledgePersistence → ReasoningContext::from_manager → actual consumer`

At each boundary record which fields survive: stable IDs; entity/relation structure; source locator; source/event time vs system cycle time; quantities and units; negation; extraction confidence vs source reliability; contradiction links; and provenance/derivation lineage. Then add the smallest backward-compatible missing contract and tests for the specific lost fields. Preserve legacy paths and use adapters rather than duplicating authority.

The system-level research contribution should be **faithful evidence resolution and explicit limits**, not a second graph/data model. This provides a much more falsifiable first milestone than an all-at-once architecture rewrite.

## Proposed architecture

~~~text
                         REQUEST / SENSOR EVENTS
                                  |
                                  v
                     [1. Typed Event Intake]
                     timestamp · source · rights
                     schema · payload identity
                                  |
                                  v
               +------------------+------------------+
               |                                     |
               v                                     v
    [2a. Typed Evidence Store]             [2b. HDC Semantic Index]
    entities / relations / units            associative and fuzzy retrieval
    source/time / validity / conflicts       concept cues / compositions
               |                                     |
               +------------------+------------------+
                                  |
                                  v
                    [3. Predictive Temporal State]
                    HDC-CfC/LTC where it fits
                    state age · horizon · surprise
                                  |
                                  v
                    [4. Bounded Work Coordinator]
                      fast → medium → deep path
                      latency / memory / compute budget
                                  |
                 +----------------+----------------+
                 |                |                |
                 v                v                v
            retrieve/query   causal/planning   exact tool/solver
                 |                |                |
                 +----------------+----------------+
                                  |
                                  v
                     [5. Independent Validation]
                     evidence · types/units · checks
                     verified / supported / unresolved
                                  |
                                  v
                      [6. Guarded Action Proposal]
                      preconditions · budget · rollback
                                  |
                                  v
                    [7. Outcome + Memory Feedback]
                    pre-update score / source trace
                    episodic capture / consolidation
~~~

This is a logical design, not a claim that these seven layers already exist as one integrated runtime.

### 1. Typed event intake

Every observation enters through a versioned envelope with at least:

- event time and receive/record time;
- source identity and source location/reference;
- payload or observation digest when stable content hashing is appropriate;
- modality and schema version;
- privacy/permission label;
- extraction method and extraction confidence, kept separate from source reliability.

Keep event time distinct from record time. A newly received report about an old event should not silently overwrite what the system knew at an earlier time. An invalid timestamp, failed parse, unknown unit, or missing source should remain visible rather than receive a plausible default.

### 2. Dual representation: HDC index plus typed evidence

**HDC is a retrieval/indexing layer, not the database of record.** Each vector or composed representation points to stable entity/evidence IDs. Similarity proposes candidates; it does not prove equality, entailment, causal direction, or source authenticity.

Typed facts should preserve subject, predicate, object/value, unit, valid-time interval, record-time interval, source reference, derivation lineage, confidence semantics, and status. Useful statuses include *observed*, *derived*, *hypothesis*, *contradicted*, *superseded*, and *unresolved*. The implementation should reuse existing knowledge encoding and causal components where their contracts fit, rather than add another competing graph or truth store.

The bridge between the two representations must be explicit and testable:

- Each HDC-indexed item resolves to an existing typed record or a declared hypothesis ID.
- Deleting or revoking evidence invalidates the associated index entries and dependent derived claims.
- A near match is never promoted to an exact identity without a typed identity-resolution step.
- Contradictory records can coexist with separate provenance; do not average mutually exclusive facts into one vector and lose the disagreement.
- A similarity score is not automatically a probability or calibrated confidence.

### 3. Predictive temporal state, not universal memory

Use the existing HDC-LTC/CfC implementation for temporal state that benefits from continuous-time evolution and variable sampling intervals. Preserve a clear separation between a state prediction and the observed event. Every prediction should state its horizon, assumptions, source-state version, and measurable target.

Do not assume CfC should own all temporal storage: its original research presents an efficient approximate closed-form model and notes limitations, including possible vanishing-gradient issues for long dependencies and the fact that CfC approximations do not necessarily retain the invertibility of ODE flows. Compare mixed-memory and simpler baselines on the workload rather than assuming continuous-time dynamics always help.

### 4. Separate memory by job

Use three linked memory roles:

- **Episodic:** what happened, when, in what context, and what was observed before/after an action.
- **Semantic:** consolidated facts, concepts, relations, and rules, each with sources, freshness, uncertainty, and invalidation links.
- **Procedural:** how to carry out a task, including preconditions, resource bounds, test fixtures, failure modes, and postconditions.

Consolidation should create derived records with lineage; it must not delete the original evidence by default. Recency and similarity are retrieval features, not proof of current truth. Memory eviction should be explicit and auditable, and privacy/retention policies must override ordinary importance scoring.

### 5. Resource-bounded coordinator

Route work using a small policy over task class, uncertainty, novelty, source conflict, deadline, and estimated tool/resource cost.

- **Fast path:** cached exact answer, high-confidence associative retrieval, deterministic simple transform.
- **Medium path:** typed relation traversal, unit-aware calculation, temporal prediction, or focused causal query.
- **Deep path:** bounded competing hypotheses, simulation, a dedicated solver, specialist modules, or a human review request.

Each escalation should be triggered by a recorded reason. Budgets must cap specialist calls, time, memory, and action scope. Include starvation prevention (ageing/priority bounds) so recurring easy work cannot indefinitely suppress urgent or rare tasks. Keep a fixed-routing control and a run-everything control to isolate whether the coordinator actually saves resources.

The policy can eventually adapt from measured quality/cost feedback, but an adaptive router must never tune itself on the held-out confirmatory set. Adaptive selection and its training data belong in the experiment manifest.

### 6. Independent verification and abstention

Before a claim is marked verified, use the strongest applicable independent checker:

- type and unit checks for quantities;
- deterministic arithmetic or interval bounds for calculations;
- constraint/solver checks for formalized engineering problems;
- source and timestamp checks for sourced facts;
- test or proof checkers where the claim has a formal contract;
- simulation plus separate safety envelope checks for modelled actions.

Verification status must be multidimensional: whether a check ran, what claim/inputs it covered, exact checker/version, output, and known limits. A checker pass means the claim satisfied that checker—not that the entire answer or world-model is true. When the original claim cannot be formalized or a verifier fails, the system should label the result *supported*, *hypothesis*, or *inconclusive*, or abstain, instead of silently lowering a threshold.

### 7. Actions as typed contracts

An action proposal should contain the intended effect, evidence supporting it, preconditions, resource bounds, expected outcome, hazard constraints, reversibility, and postcondition check. The actuator path should validate the contract and permissions independently from the component that generated the plan. After execution, score the prediction before updating the responsible model; retain failure cases and runtime context.

## Why this may outperform a single-representation design

The potential advantage comes from **placing each type of information in the representation best suited to it**:

| Need | Preferred representation/mechanism | Why |
|---|---|---|
| Fuzzy associative cueing | HDC/VSA | Composition and similarity under noisy/incomplete cues |
| Exact identity and relationships | Typed entity/relation records | Explicit identity, relation types, and schema constraints |
| Variable-rate temporal evolution | CfC/LTC when suited | Time-aware state updates without requiring every interval to be a fixed discrete step |
| Long-lived experience | Episodic records | Exact sequence and context remain recoverable |
| Reusable knowledge | Semantic store with lineage | Consolidation does not erase sources or contradictions |
| Safe exact outputs | Independent tool/constraint verifier | Checkable guarantees are separate from heuristic plausibility |
| Efficient execution | Bounded coordinator | Spend more compute only where observed need justifies it |

The key is not simply adding a symbolic database beside Symthaea. It is preventing information loss and inappropriate authority transfers at the boundaries: fuzzy similarity must not become fact; derived belief must not become observation; a model prediction must not become evidence; and successful validation of one property must not certify an entire plan.

## Evaluation plan that complements SYM-ARCH-002

Do not run a claim-bearing experiment until the underlying interfaces and the relevant SYM-ARCH-002 measurement/provenance components are accepted into a suitable integration target. First build a source-derived inventory and deterministic component tests. Then preregister a separate **system-level integration** experiment; it should not overload SYM-ARCH-002's frozen learner-mechanism campaign.

### First component tests

1. **Evidence resolution:** every HDC candidate resolves to a typed record; corrupt/duplicate/missing IDs fail closed.
2. **Contradiction preservation:** two incompatible facts with different sources/time remain separately retrievable; no lossy averaging hides the conflict.
3. **Temporal correctness:** old event time and new receive time remain distinct; stale records do not masquerade as current.
4. **Units and constraints:** unknown or incompatible units cannot pass exact verification; known-answer arithmetic/constraint fixtures are correct.
5. **Source lineage:** derived statements preserve transitive source links; revoking one source invalidates dependent results.
6. **Router determinism:** identical frozen state and request yield the same route; budget exhaustion yields an explicit degraded/inconclusive response.
7. **Prequential feedback:** prediction and latency are recorded before update; the current item cannot be scored using information learned from its label first.
8. **Faults:** missing verifier, timeout, corrupt event, conflicting state versions, and unavailable specialist produce explicit non-pass outcomes rather than fabricated success.
9. **Persisted-state integrity:** malformed fact/edge rows return errors; strict manager startup never returns a partially restored manager; legacy fallback logs clearly and drops the persistence handle so later snapshots cannot overwrite an unreadable database.

### Later end-to-end tasks

- **Source-sensitive long-horizon recall:** answer which source stated a fact, when it applied, and whether it was superseded.
- **Contradictory evolving world:** preserve multiple reports, resolve by permitted evidence rules, and surface unresolved conflict when resolution is impossible.
- **Irregular-time sensor response:** identical semantic stream under jitter, long gaps, and dropped observations, comparing timestamp-aware and fixed-step controls.
- **Multi-step engineering query:** bind concepts to typed quantities, calculate/check constraints, then provide source and checker receipts.
- **Tool-using plan:** produce preconditions, validate a plan, execute in deterministic simulation, compare predicted with observed outcome, and safely recover from injected failures.
- **Compute-budget challenge:** compare fixed routing, always-run-all, and adaptive routing under matched budgets; report quality-cost Pareto fronts and deadline misses.

### Primary measures

Keep each outcome distinct:

- exact-task correctness and constraint violations;
- source attribution and contradiction detection;
- calibration, abstention coverage, and selective error;
- p50/p95 end-to-end and component latency;
- specialist calls, bytes moved, memory footprint, and persistent-state size;
- energy only when measured on declared physical hardware;
- performance by scenario, not just overall means;
- fault recovery and failure containment.

Controls must include HDC-only retrieval where relevant, typed lookup without HDC, fixed routing, no-verifier, no-provenance, stale/shuffled evidence, and a budget-matched simple baseline. If a negative control passes or a known-answer oracle fails, the result is INCONCLUSIVE until the experiment is repaired. Do not fold all metrics into one scalar and hide trade-offs.

## Execution sequence

1. **Now:** keep this design proposal separate from runtime changes and the SYM-ARCH-002 mechanism study. Use it to define interfaces and target failure modes.
2. **After suitable experiment infrastructure is merged and the target base is explicit:** add pure Rust typed evidence/temporal contracts and unit tests, reusing existing knowledge/causal modules after call-site review.
3. **Next:** implement a read-only HDC-to-evidence resolver and tests for stale/missing/conflicted identities. Do not yet route external actions through it.
4. **Then:** add a budget-limited read-only coordinator with fixed-routing and always-run-all controls.
5. **Then:** add verification adapters for only the domains with independent checkers already available.
6. **Finally:** preregister and execute system-level experiments with separate task/split/configuration identities, paired or hierarchical uncertainty as appropriate, resource traces, negative controls, and exact-head provenance.

Stop or simplify the design if typed records alone solve the target task, if the HDC index adds cost without retrieval value, if the router does not improve the quality/resource trade-off, or if provenance and verification overhead causes unacceptable latency.

## Research basis

The literature supports investigating the *combination* of representations and explicit routing, but does not establish this specific Symthaea architecture as superior:

- Hasani et al., [“Closed-form continuous-time neural networks,” *Nature Machine Intelligence* (2022)](https://www.nature.com/articles/s42256-022-00556-7). CfC can be efficient on suitable time-series/irregular-time tasks; the authors discuss limitations and cases where other models are better suited.
- Vergés et al., [“Classification using hyperdimensional computing: a review with comparative analysis” (2025)](https://link.springer.com/article/10.1007/s10462-025-11181-2). HDC/VSA provides compositional representations, but task-specific comparisons remain necessary.
- DeLong, Fernández Mir, and Fleuriot, [“Neurosymbolic AI for Reasoning Over Knowledge Graphs: A Survey,” *IEEE TNNLS* (2025)](https://doi.org/10.1109/TNNLS.2024.3420218). Reviews logically informed embeddings, embeddings with logical constraints, and rule-learning approaches and their trade-offs.
- Liu, Wang, and Tong, [“Neural-Symbolic Reasoning over Knowledge Graphs: A Survey from a Query Perspective” (2025)](https://doi.org/10.1145/3748239.3748249). Provides a complementary view organized around knowledge-graph query/reasoning types.
- Ding et al., [“BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute,” ICML/PMLR (2025)](https://proceedings.mlr.press/v267/ding25d.html). Demonstrates a cost/quality trade-off for adaptive model routing in its tested workloads; these results are motivation, not a performance forecast for Symthaea's specialist modules.
- Hodson, Mehta, and Smith, [“The empirical status of predictive coding and active inference” (2024)](https://pubmed.ncbi.nlm.nih.gov/38030100/). Emphasizes that predictive/active-inference accounts require comparison with alternatives.
- Zhang et al., [“Trustworthy Knowledge Graphs: Practices and Approaches” (2025)](https://doi.org/10.3233/FAIA250215). Reviews transparency, accountability, and oversight requirements across knowledge-graph construction lifecycles.
- Qudus et al., [“Fact Checking Knowledge Graphs — A Survey” (ACM Computing Surveys, 2025)](https://doi.org/10.1145/3749838). Surveys approaches and evidence sources for assessing assertion veracity in knowledge graphs.

These papers support researching heterogeneous representations, exact relation systems, temporal models, and adaptive routing. They do not prove that the proposed combination is more intelligent, conscious, safe, or efficient. Only controlled experiments on declared tasks can establish the relevant engineering advantages.

## Acceptance criteria for this proposal

This document is useful only if it narrows implementation and experimentation rather than adding another grand architecture. The proposal should advance to code only when:

- each new responsibility has a named owner and does not duplicate an existing module;
- interfaces preserve source identity, time, units, contradiction, and failure states;
- HDC is used for what similarity/composition are suited to, not treated as proof;
- routing and verification costs are measured end to end;
- system-level tests remain separate from mechanism-level SYM-ARCH-002 tests;
- no architecture win is declared from a proposal, synthetic smoke test, unmerged code, or queued CI.
