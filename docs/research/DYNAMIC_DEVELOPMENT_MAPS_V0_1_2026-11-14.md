# Evidence-Grounded Dynamical Development Maps (EGDDM) — V0.1

**Date:** 2026-11-14  
**Status:** Research/architecture proposal; not an implemented or qualified capability  
**Tracking:** [DYNMAP-001](https://github.com/Luminous-Dynamics/symthaea/issues/12020)  
**Composition:** [Repository-twin #10173](https://github.com/Luminous-Dynamics/symthaea/issues/10173), [CEB intelligence #10419](https://github.com/Luminous-Dynamics/symthaea/issues/10419), [CEB collaboration #10637](https://github.com/Luminous-Dynamics/symthaea/issues/10637), [judge qualification #11772](https://github.com/Luminous-Dynamics/symthaea/issues/11772), [execution isolation #10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064)

## Executive proposal

A development map should represent four things at once:

1. **The system that exists** — components, interfaces, dependencies, tests, deployments and configuration worlds.
2. **The system that is intended** — goals, capabilities, requirements, invariants, design decisions and acceptance criteria.
3. **What is actually known** — source-backed observations, test and benchmark results, exact revisions, evaluator identity, limitations and unresolved uncertainty.
4. **How the system can evolve** — historical change, causal hypotheses, affected capabilities, candidate development paths, expected value of experiments and measured outcomes.

The proposed **Evidence-Grounded Dynamical Development Map (EGDDM)** is a typed, revision-aware graph plus an event history and a constrained planning interface. It is not a static architecture diagram, a generated roadmap, a vector database, or an autonomous agent's self-authored account of success.

The principal research opportunity is to connect repository structure, developer intent, capability traceability, evidence validity and counterfactual planning across multiple repositories, while preserving provenance, uncertainty and authority boundaries.

## 1. Research and industry landscape

We should start from a realistic account of the field.

| System / research | Demonstrated scope | What EGDDM should learn from it | Remaining opportunity |
|---|---|---|---|
| Aider repository map | Symbol/signature map, dependency graph ranking and context-budgeted selection | A map must be useful to an agent under a strict context budget | Aider's map is not a full intent/evidence/development-history model |
| Repository Intelligence Graph (RIG) / SPADE (2026) | Deterministic build/test architecture graph; evaluated with three coding agents over eight repositories | Prefer deterministic build/test facts and explicit edges over prose-only summaries | Add project intent, evidence lifecycle, temporal reasoning and cross-repository contracts |
| SWE-Time (2026-11-09) | Retrospective first-bad-commit reasoning and prospective change-impact reasoning; 1,661 cases, 65 repositories, 15 agents | Repository evolution itself deserves first-class evaluation | Extend beyond code changes to capability/evidence/contract evolution and development planning |
| CoReGraph (2026) | Repository knowledge graph integrating technical dependencies, development activity, temporal modeling and provenance | Technical and organizational history belong in one queryable representation | Add independently qualified capability claims, evidence invalidation and constrained counterfactual planning |
| Cargo metadata / rust-analyzer / SCIP | Manifest and configuration resolution; Rust semantic analysis; language-agnostic symbol/reference interchange | Preserve tool-specific authority and configuration inputs | Reconcile multiple partial oracles without pretending one is complete |
| CodeQL | Language-specific security, source/sink and data-flow analysis | Security/data-flow facts are a separate high-value graph projection | Do not treat security analysis as a universal architecture or intent oracle |

Sources:

- Aider repository map: https://aider.chat/docs/repomap.html
- RIG / SPADE, *Repository Intelligence Graph: Deterministic Architectural Map for LLM Code Assistants* (2026-01-15): https://arxiv.org/abs/2601.10112
- SWE-Time, *Benchmarking Temporal Repository Understanding* (2026-11-09): https://arxiv.org/abs/2611.12083
- CoReGraph (2026): https://doi.org/10.1016/j.procs.2026.09.172
- GitHub CodeQL query packs: https://docs.github.com/en/code-security/concepts/code-scanning/codeql/query-packs
- Cargo metadata: https://doc.rust-lang.org/cargo/commands/cargo-metadata.html
- rust-analyzer architecture: https://rust-analyzer.github.io/book/contributing/architecture.html
- SCIP: https://scip-code.org/

### What the published numbers do—and do not—say

The RIG/SPADE paper reports mean **+12.2 percentage points** accuracy and **53.9% less completion time** with its deterministic map across its eight-repository, three-agent study. That is a useful motivation for testing map utility, not an expected EGDDM result and not a direct comparison with SWE-Time.

SWE-Time reports 1,661 cases across 65 repositories and 15 agents; the highest reported score within its Claude Code evaluation is **61.9 average for Claude Opus 4.8**. This indicates substantial headroom on that benchmark. It is not a general ranking of all coding systems or a score we can compare with other benchmark families.

CoReGraph is an important nearby research direction, so we must not claim that temporal, provenance-bearing repository knowledge graphs are novel in isolation. Our differentiating research hypothesis is the compositional layer: explicit intent-to-capability traceability, evidence lifecycle and validity, cross-repository contract reasoning, and measured development-path planning.

### Important comparator limits — and what they imply for EGDDM

These limitations matter as much as the headline results:

- **SWE-Time is narrower than a general development map.** Its backward task localizes first-bad commits; its forward task predicts GitHub Actions workflow activation under a fixed pull-request event context. The authors explicitly say that other CI systems, build pipelines and non-GitHub environments remain untested, and that public histories may have appeared in pretraining. Its paired-state analysis also reports only 48.7% Pair Exact, with a 12.8% invariance error, showing that adapting predictions across changed workflow states remains difficult. EGDDM should preserve SWE-Time's official score as-is, then add separate evaluation for requirements, APIs, capabilities, evidence applicability and cross-repository effects—not relabel those extensions as SWE-Time.
- **RIG/SPADE is structurally useful but build/test-centric.** The paper's SPADE prototype automatically extracts only a subset of CMake using CMake File API/CTest; several other ecosystems in its study (including Cargo) used manually authored RIGs. Its study ran at least one full evaluation per cell and says there were too few repetitions for tight confidence intervals. The authors also say RIG omits code-level semantic/data-flow and higher-level requirements/documentation. This is a strong deterministic structural baseline, but not an end-to-end multi-ecosystem extractor or intent/evidence planner.
- **Aider optimizes context selection.** Its dependency-ranked map is valuable for token-budgeted repository navigation, but it is not designed as an authoritative longitudinal record or development-qualification system.
- **CoReGraph overlaps the temporal/provenance graph concept.** We should compete on falsifiable integration quality and measured outcome improvements, not assert that temporal repository graphs are unprecedented.
- **CodeQL is a security/data-flow specialist.** Treat its outputs as one scoped evidence stream with its own rules and versions, not the authority for all structure, intent, or maturity claims.

The differentiating research hypotheses are therefore: deterministic and auditable multi-source reconciliation; bitemporal evidence validity; explicit capability/requirement-to-proof traceability; prospective and retrospective cross-domain impact; incremental updates that converge exactly with clean rebuilds; and experimentally calibrated development planning. Each is an independent benchmark claim and must be evaluated separately.

### Retrieval and repository exploration deserve a first-class lane

Two 2026 benchmarks show why a graph's usefulness must be measured at the context budget actually available to an agent:

- **Agent Retrieval Bench (July 2026)** evaluates 427 workflow-derived samples across 25 repositories, with 345 positive cases and 82 no-gold cases over 308 frozen base-commit snapshots, 392,000 files and 7.9 million chunks. The four positive tasks are code2test, comment2context, trace2code and edit2ripple. Its current weighted results show Qwen3-Embedding-4B leading MRR at 0.2379, Qwen3-Embedding-8B leading Recall@20 at 0.7029, and RepoMap leading 8K-token budgeted context yield (BCY@8k) at 0.3788. Rankings vary by task and repository weighting; the natural no-gold abstention results expose a real calibration problem. Source: https://arxiv.org/abs/2607.24882 and https://agent-retrieval-bench.github.io/
- **SWE-Explore (June 2026)** measures ranked code-region retrieval under a fixed line budget across 848 issues, 10 languages and 203 repositories. It derives line-level ground truth from successful agent trajectories and links exploration metrics to downstream repair. In its K=5 comparison, CoSIL reports non-oracle line recall of 0.788 and F1 of 0.602, while general coding agents' line recall is around 0.14–0.19 despite much stronger file-hit rates. Context Efficiency and Recall@100 correlate strongly with downstream resolve rate. Source: https://arxiv.org/abs/2606.07297

This suggests a further paradigm shift: **the graph should not only describe a repository; it should actively guide evidence-efficient exploration.** EGDDM's retrieval lane should use typed graph expansion, change history, ownership and evidence scope to select the next useful source spans, while retaining lexical and embedding retrieval as complementary signals.

A promising hybrid is:
1. deterministic structural/semantic seed selection (exports, reverse dependencies, tests, changed symbols, owners);
2. query-aware lexical/embedding ranking for semantically relevant context;
3. temporal and configuration-world filters to reject irrelevant historical or inactive facts;
4. evidence/authority filters to prevent an unqualified declaration from being presented as a verified result;
5. budget-aware region packing with explicit abstention when the repository or available evidence cannot answer the task.

The target is not "more context." It is greater **useful evidence yield per token/line**, with lower missed-critical-context and false-context rates.

The current Agent Retrieval Bench paper also reports an RRF fusion result of **0.2713 weighted MRR and 0.7331 Recall@20**, improving over its best single-run values in that comparison. This is the stronger retrieval baseline to beat; a hybrid EGDDM ranker should not claim progress by comparing only against a single dense model or lexical search. SWE-Explore's K=5 result reports CoSIL at **0.788 line-level recall and 0.602 F1** (non-oracle), making graph-guided high-recall exploration a concrete comparator rather than a hypothetical future system. These numbers are protocol-specific and must be reproduced before a leaderboard comparison. Sources: https://arxiv.org/abs/2607.24882 and https://arxiv.org/abs/2606.07297

## 2. Existing Luminous ownership — compose, do not duplicate

The Symthaea repository already has foundational work:

- **#10173** owns the repository-twin gold set: a five-crate synthetic Rust workspace, 32 expected nodes, 37 expected edges, mutation scenarios, feature/target worlds, source anchors and change-impact measures. Its discussion already proposes a temporal, evidence-carrying graph, including exact revision, toolchain, configuration world and evidence authority.
- **#10419** owns context retrieval, code review, secure correctness and Rust semantic capability evaluation.
- **#10637** owns process adherence, human/agent collaboration, evidence-first communication and handoff quality.
- **#11772** owns independent task-judge and oracle qualification.
- **#8908 CMAQ-002** owns active causal learning / value-of-information experiment selection; EGDDM planning must consume its qualified outputs rather than implement another EVoI engine.
- **#6206 SE-FOSS-003H** owns engineering-cognition usefulness, calibration, abstention and non-authority qualification; development-plan usefulness and abstention should compose its outputs.
- **#10064** is the OS-level isolation prerequisite for executing repository-controlled code.
- Existing CEB task, run-receipt, trust and custody work remains authoritative for task identity, execution accounting, signatures and evaluator boundaries.

**Do not create a competing repository graph, generic evidence engine, task schema, run receipt, scheduler, trusted pin registry or benchmark task registry.** EGDDM is an integration and planning layer above these owners. The existing repository-twin graph remains the source of code-level structural/semantic gold evaluation.

The separate engineering design/configuration evidence lane (#10543) should remain distinct from code/repository graph accuracy. It can contribute evidence-applicability concepts without merging unlike gold sets or scores.

### Temporal retrieval is not decision-time correctness

Two new/current temporal benchmarks clarify the opportunity:

- **TEMPO (EMNLP 2026)** has 1,730 complex queries across 13 domains and 3,976 decomposed retrieval steps. Its leaderboard reports best Temporal Coverage@10 of 71.4% (DiVeR) and best NDCG@10 of 32.0. These measure temporally relevant retrieval and coverage, not whether an institutional or engineering decision used only evidence that was available to the actor when the decision was made. Source: https://aclanthology.org/2026.emnlp-main.291/ and https://tempo-bench.github.io/
- **SWE-Time (2026-11-09)** evaluates retrospective first-bad-commit reasoning and prospective change-scope/CI workflow reasoning on 1,661 cases from 65 repositories. Its best reported Claude Code result is 61.9 average for Opus 4.8. It does not directly evaluate whether the evidence cited to justify a past decision was available to the decision-maker at the decision time. Source: https://arxiv.org/abs/2611.12083
- **TEMPS (EMNLP 2026)** adds temporal embeddings to retrieval and reports TS-Retriever R@1 improving from 19.92 to 25.39 on its evaluated setup. This is a useful retrieval baseline for time-aware relevance, but still does not establish actor-specific authority or decision-time admissibility. Source: https://aclanthology.org/2026.emnlp-main.2657/

EGDDM should add **Decision-Time Evidence Admissibility (DTEA)** as a distinct research lane.

For every decision event, distinguish at least:
1. **Valid/effective time:** when the underlying fact or requirement applies in the world.
2. **Observed/recorded time:** when the system actually learned or recorded the evidence.
3. **Available-to-actor time:** when the specific decision-maker could access and was authorized to use it.
4. **Decision time:** when the consequential decision was made.
5. **Retrospective correction time:** when later evidence invalidated, amended or clarified an earlier understanding.

A document may describe an earlier fact but arrive after the decision. It may be public but outside the actor's authorized scope. A later correction may make evidence stale today without necessarily making an earlier decision unreasonable on the information legitimately available then. Conversely, hindsight cannot be used to justify a decision that was unsupported at the time.

The map must retain both the historical decision basis and the current evidence state. Never rewrite the past decision's cited evidence to match today's graph.

A decision record should bind:
- the decision subject and exact decision time;
- the policy/contract version and actor/authority scope;
- the knowledge cut-off used for the decision;
- the evidence IDs actually consulted;
- an exact owner-native authority-evidence reference (owner repository and record ID/type), binding actor ID, exact scope, policy/constitution version, source-observed time, available-to-actor time, effective validity interval and owner-record disposition;
- each cited item's availability time, authority and admissibility disposition at decision time;
- excluded or unavailable evidence with reason;
- the outcome, abstention/defer disposition and later supersession/correction links.

This enables a new class of tests: backdated evidence, delayed publication, delayed ingestion, post-decision corrections, expired credentials, revocations, contested sources, incomplete retrieval and unauthorized information. The evaluator should ask not only "is the reconstructed answer true now?" but "was this decision justified using only admissible evidence available to this actor at this time?"

### DTEA authority evidence is a reference, not a new authority engine

The decision-time record projection now requires an `actor_authority` evidence reference. It binds the decision's actor, exact scope and policy version to an owner record, plus source-observed, actor-available and effective-validity times. The validator checks reference identity, node/event agreement, actor/scope/policy equality, availability by the decision cutoff and validity across the decision time.

The current fixture's owner record is deliberately synthetic (`synthetic://mycelix/authority-grant/AG-001`) and source-declared. It is not a real Mycelix permission, cryptographic signature, trusted-root policy or grant of release authority. Mycelix remains the owner of authority records and semantics; DTEA tests whether the cited owner evidence was admissible at the decision frontier. The validator cannot authenticate an owner-native authority record or prove actor rights.

## 3. Model: four connected views

### 3.1 Structural and semantic view

Entities include repository, workspace, package/crate, module, symbol, interface, configuration world, dependency, test, benchmark, runtime, deployment unit and external dependency.

Relationships include `contains`, `exports`, `imports`, `depends_on`, `calls`, `implements`, `configured_by`, `built_by`, `tested_by` and `deployed_as`.

For Rust, keep resolution staged and separately attested:

1. Exact repository tree, Cargo manifests, lockfile, toolchain and invocation.
2. Cargo-resolved default/effective features and target-filtered dependency graph.
3. Per-crate cfg values and rust-analyzer crate graph.
4. rust-analyzer symbol/type/reference/implementation observations.
5. SCIP interchange and cross-language symbol/reference information where supported.
6. Runtime/build/test results only from an authorized, isolated execution boundary.

Cargo is authoritative for Cargo feature and target dependency resolution; rust-analyzer is not a substitute for Cargo's world selection. Unsupported worlds must remain unresolved.

### 3.2 Intent and capability view

Entities include strategic objective, capability, requirement, contract, invariant, design decision, hypothesis, milestone, acceptance criterion and owner.

A capability is not simply a directory or crate. It should have:

- a stable capability identifier and bounded description;
- required behavior and explicit non-goals;
- source requirements and invariants;
- implementing components and external contracts;
- validation methods and required evidence;
- maturity disposition and known gaps;
- dependencies and unlock relationships;
- review owner or authority source where applicable.

The critical trace is:

`Objective → Capability → Requirement / Invariant → Implementation → Test → Qualified Evidence`

Missing links are development gaps, not proof that an implementation does not exist. Conflicting links must be visible.

### 3.3 Evidence and qualification view

Every assertion must retain evidence authority. At minimum distinguish:

- `source_declared` — literal repository/document declaration;
- `syntax_extracted` — parser-derived structure;
- `manifest_resolved` — build-system/manifest resolution;
- `compiler_resolved` — compiler or language-server semantic result;
- `runtime_observed` — execution result under a named environment;
- `benchmark_observed` — outcome under a named protocol and evaluator;
- `independently_adjudicated` — external or independent review with identity and scope;
- `inferred` — model-generated hypothesis with supporting references;
- `proposed` — a planned future, not an observation.

Evidence lifecycle is separate from authority: `current`, `stale`, `superseded`, `revoked`, `unknown`, `missing` and `disputed` are not evidence authorities.

A successful test at commit A does not automatically qualify a newer commit B. A skipped or queued workflow is not a pass. A README statement is a source-declared statement, not an independently reproduced test result. An issue marked complete is not itself proof of the outcome.

### 3.4 Temporal and development view

Represent repository evolution as append-only events plus reproducible snapshots.

Required event families:

- revision observed; file/path introduced, changed, renamed, deleted;
- manifest/configuration/toolchain/world changed;
- issue/PR/decision/requirement created, amended, resolved or superseded;
- test/benchmark/CI/evaluation observed;
- evidence admitted, invalidated, expired or disputed;
- capability implementation link added or removed;
- development hypothesis proposed, accepted, rejected or superseded;
- action attempted, completed, failed, blocked or left indeterminate.

Use two clocks:
- **observation time:** when the map learned or recorded a fact;
- **validity time/revision:** the repository commit, configuration world, evaluator or artifact to which the fact applies.

Every snapshot must identify repository, exact commit, tree digest, extractor build/version and relevant world. Replaying the event stream must reconstruct the same snapshot digest. An event that cannot be ordered or reconciled should yield an explicit unknown/indeterminate state rather than fabricated continuity.

## 4. Dynamics: map reachable futures, not only current files

A candidate development action is a hypothesis with:

- target capability and expected observable change;
- prerequisite graph and blocked dependencies;
- affected components, contracts, tests and evidence;
- resource/time/compute/human budget;
- risk, uncertainty and assumptions;
- acceptance and negative-control criteria;
- rollback, supersession or recovery path;
- expected result and eventual measured result.

The planner should search for high-value actions using constrained multi-objective reasoning and value of information. Do not hide tradeoffs in one weighted score. Safety, authorization, evidence integrity, compatibility and resource constraints are hard gates.

A useful new question is not only "what should we build next?" but:

> Which smallest experiment would most efficiently distinguish between the development paths we currently believe are promising?

This turns the map into an instrument for reducing uncertainty, not an oracle that predicts success. Predicted paths remain hypotheses until measured.

### Three temporal queries

1. **Past:** What changed, why did it change, which earlier evidence or assumptions became invalid, and what is the most likely first-bad change?
2. **Present:** What is actually implemented and qualified at this exact revision and configuration?
3. **Future:** If we make this proposed change, which capabilities, interfaces, tests, risks and evidence obligations are likely to be affected?

## 5. Cross-repository architecture

Each repository owns its local map. The Luminous ecosystem view is a federation of explicit references, not a universal graph that silently merges private source or similarly named concepts.

Cross-repository edges require:

- canonical repository identity and pinned revision;
- exact exported contract/schema/interface version and digest;
- source and evidence provenance on both sides;
- compatibility disposition: compatible, incompatible, conditional, unknown or unverified;
- explicit disclosure/authorization for the exported data.

The initial targets are Symthaea, Mycelix, Symtropy and Sol-Atlas, followed by explicitly authorized public external repositories.

### Symthaea as first-class map consumer

Symthaea should use the same adapter and query interface to map itself and other repositories. Her self-map should let her inspect architecture, compare intent with implementation, find stale evidence, identify capability bottlenecks and propose bounded experiments.

Safeguards:
- observation and planning are read-only by default;
- a map fact cannot grant mutation, merge, release or deployment authority;
- the agent cannot qualify its own result;
- generated hypotheses cannot become observations without independent evidence;
- external private repositories require explicit access and purpose;
- execution remains gated by #10064 and repository-specific policy.

### Mycelix as the first external validation environment

Mycelix's public README already has a valuable maturity matrix distinguishing builds, unit tests, multi-agent tests, UI and deployment, while acknowledging that several cells are based on internal tracking and not independently re-run. Preserve those distinctions rather than converting all rows into a green/red score.

Initial capability dimensions:
- domain clusters and zome capabilities;
- bridge and cross-cluster contracts;
- Holochain roles, SDKs, unified hApp composition and deployment snapshots;
- unit-test vs multi-agent integration evidence;
- frontend surface, build/source state and deployment state;
- versioned external dependencies and compatibility;
- trust/governance/economic boundaries and operational blockers.

The Mycelix view is advisory engineering intelligence. It does not become governance, identity, trust, financial, accounting or authorization authority.

## 6. DynaMapBench — benchmark composition

DynaMapBench is a proposed evaluation profile, not a new task/receipt/evaluator platform. Reuse CEB's task schema, custody, receipts and independent judge qualification. Extend the task families owned by #10173, #10419 and #10637.

| Track | Main question | Primary measures |
|---|---|---|
| Structure | Is the graph correct? | Per-relation precision/recall/F1; symbol retrieval Recall@k; source-span accuracy |
| Retrieval / exploration | Does the map surface the next useful evidence under a fixed budget? | File/region/line precision-recall; MRR/nDCG; Recall@k; context yield at 8K tokens and fixed line budgets; natural no-gold selective success; downstream repair effect |
| Temporal backward | Can the system reconstruct why the repository is in this state? | First-bad-commit top-1/top-3; regression cause precision; supersession accuracy |
| Temporal forward | Can it forecast consequences before editing? | Affected-component/test/interface Recall@k and precision; critical-impact false negatives |
| Invalidation | Does change invalidate the right facts? | Stale-fact rate/lifetime; exact incremental vs clean-rebuild graph equivalence |
| Configuration | Does the map understand actual build worlds? | Feature/target activation accuracy; false-active/inactive facts; abstention on unsupported worlds |
| Capability traceability | Can intent be traced to proof of implementation? | Requirement-to-implementation/test/evidence coverage; unsupported-claim detection |
| Cross-repository | Can contract breaks be detected across projects? | Version mismatch recall/precision; compatibility false negatives; map reconciliation correctness |
| Planning | Does map-guided development improve outcomes? | Independently accepted capability gain per budget; expected/observed calibration; human review effort; prerequisite closure; hard-constraint violations; budget-normalized regret; Pareto-front coverage |
| Decision-time admissibility | Was the decision justified using only evidence available and authorized at the decision time? | Admissible-evidence precision/recall; future-evidence leakage; backdated/late-arrival rejection; decision-time reconstruction accuracy; abstention correctness |
| Federation integrity | Are local maps joined only through exact, owner-published identities and contracts without transferring authority? | Exact map-blob resolution; source-snapshot vs artifact-pin accuracy; cross-project edge precision; contract-version/digest match; stale-member detection; false-qualified-edge rate; authority-transfer violations |
| Safety/evidence | Does the map resist false confidence? | False-current acceptance; authority inflation; fabricated evidence detection; unauthorized mutation rate |

### Required comparison arms

1. No map / ordinary search.
2. Aider-style context-budgeted map.
3. RIG/SPADE-style deterministic build/test graph.
4. Cargo metadata plus syntax graph.
5. rust-analyzer and SCIP for supported Rust semantic worlds.
6. CodeQL for the separate security/data-flow lane.
7. Git history, issue and PR context without a synthesized graph.
8. Existing #10173 repository-twin graph.
9. Full EGDDM with intent, evidence, temporal and cross-repository layers.
10. Same-model/same-tool matched ablations to isolate the contribution of each layer.
11. Development-trajectory baselines: issue-priority/recency, dependency-only topological ordering, shortest-path planning, text-only LLM planning, EGDDM without evidence/federation, and full EGDDM under identical budgets. Reuse #8908's value-of-information outputs and #6206's usefulness/calibration/abstention measures; do not merge their scores into DynaMapBench.
12. **Agent Retrieval Bench** on its exact frozen commits and official tokenizer/cost protocol: lexical/BM25, Qwen3-Embedding-4B/8B, RepoMap, and a structural+semantic EGDDM hybrid; report per-task MRR, Recall@20, BCY@8k and the natural versus counterfactual no-gold strata separately.
13. **SWE-Explore** on the same K=5 region interface and fixed line budget: general agentic explorers, CoSIL, RepoMap-style retrieval, and EGDDM graph-guided exploration; report HitFile, HitRegion, line precision/recall/F1, nDCG@500, Context Efficiency, and restricted-context downstream resolution.
14. **TEMPO/TEMPS** as separate temporal retrieval comparators: preserve official task/scoring and evaluate map-grounded temporal evidence retrieval as an additional track, not as an interchangeable repository benchmark.

Use SWE-Time as an external temporal comparator when its public benchmark materials, task rules and license permit. Preserve the official SWE-Time scoring as a separate result; do not relabel a custom score as SWE-Time.

### Prospective pilot targets

These are research hypotheses, not achieved results or industry standards:

- 100% of admitted facts have revision-scoped provenance and an explicit authority class.
- 100% exact snapshot/evaluator identity for any evidence-qualified result.
- 100% incremental-vs-clean-rebuild graph equivalence over supported synthetic worlds.
- Zero stale facts after a completed invalidation/rebuild cycle in deterministic fixtures.
- Syntax-node precision/recall ≥0.95; compiler-resolved relation precision/recall ≥0.90 on supported gold relations (consistent with #10173's proposed targets).
- Change-impact recall ≥0.95 and precision ≥0.80 on the frozen synthetic mutation corpus.
- First-bad-commit top-1 ≥0.80 and top-3 ≥0.95 on an independently held-out pilot, subject to power analysis and baseline reproduction.
- Capability traceability completeness ≥0.95 on a clearly scoped requirement set.
- Cross-repository critical contract-mismatch recall ≥0.95, with high-severity false negatives treated as hard failures.
- Zero unsupported evidence promotion and zero unauthorized mutation in deterministic adversarial controls.
- DTEA pilot: 100% detection of the explicitly planted future-evidence and unauthorized-source negative controls; ≥0.95 precision/recall on adjudicated decision-time admissibility labels, after calibration and power analysis; report decision correctness separately from evidence-admissibility correctness.
- Federation pilot: 100% of exact artifact path/blob pins resolve at the pinned artifact commit; 100% separation of subject-source and map-artifact identity; zero independently-qualified edges without two-sided conformance evidence; missing or mismatched owner contracts remain unjoined/unknown; zero authority transfer.
- Retrieval pilot: reproduce official Agent Retrieval Bench and SWE-Explore baselines first. On held-out repository families, target a statistically supported ≥5 percentage-point improvement over the reproduced SWE-Explore K=5 line-recall baseline (including CoSIL's published 0.788 result) under the same region/line budget, without >1 percentage-point precision regression; target ≥5% relative improvement over the reproduced Agent Retrieval Bench fused-retrieval baselines (including RRF) in BCY@8k on at least three of four positive task families. Preserve task-level, repository-macro and natural no-gold results; no pooling of wrong-repository counterfactuals into the natural no-gold primary score.

Report confidence intervals, per-relation/task-family breakdowns, cost, latency, memory, human minutes and abstention. Do not publish one aggregate score that can hide a critical failure. Use repository-family/time-held-out splits and keep development, calibration and confirmatory answer custody separate.

## 7. Delivery sequence

### D0 — Architecture and research contract (this document)

- Versioned model, source authority, temporal semantics and cross-repository boundaries.
- Existing owner/non-duplication matrix.
- External baseline inventory and DynaMapBench outline.

### D1 — Read-only current-state projection

- Export the existing #10173 static repository-twin fixture without changing its ownership.
- Create a read-only Symthaea snapshot anchored to the exact source revision.
- Create a Mycelix maturity projection from README/source declarations, explicitly preserving source-declared vs independently observed status.
- Add a static consistency checker for node/edge references, provenance fields and snapshot identity.

### D2 — Temporal replay

- Replay a bounded public commit sequence.
- Record file, manifest, issue/PR and evidence events.
- Compare incremental update to deterministic full rebuild.
- Add stale-evidence and supersession controls.

### D3 — Prospective impact and capability traceability

- Build tasks from frozen pre-change revisions and developer intent.
- Predict impact before exposing the patch/future state.
- Compare with actual changed files, affected tests/contracts and independent gold labels.

### D4 — Federation and Mycelix adapter

- Pin contract references between Mycelix and Symthaea.
- Map cluster capabilities, bridge contracts and maturity assertions.
- Detect cross-repository compatibility changes without importing authority.

### D5 — Counterfactual development planner

- Generate competing bounded plans.
- Rank experiments by expected information and capability value subject to hard constraints.
- Measure actual acceptance, regressions, review time and resource consumption.
- Retain failed hypotheses and forecast calibration.

### Rust validator and decision-time fixture increment — 2026-11-14

A standalone Rust crate has been authored at `tools/development-map-validator/`. It is deliberately outside the root workspace, has its own Cargo workspace, reads one JSON file, and never follows source URLs or executes repository commands. Its current contract checks include graph references, provenance/lifecycle shape, evidence invalidation/supersession, authority promotion, event targets, containment cycles, and decision-time evidence availability. It uses RFC 3339 timestamp parsing for decision-time comparisons.

The crate now contains 59 authored unit-test cases across map snapshots, decision-time authority, development-trajectory planning, fixed-budget retrieval/exploration and federation manifests. DTEA now checks effective-time bounds at the decision instant and requires an explicit validity-completeness declaration: a null bound under `complete` means a known open interval edge, while `incomplete` applicability cannot support consulted evidence. The public DTEA expected-results fixture has 21 authored negative controls. **The Rust source and tests have not been compiled or executed.** The crate is unqualified until issue #10064's execution boundary permits a safe run; authored test count is not a passing test result.

A second public known-answer fixture now exists:

- `docs/development-map/fixtures/decision-time-admissibility-v0.1.map.json`
- `docs/development-map/fixtures/decision-time-admissibility-v0.1.expected.json`
- Tracking: [DTEA-001 #12042](https://github.com/Luminous-Dynamics/symthaea/issues/12042)

It models a decision made at 10:00Z. Evidence E1 was available by the cutoff and is cited. Evidence E2 describes a test that completed at 09:10Z but was not observed until 11:30Z and available to the actor until 11:45Z, so it cannot support the 10:00Z decision. A correction at 11:50Z later makes E1 stale in the current map, while the historical decision record remains intact and continues to state that E1 was admissible at decision time.

The fixture now has six graph nodes, six edges and seven events, including an explicit synthetic owner-authority evidence node and a preceding `authority_evidence_recorded` event. It retains the 10:00Z decision, E1 available before the decision, backdated-but-late E2 (test occurred 09:10Z but actor availability 11:45Z), and the 11:50Z correction that later invalidates E1. Its expected-results file contains 21 authored negative controls, including actor/scope/policy mismatch, late or expired authority evidence, missing owner-authority event, owner-record projection mismatch, future-effective evidence, expired/reversed intervals and incomplete applicability. The authority record URI is deliberately synthetic and source-declared; it is not a real Mycelix grant, signature or authenticated permission. Runner-independent checks re-fetched the artifacts and checked unique node/event IDs, graph/event references, provenance field presence, schema-local reference resolution, the late-arrival and authority timing relationships, the decision event's timestamp/cutoff, the owner-authority evidence reference/event/node binding, and the later invalidation linkage. These are structural and arithmetic checks over the authored JSON, not Rust test execution, a full JSON Schema validator or independent benchmark qualification.

### Runner-independent temporal fixture increment — 2026-11-14

A first public known-answer temporal invalidation fixture was added at:

- `docs/development-map/fixtures/temporal-evidence-invalidation-v0.1.map.json`
- `docs/development-map/fixtures/temporal-evidence-invalidation-v0.1.expected.json`

It models a synthetic contract-v1 → contract-v2 evolution:
- 8 graph nodes and 7 edges;
- 9 append-only events across three synthetic revisions;
- evidence E1 passes at revision A, becomes stale after the contract changes at B, and remains historically present;
- evidence E2 passes at revision C under the new v2 contract and exact declared configuration world;
- a proposed v3 action remains a proposal and cannot mutate the observed C snapshot;
- 7 authored negative controls cover false-current promotion, skipped-as-pass, world mismatch, proposal-as-observation, history erasure, path-only identity and evaluator-scope mismatch.

The provisional snapshot schema now accepts an optional event slice with separate event time and observation time, effective revision, targets, typed state effect, provenance and validity scope.

**Structural checks performed on the fetched branch artifacts:** JSON parse for map/expected/schema; all local schema references resolve; 8 unique node IDs; 7 edge endpoints resolve; 9 event target sets resolve; event kinds/node kinds/edge relations occur in the schema enumerations; E1 is stale, E2 current, and v3 remains proposed. This is an authored structural consistency check, **not** full JSON Schema meta-validation, Rust validation, event replay execution, benchmark qualification or CI.

The fixture is public and deliberately not a sealed benchmark. Its identifiers are synthetic placeholders, not real Git object hashes or cryptographic attestations.

A follow-up schema-shape audit normalized the distinction between top-level `extractor.build_digest` and per-claim `provenance.extracted_by.digest`. The Symthaea self-map, Mycelix maturity projection, Symtropy architecture projection, Sol Atlas source projection, temporal invalidation fixture, decision-time admissibility fixture and four-project federation manifest were checked against the provisional v0.1 contracts' selected required fields, enumerations, provenance/lifecycle structure, event/graph references, decision-time cutoffs, exact artifact pins and local schema references. Selected structural checks reported **zero findings**. These checks are not a full JSON Schema 2020-12 implementation, Rust execution, authenticated Git verification or independent qualification. This is not a full JSON Schema 2020-12 implementation or meta-schema validation; CI/schema-consumer validation remains pending.

### Counterfactual development trajectory planning — DYNPLAN-001

DynaMapBench's planning lane now has a dedicated child: [DYNPLAN-001 #12071](https://github.com/Luminous-Dynamics/symthaea/issues/12071). It evaluates whether a planner chooses a feasible development path under resource, dependency, evidence and authority constraints.

This is deliberately narrower than general active experiment selection:
- #8908 owns EVoI / active causal-learning methods;
- #6206 owns engineering-cognition usefulness, calibration, abstention and non-authority;
- DYNPLAN owns bounded development-trajectory cases and their feasibility/reference oracle.

A public synthetic known-answer case is authored at:
- `docs/development-map/fixtures/development-trajectory-planning-v0.1.json`
- `docs/development-map/fixtures/development-trajectory-planning-v0.1.expected.json`

It uses the actual pinned Symthaea/Mycelix map and schema identities but retains the shared contract as `contract_versioned_unqualified`, with no two-sided conformance receipt. The expected plan is evidence-first: exact pin audit, owner contract review, then conformance-fixture design; implementation remains blocked. Broad source scanning is over budget and does not close the critical contract gap; an unrelated physics benchmark is dominated; defer is a valid fallback if owner evidence is unavailable.

The case is synthetic. Its effort units are not real engineering estimates, and it does not assert integration success, owner approval, execution authority or implementation.

### Fixed-budget retrieval and exploration — DYNRETR-001

The retrieval/exploration lane now has its own child under CEB: [DYNRETR-001 #12080](https://github.com/Luminous-Dynamics/symthaea/issues/12080), composed with [CEB intelligence #10419](https://github.com/Luminous-Dynamics/symthaea/issues/10419), [repository-twin #10173](https://github.com/Luminous-Dynamics/symthaea/issues/10173), and [judge/oracle qualification #11772](https://github.com/Luminous-Dynamics/symthaea/issues/11772). It does not introduce a new task, receipt, retrieval engine or evaluator platform.

Public synthetic corpus and oracle:
- `docs/development-map/fixtures/retrieval-exploration-v0.1.schema.json`
- `docs/development-map/fixtures/retrieval-exploration-v0.1.json`
- `docs/development-map/fixtures/retrieval-exploration-v0.1.expected.json`

The corpus has seven cases and eight negative controls:
1. positive contract/type/test retrieval;
2. natural no-gold (requested protobuf migration adapter absent);
3. wrong-repository counterfactual no-gold, scored separately;
4. inactive feature/configuration decoy;
5. stale-revision evidence;
6. non-local change impact across caller, test and feature manifest;
7. lexical UI decoys versus cryptographic primitive and invariant test.

Each case now pins its own repository revision and tree; candidate regions bind exact source revisions, line spans, token estimates, authority and availability. The validator checks that available/inactive evidence belongs to that exact case revision, stale evidence differs from it, and feature requirements agree with the case's configuration world. Natural no-gold must abstain; lexical resemblance is not evidence.

The synthetic fixture and expected oracle were parsed and inspected statically. This does not measure a retrieval system, validate actual tokenizer costs, reproduce Agent Retrieval Bench or SWE-Explore, or establish downstream repair gains. The Rust validator adds eight authored retrieval tests for local shape, oracle shape, no-gold separation, config/temporal decoys, per-case revision/world isolation and oracle overlap.

### Four-repository adoption and federated directory — 2026-11-14

Four bounded, repository-local source projections now exist on separate draft branches:

| Project | Local map | Pinned source revision | Graph |
|---|---|---|---:|
| Symthaea | `docs/development-map/readme-and-workstreams-v0.1.json` | `dcffe1060ece4815cf42548a8f646cec4f075535` | 13 nodes / 18 edges |
| Mycelix | `docs/development-map/source-declared-maturity-v0.1.json` | `6089a68430a501c51934c2c4107061c2a2cb5565` | 17 nodes / 14 edges |
| Symtropy | `docs/development-map/source-declared-architecture-v0.1.json` | `aea5d824b31dd949c7b24680d27c9041d025db85` | 20 nodes / 29 edges |
| Sol Atlas | `docs/development-map/source-declared-demo-architecture-v0.1.json` | `1af10d891410cce31c1d4037e7c7129c16db5834` | 41 nodes / 51 edges |

Project review tracks:
- Symthaea: [#12024](https://github.com/Luminous-Dynamics/symthaea/pull/12024)
- Mycelix: [#6133](https://github.com/Luminous-Dynamics/mycelix/pull/6133), tracked by [#6132](https://github.com/Luminous-Dynamics/mycelix/issues/6132)
- Symtropy: [#5134](https://github.com/Luminous-Dynamics/symtropy/pull/5134), tracked by [#5132](https://github.com/Luminous-Dynamics/symtropy/issues/5132)
- Sol Atlas: [#792](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/792), tracked by [#791](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/791)

The federation directory is `docs/development-map/federation-v0.1.json`, governed by `docs/development-map/federation-v0.1.schema.json`; the versioned successor is `docs/development-map/federation-v0.1.1.json`, governed by `federation-v0.1.1.schema.json`. The v0.1.1 directory contains four pinned project members and eleven source-declared / unqualified cross-project relationships. It pins, separately:
1. each mapped project source revision and root tree;
2. each map artifact path and blob OID;
3. the branch commit and tree containing each exact map artifact;
4. the exact provisional shared map-schema blob and the commit containing that schema.

A runner-independent Git-tree check fetched each member's **exact artifact commit** and confirmed that the map path exists and its observed blob OID matches the manifest. All four exact map path/blob comparisons matched. The federation Rust validator adds local checks for unique member/repository/map identities, graph-independent relationship endpoints, source refs, exact artifact URLs, hub self-map identity, schema bindings and authority non-transfer. Federation v0.1.1 additionally requires the shared-schema candidate to match the exact hub schema blob; an independently-qualified edge requires a two-sided receipt matching both pinned member revisions/trees, exact contract blob, verifier build, configuration-world digest and passing outcome. Three new negative-control tests exercise receipt omission, one-sided/mismatched pins and endpoint-source mismatch. All Rust test cases remain uncompiled/unexecuted.

This is a **directory of independent maps, not a merged graph**. The eleven current relationships are source-declared integration/role statements or contract-versioned-but-unqualified shared-schema candidates. No API compatibility, end-to-end runtime, data exchange or cross-project authority is inferred. Member maps keep their own owner-native authority and evidence; federation edges cannot grant governance, identity, financial, deployment, sensor or physical-actuation authority.



### D1 Rust syntax projection — 2026-11-14

A syntax-only source projector was added at `tools/development-map-validator/src/rust_syntax.rs`, with a `rust-file` CLI mode requiring explicit repository/revision/tree/path/blob pins. It uses a parser over the supplied Rust text and emits `syntax_extracted` facts for modules, item declarations, functions, traits, methods, imports, visibility, attributes and item-position macro syntax. Inline module scope is preserved; out-of-line module declarations remain unresolved.

The projector does not run a compiler, expand macros, execute included code, follow module paths, resolve imports, evaluate `cfg` or produce source-line spans. These are syntax facts only, not compiler-resolved relationships. Nine additional unit-test definitions were authored; no tests or parser execution were run.

## 8. Runner-independent work and safety boundary

Safe now: architecture and schema authoring, static parsing of repository metadata/source, public Git-history analysis, manual source-anchored capability inventories, benchmark protocol, synthetic fixtures, static consistency checks and read-only map generation.

Not authorized by this document: running repository-controlled build scripts, tests, procedural macros or candidate programs on an unqualified host; merging/deploying; or treating CI skipped/queued/unknown as passed. Follow #10064 for isolated execution. Keep exact-head verification separate from branch-level or stale evidence.

## 9. Claim boundary

This document defines a research direction and proposed acceptance tests. It does not claim that EGDDM is implemented, that Mycelix or Symthaea is fully mapped, that any DynaMapBench task set is qualified, or that Symthaea exceeds SWE-Time, RIG/SPADE, Aider or other industry systems.

The field should be challenged through matched, reproducible experiments—not marketing comparisons.


### Development Intelligence and Evolution Science — 2026-11-14

The EGDDM architecture is now paired with [DEVELOPMENT_INTELLIGENCE_AND_EVOLUTION_SCIENCE_V0_1_2026-11-14.md](DEVELOPMENT_INTELLIGENCE_AND_EVOLUTION_SCIENCE_V0_1_2026-11-14.md). DIES is an integration and evaluation discipline, not a new graph engine, event store, scheduler, evidence ledger or authority service. It composes existing owners for structure (#10173), temporal decisions (#12017/#12042), retrieval (#12080), planning (#12071), causal debugging/evolutionary attribution (#10366/#10717/#12067), decision quality/evidence acquisition (#12039/#12046), and independent qualification (#11772). Its benchmark design adds paired should-act/should-defer controls, map metamorphic/noninterference properties and independently measured development outcomes. All proposed targets remain prospective.


### D1 Cargo manifest projection — 2026-11-14

A pure Rust library module was added at `tools/development-map-validator/src/cargo_manifest.rs`, exposed from the standalone crate. It accepts Cargo.toml bytes and an exact repository/revision/tree/path/blob anchor, then returns a Serde-serializable projection labelled `manifest_declared`. It preserves workspace member patterns, package metadata and inheritance declarations, normal/dev/build and target-specific dependency declarations, feature expressions, explicit targets, and build-script declarations. Inherited dependency defaults remain null/unknown unless locally explicit.

Two synthetic manifest fixtures and ten unit-test definitions were added. The parser does not execute Cargo, expand workspace globs, resolve workspace inheritance, evaluate feature combinations, infer implicit targets, inspect or execute build scripts, expand procedural macros, or produce compiler-resolved edges. A read-only `cargo-manifest` CLI subcommand now accepts explicit source pins and emits the Serde projection as JSON; it still does not authenticate the supplied Git objects. A dedicated JSON Schema and exact-head executable qualification remain future work. The crate now also contains a syntax-only Rust source projector at `tools/development-map-validator/src/rust_syntax.rs` and the explicit `rust-file` CLI mode. It records module nesting, declarations, functions, traits, methods, imports, visibility, attributes and item-position macro syntax under an exact source-file anchor, plus parser-derived start/end line and column spans. It does not follow out-of-line modules, expand macros, resolve imports, or evaluate cfg conditions. Ten additional test definitions are authored. The full crate now contains **96 authored Rust test definitions** (59 validator, 10 Cargo projection, 10 Rust syntax, 17 tree composition); none has been compiled or run.


## 9. Deterministic source-tree composition prototype

The research branch adds `tree_projection::compose_repository_tree(...)` to the standalone Rust validator. It composes Cargo-manifest and Rust-syntax projection outputs against one caller-supplied recursive Git tree inventory. Every join requires exact equality of repository, commit revision, tree OID, path and blob OID. It rejects duplicate tree paths and duplicate projection paths, malformed/noncanonical paths, inconsistent Git mode/object-type pairs, projections from another revision/tree, blob mismatches, and source-candidate paths represented by symlinks, directories or submodule gitlinks. It never follows links or invokes Git, Cargo, rustc, build scripts, tests or repository code.

The inventory is explicitly labelled partial or caller-asserted complete. Partial snapshots list all unprojected regular Cargo.toml and .rs paths and expose coverage counts. Complete snapshots are rejected unless every such source candidate has a corresponding projection. The word complete only describes the caller's assertion about the supplied recursive tree; the function cannot prove the inventory was exhaustive or authenticate Git objects. Output authority is `composed_source_anchors`, not compiler-resolved semantics or runtime evidence.

This is the next D1 source-composition increment, not a full repository twin. It deliberately does not expand Cargo workspace globs, resolve features or inherited values, evaluate Rust cfg, follow out-of-line modules, resolve imports, expand macros, or infer source-to-test / call relations. Those require separate, configuration-aware extractors with their own evidence and negative controls. Sixteen adversarial and determinism test definitions cover exact anchoring, blob mismatch, cross-revision rejection, duplicate paths, path traversal, partial/complete coverage, missing entries, symlink/gitlink boundaries, mode/type mismatch, duplicate projections and deterministic ordering. Test execution remains pending.


The tree composition API now has a JSON request representation and `tree-compose <request.json>` CLI path. Request payloads contain the exact repository/tree anchor, recursive tree entries, precomputed Cargo and Rust projections, and a caller-declared inventory completeness level. This makes composition reproducible from separately captured inputs without allowing the tool to fetch arbitrary paths or invoke repository-controlled code.
