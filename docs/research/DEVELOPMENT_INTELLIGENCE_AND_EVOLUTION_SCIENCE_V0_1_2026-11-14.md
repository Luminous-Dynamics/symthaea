# Development Intelligence and Evolution Science (DIES) — V0.1

**Date:** 2026-11-14  
**Status:** Research and integration blueprint; not a qualified autonomous development capability  
**Parent:** [DYNMAP-001 #12020](https://github.com/Luminous-Dynamics/symthaea/issues/12020)  
**Representation:** Evidence-Grounded Dynamical Development Maps (EGDDM)  
**Execution boundary:** [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064)

## Executive thesis

EGDDM should become the persistent, evidence-grounded model of an engineering system. Development Intelligence and Evolution Science (DIES) is the reasoning and evaluation discipline that uses that model to explain change, form competing hypotheses, select experiments, compare possible development paths, and measure whether an intervention actually improved a system.

DIES is **not** a second graph engine, event store, evidence ledger, task registry, scheduler, authorization service, CI system, or agent runtime. It composes the existing Symthaea, Mycelix, Symtropy, Sol Atlas and CEB owners listed below. EGDDM remains the common projection and federation layer; each domain retains its own authoritative records, semantics and permissions.

The objective is not to maximize code output. It is to maximize independently demonstrated capability improvement per constrained engineering effort, while preserving hard safety, authority, evidence and correctness boundaries.

## 1. What is already established

The 2026-11-14 EGDDM pilot has four bounded repository-local source projections:

| Repository | Map nodes / edges | Current scope |
|---|---:|---|
| Symthaea | 13 / 18 | README and selected public issue/workstream observations |
| Mycelix | 17 / 14 | README-derived maturity projection |
| Symtropy | 20 / 29 | README and architecture-document projection |
| Sol Atlas | 41 / 51 | Demo, selected source paths and evidence/decision documentation |

The combined 91 nodes and 112 edges are counts of these bounded map artifacts, **not** completeness, capability or maturity scores. The four maps remain independently owned. The current federation contains 11 directional relationships, zero compatibility receipts and zero independently qualified edges. A declaration of integration is not proof of API or runtime compatibility.

The existing owners are deliberately reused:

- **Repository structure and change impact:** [#10173](https://github.com/Luminous-Dynamics/symthaea/issues/10173).
- **Engineering context and correctness:** [#10419](https://github.com/Luminous-Dynamics/symthaea/issues/10419), [#10637](https://github.com/Luminous-Dynamics/symthaea/issues/10637), [#11772](https://github.com/Luminous-Dynamics/symthaea/issues/11772).
- **Temporal engineering decisions:** [#12017](https://github.com/Luminous-Dynamics/symthaea/issues/12017), [#12042](https://github.com/Luminous-Dynamics/symthaea/issues/12042).
- **Retrieval and development planning:** [#12080](https://github.com/Luminous-Dynamics/symthaea/issues/12080), [#12071](https://github.com/Luminous-Dynamics/symthaea/issues/12071).
- **Causal debugging and evolutionary attribution:** [#10366](https://github.com/Luminous-Dynamics/symthaea/issues/10366), [#10717](https://github.com/Luminous-Dynamics/symthaea/issues/10717), [#12067](https://github.com/Luminous-Dynamics/symthaea/issues/12067).
- **Decision quality and evidence acquisition:** [#12039](https://github.com/Luminous-Dynamics/symthaea/issues/12039), [#12046](https://github.com/Luminous-Dynamics/symthaea/issues/12046).
- **Execution isolation and independent qualification:** [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064) and the existing CEB receipt/judge owners.

These issue links are ownership references, not claims that every referenced capability is implemented or qualified.

## 2. Paradigm shifts

### 2.1 From code graphs to claim-and-capability graphs

A code graph answers which components reference other components. A capability graph answers whether the system can satisfy a stated requirement, under which configuration, and with what evidence.

The primary trace should be:

**Objective → Capability → Requirement / invariant → Implementation → Interface → Test / experiment → Evidence → Qualified claim**

A capability may span many crates, projects, contracts, operators and test environments. Conversely, the presence of code does not establish a capability. Every edge needs provenance and authority; every missing link must be visible.

### 2.2 From one notion of time to decision-time reconstruction

A repository revision timestamp is not enough to reconstruct what a person or process could legitimately know when it acted. Keep distinct:

1. Valid/effective time — when a fact applies to the subject.
2. Source-observed or published time — when the source recorded or exposed it.
3. Available-to-actor time — when the relevant actor could actually access it with the right scope.
4. Decision time and knowledge cutoff — when the decision was made and what it admitted.
5. Correction/supersession time — when later information changes present understanding.

Also bind the repository revision, configuration world, tool/evaluator identity and exact source digest. A backdated event is not evidence that the event was available earlier. A later correction changes today's understanding without rewriting yesterday's decision record.

### 2.3 From correlation to intervention-backed causal claims

A graph edge, chronology, plausible explanation or passing regression test is not proof that a change caused a capability gain.

DIES should maintain competing causal hypotheses and require discriminating evidence. When execution is authorized, use controlled interventions, sham/negative controls, condition-removed twins, exact pre/post state, independent observation and replay. Report intervention delivery, isolation and interpretability separately. Where an effect is not identifiable, preserve that result rather than inventing a causal explanation.

Reuse #10366, #10717 and #12067. Do not create a second causal engine or evolutionary lineage store.

### 2.4 From roadmaps to counterfactual capability trajectories

A development plan should be a proposed DAG of bounded actions, not a ranked issue list. Each action carries prerequisites, affected capability claims, source map pins, cost range, uncertainty, risks, reversibility, owner, authority requirements and acceptance evidence.

Hard constraints are non-compensable. A high predicted benefit cannot compensate for an unknown authority boundary or missing critical contract. When plans are incomparable, return the Pareto set rather than hiding the trade-off inside a weighted score. An evidence-acquisition action must be distinguishable from an implementation action.

Reuse #12071 for plan format and feasibility; consume existing #8908 value-of-information and #6206 engineering-cognition/abstention outputs.

### 2.5 From collecting more data to acquiring decision-sufficient evidence

The best next action may be a small, targeted observation that resolves one material uncertainty. It need not be more code or a larger scan.

For each candidate evidence request, identify the blocked claim, competent source, required fields, time scope, expected decision-loss reduction, cost, delay, disclosure burden, dependency/common-cause risk and human authorization needed. A request with unknown expected value should be deferred or abstained from, not embellished with a false precision score.

Reuse [#12046](https://github.com/Luminous-Dynamics/symthaea/issues/12046); it owns the evidence-acquisition planner.

### 2.6 From one central knowledge graph to federated intelligence

Each repository owns its local map and source authority. The federation joins exact, versioned contracts and pinned member snapshots; it does not merge permission, governance, finance, identity, deployment or physical-actuation authority.

A cross-project edge may progress from source-declared, to contract-versioned, to independently qualified only through a two-sided producer/consumer conformance receipt bound to both exact source snapshots, contract bytes, verifier build and configuration world. Same version strings are not compatibility evidence.

### 2.7 From “always act” to paired action/defer competence

Every consequential development task should have a near-identical should-defer or should-abstain twin where a critical prerequisite is absent, stale, unauthorized, ambiguous or impossible to obtain. This blocks both failure modes: reckless execution and useless blanket refusal.

Score correct action and correct abstention separately, and score the pair together. Also record when the system abstains: before action, after unnecessary investigation, or only after an irreversible effect. Post-hoc abstention is not safe abstention.

Use AgentAbstain's paired-task design as an external methodological comparator, and extend the paired controls within DTEA and DYNPLAN. Do not claim direct leaderboard comparability across different task environments.

### 2.8 From patch success to measured evolution

The learning loop must compare forecast with outcome:

**Map snapshot → Hypothesis → Predicted impact → Bounded intervention → Independent observation → Replay → Updated map**

Retain failed hypotheses, negative results, regressions, uncertainty and experiment cost. Evaluate whether the development process itself improves: fewer unnecessary investigations, better experiment selection, less rework, lower false-acceptance risk and higher independently qualified capability gain.

## 3. Architecture: compose, do not duplicate

| Function | Existing owner / input | DIES responsibility |
|---|---|---|
| Read-only repository facts | #10173; manifests, syntax, semantic indexers | Normalize source-attributed facts and configuration worlds |
| Intent and capability map | #12020 and domain requirements | Trace capability obligations to implementation and evidence |
| Temporal evidence | #12017, #12042; Mycelix #6069 / #6134 | Compare valid, observed, available and decision time without replacing owner ledgers |
| Retrieval | #10419, #12080 | Select sufficient regions under fixed budgets and exact revision/world pins |
| Causal diagnosis | #10366, #10717, #12067 | Compare hypotheses and intervention evidence |
| Decision / acquisition policy | #12039, #12046; #8908 / #6206 | Check decision loss, abstention, minimum evidence request and hard constraints |
| Development trajectories | #12071 | Compare feasible, blocked and deferred plans |
| Cross-project contracts | #12077 and owner contracts | Detect drift; require two-sided conformance before qualified joins |
| Execution and replay | #10064 and qualified Symtropy/CEB execution owners | Consume isolated execution receipts only |
| Independent qualification | #11772 and CEB custody/oracle owners | Keep task truth, execution truth and qualification independent of the proposer |

Symthaea should propose and reason. Symtropy may execute or replay a bounded task when the relevant isolation and domain qualification exists. Mycelix remains authoritative for its own trust, governance and institutional records. Sol Atlas remains authoritative for its own source and spatial evidence. No project may self-promote its claims through the federation.

## 4. Industry and research baselines

The field already includes strong, partially overlapping systems. We should not claim novelty for repository graphs, temporal knowledge graphs, code dependency extraction or retrieval alone.

| Baseline | Published / documented strength | DIES comparison |
|---|---|---|
| Aider RepoMap | Dependency-ranked symbol/signature map selected under a token budget | Match its context budget, then test capability/evidence-aware ranking |
| RIG / SPADE | Deterministic build/test graph; reported +12.2% mean accuracy and −53.9% completion time in a small 8-repository, 3-agent study | Reproduce task conditions; extend above build/test structure |
| SCIP / Sourcegraph | Language-agnostic semantic index format and precise cross-repository navigation | Consume semantic facts where available; do not reimplement compiler indexing |
| CoReGraph | Repository knowledge graph combining technical dependencies, development activity, temporal modeling and provenance | Treat this overlap seriously; differentiate through decision admissibility, causal attribution and qualified evolution |
| SWE-Time | 1,661 cases, 65 repositories, retrospective regression and prospective change reasoning | Run its official tasks separately, then add DTEA and cross-project development outcomes |
| Agent Retrieval Bench | 427 samples / 25 repositories; different leaders for MRR, Recall@20 and budgeted context yield | Reproduce official baselines and report task-stratified context yield |
| SWE-Explore | 848 issues across 203 repositories and 10 languages; ranked line regions under a fixed line budget | Compare K, line budget, line coverage, ranking and downstream restricted-context results |
| TEMPO | 1,730 temporal retrieval queries across 13 domains; temporal coverage remains incomplete | Preserve its official retrieval metrics; evaluate admissibility as a distinct task |
| AgentAbstain | 263 paired tasks across 42 environments; best reported paired accuracy is 59.5% in its evaluation | Borrow paired act/defer methodology; benchmark engineering-specific authority and evidence triggers separately |

Primary sources:
- Aider: https://aider.chat/docs/repomap.html
- RIG / SPADE: https://arxiv.org/abs/2601.10112
- SCIP: https://scip-code.org/
- CoReGraph: https://doi.org/10.1016/j.procs.2026.09.172
- SWE-Time: https://arxiv.org/abs/2611.12083
- Agent Retrieval Bench: https://arxiv.org/abs/2607.24882 and https://agent-retrieval-bench.github.io/
- SWE-Explore: https://arxiv.org/abs/2606.07297
- TEMPO: https://aclanthology.org/2026.emnlp-main.291/
- AgentAbstain: https://arxiv.org/abs/2607.10059 and https://agentabstain.github.io/

The comparisons are not interchangeable. RIG's study size and scope differ from SWE-Time; TEMPO measures temporal retrieval rather than decision-time admissibility; trajectory-derived exploration gold is not a proof that no other useful region exists; AgentAbstain's task environment is not an engineering-change benchmark. Reproduce an external benchmark with its original task, splits, tokenizer, harness, scoring and rules before making any direct comparison.

## 5. DynaMapBench: separate tracks, no compensating score

DynaMapBench should be a benchmark family composed from existing task owners, not one scalar leaderboard.

| Track | Core measure | Prospective pilot target |
|---|---|---|
| Structural fidelity | Syntax and compiler-resolved relation precision/recall | Syntax ≥0.95; compiler-resolved ≥0.90 |
| Change impact | Critical affected-component/test recall and precision | Recall ≥0.95; precision ≥0.80 |
| Temporal replay | As-of reconstruction, invalidation, history preservation | 100% deterministic replay on frozen fixtures |
| Decision admissibility | Evidence available, authorized and effective at decision time | ≥0.95 precision/recall after independent gold adjudication; zero future-evidence leakage in explicit controls |
| Retrieval/exploration | Region/line precision, recall, F1, MRR, nDCG and budgeted context yield | Reproduce external baselines first; retain the existing DYNRETR target of ≥5 percentage-point line-recall improvement over reproduced K=5 CoSIL under the same budget, with ≤1 pp precision regression |
| Development planning | Prerequisite closure, hard-constraint violations, regret, Pareto coverage and correct defer | Zero hard-constraint violations; 100% detection of explicit missing critical contracts; ≥0.95 prerequisite precision/recall |
| Causal development | Supported causal relations, predicted-vs-observed interventions, accepted repairs and regret | ≥0.90 precision/recall for supported causal relations on adjudicated gold; unresolved relations stay unresolved |
| Federation | Critical contract mismatch recall, false-compatible edges, pin drift | ≥0.95 critical mismatch recall; zero promotion without two-sided conformance |
| Paired action/defer | Pair accuracy, timely abstention, irreversible unsupported effects | Engineering-specific target ≥0.75 paired accuracy on a held-out pilot, plus zero unauthorized effects |
| Evolution outcomes | Independently accepted capability gain, regressions, cost and forecast calibration | ≥10 pp accepted-improvement gain and ≥20% lower experiment cost than strongest matched baseline, only if powered paired intervals support both claims |

These targets are hypotheses, not achieved results or industry standards. A target is not a score. Every track must publish confidence intervals, denominators, per-family results, abstention, unknowns, cost, latency, memory and reviewer effort.

### Fair comparison protocol

1. Freeze exact source commits, trees, map artifacts, configuration worlds, tools, prompts, models, tokenizers, budgets and execution policy.
2. Compare ordinary search, BM25, embeddings/RRF, Aider-style RepoMap, deterministic structure, SCIP/compiler-resolved indexing, static analysis, temporal-only retrieval, EGDDM ablations, full EGDDM, and an expert-human arm.
3. Use repository-family and time-held-out splits. Avoid putting both near-identical versions of a task or answer key into train/development and confirmatory sets.
4. Pair tasks that differ by one decisive fact: current vs stale evidence, exact vs mismatched contract blob, authorized vs unauthorized actor, active vs inactive feature, should-act vs should-defer.
5. Keep grader/gold custody outside candidate-visible context. Use independent adjudication and separately implemented reference checks for evaluator correctness.
6. Report a metric only when its task scope, denominator and evidence are valid. Do not pool official external benchmark scores into a DynaMapBench composite.

### New metamorphic and noninterference tests

Use the existing #10173 mutation gold set and extend it with map-level invariants rather than creating a duplicate corpus:

- A one-file or one-contract mutation changes the predicted affected subgraph and leaves unrelated claims invariant.
- A source revision change invalidates only evidence whose declared scope depends on that revision/configuration.
- A map artifact update does not masquerade as a source-subject update.
- An irrelevant file addition or event reordering cannot improve answer quality by leaking gold or changing unrelated conclusions.
- A backdated but late-observed event cannot change the actor's historical decision view.
- A stale map cannot be silently treated as current.
- Incremental update and clean rebuild produce byte-identical canonical snapshots for the same frozen inputs.
- A missing/ambiguous source must cause unknown or abstention, not a plausible invented edge.

These are properties to test, not guarantees the current prototype already satisfies.

## 6. Rust-first implementation path

### D1 — Read-only source projection

Build a small standalone Rust mapper that reads a frozen repository tree and never executes repository-controlled code.

- Parse Cargo manifests as data; extract workspace/package identity, members, dependencies, optional dependencies, feature declarations, targets and declared build-script presence.
- Parse Rust syntax into source spans, modules, imports, declarations, public APIs, tests and conditional compilation attributes.
- Treat syntax-extracted edges and manifest-declared edges as such. Do not label them compiler-resolved.
- Do not run build scripts, procedural macros, tests, cargo check, arbitrary commands or repository code.
- Produce deterministic content-addressed snapshots, explicit extraction limitations, source provenance and configuration-world identities.
- Add qualified semantic-indexer adapters later; rust-analyzer/SCIP output must carry its own exact tool, build, target and configuration identity.

### D2 — Temporal event replay

Implement deterministic as-of projection over append-only events. Preserve event time, observation time, availability, effective revision, correction and supersession. Compare incremental updates with a clean rebuild; keep historical views after current evidence becomes stale.

### D3 — Change-impact and capability traceability

Predict affected symbols, interfaces, tests, requirements and capabilities from the pre-change view. Score prospective predictions before exposing the changed source or future snapshot. Reuse #10173 gold tasks and independently adjudicated requirement obligations.

### D4 — Fixed-budget retrieval

Combine deterministic structural expansion, semantic/lexical retrieval and temporal/configuration/authority filtering. Pack exact source regions under fixed token and line budgets. Separate natural no-gold from wrong-repository controls; permit explicit abstention.

### D5 — Causal and counterfactual planning

Consume existing causal and value-of-information owners. Compare feasible development trajectories, preserve hard constraints and select the smallest informative experiment. Forecast benefit and cost, then compare them with independently observed outcomes.

### D6 — Federated adoption

Keep each repository's map and authority local. Add consumer/producer conformance only after exact contract and two-sided receipts exist. Expand beyond the current four bounded projections only when the first Rust mapper and temporal replay are independently qualified.

## 7. Current no-runner increment and claim ceiling

On 2026-11-14, the standalone development-map-validator received a DTEA hardening increment:

- Consulted evidence is checked against effective-time start/end bounds at the decision timestamp.
- Reversed validity intervals are rejected.
- Equality at the start/end boundary is accepted under closed-interval semantics.
- An explicit validity_completeness declaration distinguishes a known open interval edge (complete with a null bound) from incomplete applicability (incomplete).
- Incomplete effective-time applicability cannot justify consulted evidence; it can remain explicitly excluded.
- Five additional Rust test definitions were authored, bringing the source to **59 authored tests**, and the public DTEA expected-results fixture now contains **21 negative controls**.

These are source changes and authored checks. The tests have not been compiled or executed. Exact-head CI workflows were observed as skipped, not passed. Full JSON Schema 2020-12 validation, runtime replay, independent benchmark qualification and #10064 isolation qualification remain outstanding. No real Mycelix authorization record is authenticated by the synthetic DTEA fixture.

## 8. Decision

Pursue DIES as a first-class Symthaea research and engineering discipline, but keep EGDDM as its common map representation and keep the existing owner boundaries intact.

The practical next engineering milestone is D1: a read-only Rust source projector with explicit authority labels, followed by D2 temporal replay. The highest-value research milestone is a preregistered, matched DynaMapBench evaluation that demonstrates whether maps improve decisions and independently verified capability outcomes—not merely whether they make better diagrams.

The test of success is not that Symthaea can describe her own architecture. It is that she can identify a real gap, select a justified experiment, predict its result, respect authority and safety constraints, and learn from an independently verified outcome.
