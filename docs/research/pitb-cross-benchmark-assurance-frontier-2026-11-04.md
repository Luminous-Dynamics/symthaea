# PITB cross-benchmark assurance frontier

**Date:** 2026-11-04  
**Status:** research synthesis and proposed qualification protocol. No PITB end-to-end SOTA gain is claimed.

## Executive direction

PITB should evolve from a theorem-proving benchmark into an **evidence-qualified proof engineering benchmark**. Its scored unit should no longer be only a theorem that a model makes pass. The unit should be a complete, independently replayable contribution:

source task → intended specification → pinned Lean target → retrieved proof route → kernel proof → clean repository replay → maintained proof state

A proof assistant kernel checks whether a proof proves a formal proposition. It does not establish that the proposition is the intended one, that its hypotheses are meaningful, that an evaluation harness is resistant to trivial solutions, or that the result survives a later repository change. Recent benchmark audits make these failures measurable, not hypothetical.

PITB's differentiation should therefore be the combination of:
1. exact target and source provenance;
2. counterfactual support-route measurement;
3. typed graph retrieval with untrusted candidates kept separate from formal evidence;
4. budget-aware proving and independently verified completion;
5. selective proof maintenance under repository evolution.

This is a stronger product thesis than claiming one more point on a theorem pass-rate leaderboard.

## What the current benchmark landscape measures

These benchmark families are complementary, not directly rankable against one another.

| Benchmark/system | Published scope and result | What it stresses | PITB opportunity |
|---|---|---|---|
| **AutoformBench** | 341 audited Lean formalization tasks from 24 advanced mathematical sources; leading published run 281/341 (82.4%) | Building missing definitions, lemmas and supporting formal infrastructure, not only filling an isolated hole | Measure proof-route discovery, infrastructure planning, target fidelity and cost under matched budgets |
| **SorryDB-2601** | 2,601 real-project sorry tasks; 1,000-task paper evaluation split; project README reports 35.7% combined Pass@1, 30.3% for Gemini Flash 3 Agentic | Completing real Lean obligations inside their project, imports, toolchain and build context | Independently reproduce every source goal, audit semantic integrity, split by repository and time, and replay the entire project |
| **FVSpec** | 11,039 property-based tests scraped; 2,772 source tests translated into 9,415 Lean challenges; 333 repositories / 281 owners; published sample averages 70% easy and 49% hard | Translating imperative software properties into Lean, including Python semantics and specification inference | Make source-test-to-Lean-target correspondence and non-vacuity first-class scored outcomes |
| **LeanSearch v2** | Global premise retrieval: nDCG@10 0.62 vs 0.53 next best; on 69 research-level queries, 46.1% of annotated premise groups recovered within top 10; downstream proof success 20% vs 16% next best and 4% without retrieval | Recovering jointly useful premise groups, not merely one relevant declaration | Measure complete route sufficiency, minimal-route coverage and verified downstream solve/cost |
| **LeanGraphSearch** | Dependency graph expansion improves recall by up to 12% over LeanSearch v2 and covers up to 3.7% more complete annotated premise groups; end-to-end pass rate increases at fixed reflection budgets | Multi-hop retrieval through formal dependencies | Combine graph expansion with typed, independently justified edge semantics and counterfactual route validation |
| **TheoremGraph / LeanGraph** | 11.7M informal theorem-like environments, 18.3M candidate informal dependencies; 388,105 formal nodes and 11.3M typed Lean edges across 25 projects | Large-scale statement-level graph, with extractor labels preserving provenance/precision trade-offs | Preserve provenance and candidate-vs-observed distinctions; never promote a similarity edge to a formal dependency |
| **VeriSoftBench** | 500 tasks from 23 real-world Lean projects; best reported model solves 41% with curated context and 35% with raw repository context | Deep project-specific definitions, APIs and software semantics | Quantify the value and cost of PITB evidence/context preparation under identical models |
| **Vero** | 43 multi-module instances, 743 APIs and 2,705 specifications; strongest evaluated configuration solves 27/43 code-and-proof and 25/43 proof-only | Coherent implementation and proof synthesis across repositories | Add source/implementation/target binding and long-lived repository maintenance |
| **Aegis / EasyProver** | 2,045 differentially validated Rust-Lean pairs, 11,533 candidate specifications; EasyProver proves 1,054/2,000 sampled statements (52.7%) within 30 minutes | Implementation-derived obligations, fixed-width arithmetic, failure, loops, casts and mutation; recursive agent orchestration | Use Aegis as a hard software-verification cohort and compare proof correctness, spec fidelity, cost and repair under pinned conditions |
| **CoCo-Prover** | Two-level AND/OR proof graph plus declaration dependency graph; reports up to 100% on two benchmarks and up to 30.9% cost reduction vs its strongest baseline | Cost-efficient orchestration of specialist proving actions | PITB must match its success-cost frontier while adding source/spec assurance and proof lifecycle verification |

**Do not compare the headline percentages directly.** The cohorts, source visibility, task types, model versions, scaffolds, K budgets and stopping policies differ. Treat published results as reference points until a matched PITB rerun is complete.

## The reliability gap PITB should own

An ICML 2026 benchmark audit of five widely used Lean theorem-proving benchmarks reported 4,833 findings, including 398 mechanically certified issues such as counterexamples, vacuous theorems and unsound axioms. The important distinction is:

- **Kernel correctness:** the proof term checks against the supplied formal type.
- **Specification fidelity:** the formal type actually encodes the intended mathematical/software claim.
- **Evaluation integrity:** the harness admits only the registered task and permitted proof contribution.
- **Repository integrity:** the exact change builds and replays in the pinned project.
- **Lifecycle validity:** later dependency/environment changes do not leave a stale proof marked green.

These should be separate gates and separate metrics. A pass-rate improvement cannot compensate for a failed integrity gate.

## Paradigm shifts

### 1. Score contributions, not disappearance of sorry

The completion event should be a byte-bound patch which changes only the registered source span, preserves the declaration statement, introduces no forbidden axioms or sorryAx, passes the exact Lean kernel target, and survives a clean repository build.

A "sorry removed" string check is not sufficient. A theorem statement can be weakened, hypotheses added, or an axiom introduced while still producing a kernel-checked artifact.

### 2. Make source-to-formal target correspondence a first-class object

For every task, bind:
- upstream repository/file/commit and exact source range;
- original PBT/textbook/informal statement or source problem identity;
- formal declaration fully-qualified name and elaborated type digest;
- normalized statement AST digest, local context and imports;
- Lean/Mathlib/toolchain/environment digest;
- translation/extractor version, configuration and provenance;
- human review or independently reproducible source-to-target correspondence receipt.

A target can be kernel-checked while being semantically wrong. PITB should make that distinction visible and block unqualified tasks from the primary score.

### 3. Replace premise ranking with proof-route retrieval

A ranked list does not establish that the retrieved declarations jointly close a proof. Represent support as alternative AND-bundles and measure the minimal sufficient antichain under a frozen target, context, prover, toolchain and budget.

For example, if {h} and {hEq} each solve the exact target, neither is common to every minimal sufficient route. If only {h1,h2} succeeds, both are required in that observed minimal route. This is finite, policy-specific counterfactual evidence—not a theorem of logical necessity.

### 4. Treat external graphs as candidate evidence until re-extracted

DAPS, LeanGraphSearch and TheoremGraph are valuable candidate generators. Their edges do not all have the same semantics or trust:
- DAPS's USES_DEFINITION label does not prove a transparent definition-body dependency;
- an informal TheoremGraph link is a candidate, not a Lean elaborator event;
- retrieval scores are not probabilities and are not comparable across retrievers.

Preserve the raw source ID, relation, extractor, checkpoint and graph snapshot. Promote an edge only after an independent Lean extractor reproduces the exact typed relation under the pinned environment.

### 5. Optimize the entire verified frontier

CoCo-Prover demonstrates that the cost frontier is a first-class research result. PITB should measure cost per independently verified task, energy/GPU use, p50/p95 time-to-green and proof repair/replay cost. Use multi-arm orchestration with bounded, separately priced actions, but let deterministic proof DAG and verifier constraints control admissible work.

### 6. Give proofs a lifecycle

A proof is not permanently valid just because it once checked. Changes to declaration signatures, transparent bodies, instances, imports, axioms, modules or toolchains should create typed invalidation events. Preserve unaffected artifacts only when the dependency inventory is complete and the new state is independently replayed. A stale proof acceptance rate of zero is a hard gate.

## PITB benchmark portfolio

Use distinct tracks with their own scorecards.

| Track | Benchmark cohort | Primary outcome | Required integrity gates |
|---|---|---|---|
| A — Advanced formalization | AutoformBench | Exact audited target completion | source access pinned; allowed axioms; clean-room comparator and independent replay |
| B — Real mathematical projects | SorryDB-2601 plus temporal SorryDB cohort | Verified exact sorry completion and repository build | exact repository commit/Lean; source span and goal reproduction; patch-only replacement; full repo replay |
| C — Software property formalization | FVSpec | Correct PBT-to-Lean statement plus verified proof | source-test traceability; semantic review; hypothesis/non-vacuity audit; source implementation correspondence |
| D — Repository software verification | Vero, VeriSoftBench, Aegis | Verified APIs/specifications or exact repository instance completion | implementation/spec bindings; no altered specification; full dependency environment and replay |
| E — Global premise retrieval | MathlibQR, MathlibMPR, LeanSearch v2 evaluation queries | Complete useful premise group and downstream verified solve | same target/pool/context and budget; upstream candidate provenance; uncalibrated raw scores |
| F — Support antichains | Frozen exact-target ablation groups | Minimal route recovery and sufficient-bundle rate | exhaustive subsets, both proof gates, fixed budget/environment/context |
| G — Proof maintenance | PITB controlled mutation suite + real repository patches | Safe affected-set prediction and maintained verified state | zero stale acceptance; complete dependency inventory; independent full-state replay |

## Proposed qualification gates

These are **proposed targets**, not achieved measurements.

### SorryDB-2601 (1,000-task evaluation split)

Published reference in SorryDB's README: 35.7% combined Pass@1. On the same frozen runner, source visibility, prover and budget:

| Gate | Target | Count |
|---|---:|---:|
| Baseline replication | 35.7% ±1.0 pp | 347–367 / 1,000 |
| Competitive | ≥40.7% | ≥407 / 1,000 |
| Stretch | ≥45.7% | ≥457 / 1,000 |

Cost target: at least 20% lower cost per independently verified task without more than 1 pp completion regression. All reported successes must pass patch integrity, exact-target kernel replay and clean repository build.

### AutoformBench (341 tasks)

The top published source-present leaderboard result is 281/341 (82.4%):
- competitive arithmetic gate: 299/341 (first integer count clearing +5 pp);
- stretch gate: 316/341 (first integer count clearing +10 pp).

Do not import textbook source excerpts, hidden traces or gold proofs into training. The public AutoformBench release is CC BY-NC 4.0 and explicitly intended for research/evaluation rather than training.

### FVSpec / Vero / Aegis

First replicate each benchmark's published evaluation procedure and exact model/scaffold/budget. Then target +5 pp verified task completion or a ≥20% cost reduction at non-inferior completion, with no deterioration in source/specification fidelity. Report by repository/owner and easy/hard or obligation-complexity stratum.

### Retrieval and support

- Report Recall@K/nDCG@K only as diagnostics.
- Primary retrieval metric: complete annotated proof-route coverage and useful-novel-premise rate.
- Primary downstream metric: verified exact-target completion under the same prover budget.
- Support metric: route sufficiency, exact minimal-route recovery, necessary-across-antichain precision/recall and probability calibration.
- Compare BM25, LeanSearch v2, LeanGraphSearch, DAPS, TheoremGraph, PITB typed bundle and their union with the same candidate pool and prover budget.
- Never accept score calibration claims without a held-out calibration set.

### Proof maintenance

- stale proof acceptance: **0**;
- stale receipt replay acceptance: **0**;
- curated critical-change recall: **100%** before hidden testing;
- hidden impact recall: target ≥0.99 with project/patch-family confidence intervals;
- impact precision: target ≥0.95 on adjudicated changes;
- cost per verified maintained change: target ≥20% below full rebuild at non-inferior safety;
- full required repository replay: 100% of counted maintained changes.

Finite mutation suites do not establish universal guarantees. Treat repositories or independently sampled patch families as inference clusters; do not inflate sample size by counting individual edges as independent examples.

## Experimental controls

Every comparison must freeze:
1. benchmark release, source visibility and task membership;
2. model/version, system prompt, agent/tool versions, retrieval checkpoint and configuration;
3. Lean/Mathlib/Lake versions and environment digest;
4. target statement and source binding;
5. candidate pool, context policy and allowed axioms;
6. total wall-time, token, CPU/GPU and monetary budgets;
7. stopping, retry, self-consistency and parallelism policy;
8. verifier and clean-replay procedure.

Use project/owner-disjoint and temporal cohorts. Deduplicate by statement/goal hashes and source lineage. Keep public development cohorts separate from hidden confirmatory evaluation. Bootstrap by source repository/group, not theorem row.

## Execution sequence

1. **Integrity foundation:** complete the pinned SorryDB-2601 file/record/split audit; keep source reproduction and semantic fidelity unqualified until independently verified.
2. **Cross-benchmark binding:** adopt one source-to-formal-target receipt across SorryDB, FVSpec, AutoformBench and software verification tasks.
3. **Retrieval integration:** map DAPS, LeanSearch v2, LeanGraphSearch and TheoremGraph candidate outputs into query-level receipts and the typed PITB graph.
4. **Counterfactual support:** use exhaustive ablations on small contexts to validate route sufficiency and alternative support bundles.
5. **Proof controller:** combine typed dependency DAGs and AND/OR proof bundles with cost-aware action routing; retain full rebuild as safety baseline.
6. **Maintenance:** apply controlled patch families, compute invalidation closures, selectively repair, then replay the whole required repository state.
7. **Confirmatory claims:** freeze the method on development repositories and report only on unseen owners/projects and later commits.

## Current implementation workstreams

- [Typed proof graph evidence](https://github.com/Luminous-Dynamics/symthaea/pull/10317)
- [AutoformBench dependency-aware formalization preregistration](https://github.com/Luminous-Dynamics/symthaea/pull/10335)
- [Signed graph snapshot attestation](https://github.com/Luminous-Dynamics/symthaea/pull/10336) — requires exact-head repair/qualification
- [Typed proof impact and maintenance analyzer](https://github.com/Luminous-Dynamics/symthaea/pull/10337)
- [DAPS candidate-only graph import](https://github.com/Luminous-Dynamics/symthaea/pull/10351)
- [Rust support-antichain analyzer](https://github.com/Luminous-Dynamics/symthaea/pull/10355)
- [Per-query retrieval evidence receipts](https://github.com/Luminous-Dynamics/symthaea/pull/10360)
- SorryDB-2601 source snapshot integrity work: tools/pitb-sorrydb-integrity/

## References

- AutoformBench: https://github.com/facebookresearch/autoformbench and https://ai.meta.com/research/publications/autoformbench-a-lean-formalization-benchmark-for-advanced-mathematics/
- SorryDB paper: https://proceedings.mlr.press/v306/letson26a.html; repository and baseline documentation: https://github.com/SorryDB/SorryDB
- FVSpec: https://fvspec.galois.com/ and https://arxiv.org/abs/2606.01008
- LeanSearch v2: https://arxiv.org/abs/2605.13137
- LeanGraphSearch: https://arxiv.org/abs/2610.22339
- TheoremGraph: https://arxiv.org/abs/2606.25363
- CoCo-Prover: https://arxiv.org/abs/2610.09681
- Aegis / EasyProver: https://arxiv.org/abs/2610.38174
- Vero: https://vero.verina.io/ and https://arxiv.org/abs/2608.13522
- VeriSoftBench: https://utopia-group.github.io/VeriSoftBench/
- Benchmark audit: https://proceedings.mlr.press/v306/ammanamanchi26a.html
