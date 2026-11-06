# PITB cross-benchmark assurance frontier

**Date:** 2026-11-06  
**Status:** research synthesis and qualification plan; no new cross-benchmark score or SOTA gain is claimed.

## Executive position

The field has moved beyond isolated theorem completion. Current benchmarks now test materially different work:

- **SorryDB:** fill a sorry inside the original repository, commit, Lean version and local context.
- **AutoformBench:** translate advanced mathematical development into Lean while constructing missing definitions, intermediate lemmas and supporting infrastructure.
- **FVSpec:** recover software properties from real property-based tests, formalize them in Lean, and discharge the resulting verification obligations.
- **MathlibQR / MathlibMPR and premise-retrieval suites:** retrieve relevant declarations or complete premise groups, sometimes using multi-hop dependency structure.
- **Proof-maintenance / change-impact tasks:** determine which previously valid proof assets became stale after source, dependency, instance, import or environment changes.

These are different task distributions and success conditions. Their headline percentages must **not** be pooled or ranked as if they were samples from one common benchmark.

PITB's opportunity is a shared *assurance and evidence layer* across those task types: freeze the task, reconstruct its source and environment, audit specification fidelity, retrieve typed support, construct a proof/formalization, independently verify it, then test whether the artifact remains valid after controlled changes.

## Current benchmark portfolio

| Benchmark | What it measures | Public scale / reported reference | What PITB must preserve |
|---|---|---|---|
| SorryDB-2601 | Fill existing Lean proof holes in real repositories | 1,000-task published evaluation; official page shows 57.2% Pass@1 for Aleph Prover and 63.2% union coverage across listed systems | Exact repo commit, Lean toolchain, source coordinate, live goal, build and kernel receipt |
| AutoformBench | Advanced-mathematics formalization, including definitions and supporting infrastructure | 341 tasks / 24 source groups; GPT-6 Astra 281/341 (82.4%) in the published source-present condition | Exact source visibility, formalization target, clean-room workspace, permitted axioms and comparator/kernel replay |
| FVSpec | Translate real Python property-based tests into Lean specifications and proofs | 11,039 PBTs, 9,415 Lean challenges, 333 source repositories; 62% hard by the published classifier | Original PBT bytes, Python semantics, translation lineage, Lean specification audit, project/source identity |
| LeanSearch v2 / MathlibQR / MathlibMPR | Global premise retrieval and recovery of proof-support groups | LeanSearch v2 reports nDCG@10 0.62 and 46.1% recovery of annotated premise groups within 10 candidates on its research-level set | Candidate-pool identity, query identity, exact ranking receipt, uncalibrated score semantics and downstream proof outcome |
| LeanGraphSearch | Multi-hop dependency graph retrieval | Reports up to +12% recall over LeanSearch v2 and improved complete-route coverage on MathlibMPR | Graph snapshot and relation provenance; report retrieval recall separately from proof completion |
| DAPS | Dependency-aware premise selection and Blueprint/Mathlib adaptation | Reports mean Recall@32 of 89.25 on Mathlib4-Heldout, +11.41 over LeanHammer | Model checkpoint, graph and candidate-pool identity, split, source of each graph edge, per-query ranked output |

Sources: [SorryDB paper and evaluation](https://proceedings.mlr.press/v306/letson26a.html), [SorryDB public leaderboard](https://sorrydb.org/), [AutoformBench release](https://github.com/facebookresearch/autoformbench) and [Meta paper description](https://ai.meta.com/research/publications/autoformbench-a-lean-formalization-benchmark-for-advanced-mathematics/), [FVSpec leaderboard](https://fvspec.galois.com/) and [paper](https://arxiv.org/abs/2606.01008), [LeanSearch v2](https://arxiv.org/abs/2605.13137), [LeanGraphSearch](https://arxiv.org/abs/2610.22339), [DAPS](https://arxiv.org/abs/2610.41536).

### Critical interpretation limits

**SorryDB:** the fixed evaluation split is task-held-out, not project-disjoint. It spans all 78 repositories in the pinned full snapshot, with 107 repo/commit pairs and 339 repo/file clusters. It is therefore possible for development tasks and evaluation tasks to share substantial project-level context. The current PITB snapshot audit also surfaced a dataset-documentation mismatch: the upstream README calls the full snapshot 2,601 tasks, while the exact pinned JSON blob contains 5,663 records. PITB records both values and treats the pinned bytes as authoritative.

**AutoformBench:** its 82.4% result is not comparable to SorryDB Pass@1. The tasks require building formal infrastructure, and the official leaderboard explicitly distinguishes the source-present condition and asks readers to consider each model's scaffold, source access, tools, budgets and stopping policy. The release is CC BY-NC 4.0 and is marked for evaluation/research, not model training. PITB must respect the benchmark and third-party source licenses.

**FVSpec:** the corpus has 9,415 Lean challenges, but the published headline easy/hard baselines are described on a sampled 100 easy + 100 hard problems. Do not present the 70% easy / 49% hard model average as a full-corpus success rate.

**Retrieval benchmarks:** Recall@K, nDCG and annotated premise-group recovery are retrieval outcomes, not end-to-end proof success. Raw scores from different retrievers are not calibrated probabilities and cannot be compared numerically without score normalization.

## The benchmark-integrity problem is now a first-class research problem

The ICML 2026 audit *Faults in Our Formal Benchmarking* reports 4,833 findings across five Lean theorem-proving benchmarks and forks, including 398 mechanically certified issues such as counterexamples, vacuous theorems and unsound axioms. It also identifies missing hypotheses, simplification, incorrect translations, evaluation loopholes and Lean-specific specification hazards.

This leads to a strict distinction:

1. **Kernel validity:** the proof establishes the exact formal proposition Lean checked.
2. **Source fidelity:** the proposition is the one actually encoded at the pinned source location.
3. **Intent fidelity:** the formal proposition faithfully expresses the original informal mathematical or software requirement.
4. **Evaluation integrity:** the harness cannot accept vacuous solutions, sorryAx, injected axioms, weakened statements or hidden denominator changes.
5. **Maintenance validity:** the proof remains valid under the intended source and dependency evolution policy.

A kernel pass establishes only the first. PITB must report each gate separately.

Reference: [Faults in Our Formal Benchmarking, ICML 2026](https://proceedings.mlr.press/v306/ammanamanchi26a.html).

## Paradigm shifts

### 1. From proof generation to an assurance lifecycle

The benchmark unit should not be only a theorem statement. It should be a versioned assurance package:

source -> formal statement -> goal/context -> typed support -> proof artifact -> checker receipt -> semantic review -> maintenance receipt

Every transition has an identity, provenance and a failure taxonomy. A missing source replay is not a failed proof attempt; a kernel rejection is not a semantic mismatch; and an accepted Lean proof is not an informal-intent certificate.

### 2. From ranked premises to verified support routes

A ranked list is a weak interface to proof search. The useful object is an AND/OR set of possible support routes:

- each route is a set of premises that is jointly sufficient under one frozen target, environment, prover policy, context and budget;
- alternative minimal routes remain separate;
- the intersection of all observed minimal routes is only *necessary within that finite experiment*, not a general logical-necessity theorem;
- retrieved candidates become scheduler-eligible only after exact identity and semantic role are independently verified.

The Rust support-antichain analyzer and per-query retrieval receipts implement the measurement boundary needed for this shift.

### 3. From one dependency graph to a typed, provenance-bearing assurance graph

A graph edge must say what was observed: type reference, signature reference, proof-term reference, transparent definition-body reference, instance synthesis, module/environment dependency, documentation link, or unverified retrieval candidate. Similarity and occurrence are not logical necessity. External DAPS edges remain candidate-only until an independent Lean extraction reproduces their exact typed meaning.

### 4. From full rebuild versus optimistic caching to safe proof maintenance

After a source or environment change, a prover should calculate the minimum *safe* revalidation frontier. The optimization target is not minimum rebuild count by itself. It is minimum maintenance cost subject to zero stale-proof acceptance and complete independent replay of the required release state.

The typed proof-impact analyzer separates reprove, revalidate, rebuild, retrieval refresh and unknown-conservative actions. Its output is a plan, not a proof certificate.

### 5. From IID leaderboard statistics to hierarchical, paired inference

Tasks within the same repository and source file are dependent. Compare systems on identical task IDs, treat unattempted tasks as unsolved while reporting attempt coverage, collapse retries to one task-level success, and resample the project/file hierarchy. A fixed SorryDB cohort still cannot establish unseen-project generalization: that requires a repository-disjoint or later temporal cohort.

### 6. From a scalar SOTA score to an assurance-adjusted frontier

A meaningful comparison is a vector, not one percentage:

- source reconstruction and goal-replay coverage;
- specification-fidelity and vacuity/assumption audit coverage;
- exact-target verified completion (Pass@1 and Pass@k);
- retrieval route coverage and useful-novel-premise rate;
- unique tasks solved and cross-system union;
- cost, tokens, wall time and GPU/CPU resources per verified completion;
- stale-proof false-acceptance and invalidation recall;
- repository/file/temporal generalization.

A system that gains 3 points of raw pass rate while weakening the specification, skipping source replay or increasing proof cost has not improved the assurance frontier.

## SOTA comparison protocol

For every pairwise comparison, freeze:

- benchmark snapshot and full/evaluation artifact digests;
- exact task IDs, repository commits, Lean/toolchain and dependency environment;
- informal source visibility and target statement;
- candidate pool, retrieval graph and retriever/checkpoint;
- base prover, tool access, model, scaffold, reflection/turn limits and stopping rule;
- wall, CPU, GPU, token and monetary budgets;
- checker, permitted axioms, semantic-gate rubric and independent replay procedure.

Use the same tasks and total budget for each system. Do not compare model A with Pass@32 against model B at Pass@1. Publish per-task results, attempt coverage, unique solved IDs, cost and the union of complementary systems.

The current SorryDB implementation now emits:
- paired task-level differences on the fixed 1,000-task denominator;
- repository-cluster bootstrap intervals (10,000 replicates);
- nested repository/file bootstrap intervals (10,000 replicates);
- repository-cluster paired sign-flip tests (50,000 replicates);
- Holm correction across the complete strategy-pair family;
- deterministic seeds and explicit cluster denominators.

These are cohort-level inference tools, not a substitute for repository-disjoint and temporal evaluation.

## Proposed benchmark targets

Targets below are proposed qualification thresholds, not measured PITB results.

| Workstream | Public comparison point | Competitive target | Stretch target |
|---|---|---|---|
| SorryDB proof completion | Public leaderboard: Aleph Prover 57.2% Pass@1; listed-system union 63.2% | Beat 57.2% Pass@1 under a matched, assurance-adjusted protocol | Exceed 63.2% unique-task union under the same frozen split and budget, with source/semantic gates |
| AutoformBench formalization | GPT-6 Astra 281/341 (82.4%) source-present | At least 299/341 (87.7%) | At least 316/341 (92.7%) |
| FVSpec hard subset | Published sampled baseline average 49% hard | At least 54% proved on the same frozen hard sample and budget | At least 59%, while easy performance does not regress by >1pp |
| Premise retrieval | LeanSearch v2: 46.1% annotated groups within 10 candidates; LeanGraphSearch up to +12% recall | +3pp complete-route recovery over the best matched baseline | +5pp route recovery and lower cost per verified exact target |
| Proof maintenance | No directly comparable public global score established | 100% critical-control invalidation recall; zero stale-proof accepts | >=0.99 hidden impact recall, >=0.95 precision and >=20% lower cost than full rebuild at non-inferior safety |
| Source/specification integrity | 398 mechanically certified benchmark defects reported in the 2026 audit | 100% of counted PITB tasks have source/build/goal receipts; all discovered findings adjudicated | Project-disjoint and temporal cohorts plus independent semantic review on every flagged task |

Numerical targets must be revisited after a reproducible pilot and power analysis. The 341-task AutoformBench thresholds are arithmetic gates only; they do not establish a matched model win unless the source-visibility condition and resource budgets also match.

## Execution sequence

1. Complete exact-head qualification of the current SorryDB snapshot/source/goal audit and clustered comparison code.
2. Use the pinned real SorryDB integration task to qualify fresh checkout, exact source range, toolchain, full project build and live goal replay.
3. Run matched retrieval arms (BM25, LeanSearch v2, LeanGraphSearch, DAPS, TheoremGraph and PITB union) with per-query receipts and identical downstream prover budgets.
4. Build support-ablation cohorts for selected tasks; measure minimal route recovery and sufficiency, not only Recall@K.
5. Add a project-disjoint partition using repository remotes as the grouping unit; publish its deterministic split manifest and task denominators before using it for confirmation.
6. Add a rolling temporal cohort from later SorryDB snapshots, freezing the crawl cut and source commits.
7. Run controlled proof-maintenance mutations across opaque proof bodies, transparent definitions, instances, imports, toolchain and semantic-gate evidence.
8. Extend the same assurance envelope to FVSpec property translation and AutoformBench formalization without pooling their success rates with SorryDB.
9. Publish full cost/safety/coverage frontiers and independently replay all counted successes.

## Current implementation references

- SorryDB snapshot, source and goal replay, assurance-adjusted proof-run receipt: tools/pitb-sorrydb-audit/
- Signed semantic/evidence closure: tools/pitb-sorrydb-evidence-closure/
- SorryDB task integrity contract: tools/pitb-sorrydb-integrity/
- Typed graph evidence and snapshot attestations: tools/pitb-graph-evidence/
- DAPS candidate-only importer: tools/pitb-daps-graph-import/
- Per-query retrieval evidence: tools/pitb-retrieval-evidence/
- Exact support antichain: tools/pitb-support-antichain/
- Change-impact / proof maintenance: tools/pitb-proof-impact/

None of these components, by itself, demonstrates a measured SOTA gain. Their purpose is to make later gains reproducible, falsifiable and safe to claim.

## External baseline update (6 November 2026)

The component-level comparison frontier is now clearer. These published figures are useful targets, but each belongs to its own benchmark and must be reproduced under its own task and resource contract.

### Retrieval and proof-support systems

- **DAPS** reports mean Recall@32 of 89.25 on Mathlib4-Heldout, +11.41 over LeanHammer, and reports downstream gains on miniF2F and IMO-ProofBench. It contributes a structural neighborhood encoder, group-level contrastive learning, Mathlib-to-Blueprint adaptation, 53 mined Blueprint projects and 3,805 manually curated informal/formal nodes. This is a high bar for candidate retrieval—not a proof completion score. [DAPS paper](https://arxiv.org/abs/2610.41536)
- **LeanGraphSearch** reports on the shared 810-query MathlibQR subset nDCG@10 0.629 and Recall@10 0.794, compared with LeanSearch v2 Standard at 0.616 and 0.765. On MathlibMPR, complete-route coverage@50 is 44.2% vs 41.3% for LeanSearch v2 Reasoning. Its iterative proving integration reports gains up to 10% over LeanSearch v2 in some settings. PITB's retrieval target remains >=0.65 nDCG@10, >=0.82 Recall@10 and >=50% complete-route coverage@50, followed by a matched downstream proof-success comparison. [LeanGraphSearch paper](https://arxiv.org/abs/2610.22339) · [project tables](https://pluslabnlp.github.io/LeanGraphSearch/)
- **TheoremGraph** reports 11.7M informal mathematical statements and 18.3M candidate informal dependency edges, alongside 388,105 formal Lean declarations and 11.3M typed Lean dependency edges across 25 projects. Its informal edges are extractor-proposed candidates; the formal graph and the cross-formality links have distinct evidence levels. PITB should ingest this as a provenance-rich candidate source, never flatten its informal edge hypotheses into observed Lean dependencies. [TheoremGraph](https://www.theoremsearch.com/theorem-graph) · [dataset summary](https://huggingface.co/datasets/uw-math-ai/math-graph)
- **CoCo-Prover** is an especially relevant orchestration comparator: its two-level AND/OR proof graphs and metalevel routing optimize the success-cost frontier, reporting up to 30.9% lower cost than a strong baseline on five Lean program-verification benchmarks. PITB's opportunity is to combine this cost-aware action selection with evidence-qualified support routes and maintenance obligations, then compare under identical budgets. [CoCo-Prover paper](https://arxiv.org/abs/2610.09681)
- **FVSpec** broadens the target beyond mathematics: 11,039 scraped property-based tests, 9,415 Lean challenges from 2,772 PBTs, 333 source repositories, and 62% classified as hard. Published model baselines average 70% on a random 100-problem easy sample and 49% on a random 100-problem hard sample—not on the entire 9,415-task corpus. A PITB software-assurance track must score both translation fidelity and proof validity against the original PBT semantics. [FVSpec leaderboard](https://fvspec.galois.com/) · [paper](https://arxiv.org/abs/2606.01008)

### Current SorryDB reproduction witness

On the exact PR head, the sorrydb-snapshot-integrity job completed every step, including one end-to-end source replay:

- selected a task from the pinned SorryDB_2601 evaluation split;
- checked out its immutable repository commit ddcb6542f2b62883c8642d3db4ecc115afcce33e;
- verified the source-file, lean-toolchain and Lake-manifest Git blob identities;
- installed Lean 4.26.0 and successfully built the source project;
- verified that the exact committed span at PrimeNumberTheoremAnd/FioriKadiriSwidinsky.lean:44:137-142 is the stored sorry token;
- replayed the live Lean goal and matched it to the frozen SorryDB goal, retaining goal, manifest and replay-log digests.

This is one real source/build/goal-replay witness. It does not estimate the percentage of all 1,000 tasks that can be reproduced, and it is not a theorem-solving success. The remaining evaluation tasks still need source reproduction and semantic-gate evidence before their proof outcomes can enter an assurance-qualified denominator.

### General matched-comparison engine

The standalone Rust comparison engine is being qualified separately in [PR #10732](https://github.com/Luminous-Dynamics/symthaea/pull/10732). Its exact-head workflow now passes five Rust tests and the cohort/run/report schemas. The synthetic paired cohort demonstrates the intended separation: four kernel-verified challenger proofs, but only three assurance-qualified completions because one task lacks its required specification audit. It requires exact task-set identity, repository revision, Lean version, environment, arm/scaffold/implementation, budget policy and declared repository or repository-file bootstrap unit.

The SorryDB-specific audit already has its own paired cluster bootstrap, nested source-file sensitivity, cluster-level sign-flip tests and Holm correction. The general engine provides a common interface for matched results from SorryDB, formalization, software-property translation and other tracks; it does not pool their scores.

## What PITB should build beyond the current SOTA

The next genuine architectural leap is an **assurance-constrained proof operations controller**. Its planning state should be a typed AND/OR route graph, not a flat list of candidate premises. At each step it should choose among:

1. expand the typed retrieval frontier (BM25, LeanSearch v2, LeanGraphSearch, DAPS, TheoremGraph or a union);
2. test an alternative support bundle against an exact-target prover budget;
3. invoke a bounded specialist prover or tactic search;
4. request a counterfactual support ablation when route necessity or complementarity is uncertain;
5. revalidate/rebuild a stale proof after a source, dependency, instance, import or toolchain mutation;
6. spend a human-review budget where source/intent ambiguity makes automation unsafe.

The controller should maximize expected **assurance-qualified unique completions** per unit cost, subject to hard constraints: no task/split leakage, no weakened gate, no stale receipt, no proof accepted against a changed target, and no uncertain external edge promoted into a formal dependency. Each action needs a predicted value of information, a measured resource cost, a termination condition and an independently checked result. An action that improves retrieval recall but not exact-target verified completion should lose budget to one that closes real proof routes.

Evaluation should compare four layers on the same frozen task IDs and total budget: (A) strongest standalone prover; (B) standalone prover plus strongest retrieval; (C) typed multi-source retrieval plus support-antichain route planning; (D) assurance-constrained adaptive controller with proof-maintenance checks. Report both the raw and assurance-adjusted frontier, the system union, route sufficiency, failure reasons, project-cluster uncertainty and cost per verified completion. Reserve project-disjoint and chronological cohorts for confirmation, not tuning.

**Near-term qualification gates:**
- SorryDB: reproduce the strongest public baseline that can actually be run on the exact 1,000 tasks and same resource policy; then target +5 percentage points with a repository-cluster 95% interval above zero. Treat the current public 57.2% Aleph Pass@1 result and 63.2% union coverage as external references, not matched controls. [Leaderboard](https://sorrydb.org/)
- AutoformBench: reproduce the published GPT-6 Astra 281/341 source-present result before claiming a gain; 299/341 is the first integer threshold above +5 percentage points, 316/341 above +10 points. Preserve the release's CC BY-NC 4.0 evaluation-only restriction. [Release](https://github.com/facebookresearch/autoformbench)
- Retrieval: beat the strongest matched retrieval baseline on complete support-route coverage, then demonstrate a downstream exact-target completion gain under the same prover, tool, reflection and token budgets.
- Proof maintenance: 100% invalidation recall on registered critical mutations and zero stale-proof acceptance; then show at least 20% lower revalidation/rebuild cost than full rebuild at non-inferior safety.
- Assurance: no known vacuous proposition, goal mismatch, or unapproved axiom may be counted as an assurance-qualified success; independently authenticate every source/build/checker/semantic receipt that enters the public score.

These thresholds are research targets, not current PITB scores. A public claim of superiority requires a matched baseline, a frozen split, the exact-head receipt chain, and uncertainty analysis at the project cluster level.
