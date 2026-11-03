# PITB AutoformBench residual frontier analysis

**Date:** 2026-11-03  
**Status:** public task-level leaderboard audit and experiment design. No PITB run, model reproduction, or improvement is claimed.

## Executive finding

AutoformBench should be treated as a separate **formalization-infrastructure construction** endpoint, not as another proof-hole completion score. Its 341 tasks require an agent to formalize advanced mathematical developments by constructing missing definitions, intermediate lemmas, and supporting infrastructure before completing the target. The official release records 1,155 graded declarations across its 341 problem workspaces.

The most useful signal in the task-level results is not only the 82.4% solve rate of the leading published run. It is the asymmetric cost of the residual frontier:

- GPT-6 Astra: 281/341 solved (82.4%); 60 unresolved.
- The 60 unresolved tasks account for $6,817.01 of $14,763.24 recorded task cost (46.2% of spend).
- Mean recorded cost is $113.62 for an unresolved task versus $28.28 for a solved task (4.02×).
- The unresolved median is 41.15M input tokens and 124 turns, versus 3.79M input tokens and 37 turns for solved tasks.
- 39 unresolved tasks end with `model_gave_up`; 21 end with `budget_consumed`.

This pattern is consistent with expensive, low-yield exploration on the hard tail. The published CSVs do not include full model traces, so the mechanism is not established. PITB should test whether typed decomposition, dependency-aware context, and adaptive stop/escalate decisions reduce this cost without sacrificing verified completion.

## Frozen source identity

The analysis uses the public `facebookresearch/autoformbench` release at commit:

`af1aa38fd44e019e566bce0b5a168609a886d8e6`

The release pins Lean `v4.32.2` and Mathlib commit `905b95818eb32af7874a58b427f50c1711a5e96c`.

Git object IDs of the exact inputs fetched for this analysis:

| File | Git blob SHA-1 |
|---|---|
| `MANIFEST.csv` | `45d5cac87088c19427b742072c6dc3deb2d6e433` |
| `leaderboard_data/gpt-6-astra.csv` | `d2994a38a2213c3ad1c240080e8dc531b77e64c8` |
| `leaderboard_data/opus-5.csv` | `f928544f58138bf1c0616c0fbfb1c7296f6bd923` |
| `leaderboard_data/gpt-5.6-sol.csv` | `b6f314bee4e20cfb1fdef8c6ca19f03dbc42249f` |
| `leaderboard_data/deepseek-v4.1-flash.csv` | `c9f226f7fdc0a0b536779cd1c6d7221ee39d3371` |

These are Git blob IDs, not SHA-256 checksums. The release commit and blob IDs together identify the source snapshot. The calculations below were recomputed from all 341 rows in each task-level CSV and cross-checked against the published aggregate solve counts. No task prompts, source excerpts, model traces, or accepted submissions were copied into PITB.

## Published leaderboard and recorded cost

| Published run | Solved | Solve rate | Recorded total cost | Cost per solved task* |
|---|---:|---:|---:|---:|
| GPT-6 Astra | 281/341 | 82.4% | $14,763.24 | $52.54 |
| Opus 5 | 211/341 | 61.9% | $15,097.80 | $71.55 |
| GPT-5.6 Sol | 201/341 | 58.9% | $14,985.84 | $74.56 |
| DeepSeek V4.1 Flash | 180/341 | 52.8% | $4,029.84 | $22.39 |

*Total recorded task-level cost divided by successful tasks; unsuccessful attempts are included in the numerator.

The CSVs also permit descriptive same-task comparisons. GPT-6 Astra succeeds where Opus 5 does not on 71 tasks, while Opus 5 succeeds where GPT-6 Astra does not on one task. Against DeepSeek V4.1 Flash the corresponding counts are 101 and zero. These are task-aligned outcome differences, **not controlled causal comparisons**: the model, scaffold, token economics, source access, budgets, and stopping policies differ. Do not use these counts to attribute the gap to one model or PITB intervention.

The cost frontier is more nuanced than the solve leaderboard. GPT-6 Astra has the best reported solve rate, but DeepSeek's recorded cost per successful task is lower. That is a legitimate Pareto trade-off to investigate, not evidence that the cheaper run is the better prover under equal conditions.

## Residual-task anatomy

The GPT-6 Astra run leaves 60 of 341 tasks unsolved. Failure reasons:

| End reason | Count |
|---|---:|
| `model_gave_up` | 39 |
| `budget_consumed` | 21 |

The six largest residual subject groups contain 49/60 failures (81.7%):

| Subject group | Tasks | Solved | Unresolved | Solve rate |
|---|---:|---:|---:|---:|
| Elliptic curves | 27 | 11 | 16 | 40.7% |
| Lie groups and Lie algebras | 28 | 18 | 10 | 64.3% |
| Algebraic topology I | 16 | 10 | 6 | 62.5% |
| Introduction to arithmetic geometry | 26 | 20 | 6 | 76.9% |
| Number theory I | 24 | 18 | 6 | 75.0% |
| Noncommutative algebra | 22 | 17 | 5 | 77.3% |

Additional unresolved tasks occur in double Hecke algebras (3/15), graph theory and additive combinatorics (3/9), algebraic geometry I (2/8), and four other groups with one failure each. Some groups show 100% solve rates, but several have very small denominators; those are not evidence of general mastery.

The 60 unresolved tasks cost $6,817.01 in total. Their mean recorded input-token count is 41.67M versus 10.38M for solved tasks; mean turns are 120.1 versus 49.9; mean added code is 1,009 versus 471 LOC. The median cost is $109.62 on unresolved tasks versus $8.57 on solved tasks. The largest individual unresolved attempts approach the published per-task budget ceiling.

The manifest contains 1,155 graded declarations across 341 workspaces (3.39 declarations per workspace on average). Multi-declaration tasks can require substantial supporting formalization, so a final theorem-only retrieval metric will miss much of the engineering burden.

## Graded-declaration workload and the hard tail

The official manifest exposes a useful **workload proxy**: the number of graded declarations attached to each task. It is not a complete count of all required helper lemmas or dependencies, but it offers an independently available measure of how much formalization surface the task requires.

| Graded declarations per task | Tasks | Solved | Solve rate | Mean recorded cost |
|---|---:|---:|---:|---:|
| 1 | 230 | 208 | 90.4% | $28.87 |
| 2–4 | 53 | 41 | 77.4% | $63.29 |
| 5+ | 58 | 32 | 55.2% | $82.23 |

Tasks with five or more graded declarations are 17.0% of the benchmark but account for 26/60 (43.3%) of the leading run's unresolved tasks. All tasks with two or more declarations comprise 111/341 (32.6%) of the benchmark but account for 38/60 (63.3%) of unresolved tasks and $4,388.04/$6,817.01 (64.4%) of unresolved-task spend.

The same stratification across the four published model runs shows that this is a benchmark-wide hard-tail pattern, not a peculiarity of one system:

| Published model run | 1 declaration | 2–4 declarations | 5+ declarations | Recorded spend per 5+ solved task* |
|---|---:|---:|---:|---:|
| GPT-6 Astra | 90.4% | 77.4% | 55.2% | $149.04 |
| Opus 5 | 74.8% | 47.2% | 24.1% | $249.28 |
| GPT-5.6 Sol | 70.9% | 45.3% | 24.1% | $242.51 |
| DeepSeek V4.1 Flash | 64.3% | 39.6% | 19.0% | $85.55 |

Every run has a lower solve rate as the graded-declaration count grows. Within each run, the recorded spend per verified completion in the 5+ stratum is about 4.3–5.4 times that in the one-declaration stratum. These are within-run cost ratios; absolute costs across models are not directly comparable because their prices, scaffolds, and resource policies differ.

*Total recorded spend for the stratum, including unsuccessful attempts, divided by its number of solved tasks. This is not the marginal cost of a successful attempt.

To reduce subject confounding, I repeated the five-plus versus under-five comparison **within source groups that contain both task-size strata**. Thirteen of 24 source groups have both. Within those groups:

- Five-plus: 26/58 unresolved (44.8%).
- Under-five: 29/191 unresolved (15.2%).
- Pooled within-source risk difference: +29.6 percentage points.
- Pooled within-source risk ratio: 2.95.
- A deterministic 50,000-replicate bootstrap resampling source groups (seed `8badf00d`) gives a 95% interval of +13.6 to +48.3 percentage points for the risk difference, and 1.75 to 5.74 for the risk ratio.

This is exploratory, not causal. Only 13 source groups contribute to the within-group contrast, several cells are very small, the task-size proxy is imperfect, and the public model run is a single published configuration. It does, however, support a focused next experiment: test whether planning and verifying the declaration dependency DAG reduces the disproportionate failure and resource cost on multi-declaration tasks. Compare it with an equally budgeted direct-formalization baseline; do not infer that decomposition alone is the cause.

## Interpretation: what the benchmark is measuring

AutoformBench is not simply “find a tactic for a known proposition.” The agent receives an informal proof and must produce the Lean-side mathematical infrastructure around it. The relevant capability stack is:

1. **Conceptual decomposition:** identify definitions, intermediate claims, and the dependency structure needed to express the informal argument.
2. **Library alignment:** find the right Mathlib abstractions and translate source-level concepts into compatible types and interfaces.
3. **Dependency construction:** build a coherent DAG of supporting declarations, with no circular or semantically misplaced edges.
4. **Proof development:** prove the helper declarations and compose them into the exact target.
5. **Build and trust integrity:** satisfy the pinned Lean/Mathlib environment, permitted-axiom policy, comparator, and independent-kernel replay.
6. **Budget control:** notice when the current route is stalled, choose a different abstraction or dependency route, and stop unproductive search.

The residual distribution suggests prioritizing the first three capabilities before simply adding more parallel tactic attempts. That is a hypothesis: the public CSVs do not expose the traces required to establish the causal source of failure.

## Proposed PITB intervention: formalization as typed infrastructure synthesis

Keep this as a separate AutoformBench evaluation track. Do not merge its solve percentage with SorryDB, Aegis, LeanLean, or research-family transfer.

| Arm | System | Isolated question |
|---|---|---|
| F0 | Matched direct formalization agent | Baseline for the pinned task set and budget |
| F1 | F0 + semantic seed retrieval | Does stronger concept matching help? |
| F2 | F1 + multi-hop dependency-graph retrieval | Does route recovery beat nearest-declaration retrieval? |
| F3 | F2 + typed proof insights with explicit preconditions and failure boundaries | Does strategy transfer help build missing definitions/lemmas? |
| F4 | F3 + explicit declaration-DAG planning and incremental compile gates | Does infrastructure-first decomposition reduce rework? |
| F5 | F4 + adaptive budget controller and bounded private-workspace branches | Does search allocation improve success-cost efficiency? |

### Retrieval baseline correction: DAPS and TheoremGraph

The earlier retrieval design is incomplete unless it explicitly includes **DAPS** and **TheoremGraph**. These are current dependency-aware baselines, and their results should change how PITB tests the retrieval layer.

**DAPS (arXiv submitted 30 October 2026; project page labels NeurIPS 2026)** combines:
- a typed, heterogeneous Mathlib dependency graph;
- a structural neighborhood encoder;
- dynamic subgraph sampling and two-hop bidirectional context;
- a group-level contrastive objective that learns jointly useful premise sets, rather than only independent query-to-lemma similarity;
- optional continual adaptation on Blueprint data with rare research-level and informal/formal nodes.

Its reported Recall@32 is 89.25 on Mathlib4-Heldout (+11.41 points over LeanHammer), 23.53 on DAPS-Collection (+20.34 points over LeanHammer), and 50.08 on DAPS-Blueprint after Blueprint adaptation (+9.66 over LeanHammer). Supplying its premises raised Goedel-Prover-V2-8B miniF2F-test Pass@4 from 49.90 to 54.03. These results are on DAPS's own tasks, splits, models and budgets; they are not an AutoformBench score or a direct PITB result. DAPS's own ablations are especially informative: removing structural fusion or reducing the dependency neighborhood from two hops to one causes large Recall@32 reductions on Mathlib4-Heldout. The result supports graph structure and jointly-selected premise bundles, not just a better embedding model.

**TheoremGraph / LeanGraph** supplies another important source:
- its reported formal graph has 388,105 Lean declaration nodes and 11.3 million typed edges over 25 projects;
- its informal graph contains 11.7 million theorem-like environments and 18.3 million *candidate* directed dependencies;
- 47,952 informal/formal matches are LLM-affirmed above a cosine threshold, with an affirmation rate of 48% above 0.8 and 87% above 0.9;
- name/signature plus graph expansion reports Recall@10 of 0.775, close to LeanSearch v2's 0.780 reranked result.

TheoremGraph's elaborator-derived typed formal edges are a strong graph-resource baseline. Its informal dependencies and cross-formality matches are *candidate evidence*, not kernel-checked mathematical dependencies: retain extractor provenance, confidence, review status and source locator. Do not turn an LLM-affirmed match into a trusted proof edge. Also distinguish TheoremGraph's graph snapshot from DAPS's separately built Mathlib graph; their node and edge counts are not directly comparable.

#### Matched retrieval experiment required for PITB

On every selected AutoformBench development and confirmatory task, use the same target/query, source visibility, formalizer (where needed), LLM, prompt, toolchain, candidate-generation policy and total resource cap. Compare:
1. lexical/BM25 and name/signature retrieval;
2. LeanSearch / LeanExplore API baselines where reproducible;
3. LeanGraphSearch multi-hop retrieval;
4. DAPS with its published checkpoint and two-hop, bidirectional, group-aware premise selection;
5. TheoremGraph name/signature graph expansion, keeping informal candidate links in a lower trust tier;
6. a deduplicated union/reranker over the strongest available sources;
7. PITB typed support-bundle selection, which must beat the strongest validation-selected arm in *downstream verified completion*, not merely retrieval recall.

Use both K=16 and K=32 premise budgets. Report Recall@K, precision@K, macro-F1, complete reference-route coverage, type/instance compatibility, novel-useful-premise rate, duplicate/frequency skew, retrieval latency, prompt tokens, and downstream kernel-verified solve rate/cost. DAPS uses a formalized Lean query for informal prompts; record the formalizer and measure query-statement failure separately. Do not compare a DAPS Recall@32 result with PITB Pass@1.

**Leakage and generalization controls:** DAPS-Blueprint's 236-query test is a node-level split among 53 projects; the paper explicitly does not claim unseen-project generalization. Before using it for training or tuning, audit project/source overlap with PITB's training families and AutoformBench source groups. PITB's confirmatory graph-transfer claim must use source/project-disjoint holdouts and, separately, a later temporal holdout. The DAPS collection and Blueprint training resources are comparator/training data candidates only after licensing, snapshot, and contamination review.

**Implementation boundary:** DAPS's public implementation is Apache-2.0 and its reproduction stack uses Python, PyTorch, Transformers, Sentence-Transformers and PyTorch Geometric. PITB should first use pinned DAPS outputs/checkpoints through a thin importer/adapter and compare them with Rust-native graph extraction, indexing and orchestration. Do not import its Python training stack into production merely to claim parity; any reproduced paper run must preserve its stated environment separately.

#### The graph model PITB should build

Use a typed, multi-relational **proof-support hypergraph**, not an untyped similarity graph:

| Relation class | Meaning | Trust / use |
|---|---|---|
| `type_reference` | Constants referenced by a declaration's elaborated type | High-confidence formal edge, exact Lean/toolchain identity |
| `proof_term_reference` | Constants used by the checked proof term | High-confidence proof-support edge; does not by itself equal necessary support |
| `instance_synthesis` | Local/global instance evidence used to elaborate the target or proof | High-confidence, environment- and scope-bound |
| `blueprint_dependency` | Human-authored dependency edge from a formalization blueprint | Curated but independently reviewable |
| `informal_candidate_dependency` | Extractor-proposed relation between informal results | Candidate only; retain extractor and source span |
| `cross_formality_match` | Proposed informal-to-Lean statement match | Candidate only until independently reviewed |
| `module_environment_dependency` | Imports, options, notation, macros, elaborators and environment state needed to rebuild | Build evidence; do not treat as theorem premise |

A proof strategy usually needs a *set* of premises with joint compatibility. Model that as an AND/OR hypergraph: one obligation may have multiple alternative sufficient bundles (OR), while each bundle may require several premises jointly (AND). Record the provenance of every node/edge, exact declaration and environment identities, evidence status, and whether it was observed, proposed, checked, or independently reviewed.

Do not infer that every declaration in a proof term is logically necessary. Preserve the distinction between typing closure, value provenance, actual elaborated support, and experimentally sufficient support. Support-antichain recovery and controlled ablation can estimate sufficient bundles under a declared context and budget; they cannot certify universal minimality without a stronger proof.

**F4 should not be scored on graph construction alone.** It must reduce end-to-end verified-task cost while preserving the exact target, and it must report graph-edge precision/recall on independently audited dependencies, support-bundle recovery, cycle/closure defects, incremental re-elaboration cost, helper reuse, and collateral regressions.

Required controls:

- Same model/version, prompt family, source visibility, tool set, environment, per-task budget, wall-time cap, and total resource accounting across matched arms.
- Pin the exact task manifest, every workspace, Lean and Mathlib revisions, Harbor adapter, grader/comparator, allowed axioms, and independent replay process.
- Use the official clean-room grader and exact challenge targets. A candidate must not change or weaken any target statement.
- Maintain separate ledgers for target declarations, helper declarations, compilation, comparator acceptance, axiom closure, and independent replay.
- Report task-level outcomes and all incomplete, failed, abstained, timed-out, and budget-exhausted tasks; do not drop difficult tasks from the denominator.
- Use paired inference clustered by source group (24 groups), not an IID task bootstrap. Freeze any validation/tuning split before opening confirmatory source groups.
- Record total model tokens, cached input, output tokens, calls, turns, code LOC, Lean CPU, elaboration/build time, wall-clock, memory, and recorded monetary cost.
- Keep source excerpts and any third-party textbook materials outside the PITB data store unless separately licensed and approved.

### Qualification targets

These are proposed gates, not results:

1. Reproduction gate: reproduce the public task set and checker with 100% task-manifest agreement, 100% clean-room environment reconstruction, and zero target/axiom policy drift.
2. Pilot signal gate: reach at least 291/341 (85.3%) on the same 341 tasks and matched budgets, with a paired, source-group-clustered 95% interval above matched F0. This is only an early signal (+2.9 percentage points over 281/341), not a five-point SOTA-beating claim.
3. Competitive gate: reach at least 299/341 (87.7%), which clears a +5 percentage-point absolute improvement over 281/341, with the paired 95% interval lower bound above zero.
4. Efficiency gate: reduce recorded cost per verified task by at least 20% versus matched F0 while remaining non-inferior in solve rate (non-inferiority margin no worse than 1 percentage point).
5. Stretch gate: reach at least 316/341 (92.7%), clearing +10 percentage points over the published baseline, without relaxing integrity, source, budget, or checker conditions.
6. Reliability gate: 100% of counted successes survive exact-target comparator checks and independent clean-room replay; zero accepted target mutations or unapproved axiom additions.
7. Generalization gate: replicate the direction of gain on held-out source groups or a later independent advanced-mathematics formalization cohort.

With a fixed 341-task denominator, 298/341 is still fractionally below +5.0 percentage points; the first integer threshold clearing +5 points is 299/341. Likewise, 315/341 is fractionally below +10 points; the first integer threshold is 316/341. These exact thresholds should be pre-registered. Do not describe 291/341 as a five-point improvement: it is approximately 85.3%, or +2.9 percentage points.

## Reproducible audit utility

The research-only auditor is at `scripts/research/pitb_autoformbench_audit_v0.py`. It reads only the pinned release commit's manifest and public task-level leaderboard CSVs; it does not open Lean workspaces, textbook material, accepted solutions, submissions, or traces.

```sh
python3 scripts/research/pitb_autoformbench_audit_v0.py --self-test
python3 scripts/research/pitb_autoformbench_audit_v0.py --repo /path/to/autoformbench
```

The repository clone must contain the pinned release commit. The auditor verifies the five Git blob identities, manifest cardinalities, declaration totals, exact task-set equality, source-group joins, numeric fields, official model solve counts, and then recomputes cost/residual/paired tables. The synthetic self-test was run locally on the analysis implementation; the current GitHub branch version has not yet been executed in a runner or against a local checkout of the pinned external repository. The report's statistics above were independently computed from the fetched public CSV rows.

## Immediate work items

1. Execute the checked-in offline audit against a local clone of the pinned AutoformBench release and compare its JSON output byte-for-byte (after canonical JSON normalization) with the registered summary tables.
2. Add a source-group dashboard for solve rate, cost per verified task, unresolved-cost share, and budget-exhaustion reason. Preserve the raw denominators.
3. Reproduce one modest, pre-registered development subset under Lean v4.32.2 / Mathlib `905b958...` using the official Harbor adapter. This is a pipeline smoke, not a benchmark result.
4. Implement dependency-aware declaration planning and type-directed search before expanding agent fan-out. Track whether a branch discovers a missing definition, proves a helper, finds an existing library route, detects a type mismatch, or abstains.
5. Add adaptive stopping with an explicit expected marginal verified-success-per-cost estimate. Evaluate against fixed-budget and fixed-step baselines; do not let the controller silently reduce the denominator or truncate hard tasks.

## License and data boundary

The AutoformBench repository states that the release is CC BY-NC 4.0 and intended for evaluation/research, **not training data**. The public repository omits textbook excerpts, source PDFs, informal source files, traces, accepted submissions, gold proofs, and internal audit records. Its third-party source materials retain their own licenses. PITB may treat the public task workspaces and task-level leaderboard data as an external evaluation reference only after a project-specific license review. Do not train, fine-tune, build a persistent training corpus, or redistribute source excerpts from this release. Commercial use or integration requires separate legal review; this report is not legal advice.

## Sources

- AutoformBench release and leaderboard: https://github.com/facebookresearch/autoformbench
- AutoformBench paper description (13 October 2026): https://ai.meta.com/research/publications/autoformbench-a-lean-formalization-benchmark-for-advanced-mathematics/
- DAPS: https://arxiv.org/abs/2610.41536; project/code: https://daps-eth.github.io/ and https://github.com/DAPS-ETH/DAPS (Apache-2.0).
- TheoremGraph / LeanGraph: https://arxiv.org/abs/2606.25363; search service: https://theoremsearch.com/.
- LeanGraphSearch: https://arxiv.org/abs/2610.22339 and https://pluslabnlp.github.io/LeanGraphSearch/
- LeanGraphSearch: https://arxiv.org/abs/2610.22339 and https://pluslabnlp.github.io/LeanGraphSearch/
- T² semantic theorem testing: https://aclanthology.org/2026.acl-industry.150/
- BlueprintRepair / BlueprintTrace: https://aclanthology.org/2026.mathnlp-main.3/
- PITB external benchmark portfolio: `docs/research/pitb-external-benchmark-portfolio-2026-10-31.md`

---

**Evidence status:** The published CSVs and manifest were fetched at the release commit and their task-level counts reconcile to the official leaderboard totals. The aggregate calculations above were executed directly over the CSV rows. PITB has not yet run AutoformBench, reproduced any listed model, received source texts, or tested the proposed interventions.