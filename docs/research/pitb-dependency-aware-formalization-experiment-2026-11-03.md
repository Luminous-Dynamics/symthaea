# PITB dependency-aware formalization: preregistered experiment design

**Date:** 2026-11-03  
**Status:** experiment protocol and external-benchmark analysis only. No PITB solve-rate gain, retrieval run, or SOTA improvement is claimed.

## Executive decision

The next frontier should not be pursued as a contest to produce the most relevant top-K list. The research opportunity is to build a **target-preserving proof-support planner** that combines several imperfect knowledge sources, constructs candidate AND/OR premise bundles, checks type and environment compatibility, plans missing declarations as a DAG, and accepts progress only after exact-target independent verification.

This is a different endpoint from retrieval Recall@K:
- Retrieval asks whether a reference premise was returned.
- Bundle construction asks whether the returned premises form a jointly useful support route.
- Formalization planning asks whether missing definitions and helper lemmas are identified and ordered correctly.
- Verification asks whether the unchanged target is proved in the exact environment and survives independent replay.
- Maintenance asks which proof artifacts become invalid when an upstream declaration or environment changes.

PITB should optimize the last three, while retaining the first two as measured components.

## What the public AutoformBench task-level data tells us

The analysis is pinned to the public AutoformBench Git commit:

`af1aa38fd44e019e566bce0b5a168609a886d8e6`

The manifest has 341 tasks, 24 source groups and 1,155 graded declarations. The declared-declaration count is a workload proxy, not a full measure of helper-lemma, proof-route or dependency complexity.

### Published solve-rate stratification

| Published model run | 1 declaration (n=230) | 2–4 declarations (n=53) | 5+ declarations (n=58) |
|---|---:|---:|---:|
| GPT-6 Astra | 208/230 (90.4%) | 41/53 (77.4%) | 32/58 (55.2%) |
| Opus 5 | 172/230 (74.8%) | 25/53 (47.2%) | 14/58 (24.1%) |
| GPT-5.6 Sol | 163/230 (70.9%) | 24/53 (45.3%) | 14/58 (24.1%) |
| DeepSeek V4.1 Flash | 148/230 (64.3%) | 21/53 (39.6%) | 11/58 (19.0%) |

All four published runs degrade with the graded-declaration count. For GPT-6 Astra, the 5+ stratum is 17.0% of tasks but accounts for 26/60 (43.3%) of unresolved tasks. All tasks with 2+ declarations are 111/341 (32.6%) but account for 38/60 (63.3%) of unresolved tasks and about 64.4% of unresolved-task spend.

Recorded spend per solved task in the 5+ stratum is approximately 4.3–5.4 times the one-declaration stratum across the four published runs. This quantity is total recorded stratum spend, including failed attempts, divided by its number of solved tasks; it is not marginal successful-attempt cost. Absolute model costs are not directly comparable across providers or scaffolds.

To reduce source-group confounding, a within-source comparison was computed over the 13 of 24 source groups containing both 5+ and under-five tasks. On GPT-6 Astra:
- 5+ declarations: 26/58 unresolved (44.8%);
- under-five declarations: 29/191 unresolved (15.2%);
- pooled difference: +29.6 percentage points;
- risk ratio: 2.95;
- deterministic 50,000-replicate source-group bootstrap (seed `8badf00d`): 95% interval +13.6 to +48.3 points for the difference and 1.75 to 5.74 for the ratio.

This is exploratory, not causal. It has 13 contributing clusters, sparse cells, one published system configuration and an imperfect complexity proxy. It motivates a focused experiment; it does not prove that DAG planning is the cause of the hard tail.

## Industry baseline and what each system contributes

These results come from different datasets, environments and budgets. Do not compare their headline metrics as if they were one leaderboard.

| System | Published evidence | Required PITB treatment |
|---|---|---|
| DAPS | Mean Recall@32 89.25 on Mathlib4-Heldout, +11.41 points over LeanHammer; 23.53 on DAPS-Collection; 50.08 on DAPS-Blueprint after Blueprint adaptation. DAPS premises raise Goedel-Prover-V2-8B miniF2F Pass@4 from 49.90 to 54.03. | Strong group-aware, dependency-aware premise selector. Reproduce at its published K and frozen checkpoint; measure downstream proof completion separately. |
| LeanGraphSearch | Recall@100 94.7% on an 810-query MathlibQR fair subset versus 82.7% for LeanSearch v2; MathlibMPR Covered@20 rises from 39.1% to 42.8%; own downstream proof loop reports gains over vanilla and LeanSearch v2. | Multi-hop dependency traversal and route coverage baseline. Preserve query subset, retrieval policy and downstream prover setup. |
| TheoremGraph / LeanGraph | 388,105 formal declaration nodes and 11.3M typed formal edges across 25 Lean projects; 11.7M informal theorem-like environments and 18.3M candidate informal dependency edges; name/signature plus graph expansion Recall@10 0.775 versus LeanSearch v2 0.780 reranked. | Large graph-resource baseline. Keep informal candidate edges separate from elaborator-derived formal edges and retain extractor provenance. |
| CoCo-Prover | AND/OR graph scheduling, cross-declaration lemma dependency DAG and specialist action routing; reports up to 30.9% cost reduction in its own Lean program-verification benchmarks. | Proof orchestration and resource-allocation comparator, not a direct AutoformBench result. |
| VeriSoftBench / Vero | Repository context and multi-module completion remain materially harder than isolated theorem tasks. | Test project-scale completeness and clean-checkout replay; do not use per-obligation pass rate as a substitute for full-instance completion. |

### Critical DAPS split limitation

DAPS-Blueprint contains 236 test queries produced by an 8:1:1 **node-level** split across 53 highly unequal projects. The paper explicitly states this does not establish generalization to wholly unseen projects. PITB must not repeat that claim. Add separate project-disjoint and temporal holdouts for any cross-project or research-mathematics generalization claim.

DAPS also identifies the premise-conditioned generation stage as a remaining bottleneck: retrieval gains do not automatically produce equivalent downstream proof gains. This is precisely why PITB's primary endpoint must be verified completion, not recall.

### Sources

- DAPS: https://arxiv.org/abs/2610.41536 and https://github.com/DAPS-ETH/DAPS
- LeanGraphSearch: https://arxiv.org/abs/2610.22339
- TheoremGraph / LeanGraph: https://arxiv.org/abs/2606.25363 and https://www.theoremsearch.com/theorem-graph
- CoCo-Prover: https://arxiv.org/abs/2610.09681
- AutoformBench public release: https://github.com/facebookresearch/autoformbench

## Proposed architecture: retrieve, compose, formalize, verify, maintain

### 1. Query and target identity

Every task begins with an immutable target identity:
- exact Lean statement bytes and elaborated type digest;
- repository commit, Lean toolchain, Mathlib revision, Lake manifest and root module digests;
- ordered local context and local instance registry identity;
- source coordinate and expected informal statement identity, where applicable;
- fixed resource and action-policy identity.

The target is never changed to improve a score. A changed target is a different task and is excluded from the original task's success count.

### 2. Multi-source retrieval

Query all baselines under the same target and candidate budget:
1. BM25/name/signature retrieval;
2. LeanSearch v2 / LeanExplore where the released API and snapshot can be reproduced;
3. LeanGraphSearch multi-hop graph retrieval;
4. DAPS group-aware premise selection (K=16 and K=32);
5. TheoremGraph / LeanGraph typed edge expansion;
6. a union/reranker over the strongest validation-selected retrieval arms.

Keep candidate origins, checkpoint hashes, graph snapshot, query formalizer, retrieval scores and K budgets. A candidate link from an informal extractor or LLM alignment is never promoted to formal evidence because it scores highly.

### 3. Typed compatibility filter

Before a candidate can enter a proof context, verify:
- exact declaration identity and current repository/environment;
- elaborated type compatibility with the target subgoal;
- required universe/implicit argument and coercion conditions;
- instance-synthesis closure and local/global scope;
- module, import, notation, macro and elaborator prerequisites;
- source and extractor provenance.

Classify rejections rather than dropping them: type mismatch, wrong instance, stale environment, wrong sibling theorem, wrong project/topic, unsupported declaration kind, or unverified cross-formality link. DAPS's reported type-generalization and topical/sibling drift cases should become explicit negative controls.

### 4. AND/OR support-bundle planner

A theorem often requires multiple premises jointly, and the same goal may have several alternative sufficient premise sets. The unit of planning is therefore an AND/OR support structure:
- AND: a bundle of premises and environment requirements that must be available together;
- OR: alternative bundles or proof routes for the same goal;
- dependency DAG: helper definitions and lemmas that must be created before dependent declarations can be completed.

The planner must not equate proof-term occurrence with logical necessity. A premise observed in one proof is an observed support edge. A bundle is only *experimentally sufficient* when the exact target is proved in the restricted context with an independently checked proof. A bundle is only *exhaustively minimal* when the registered proper-subset lattice has been tested and every proper subset failed under the same frozen solver, environment and budget. For more than 16 mutable premises, label the analysis approximate/designed-sampling, never exhaustive.

### 5. Formalization DAG and incremental compilation

Use explicit declaration states: proposed, open, provisional, stable, invalidated and discarded. A helper can be provisional while being proved, but its dependents cannot advance until it is stable and kernel verified. Build gates should be incremental and state-bound. When a dependency changes, invalidate exactly its transitive dependents; preserve unrelated verified work. Capture build graph, environment identity and rollback events.

### 6. Independent verification

A counted success must:
- compile the unmodified target statement;
- contain no sorryAx or unapproved axiom additions;
- pass the registered exact-target comparator;
- pass independent clean-checkout/kernel replay;
- pass the semantic target-to-source gate where target correspondence is in scope;
- retain authenticated proof, checker, environment and source receipts.

The typed graph evidence verifier is only the structural/byte-integrity layer. Its current `scheduler_admission: false` is intentional. Do not feed unverified graph edges to the scheduler before signed extractor receipts and independent semantic re-derivation exist.

## Preregistered arms

| Arm | Retrieval / planning stack | Isolated question |
|---|---|---|
| F0 | Direct formalization, no retrieval, no persistent declaration DAG | Matched baseline for the exact benchmark snapshot |
| F1 | F0 + BM25/name/signature and LeanSearch v2 | Does conventional semantic retrieval help? |
| F2 | F1 + LeanGraphSearch multi-hop traversal | Does dependency expansion improve route recall and solve/cost? |
| F3a | F1 + DAPS checkpoint and published group premise selector | Does group-aware, graph-aware premise selection help on this cohort? |
| F3b | F1 + TheoremGraph/LeanGraph typed expansion | Does the larger formal graph resource improve retrieval and downstream proof completion? |
| F4 | Best validation-selected retrieval arm + PITB typed AND/OR support bundles, instance-aware context and counterfactual bundle checks | Does bundle composition beat ranked premise lists? |
| F5 | F4 + declaration-DAG planning, incremental elaboration and selective rollback | Does explicit infrastructure planning close the multi-declaration hard tail? |
| F6 | F5 + calibrated multi-resource controller and bounded parallel branches | Does adaptive allocation improve the success-cost/latency frontier? |

Do not compare a DAPS Recall@32 number with AutoformBench solve rate. The arms must use the same AutoformBench targets, base prover, prompts, allowed tools, source visibility, per-task model/tool/token/Lean CPU/wall/currency budgets, grader, kernel replay and action accounting. Reproduce a released baseline with its own published settings first; then run a matched PITB protocol as a separate experiment.

Select the strongest F1–F3 arm on development-only tasks and freeze it before F4–F6 confirmatory evaluation. No test-set tuning, silent retries, dropped unresolved tasks, or hidden budget increases.

## Statistical plan and split policy

- The 341 AutoformBench tasks and their four published runs are a public development/reference cohort, not a blind confirmatory set. Do not claim novel generalization from re-tuning on these tasks.
- Use source group as the clustering unit. The 24 source groups are not 341 independent observations. Preserve all denominators and provide source-group paired differences.
- Report paired source-group bootstrap intervals, source-group blocked permutation/randomization inference and a hierarchical sensitivity analysis. Because there are only 24 clusters and some are small, include small-cluster caveats and raw group-level outcomes; the 13-group high-vs-low analysis is exploratory.
- Pre-register the target-count strata: 1, 2–4 and 5+. Report all arms separately by stratum. The primary moderator is 5+ declaration workload; this is not a complete dependency-complexity measure.
- For the primary confirmatory test, create a new advanced-mathematics cohort from projects not used for training, retrieval fine-tuning or prompt tuning. Hold out complete source projects and a later temporal snapshot. A random task split is not sufficient.
- Correct multiple confirmatory arm comparisons (Holm or a preregistered hierarchical test). Report effect sizes and intervals, not only p-values.
- Report all tasks as solved, unresolved, failed, timed out, budget exhausted, invalid input or abstained. The fixed denominator never shrinks.

## Metrics

### Primary endpoint

**Verified exact-target completion rate** under a shared fixed budget, with every success surviving independent clean-checkout/kernel replay and the relevant semantic target gate.

### Secondary endpoints

| Area | Required measures |
|---|---|
| Retrieval | Recall@16/32, precision@K, macro-F1, nDCG, complete proof-route coverage, useful-novel-premise rate, long-tail recall, candidate frequency skew |
| Compatibility | Type-check acceptance, instance-synthesis correctness, universe/coercion failures, wrong-sibling false positives, stale-environment rejection |
| Bundle construction | Exact bundle recovery, joint sufficiency, alternative route coverage, duplicate-premise rate, support-antichain recovery, minimality coverage and abstention |
| Formalization | Target declaration completion, helper declaration completion, dependency-DAG precision/recall, missing-node discovery, cycle and stale-edge rates, incremental build time |
| Verification | Exact-target comparator pass, axiom policy, sorryAx detection, independent replay, semantic false-accept rate |
| Economics | Total model/input/output/cache tokens, tool calls, Lean CPU, wall clock, peak memory, actual monetary cost, cost per verified completion, cost-quality AUC |
| Maintenance | Stale receipt rejection, impact-set precision/recall, unaffected-proof preservation, repair cost, full-repository replay |

Retrieval metrics are diagnostic. They cannot substitute for verified completion and economics.

## Proposed qualification targets

These are gates to aim for, not achieved results:

1. **Reproduction:** 100% task-manifest agreement; exact source and toolchain pin; zero target or axiom-policy drift.
2. **Overall competitive:** at least 299/341 on the current public 341-task cohort *only as an external reference threshold* (first integer above +5 percentage points versus 281/341). A competitive claim additionally requires a matched F0 and a paired source-group interval whose lower bound is above zero.
3. **Stretch:** at least 316/341 on the same denominator (first integer above +10 percentage points versus 281/341), without relaxing budgets or trust gates.
4. **Hard-tail breakthrough:** at least +10 percentage points in the 5+ declaration stratum versus matched F0, with no more than a 1-point absolute regression in the one-declaration stratum. Treat the subgroup result as a preregistered secondary/interaction endpoint; power it with enough independent projects before claiming generalization.
5. **Efficiency:** at least 20% lower cost per verified completion at non-inferior solve rate (margin no worse than 1 percentage point); report a Pareto curve across shared budget caps.
6. **Bundle quality:** at least 0.90 exact bundle recovery on a powered human-adjudicated subset; at least 0.95 measured sufficiency at at least 0.80 non-abstaining coverage; no bundle is called minimal without a complete, frozen counterfactual lattice.
7. **Integrity:** zero false accepts in the registered semantic/source/target mutation suite; 100% independent replay for counted successes.
8. **Generalization:** replicate the direction of the gain on a project-disjoint advanced-mathematics cohort and a later temporal snapshot.

The 299/341 and 316/341 values are arithmetic thresholds relative to the published 281/341 GPT-6 Astra result, not claims that GPT-6 Astra is a matched PITB baseline. The public set has already been analyzed and must not be presented as blind evaluation.

## Adversarial controls

At minimum, mutate:
- exact statement while preserving a similar name/slogan;
- typeclass or instance type while keeping the same class name;
- theorem sibling with a near-identical prefix/suffix;
- project/topic node while preserving topical similarity;
- edge direction, relation type, source repository commit and Lake environment;
- one required premise in an AND bundle;
- an OR alternative while retaining a false bundle;
- local instance registry and notation/macro/elaborator scope;
- helper lemma statement and transitive dependents;
- proof/receipt digest, signer key status, semantic review status and source span;
- one budget dimension after the action is scheduled.

Negative controls should demonstrate that high retrieval similarity cannot overcome type mismatch, a missing conjunctive premise, stale environment, invalid source correspondence or an unauthenticated receipt.

## Implementation sequence

1. Complete and qualify the standalone typed graph evidence contract and preserve the exact runner summary.
2. Add an authenticated snapshot/receipt envelope bound to the pinned repository, toolchain, extractor build/config, typed graph digest and reviewer/extractor key registry.
3. Implement independent Lean extraction replay for signature/proof/definition/field/extends/docref/instance edge categories; compare the exact extracted edge set and environment identity. Signatures authenticate a producer, not semantic truth.
4. Add the AND/OR support-bundle schema with explicit candidate/compatible/sufficient/exhaustively-minimal lifecycle and exact target/environment binding.
5. Build thin DAPS and TheoremGraph import adapters; keep their native checkpoints and provenance. Do not pull Python training stacks into production.
6. Run retrieval-only, bundle-only and DAG-only ablations on development tasks with matched budgets.
7. Pre-register a project-disjoint confirmatory cohort, test the high-declaration moderator and compare full success-cost-latency frontiers.
8. Extend into change-impact/proof-maintenance experiments: mutate a shared helper, invalidate only exact dependents, and measure repair/clean replay.

## Data and license boundary

AutoformBench's repository labels its release CC BY-NC 4.0 and states it is intended for evaluation/research rather than training. Its third-party textbook/source materials retain their own rights. This PITB track uses the published public task-level result CSVs for descriptive analysis and should use the official evaluation interface only after project-specific license review. Do not train, fine-tune or create a persistent training corpus from AutoformBench materials. DAPS code is Apache-2.0, but datasets/checkpoints and third-party Blueprint sources need their own license and provenance review. TheoremGraph datasets and services also require asset-level license/redistribution review. This document is research planning, not legal advice.

## Reproducibility references

- Pinned AutoformBench release: `af1aa38fd44e019e566bce0b5a168609a886d8e6`
- Analysis implementation: `scripts/research/pitb_autoformbench_audit_v0.py` (research-only; published CSV analysis, no training-data import)
- Typed graph evidence contract: `tools/pitb-graph-evidence/`
- Existing two-level scheduler: `docs/research/pitb-proof-graph-scheduler-2026-11-01.md`
- Support-antichain protocol: `docs/research/pitb-support-antichain-2026-10-30.md`
