# PITB cross-benchmark competitive frontier (2026-11-06)

This registry is a dated map of the external benchmark frontier and the targets PITB should pursue. **It is not a performance report.** All figures are external reference results; all targets are prospective engineering goals.

## Why seven tracks, not one leaderboard

The benchmark families ask different questions:

1. **SorryDB** — can a system complete real open Lean proof obligations from working mathematics projects?
2. **AutoformBench** — can it construct the definitions, auxiliary declarations and proofs needed to formalize advanced mathematics?
3. **FVSpec** — can it faithfully translate real software property-based tests and then prove the resulting Lean specifications?
4. **Aegis** — can it verify generated specifications against real Rust implementations, including machine arithmetic and imperative behavior?
5. **Mathlib retrieval** — can it recover useful premises and complete proof routes, then turn those routes into verified proofs?
6. **CoCo-Prover** — can multi-agent proof orchestration improve the success-versus-cost frontier on many dependent program obligations?
7. **Benchmark assurance** — can the task, specification, source, verifier, split and result be independently trusted?

These numbers cannot be pooled. A 57.2% SorryDB Pass@1, an 82.4% AutoformBench formalization completion rate, an FVSpec easy-task mean, and Mathlib Recall@100 do not share a denominator or task definition. The benchmark audit literature also shows why “the Lean kernel accepted it” is not enough: a proof can be impeccable relative to a vacuous, incomplete or incorrectly translated proposition.

## Current external reference points

| Track | Published reference | PITB target |
|---|---|---|
| SorryDB_2601 | Aleph Prover 572/1,000 Pass@1; 632/1,000 combined coverage across the listed leaderboard systems | ≥603/1,000 matched single-system Pass@1; ≥680/1,000 matched union coverage |
| AutoformBench | GPT-6 Astra 281/341 (82.4%), source-present | ≥299/341 (first integer count clearing +5 pp); stretch ≥316/341 |
| FVSpec | mean baseline 70% easy / 49% hard for three reported models | ≥75% easy / ≥54% hard, with verified PBT-to-spec fidelity |
| Aegis | EasyProver 1,054/2,000 (52.7%) within 30 min per statement | ≥1,200/2,000 (60%) on the same stratified cohort and cap |
| Mathlib retrieval | LeanGraphSearch: Recall@100 94.7%; Covered@20 42.8%; Covered@50 44.2% | ≥96%; ≥44.8%; ≥46.2%, respectively, with downstream proof utility |
| CoCo-Prover | up to 30.9% cost reduction in one reported benchmark/system cell | a better matched success-cost frontier on at least three of five tasks |
| Assurance | 4,833 reported audit findings, 398 mechanically certified issues across five benchmark families | 100% source/spec receipt closure for admitted tasks; zero silent denominator loss |

Exact source conditions, denominators, cohort identities, dates and interpretation boundaries are in [the JSON registry](./pitb-cross-benchmark-frontier-v0.json). The JSON Schema is kept beside it.

## Paradigm shift: proof engineering as a complete assurance loop

PITB should not be “another prover with retrieval.” Its competitive claim should be that it controls the entire proof-engineering lifecycle:

- **Source-to-specification fidelity:** establish that the theorem actually expresses the source task, with exact source, version, range, hypotheses, type and environment.
- **Typed knowledge graph:** separate observed formal dependencies from proof-term support, definition-body references, instance search, module/build edges, informal hints and external candidates.
- **Route retrieval, not isolated premise ranking:** retrieve alternative and conjunctive premise bundles; evaluate complete sufficient routes, not only Recall@K.
- **Counterfactual support discovery:** compute the finite antichain of minimal sufficient premise sets under a frozen target, context, Lean version, solver policy and budget. Do not infer logical necessity from one proof term.
- **Adaptive proof orchestration:** allocate bounded specialist attempts according to expected marginal verified progress, while charging all probes, failed attempts and checker work.
- **Independent proof and semantic replay:** independently open and authenticate source, run, proof, checker and semantic-gate evidence. A hash-shaped string is not an authenticated receipt.
- **Proof maintenance:** after an upstream definition, import, typeclass or implementation changes, determine which proofs remain valid, which need repair, and whether any statement semantics changed.
- **Prospective generalization:** use repository-disjoint and future-arrival cohorts. A random task split inside the same project is not an unseen-project test.

The success metric is the **assurance-adjusted Pareto frontier**: unique exact-target proofs with verified semantics, by wall time and total cost, under a fixed budget. Raw pass rate remains important, but it cannot compensate for incorrect task translation, unverified receipts, contamination or silent task removal.

## SorryDB: task-held-out versus project-held-out

PITB has independently audited the frozen SorryDB_2601 artifacts:

- SorryDB repository commit: `74b3b63f8ba890d675f362924b21ac55db410383`
- Full database blob: `5fafddac467af6d30d8f5c8731d6edec9c273a47`
- 1,000-task evaluation split blob: `af3b663429dc4689ee728d2fb5c7d8d5034476f7`
- Full snapshot: 5,663 tasks (the README documents 2,601; the count drift is retained in the audit), 78 repositories.
- Evaluation split: 1,000 tasks across the same 78 repositories, 107 repository/commit pairs and 339 repository/file clusters.

Therefore SorryDB_2601 is task-held-out, **not project-disjoint**. PR #10667 adds deterministic project-disjoint folds: five folds of exactly 200 tasks, holding whole repositories together (15–16 repositories per fold). This is project-held-out generalization within the same historical snapshot; it does not guarantee novelty relative to model pretraining. PITB must report the original task-held-out result and the project-disjoint results separately.

## Experimental protocol that makes a SOTA claim credible

Every direct comparison should freeze and publish:

- task IDs, immutable source commit, source/goal digests, Lean and Mathlib versions, graph snapshot and split manifest;
- model identifier/version, system prompt, query formulation, retrieval index, checkpoint, agent scaffold and tool permissions;
- attempt/retry policy, reflection turns, wall-clock, CPU/GPU, token and monetary budgets;
- checker/verifier version, semantic/vacuity/assumption policies, and the exact evidence closure requirements.

Use paired task results and preserve failure classes. Infrastructure errors, timeouts, abstentions, unproved tasks, semantic mismatches and quarantined tasks must remain visible; do not delete them to improve the denominator. For SorryDB, resample repositories and files as clusters. For AutoformBench, preserve source-group stratification and distinguish source-present from source-free. For FVSpec, group multiple formalizations by originating PBT. For Aegis, group by implementation/specification pair.

The minimum direct-comparison arm set should be:

- no-retrieval + symbolic tactics;
- BM25;
- LeanSearch v2;
- DAPS;
- LeanGraphSearch;
- PITB typed graph and route retrieval;
- PITB without support-antichain guidance;
- a multi-agent union under the same total budget.

First measure candidate and route retrieval. Then prove the exact same target under a common downstream prover budget. The primary retrieval endpoint is verified task completion, not a candidate-ranking metric.

## How to beat the frontier

### 1. Win on coverage without buying unbounded compute

A bounded router should decide when to use symbolic tactics, local premise search, multi-hop graph expansion, LLM proof construction, independent checking, or abstention. Use marginal expected *assurance-qualified* success per unit cost, not token probability or self-reported confidence.

### 2. Beat retrieval SOTA on useful route recovery

LeanGraphSearch already shows the benefit of multi-hop dependency retrieval: Recall@100 94.7% versus LeanSearch v2 82.7%, plus better complete-route coverage. The next gain is not merely more recall. Measure:
- useful-novel-premise rate;
- sufficient-route coverage and minimal-route recovery;
- candidate precision conditioned on type compatibility;
- downstream exact-target completion under the same prover and budget;
- cost per assurance-qualified completion.

### 3. Beat proof-systems SOTA on the cost frontier

CoCo-Prover shows that two-level AND/OR proof graphs and metalevel specialist orchestration can reduce cost while improving solve rates. PITB should reproduce matched comparisons, charge parallel and failed probes, and target lower total cost at equal assurance-adjusted success—not claim a universal saving from one paper's best cell.

### 4. Expand to hard, real software obligations

Competition math is nearing saturation on some suites. Aegis and FVSpec create a harder frontier: translation fidelity, implementation semantics, fixed-width arithmetic, failure paths, loops and mutation. PITB should become cross-language: Python PBT → Lean and Rust implementation → Lean, while preserving original source semantics.

### 5. Turn benchmark integrity into a first-class result

Publish the defect taxonomy, source/spec correspondence receipts, exact split custody, verifier replay and negative-control results with each score. This creates an assurance qualification profile alongside completion, cost and coverage.

## Current implementation map

The following work is in separate reviewable PRs, not merged production functionality:

- [#10317 Typed graph evidence](https://github.com/Luminous-Dynamics/symthaea/pull/10317)
- [#10351 DAPS candidate-only graph import](https://github.com/Luminous-Dynamics/symthaea/pull/10351)
- [#10360 per-query retrieval receipts](https://github.com/Luminous-Dynamics/symthaea/pull/10360)
- [#10355 exact support-antichain analyzer](https://github.com/Luminous-Dynamics/symthaea/pull/10355)
- [#10371 SorryDB snapshot and semantic integrity](https://github.com/Luminous-Dynamics/symthaea/pull/10371)
- [#10667 project-disjoint benchmark folds](https://github.com/Luminous-Dynamics/symthaea/pull/10667)
- [#10518 paired repository/file cluster statistics](https://github.com/Luminous-Dynamics/symthaea/pull/10518)
- [#10654 Aegis Rust-to-Lean integrity and polarity scoring](https://github.com/Luminous-Dynamics/symthaea/pull/10654)

These PRs have different bases and validation states; do not assume the stack is merge-ready because component checks passed. PR #10371 in particular is a large integration stack and should be split/reviewed in narrow mergeable slices.

## Licensing and evaluation hygiene

- AutoformBench's public release is CC BY-NC 4.0 and explicitly for evaluation/research, not training. Referenced mathematical sources have separate licenses. PITB must not import its source excerpts, traces or gold solutions as training material.
- SorryDB's source repository is Apache 2.0; each indexed project still has its own license and attribution.
- FVSpec and Aegis retain their own corpus/source licensing requirements. Verify exact release terms before copying or redistributing content.

## References

- SorryDB leaderboard: https://sorrydb.org/
- SorryDB paper: https://proceedings.mlr.press/v306/letson26a.html
- AutoformBench: https://github.com/facebookresearch/autoformbench
- AutoformBench paper: https://ai.meta.com/research/publications/autoformbench-a-lean-formalization-benchmark-for-advanced-mathematics/
- FVSpec: https://fvspec.galois.com/paper and https://arxiv.org/abs/2606.01008
- Aegis / EasyProver: https://arxiv.org/abs/2610.38174
- LeanGraphSearch: https://arxiv.org/abs/2610.22339
- CoCo-Prover: https://arxiv.org/abs/2610.09681
- Formal benchmark audit: https://proceedings.mlr.press/v306/ammanamanchi26a.html and https://github.com/Shashi456/atp-checkers
