# PITB external benchmark portfolio and competitive strategy

**Date:** 2026-10-31  
**Status:** research synthesis and targets; no cross-benchmark PITB result is claimed.

## The field is not one leaderboard

The current proof-engineering landscape measures several different capabilities. A credible PITB comparison should be a vector of separately reported outcomes, not a blended score or an accuracy chart that mixes incompatible denominators.

| Capability | External benchmark | What it measures | Current evidence / comparison |
|---|---|---|---|
| Open proof-hole completion in original projects | SorryDB_2601 | Fill real open Lean obligations against the source project and pinned environment | Aleph Prover 57.2% Pass@1; Gemini Flash 3 Agentic 30.3% Pass@1; general Claude Opus 4.5 7.8% Pass@1 / 15.4% Pass@32 |
| Repository-scale software verification | VeriSoftBench | Proof obligations with project-specific libraries and cross-file dependencies | 500 tasks from 23 projects; best reported model 41% with curated context vs 35% with raw repository context |
| Long-structure mathematical proving | TheoremBench | Nearly 100 classical theorems in plain and premise-expanded forms; partial theorem structure and token efficiency | Explicitly exposes theorem-level coverage and easy-subtheorem bias |
| Semantic theorem correctness | T² | Check generated theorem statements through dependent successor theorems | 2,206 problems across 5 repositories, ~41 successors per problem; best reported testing accuracy 38.9% despite stronger compilation performance |
| Repair efficiency and trajectory quality | BlueprintTrace / BlueprintRepair | Typed local repair actions vs exact source patches vs whole-module rewrites | 142 controlled failure states; similar coverage, with typed repair cheapest per solved state in reported runs |
| Premise retrieval | Legendre / MathlibQR and MathlibMPR | Relevant-declaration recall and complete multi-premise route coverage | Lean Finder v2: 97.3% Recall@100 on 894 resolved queries; LeanGraphSearch: 94.7% on a different 810-query subset and 42.8% Covered@20 on MathlibMPR |
| Graduate-level formalization | ProofBench v1.1 | Natural-language problems paired with reviewed Lean statements | As of 2026-10-29, four systems tie at 100% on the reported leaderboard; the benchmark uses 100-problem public and private splits and a 40-turn tool budget |
| Cross-family strategy transfer | PITB frozen research-math split | Whether typed proof insights improve exact-target proofs on held-out mathematical families | PITB's unique research claim; no measured result yet |

These rows are not head-to-head comparisons. Even the retrieval numbers differ in query denominator; the Legendre board explicitly resolves 894 of MathlibQR's 946 queries and warns that the original system papers used different corpora/metrics. The theorem and project benchmarks differ in task source, formal environment and acceptance criterion. Keep all of those facts visible.

## What the frontier implies

### 1. Context quality is a first-class variable

VeriSoftBench's curated-context advantage over raw repository context is direct evidence that more context is not automatically better. PITB should measure a context curve: full project context, dependency-closure context, mechanically projected local context, and evidence-selected support context. Score both proof success and the cost/latency required to reach it.

A projected proof context must preserve Lean elaboration resources, especially local instances. Typeclass synthesis searches local and global instances, so a declaration-only context ablation can produce invalid or misleading experiments if the instance registry is not projected consistently. This is exactly why the PITB ContextEvidenceV2 track captures both the ordered LocalContext and LocalInstances.

### 2. Compilation is necessary, not enough

T² treats successor theorems as integration tests for the semantic meaning of generated declarations. Its low testing accuracy relative to compile success shows why a proof checker alone cannot validate that a generated or transferred statement means the intended thing. For SorryDB the target theorem statement should be preserved byte-for-byte at source level and by an elaborated type digest. For theorem-generation or repair tracks, dependent-successor and nearby mutation checks should be separate gates.

### 3. Typed repair is a promising interface

BlueprintRepair uses schema-checked local operations that identify the node being changed and prevent silent target mutation. Its reported typed-edit interface costs less per solved state than unrestricted patching or rewriting, without a clearly established coverage deficit in the tested localized-failure benchmark. PITB should make typed repair the default action space for theorem-blueprint and context-graph interventions, while retaining free-form rewriting as a measured comparator.

Do not assume this result transfers automatically to SorryDB source holes: BlueprintTrace is a controlled repair benchmark. Test the interface under matched source, feedback, model and budget on SorryDB before claiming the cost ordering generalizes.

### 4. Retrieval should recover routes, not just nearest declarations

A 97.3% Recall@100 result is strong retrieval evidence but not a 97.3% proof completion rate. Multi-premise route coverage is a more demanding retrieval measure, and downstream proof completion/cost must remain the final utility endpoint. PITB should combine intent-aware query rewriting, semantic seed retrieval, dependency graph traversal, instance-aware support projection and a Lean feedback loop.

### 5. Use compute as a controlled decision variable

Pass@32 can obscure how resources were spent. PITB should report a success-cost frontier at several shared budget caps, not just one maximum-k score. The controller should choose among retrieval, context projection, repair and further candidate generation by expected marginal verified success per unit cost, and stop when the remaining options have low calibrated value. This is a hypothesis to test; it is not established by adding an HDC/LTC controller.

## PITB's proposed capability vector

Keep at least these dimensions independently visible:

1. **Exact proof completion:** Pass@1, Pass@8, Pass@32 on a pinned cohort.
2. **Transfer utility:** paired effect of typed insights against same-model, same-budget no-insight controls.
3. **Retrieval:** Recall@k, nDCG@k, premise-route coverage, and downstream completion.
4. **Context efficiency:** solve rate against context size, support size, instance retention, tokens and time.
5. **Semantic preservation:** exact statement/type invariance, valid source scope, downstream successor checks, mutation false-accept rate.
6. **Typed repair:** accepted actions, repair cost, unnecessary edits, theorem-target preservation and repair trajectory completeness.
7. **Reliability:** exact independent replays, repository-clustered confidence intervals, selective risk and calibration.
8. **Economics:** cost per verified success, latency, build/checker time and the full success-cost Pareto frontier.
9. **Generalization:** temporal performance on newer SorryDB snapshots and cross-project/cross-domain performance on VeriSoftBench-like obligations.

Do not add these to one scalar. Publish the vector and declare which dimensions are qualification gates.

## Competitive targets

- **Near-term causal claim:** at least +5 percentage points absolute Pass@1 on SorryDB over the same-model PITB S0 arm, with a repository-clustered 95% interval excluding zero.
- **Efficiency claim:** at least 10% lower cost per verified success at matched solve rate, or a strictly better success-cost Pareto frontier.
- **Industry stretch:** at least 60% Pass@1 on SorryDB_2601, above the published 57.2% Aleph Prover reference, with matched attempt and trust-tier semantics. If exact comparability cannot be established, label it an external reference only.
- **Retrieval:** reproduce results on one pinned common query set; then target >97.3% Recall@100 and higher route coverage, but require downstream solve/cost benefit before claiming retrieval improvement.
- **Semantic integrity:** zero false accepts in a pre-registered mutation suite and 100% independent replay of counted successes.
- **Forward generalization:** replicate the direction of improvement on a temporally later frozen SorryDB cohort that was not used for tuning.

These targets are deliberately separated. A high Pass@1 with theorem edits is a failure of the integrity gate; high Recall@100 without more verified proofs is not a proof-engineering win.

## Recommended execution order

1. Pin the SorryDB_2601 split bytes and exact upstream version, then build a Rust importer and task-manifest audit.
2. Complete a two-project Lean smoke run proving that repository reconstruction, sorry-span replacement, elaborated type identity and clean independent checking work end to end.
3. Run matched S0–S4 arms on a development-only subset to test the retrieval/context/insight hypotheses without opening confirmatory targets.
4. Freeze prompts, model/config hashes, resource caps and analyzer version. Run the 1,000-task SorryDB cohort with complete accounting and repository-clustered analysis.
5. Add T²-style successor testing where relevant, and separately run typed repair comparisons on BlueprintTrace or an equivalent controlled subset.
6. Evaluate on a later temporal SorryDB snapshot and on repository-scale software-verification tasks. Keep each report and denominator separate.

The key paradigm shift is not a larger prover. It is a proof-engineering controller whose every intervention has explicit preconditions, typed effects, a measured budget and an independent verifier. PITB wins only when this architecture creates more verified value per unit cost without altering the proposition being proved.

## Sources

- SorryDB: https://sorrydb.org/ and https://proceedings.mlr.press/v306/letson26a.html
- VeriSoftBench paper and results: https://arxiv.org/abs/2602.18307 and https://utopia-group.github.io/VeriSoftBench/
- TheoremBench: https://arxiv.org/abs/2606.09450
- T² / Benchmarking Testing in Automated Theorem Proving: https://aclanthology.org/2026.acl-industry.150/ and https://github.com/ldilab/T2
- BlueprintRepair / BlueprintTrace: https://aclanthology.org/2026.mathnlp-main.3/ and https://github.com/Karifannaa/BlueprintRepair
- Legendre Mathlib retrieval board: https://www.legendre-leaderboard.com/
- LeanGraphSearch: https://pluslabnlp.github.io/LeanGraphSearch/
- ProofBench v1.1: https://www.vals.ai/benchmarks/proof_bench
- Lean 4 local instance synthesis: https://lean-lang.org/doc/reference/latest/Type-Classes/Instance-Synthesis/
