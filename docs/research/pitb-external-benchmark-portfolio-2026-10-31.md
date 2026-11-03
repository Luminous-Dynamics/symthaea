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
| Dependency-aware premise selection | DAPS | Typed graph neighborhoods, two-hop bidirectional dependency context, group-level contrastive premise bundles, and Blueprint adaptation | Recall@32: 89.25 on Mathlib4-Heldout (+11.41pp vs LeanHammer), 23.53 on DAPS-Collection (+20.34pp), 50.08 on DAPS-Blueprint (+9.66pp); premise conditioning raises Goedel-Prover-V2-8B miniF2F Pass@4 from 49.90 to 54.03. Different cohorts; DAPS-Blueprint is node-heldout, not project-heldout |
| Formal/informal theorem graph | TheoremGraph / LeanGraph | Large-scale typed Lean declaration graph bridged to candidate informal theorem dependencies | 388,105 formal nodes and 11.3M typed edges across 25 projects; 11.7M informal statements and 18.3M extractor-proposed edges; name/signature plus graph expansion Recall@10 0.775 vs LeanSearch v2 reranked 0.780. Informal edges and LLM-affirmed cross-formality matches remain candidate evidence |
| Graduate-level theorem proving | ProofBench v1.1 | Natural-language problems paired with reviewed Lean statements | Private 100-problem split with an agentic 40-turn tool budget; current scores must be read with model/harness/cost settings |
| Open, evolving Lean formalization | LeanEval v1 | Hard mathematical problems with reviewed statements and automatic comparator verification | Small, actively updated public leaderboard; useful as an external challenge set, not a stable fixed-snapshot comparison |
| Advanced mathematics formalization | AutoformBench | Formalize proof developments with missing definitions, intermediate lemmas and supporting infrastructure | 341 audited tasks from 24 sources; current source-present leaderboard shows GPT-6 Astra 281/341 (82.4%) and Opus 5 211/341 (61.9%); separate from fill-in-the-sorry |
| Theory-scale autoformalization | LCS-Bench | Build coherent formal libraries across many interdependent textbook items | 327 items, 4,076+ declarations and 85K+ Lean LOC; reported best autoformalization result 20.1% |
| Implementation-level software verification | Aegis | Prove or refute generated specifications for translated, real safe-Rust implementations | 2,045 Rust–Lean pairs and 11,533 candidate specifications; EasyProver solves 52.7% of a stratified 2,000-statement sample within 30 minutes per statement |
| Repository-wide proof compression | LeanLean (64 repositories) | Reduce repository token/declaration size while preserving protected theorem validity; exercises dependency restructuring, math simplification and syntax optimization | Opus 5 paper baseline: 48.3% average compression; 64 repos up to 174k LOC, 12h each; separate from fill-in-the-sorry and theorem-family transfer |
| Proof-graph scheduling and specialist orchestration | CoCo-Prover; EasyProver / Aegis | AND/OR proof hypergraph per declaration + lemma-dependency DAG across declarations; symbolic ready-goal scheduler and priced specialist actions | CoCo-Prover reports up to 30.9% lower cost and best solve rate on five Lean program-verification benchmarks; EasyProver reports near-saturation on classic sets and introduces harder implementation-derived Aegis tasks. Different task/protocol scope; PITB has no matched result |
| Adaptive compute allocation | PITB controller + published Lean cost-aware routing | Decide continue/retrieve/redecompose/repair/tactic-probe/snapshot-branch/restart/stop using action-conditioned verified-success probability under a fixed horizon under a shared multi-resource budget | Published route policy reports 25.8% lower average cost than fixed-step on a PutnamBench subset; a separate simulation reports +7.9% cost-quality AUC / 28.9% lower budget to matched accuracy. Different cohorts/protocols; PITB has no measured result yet |
| Cross-family strategy transfer | PITB frozen research-math split | Whether typed proof insights improve exact-target proofs on held-out mathematical families | PITB's unique research claim; no measured result yet |
| Benchmark target integrity | PITB benchmark-integrity gate / ICML 2026 audit | Detect source/statement drift, vacuity/counterexamples, unresolved metavariables, unapproved axioms and semantic mismatch before scoring a task | Audit of five popular Lean benchmarks/forks reported 4,833 findings, including 398 mechanically certified defects; not a prover leaderboard result |

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

A 97.3% Recall@100 result is strong retrieval evidence but not a 97.3% proof completion rate. DAPS materially raises the dependency-aware retrieval frontier: its 89.25 Recall@32 on file-heldout Mathlib queries and +4.13pp miniF2F Pass@4 change the baseline PITB should compare against. TheoremGraph offers a much larger typed formal dependency resource, while its informal edges and LLM-affirmed cross-formality links must remain lower-trust candidates. LeanGraphSearch remains a distinct two-hop graph retrieval comparator. None of these headline metrics are interchangeable: task collections, K, model, query formalizer, train/test splits and checkers differ.

PITB should compare these systems on the same fixed target/query set and K=16/32 budgets, then measure downstream exact-target proof success and cost. Include group-level premise-set utility, complete route coverage, precision, recall, macro-F1, type/instance compatibility, source-group/project holdout, and frequency/recency bias. DAPS-Blueprint's test split is node-level among 53 projects, not unseen-project generalization; PITB needs project-disjoint and temporal holdouts. Cross-formality candidate matches should carry provenance, confidence and independent-review status. The differentiator is not merely another graph index: it is joining these heterogeneous sources into an evidence-qualified AND/OR support planner that recognizes when an existing lemma bundle is sufficient and when missing infrastructure must be constructed.

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



## AutoformBench residual-frontier audit (3 November 2026)

The task-level residual and cost analysis is maintained separately at [PITB AutoformBench residual frontier](pitb-autoformbench-residual-frontier-2026-11-03.md). It pins the public release commit and five Git blob identities, reconciles all 341 tasks and 1,155 graded declarations, and separates descriptive cross-model outcomes from matched experiments. The corresponding research-only offline auditor is `scripts/research/pitb_autoformbench_audit_v0.py`. The report identifies an expensive residual tail and proposes a distinct formalization-infrastructure synthesis endpoint; it does not claim a PITB solve-rate improvement.

## Sources

- SorryDB: https://sorrydb.org/ and https://proceedings.mlr.press/v306/letson26a.html
- VeriSoftBench paper and results: https://arxiv.org/abs/2602.18307 and https://utopia-group.github.io/VeriSoftBench/
- TheoremBench: https://arxiv.org/abs/2606.09450
- T² / Benchmarking Testing in Automated Theorem Proving: https://aclanthology.org/2026.acl-industry.150/ and https://github.com/ldilab/T2
- BlueprintRepair / BlueprintTrace: https://aclanthology.org/2026.mathnlp-main.3/ and https://github.com/Karifannaa/BlueprintRepair
- Legendre Mathlib retrieval board: https://www.legendre-leaderboard.com/
- DAPS (NeurIPS 2026 project; submitted 30 October 2026): https://arxiv.org/abs/2610.41536; https://daps-eth.github.io/; code: https://github.com/DAPS-ETH/DAPS (Apache-2.0).
- TheoremGraph / LeanGraph: https://arxiv.org/abs/2606.25363; TheoremSearch: https://theoremsearch.com/.
- LeanGraphSearch: https://pluslabnlp.github.io/LeanGraphSearch/
- ProofBench v1.1: https://www.vals.ai/benchmarks/proof_bench
- LeanEval: https://lean-lang.org/eval/
- AutoformBench: https://github.com/facebookresearch/autoformbench and https://ai.meta.com/research/publications/autoformbench-a-lean-formalization-benchmark-for-advanced-mathematics/
- LCS-Bench: https://arxiv.org/abs/2606.26525
- EasyProver / Aegis: https://arxiv.org/abs/2610.38174
- Lean 4 local instance synthesis: https://lean-lang.org/doc/reference/latest/Type-Classes/Instance-Synthesis/


## Frontier update: EasyProver and Aegis (28 October 2026)

A new preprint, *Scaling Lean Theorem Proving for Real-World Code Verification* (28 October 2026), adds an important architecture baseline. EasyProver reports an off-the-shelf frontier model, without additional training, proving 669/672 PutnamBench tasks, all six IMO 2026 formalizations, 597/600 NTP4VC verification conditions, and all 189 VERINA tasks by proof or refutation. These are different cohorts and acceptance endpoints, not scores that can be compared directly with SorryDB Pass@1 or PITB's family-transfer split.

The system architecture is especially relevant:
- Recursive model-directed delegation with child agents in fresh contexts and private project workspaces.
- Asynchronous returns: completed children can redirect the parent before slower siblings finish.
- Parallel tactic probing, with successful local continuations returned and failed probe diagnostics kept out of the parent context.
- Explicit concurrency admission controls for agents and expensive Lean queries.
- A stateless, elastic elaboration service with warm Mathlib workers, decoupled from proof-search agents.
- Lock-protected candidate promotion followed by independent file checking and revert-on-failure.

This motivates the next PITB paradigm: **bounded speculative proof engineering**. Each branch has a frozen task/environment/proof-state identity, a private workspace, a budget, and a typed result. Branches can explore aggressively, but only a single independent promotion gate may turn a candidate into a verified success.

### PITB runtime ablation

Register a distinct runtime ablation on SorryDB after the environment smoke is complete:

| Arm | Runtime | Hypothesis |
|---|---|---|
| R0 | Serial agent with one shared context | Matched reference |
| R1 | Independent restarts with isolated contexts | Value of context reset |
| R2 | Recursive delegation with private workspaces and asynchronous returns | Value of adaptive branching |
| R3 | R2 plus parallel tactic probes and separate agent/Lean admission caps | Value of local parallelism |
| R4 | R3 plus PITB context evidence, typed insights, support selection and repair routing | Value of PITB's evidence layer |
| R5 | R4 plus post-elaboration proof-state snapshot reuse, immutable shared environment, and branch-local metavariable state | Value of Level-2 live-state branching; requires a separate snapshot/cache-isolation receipt |

Match the same model, task set, toolchain, prompt family, checker, trust policy, and total resource envelope. Measure branch fan-out, useful-child-return rate, duplicate-work rate, branch survival, tactic-probe success rate, total Lean CPU time, elaborator queue time, warm-cache hit rate, p50/p95 time-to-first-verified-proof, model tokens, and cost per verified proof. Compare multiple budget caps. A larger search tree is not itself a result.

Start with small worker tiers (1/2/4/8), separate the concurrency caps for agents and expensive Lean queries, and use backpressure when the elaborator or build service saturates. Share immutable build artifacts where safe; keep mutable proof workspaces isolated until promotion.

### Aegis: a distinct future cohort

The same preprint introduces Aegis, a generated implementation-level verification benchmark. It synthesizes safe Rust implementations, differentially checks behavior against Aeneas-translated Lean models, generates frozen specifications, and labels statements with kernel-checked proofs or refutations. The paper reports 2,045 validated Rust–Lean program pairs and 11,533 candidate specifications. On a stratified 2,000-statement sample, EasyProver proved 1,054 (52.7%) within a 30-minute per-statement budget; it left 943 unresolved and did not attempt three. Results vary by task difficulty and specification category.

Aegis is not another SorryDB leaderboard row. It probes implementation semantics, fixed-width arithmetic, error behavior, loops, casts and mutation. PITB should add it only after pinning the exact code/data release, license, generated statement set, translation artifacts, differential-validation coverage and proof/refutation checker. Kernel-checked labels establish propositions about the translated Lean program; they do not automatically establish that a generated specification captures the intended user requirement.

The best future Aegis report should separately measure proof, refutation, unresolved, invalid-specification and abstention rates; Rust/Lean differential mismatches; mutation-control performance; and resource cost. Keep it distinct from SorryDB and the research-math family split.

### New PITB qualification hypotheses

These are targets, not measured results:

1. Zero cross-branch file contamination; every branch bound to source commit, task ID, Lean environment and proof-state digest.
2. Lower p95 time-to-first-verified-proof at matched total budget, without reduced exact-task reliability.
3. Measurable useful-child-return rate and more verified solves per 1,000 branch tokens.
4. Measurable tactic-probe efficiency: success per probe and per Lean CPU-second.
5. 100% independent replay of promoted candidates, with target/type identity, patch scope, and axiom policy preserved.
6. Forward generalization on a later SorryDB cohort, with the Aegis set kept separately frozen.

**Primary source:** Ganguly et al., *Scaling Lean Theorem Proving for Real-World Code Verification*, arXiv:2610.38174, 28 October 2026: https://arxiv.org/abs/2610.38174.


## Frontier update: AutoformBench and LCS-Bench

AutoformBench (Meta AI, 13 October 2026) addresses advanced-mathematics formalization rather than filling an existing proof hole. It contains 341 audited Lean 4 tasks from 24 mathematical sources. The agent receives an informal proof but must build missing definitions, intermediate lemmas and infrastructure before proving the target. The current public source-present leaderboard lists GPT-6 Astra at 281/341 (82.4%), Opus 5 at 211/341 (61.9%), GPT-5.6 Sol at 201/341 (58.9%), and additional systems. The repository pins Lean v4.32.2 and a Mathlib commit. The release is source-free with respect to textbook excerpts and gold proofs and is licensed CC BY-NC 4.0; check licensing and obtain source texts independently before any reuse. Do not compare these percentages to SorryDB or PITB Pass@1.

LCS-Bench tests theory-scale autoformalization of logics for computer science: 327 textbook items, more than 4,076 declarations and over 85K lines of Lean. Its reported state-of-the-art autoformalization score is 20.1% across its task setup. It is valuable for evaluating whether insights and typed dependencies scale from individual holes to whole formal libraries.

**PITB implication:** add external replication targets by task type, not leaderboard score. SorryDB tests completion of project-local holes; AutoformBench tests filling in missing formal infrastructure around advanced mathematics; LCS-Bench tests coherent theory-scale construction; Aegis tests correctness obligations for real implementations. PITB's unifying contribution could be a typed evidence-and-search controller, but the success metric and denominators remain separate for each cohort.


## Experimental architecture correction: branch wall-work versus elapsed time

Parallel proof exploration needs two clocks. The sum of per-branch wall durations is **branch wall-work**; it can exceed end-to-end elapsed wall time when branches overlap. Do not add overlapping branch durations and call the result user-observed latency. The runtime ledger now records branch wall-work separately from elapsed wall-clock. Tokens, model calls, Lean CPU and cost remain additive across branches; elapsed wall time is a separate end-to-end measure. Report both branch wall-work and elapsed time, along with p50/p95 time-to-first-verified-proof.

Promotion is now a content-addressed transaction. A branch candidate carries a digest; the promotion record must name the exact candidate digest, the checker and independent replay each state that exact digest, and the target type must match the source task before and after. The checker and replay identities must be distinct. A label such as candidate_available cannot authorize promotion by itself.


## Bounded speculative proof ledger

PITB's R0–R4 runtime comparison is recorded in a branch ledger. Branches may return retrieval routes, partial lemmas, repair plans, counterexamples, or complete proof candidates, but all remain untrusted. A proof candidate is represented by a SHA-256 of the exact candidate bytes. Promotion must bind that digest to the source branch, the sorry-only patch digest, the original target-type digest, a checker identity/receipt, and a distinct independent replay identity/receipt. The checker and replay must attest the same candidate bytes. Changing the target statement, promoting a different branch artifact, adding unapproved axioms, or escaping the sorry span blocks success.

Resource accounting separates elapsed wall-clock from summed branch wall-work. Parallel branches may accumulate 180 branch-ms while completing in 120 elapsed-ms; that is not an accounting contradiction. Token count, model calls, Lean CPU and cost remain additive. The Rust `analyze_sorrydb_branch_ledger` binary validates these joins and budget invariants and has a synthetic negative-control suite. The fixture is contract-only and does not record an executed branch or proof.


LeanEval is a useful complementary destination for open challenge problems: it aims to capture difficult Lean formalization tasks with reviewer-vetted statements and automatically comparator-checked submissions. Its public leaderboard is evolving and currently small, so treat it as a prospective external challenge set rather than a stable fixed-cohort benchmark. See https://lean-lang.org/eval/.


### Live elaboration snapshot baseline (R5)

The R5 systems arm is motivated by *Keep the Proof State Live* (2026). The paper reports 5.6–50× wall-time speedups across 48 miniF2F-v2 problems (45 prove-phase cases and three full end-to-end runs), with a 9.7× median and 14× average on the 45 hand-crafted prove-phase cases. Its mechanism captures the post-elaboration snapshot (Environment, MetavarContext, LocalContext), shares the immutable Environment, and copies the mutable MetavarContext per branch. It is not just an import cache: the branch avoids re-elaborating the theorem body. The paper's patched binary is based on Lean 4.25.1; PITB must treat it as a separate implementation/toolchain baseline, not assume the patch works on the SorryDB toolchains. Source: https://arxiv.org/abs/2605.25556.

R5 must match R4's model, task, branch portfolio, checker, trust tier and total resource envelope. The only intended intervention is post-elaboration state reuse. It must record the exact snapshot receipt, branch-local metavariable-context copy, environment identity, and cache policy. The conservative v0 cache policy is **discard mutable tactic/typeclass caches at branch fork and rebuild them branch-locally**; a future dependency-validated shared-cache arm is separate. This is particularly important because Lean 4.36.0-rc2 adds cross-command typeclass cache persistence and explicit invalidation for environment/instance/reducibility/option changes, and fixes stale cache entries surviving SavedState restore (release-candidate notes, 27 October 2026): https://lean-lang.org/doc/reference/latest/releases/v4.36.0/.

Required metrics: snapshot capture time, branch restore/start p50/p95, branches started, elapsed wall time, summed branch wall-work, Lean CPU time, memory, cache invalidations, verified branches per second, and cost per independently verified proof. Compare R4 cold re-elaboration against R5 post-elaboration snapshotting at equal total work and budget. Do not cite wall-time speedup alone if total CPU/memory or proof yield deteriorates.


### LeanLean: a separate repository-compression endpoint

LeanLean belongs beside SorryDB, not inside it. It evaluates whole-repository compression while preserving protected theorem validity. The public paper reports 48.3% average compression for Opus 5 across 64 real repositories; the task budget is 12 hours per project. PITB has added a dedicated evaluation design and proposes matched L0–L5 arms for graph-only, support-antichain, typed repair, insight transfer and adaptive allocation. This is a separate endpoint; no LeanLean task has been downloaded or evaluated by PITB. See [the LeanLean evaluation design](pitb-leanlean-evaluation-2026-11-01.md).


### Adaptive proof-search control: a separate systems/cost endpoint

The current cost-aware routing literature is now part of the comparator set. The ICML 2026 action-routing work reports 25.8% average cost reduction against fixed-step proving while preserving performance on a PutnamBench subset. *Smarter Spending* reports 7.9% cost-quality AUC gain and 28.9% lower budget to matched accuracy in simulation over 64 seeds on an 85-problem subset. These are not SorryDB scores or PITB measurements; see the adaptive allocation design at `docs/research/pitb-adaptive-proof-search-allocation-2026-11-01.md`.

PITB's first implementation is a conservative deterministic LCB/shadow-price action selector. Its intended differentiator is a wider action set and a multi-resource feasibility check tied to exact proof-state identity. It is not yet a learned router: supplied action-conditioned success estimates and standard errors remain untrusted until measured on authenticated trajectories and interval coverage is checked. The controller now verifies exact calibration-receipt bytes and recomputes per-action ECE/Brier statistics, but source checker/replay receipts are not yet individually authenticated and the checked-in calibration data are synthetic.


### Benchmark integrity is a separate capability axis

The ICML 2026 audit of five Lean theorem-proving benchmarks/forks reported 4,833 findings, including 398 mechanically certified defects such as counterexamples, vacuity and unsound axioms. PITB has added a per-target target-integrity receipt and Rust verifier. This is not a solve-rate metric; it determines whether a task is eligible for a performance denominator. No public benchmark cohort has been audited by PITB yet. Measured eligibility now requires exact evidence artifacts and an Ed25519 reviewer attestation checked against an externally pinned registry digest; no production reviewer keys are committed. See `docs/research/pitb-benchmark-integrity-gate-2026-11-01.md` and https://proceedings.mlr.press/v306/ammanamanchi26a.html.


### Frontier update: graph-based cost-efficient orchestration (1 November 2026)

CoCo-Prover (7 October 2026) reports up to 30.9% lower monetary cost than the strongest baseline/model on five Lean program-verification benchmarks. Its core design joins intra-declaration AND/OR proof hypergraphs to an inter-declaration lemma-dependency DAG, builds a dependency-ready frontier symbolically, then routes goal groups to bounded specialists. It reports cost gains but also notes slower wall-clock time in some settings, reliance on LLM semantic judgment, and only two repository-level benchmarks. Sources: https://arxiv.org/abs/2610.09681.

EasyProver/Aegis (28 October 2026) reports 669/672 PutnamBench, 597/600 NTP4VC and 189/189 VERINA obligations by proof or refutation. Aegis adds harder code-verification tasks derived from real safe-Rust implementations: 2,045 Rust-Lean pairs and 11,533 candidate specifications; EasyProver solves 52.7% of a stratified 2,000-statement sample within 30 minutes per statement. Source: https://arxiv.org/abs/2610.38174.

PITB's new proof-graph state contract borrows the two-level graph abstraction but focuses on independently auditable proof-state identities, stable/provisional/discarded helper lifecycles, exact rollback propagation, deterministic frontier recomputation and multi-resource reconciliation. The checked-in graph is synthetic; there is no live Lean capture or measured performance yet. See `docs/research/pitb-proof-graph-scheduler-2026-11-01.md`.


### Causal intervention evaluation is a separate capability

PITB added a matched randomized support-antichain experiment with full-context, predicted-support, oracle-support and size-matched random-support arms. The paired analyzer reports intention-to-treat risk differences and component-cluster bootstrap intervals, with exact state/target/seed/resource joins. Every measured arm row now requires a signed target-integrity receipt verified against an externally pinned reviewer-key registry; checker/run/successor receipt structures and identity joins are validated. This is experimental infrastructure, not a measured treatment effect; confirmatory status remains blocked pending trusted runner-origin attestation. See `docs/research/pitb-causal-proof-intervention-trials-2026-11-01.md`.


### Mutation-aware bidirectional implementation verification

The Aegis paper says mutated program variants were generated but not evaluated. PITB has opened a separate mutation-aware cohort that scores proof and refutation polarity, source-version binding, one-site mutation sensitivity and unseen-operator transfer. It is not part of the published Aegis 2,000-task result and no mutation cohort has been generated by PITB yet. See `docs/research/pitb-mutation-aware-bidirectional-verification-2026-11-01.md`.
