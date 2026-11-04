# PITB: assurance-adjusted, cross-benchmark proof engineering programme

**Research note · 4 November 2026**

## Thesis

PITB should not become one more leaderboard of proof attempts. It should become an **evidence-qualified proof-engineering control plane**: a system that connects task discovery, specification reconstruction, typed premise retrieval, proof-route planning, exact source repair, independent checking, and long-term maintenance.

The score should mean: *the agent repaired the intended obligation in the exact original project context, with independently auditable evidence, within a declared resource budget*. A kernel-accepted term is necessary; it is not, by itself, proof that the intended task was solved faithfully.

The current SorryDB receipt contract therefore requires more than compilation: the target must be a `Prop`, the exact source span must be replaced once, the original target `sorry` and its `sorryAx` dependency must disappear, the declaration-signature digest must remain identical across replacement, the kernel and independent semantic/source-binding/vacuity gates must pass, and axiom/`sorry` deltas must remain zero. This blocks a class of false wins where an agent changes or weakens the theorem instead of proving it.

## What the current frontier measures

These benchmarks are complementary, not interchangeable.

| Benchmark | Task unit and scale | Public reference point | PITB role |
|---|---|---|---|
| [SorryDB](https://proceedings.mlr.press/v306/letson26a.html) | Open `sorry` obligations from 78 real Lean projects; a frozen 1,000-task evaluation slice | The live [leaderboard](https://sorrydb.org/) reports 57.2% Pass@1 for the top listed individual approach and 63.2% combined coverage | Exact repository reproduction, proof replacement, real-project proof reliability and repair maintenance |
| [AutoformBench](https://github.com/facebookresearch/autoformbench) | 341 audited advanced-mathematics formalization tasks from 24 sources, requiring missing definitions/lemmas/infrastructure | GPT-6 Astra: 281/341 (82.4%) in the reported source-present condition | Formalization planning and infrastructure construction; distinct from filling an existing proof hole |
| [FVSpec](https://fvspec.galois.com/) | 9,415 Lean challenges derived from 11,039 real-world Python property-based tests across 333 repositories | Public baselines average 70% on easy and 49% on hard samples for the reported model set | Software-property interpretation, Python semantics, translation fidelity and proof of real engineering intent |
| [EasyProver / Aegis](https://arxiv.org/abs/2610.38174) | 2,045 differentially validated Rust–Lean program pairs and 11,533 candidate specifications from real implementations | EasyProver proves 1,054/2,000 sampled Aegis statements (52.7%) within 30 minutes per statement | Implementation-derived verification, counterexample/refutation support, mutation/cast/overflow/error-path correctness |
| [MathlibQR / MathlibMPR](https://arxiv.org/abs/2610.22339) | 810 fair retrieval queries over 171 declarations; 69 research-level theorems with annotated premise groups | LeanGraphSearch: Recall@10 0.794 / Recall@100 0.947; 44.2% complete-route coverage@50 on MPR | Retrieval diagnostics and premise-route construction, not end-to-end proof assurance |

The live SorryDB page defines Pass@1 as one attempt per task and combined performance as the fraction solved by at least one listed approach. These are different statistics. PITB must report both **attempt success** and **unique task coverage**, never substitute one for the other.

### Current retrieval systems must be included

The matched retrieval portfolio should include:
- BM25 / lexical retrieval;
- LeanSearch v2;
- Lean Finder V2 (short/ambiguous-query disambiguation and query rewriting);
- LeanGraphSearch (multi-hop traversal over formal declaration dependencies);
- DAPS selector;
- TheoremGraph-derived candidates where license and provenance permit;
- a PITB union/reranker;
- no-retrieval control.

Lean Finder V2 reports at least an 11% relative improvement over previous Mathlib semantic search systems on MathlibQR through adversarial training and query rewriting. LeanGraphSearch reports up to 12% recall gains and improved end-to-end pass/cost efficiency in its own reported settings. These are not directly comparable to SorryDB or AutoformBench. They motivate matched experiments; they are not PITB results.

The retrieval figures need to be kept task-specific. On the 810-query MathlibQR fair subset, LeanGraphSearch reports Recall@10 0.794 and Recall@100 0.947, while LeanSearch v2 Standard reports 0.765 and 0.827 respectively. On MathlibMPR, the reported complete-route coverage at 50 candidates is 44.2% for LeanGraphSearch Reasoning and 41.3% for LeanSearch v2 Reasoning. LeanSearch v2 separately reports 46.1% of annotated premise groups covered within 10 candidates on its 69-query reasoning benchmark. “Premise-group coverage” and “complete-route coverage” are not interchangeable denominators.

TheoremGraph's informal edges are also quality-stratified: its page reports estimated precision of 98.8% for deterministic edges, 76.6% for heuristic edges and 42.7% for notation-derived edges, with a 68.1% aggregate estimate. This reinforces the PITB rule that external or heuristic edges remain candidate evidence until independently corroborated.

## Paradigm shifts

### 1. From proof pass/fail to an evidence lifecycle

Represent each task as an explicit state machine:

`discovered → snapshot-pinned → source-reconstructed → goal-reproduced → specification-audited → retrieved → planned → proof-candidate → kernel-checked → semantics-checked → repair-accepted → maintenance-tested`

Every transition produces a receipt. A task may be counted in a given metric only at the corresponding stage. A failure at source reconstruction, target fidelity, vacuity, kernel checking or maintenance remains visible and cannot be hidden by a later aggregate success rate.

**Why it matters:** the 2026 benchmark audit of Lean corpora reported 4,833 findings, including 398 mechanically certified defects such as counterexamples, vacuous statements and unsound axioms. A proof kernel checks the formal proposition; it does not establish that the proposition faithfully encodes the intended task.

### 2. From ranked premises to typed proof routes

Retrieval should return provenance-bound candidates, but planning should operate over typed AND/OR bundles:
- candidate edge: useful search hint, not a proof dependency;
- observed typed edge: independently extracted Lean relationship;
- sufficient route: a set of declarations that closes the exact target under a frozen environment and budget;
- minimal route: no proper subset succeeds under the same experimental policy;
- antichain: all inclusion-minimal sufficient routes.

The support-antichain analyzer distinguishes one proof term's imports from declarations common to every observed minimal route. The new retrieval-route coverage join now measures whether a returned candidate bundle covers an entire minimal sufficient route, while requiring an explicit graph-node-to-declaration crosswalk. Its paired comparison reports same-query discordances, exact McNemar tests and component-clustered bootstrap intervals; a one-cluster sample deliberately receives no interval.

Primary retrieval-to-proof measures:
- Recall@K and nDCG@K (diagnostic only);
- complete-route coverage@K;
- useful-route-premise rate (route-membership precision, not baseline novelty);
- minimal-route recovery;
- baseline-relative novel premise rate only when an exact matched candidate-set difference is available;
- verified target completion after a fixed downstream proof budget;
- cost per unique assurance-adjusted task solved.

### 3. From static benchmarks to two-clock evaluation

Keep two complementary clocks:
- **frozen clock:** immutable public historical snapshots for reproducible paired comparisons;
- **fresh clock:** sequestered tasks from later repository commits/crawls for contamination resistance and real-world transfer.

Never train or tune on hidden fresh tasks. Record when a task first entered public history and whether its relevant source/solution existed before model release.

### 4. From a single solver to a portfolio controller

A system should allocate resources across retrieval, elaboration, proof search, repair and independent checking. Decisions should use calibrated uncertainty and marginal expected value, not only one opaque confidence score.

Research questions:
- When should the system switch from semantic search to graph traversal?
- When does a proof-route bundle beat top-K flat retrieval?
- When should it abstain rather than exhaust a budget?
- What is the cheapest checker sequence that still satisfies the assurance contract?
- Which failure modes justify another attempt, and which indicate a stale/invalid task?

### 5. From proof-only scoring to proof-or-refutation on implementation-derived tasks

For software properties, a useful prover must distinguish at least four terminal states: `proved`, `refuted`, `unresolved`, and `invalid_translation_or_task`. A correct counterexample is not a failed attempt; it is a validated result that the proposed property is false of the frozen implementation.

The EasyProver/Aegis pipeline provides a concrete design pattern: freeze the Rust implementation and generated Lean statement before the proof agent sees it; translate through Aeneas; differentially validate Rust/Lean behavior; and label the statement or its negation with a kernel-checked proof. The Aegis paper reports 2,045 validated Rust–Lean pairs and 11,533 candidate specifications, with 1,054/2,000 sampled statements proved within 30 minutes by EasyProver.

PITB should make this a separate software-verification track:
- immutable Rust source and compiler/toolchain identity;
- exact translated Lean program and translation digest;
- differential tests or executable counterexamples binding Rust and Lean behavior;
- a frozen property statement with declaration signature fixed before proving;
- proof-or-refutation receipt, including the exact proposition and its polarity;
- independent kernel replay, axiom policy, target-hole/axiom absence;
- semantic classification of missing hypotheses, vacuity and translation mismatch.

A proof of the Lean translation is not automatically a proof of the source program unless the translation correspondence is separately established. Conversely, a counterexample in the translated model is useful only when its input and observable behavior reproduce against the pinned source implementation.

### 6. From one-shot correctness to proof maintenance

A real engineering proof is not only correct once. It must survive upstream dependency changes and project evolution.

Measure:
- proof survival under non-semantic source updates;
- proof breakage under dependency/API changes;
- repair success within fixed budgets;
- maintenance latency;
- reviewer/maintainer acceptance of proposed source changes;
- regression rate after merge.

This becomes the “SWE-bench for proof obligations” dimension of PITB, with exact source patches and independently checked builds.

The proof-search runtime should also be treated as a controlled portfolio, not one serial agent loop. EasyProver's current design combines recursive isolated contexts, asynchronous returns, parallel tactic probing, independent final acceptance and an elastic elaboration service; its paper describes per-process concurrency limits of 64 active agents and 16 expensive Lean queries. PITB should reproduce the architectural idea under a **fixed total CPU/GPU/token/wall-time budget**, with early-accepted proof cancellation and retained branch receipts. Comparing raw agent counts or reflection turns would be misleading when parallelism and elaboration resources differ.

## Experimental protocol

### A. Freeze the comparison

For every arm, pin:
- exact task split, task order and source visibility;
- exact repository commit, Lean toolchain and Lake manifest;
- model/checkpoint and strategy implementation digest;
- query formulation, retrieval candidate pool and reranker config;
- total wall/CPU/GPU, token and monetary budget;
- number of attempts, reflection turns and stopping policy;
- checker and semantic-audit policy.

Run every arm against the same task IDs. Use a paired per-task analysis and cluster confidence intervals or bootstrap by repository and source file, not an IID bootstrap over individual tasks. For multiple retrieval arms, compare paired task outcomes and use cluster-aware uncertainty.

### B. Primary outcomes

1. **Unique assurance-adjusted task coverage**: unique tasks with a fully qualifying proof receipt divided by eligible tasks.
2. **Assurance-adjusted Pass@1**: one attempt per task, all source/proof/semantic gates required.
3. **Cost per unique verified task**: total cost, including failed attempts and infrastructure failures, divided by unique tasks solved.
4. **Proof-route sufficiency**: fraction of proposed/retrieved bundles containing a route independently shown sufficient under the frozen ablation policy.
5. **Semantic rejection rate**: fraction of kernel-accepted candidates rejected by source/specification/vacuity/axiom gates.
6. **Maintenance survival and repair rate**: proof validity after pinned upstream changes and successful repair within budget.
7. **Coverage/abstention and calibration**: report how often the system declines to claim a result and whether its route-sufficiency probabilities are calibrated.

Keep infrastructure failures and invalid targets in separate denominator views. Publish both intention-to-treat coverage and conditional proof success after a task is reproduced; never remove hard failures silently.

### C. Qualification thresholds (proposed, not achieved)

| Area | Pilot qualification | Strong target |
|---|---:|---:|
| SorryDB historical frozen split | At least 60% unique assurance-adjusted task coverage on the same 1,000 tasks, compared with the published 57.2% individual Pass@1 reference under a carefully matched budget | Statistically supported improvement with repository/file-clustered intervals; separately exceed 63.2% combined public union coverage without multiplying attempt budgets |
| SorryDB efficiency | Establish full cost and wall-time baseline | At least 25% lower cost per unique verified task without reducing coverage |
| AutoformBench | 299/341 or more (first integer clearing +5 pp over 281/341) | 316/341 or more (first integer clearing +10 pp) in the same source-visibility condition |
| FVSpec | Reproduce the 200-task easy/hard baseline protocol and semantic-audit a sample | Improve hard-set verified coverage while maintaining source/spec translation fidelity; report results per property family |
| MathlibQR / MathlibMPR retrieval | Reproduce the 810-query and 69-query task/split/protocols; match LeanGraphSearch's 0.794 Recall@10 / 0.947 Recall@100 and 44.2% complete-route@50 figures on the same definitions | Exceed the strongest matched baseline with paired query-level results and component-clustered intervals; then show higher exact-target proof completion under one shared prover budget |
| PITB support-antichain route retrieval | Establish no-retrieval, BM25, LeanSearch v2, Lean Finder V2, LeanGraphSearch, DAPS and union baselines using exact node-to-declaration crosswalks | Statistically supported increase in complete sufficient-route coverage and unique assurance-adjusted solves per cost; do not substitute MathlibMPR curated-route coverage for ablation-measured sufficiency |
| Proof maintenance | Establish survival/repair baseline on pinned updates | 90% of non-semantic updates survive unchanged or are repaired within a declared budget |
| Integrity | 100% exact bytes, task membership and checker receipts for all scored runs | 100% source reconstruction and semantic gate coverage for every claimed assurance-adjusted success |

The 60% SorryDB target is a proposed pilot gate, not a claim that a 2.8 percentage-point uplift is statistically significant. The published public leaderboard's model settings and budgets must first be matched or clearly qualified.

## Immediate execution order

1. Qualify the existing SorryDB data/split auditor and optional proof-run receipts on exact-head CI.
2. Integrate retrieval receipts from DAPS, Lean Finder V2, LeanGraphSearch, LeanSearch v2 and BM25.
3. Join retrieval candidates to typed graph evidence without promoting external candidate edges.
4. Run a small frozen task pilot with no-retrieval, individual retriever and union arms, same downstream solver and budgets.
5. Select a tractable subset for exhaustive support-antichain ablations and compare predicted routes against measured sufficiency.
6. Add semantic audit gates and proof-maintenance updates before reporting any public “assurance-adjusted” score.
7. Only after a reproducible pilot should we spend on large-scale model sweeps.

## Non-claims

The current Rust tools qualify data contracts, graph/receipt integrity and synthetic controls. They do not yet demonstrate a new real-world proof-solving gain, an independent semantic audit of every benchmark task, or a cross-benchmark model win. The next performance claim must be based on a fully pinned, paired evaluation and retained proof/checker receipts.

## Source references

- SorryDB paper: https://proceedings.mlr.press/v306/letson26a.html
- SorryDB public leaderboard: https://sorrydb.org/
- AutoformBench Meta release: https://ai.meta.com/research/publications/autoformbench-a-lean-formalization-benchmark-for-advanced-mathematics
- AutoformBench repository/results: https://github.com/facebookresearch/autoformbench
- FVSpec leaderboard: https://fvspec.galois.com/
- FVSpec paper: https://arxiv.org/abs/2606.01008
- Lean Finder V2 paper: https://arxiv.org/abs/2610.30843
- LeanGraphSearch paper: https://arxiv.org/abs/2610.22339
- Faults in Our Formal Benchmarking (ICML 2026): https://proceedings.mlr.press/v306/ammanamanchi26a.html
