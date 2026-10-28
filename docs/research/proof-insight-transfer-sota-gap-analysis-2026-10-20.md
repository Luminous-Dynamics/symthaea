# Proof Insight Transfer: state-of-the-art gap analysis and next experiments

**Date:** 2026-10-22  
**Status:** literature-informed research plan; the figures below are published reference results from their stated setups, not PITB results. **Symthaea has no measured insight-transfer result yet.**

## Executive decision

The differentiated research question is not “can a retriever find a similar proof?” Strong premise retrieval already helps Lean proving. It is:

> Can a proof-derived, typed strategy—with explicit preconditions, state/obligation change, evidence, and predicted failure boundary—cause more **verified exact-target successes** on a previously unseen mathematical family than the strongest matched retrieval baseline?

The most important missing comparator was a strong global-premise retriever. PITB now preregisters arm I alongside A–E for validation-only baseline selection. The primary outcome remains F (typed insight) minus the strongest preregistered validation-selected baseline, with semantic-gate success as the endpoint and no claim until all gates pass. This is a pre-outcome amendment; re-hash the current protocol bytes into any future run manifest.

## What the field already does well—and where PITB must be stricter

### Retrieval is a serious baseline, not a straw man

LeanSearch v2 reports **nDCG@10 = 0.62 vs. 0.53** for the next-best system on its single-query retrieval benchmark. On its 69-query research-Mathlib premise-group benchmark, its reasoning mode recovered **46.1%** of ground-truth premise groups within ten candidates, compared with **38.0%** for strong reasoning retrieval and **9.3%** for the listed premise-selection baselines. Its fixed-prover downstream experiment reported **20%** proof success with LeanSearch v2, **16%** with the next-best retriever, and **4%** without retrieval ([paper](https://arxiv.org/abs/2605.13137); [reproduction artifacts](https://github.com/frenzymath/LeanSearch-v2)).

Those numbers are **not** comparable to PITB's eventual success rate: the task sets, model loop, corpora, budget, and outcome units differ. They do show why a weak lexical-only baseline would be inadequate. Arm I must pin the retriever implementation, source/index snapshot, candidate budget, and model/service version. If the published LeanSearch v2 system cannot serve a particular eligible target, PITB must identify the locally reproducible same-corpus substitute and not label it as a reproduction of the paper.

### Dependency-aware retrieval is another current high-water mark

LeanGraphSearch, published 15 October 2026, starts from semantically matched declarations, traverses Mathlib's forward-dependency graph to surface multi-hop supporting lemmas, then reranks the expanded candidate set. In its own evaluations it reports up to **+12% recall over LeanSearch v2 on MathlibQR**, up to **+3.7% coverage of complete annotated proof routes on MathlibMPR**, and downstream prover-loop gains of up to **40% over the vanilla prover** and up to **10% over LeanSearch v2** across its stated benchmarks ([paper](https://arxiv.org/abs/2610.22339)).

These are published reference results for their declared tasks—not PITB measurements and not directly comparable to PITB's exact-target semantic-gate outcome. They nevertheless affect baseline design. PITB's dependency-graph arm D must pin its graph construction, candidate budget, index snapshot and reranker. A locally implemented graph retriever is **not** a LeanGraphSearch reproduction unless its implementation and evaluation setup actually match. Where a compatible reproduction cannot run against PITB's pinned target collection, report that limitation and keep a separate, independently reported Mathlib retrieval track rather than silently dropping the strongest relevant baseline or merging unlike scores.

### Compilation alone is not a sufficient correctness endpoint

The ACL 2026 T² study covers **2,206 problems** from five real Lean 4 repositories with an average of 41 dependent successor theorems per problem. In its published setup, Claude Sonnet 4.5 reached **38.9% Testing Accuracy** even with natural-language proof and successor theorems supplied as context ([ACL Anthology](https://aclanthology.org/2026.acl-industry.150/); [code/data](https://github.com/ldilab/T2)). This motivates separating kernel/checker acceptance from statement correspondence, dependency/axiom audit, dependent-successor checks, and frozen mutation controls.

Successor checks are not magically semantic oracles: failures can result from unrelated build problems, and a dependent theorem may not exist for a target. PITB should retain exact receipts, distinguish relevant semantic regression from unrelated failure, and record a defensible not-applicable decision rather than inventing proxy tests.

### Formal-math coverage and proof efficiency are separate axes

TheoremBench uses nearly **100 classical theorems**, with both a main-target version and a premised version that exposes related supporting subtheorems. It reports theorem-level coverage and token efficiency, highlighting that performance on easy subtheorems and lengthy tactic traces can obscure weak full-theorem progress ([paper](https://arxiv.org/abs/2606.09450)). PITB should therefore report target-level exact success, per-family coverage, tokens/time/cost per verified solve, and complete-family coverage—not only a pooled pass percentage.

### Generalization beyond competition-style mathematics matters

VeriSoftBench contains **500 Lean 4 proof obligations across 23 real-world repositories**. Its authors report poor transfer from Mathlib-style provers to repository-centric software verification and improved performance when context is limited to curated dependency closures rather than the full repository ([paper](https://arxiv.org/abs/2602.18307); [repository and evaluation instructions](https://github.com/utopia-group/VeriSoftBench)). PITB should eventually test whether insight records help under dependency-rich context, not just on the source corpus where the insight was extracted.

### Informal research-mathematics value has different costs and scoring

First Proof Batch 3, published **14 October 2026**, tested five systems on ten research problems for 24 hours per system without human intervention; 30 expert reviewers graded correctness, exposition, and attribution. ProofCouncil was graded correct on **9/10**, while the direct GPT-6 Astra entry reached **7/10**. The reported costs were **$1,888** and **$36**, respectively, and adequate citations were judged on **7/10** ProofCouncil outputs versus **4/10** direct Astra outputs ([official press release](https://1stproof.org/third-batch-press-release.html); [benchmark page](https://1stproof.org/third-batch.html)). This is a distinct, human-reviewed informal-proof setting, not a Lean pass-rate baseline. It does establish that abstention, citation provenance, exposition, and resource cost deserve explicit measures.

### Open proof generation has moved beyond older MiniF2F reference numbers

The ICLR 2026 Goedel-Prover-V2 paper reports **84.6% pass@32 on MiniF2F for its 8B model**, and **88.1% pass@32** for its 32B model in standard mode, rising to **90.4%** with its self-correction setup. It also reports 86 PutnamBench problems solved at pass@184 for its flagship model ([ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html)). These results supersede older 26.5%-class MiniF2F figures as a useful description of the open proof-generation frontier.

**This is not a direct PITB baseline.** MiniF2F pass@32 measures generated formal proofs under a particular task set and search budget. PITB's primary outcome is causal utility from transferring typed proof insights to exact held-out targets under matched treatment/control arms. PITB must not compare its future solve percentage to Goedel's MiniF2F score as if they shared a denominator.

The comparison plan therefore has two distinct tracks:

- **Proof-generation strength:** on a frozen common theorem set, compare the strongest reproducible prover and the same prover with/without the insight intervention, under the same wall-clock, token, checker, and test-time-search budgets. If a full Goedel-Prover-V2 reproduction is infeasible for the pinned corpus, state that explicitly and do not label a substitute as a reproduction.
- **Transfer-mechanism utility:** retain the preregistered retrieval and proof-context arms, select baselines on validation families only, and estimate the paired difference between the typed-insight arm and the strongest eligible baseline on confirmed exact-target semantic-gate success. Report a clustered confidence interval and all exclusions/unknowns.

The 2026 result strengthens the case for self-correction and test-time search as controls, not just as competitor names. It also makes verifier cost and search budget part of the benchmark identity: any gain attributed to insight transfer must survive a matched budget and cannot borrow extra search for one arm.

## Paradigm shifts worth pursuing

1. **From similarity search to executable strategy contracts.** Represent each insight as a structured operation: applicable object/domain, preconditions, proof-state/obligation transformation, expected dependency effect, evidence locator, and known failure modes. A candidate is only a hypothesis until its preconditions are checked and target-side proof gates pass. Long-term, compile the schema into small typed tactic or proof-plan combinators that the kernel can reject—not into trusted natural-language assertions.

2. **From correlation to intervention.** Compare the typed insight treatment against raw proof, lexical/AST/graph retrieval, a strong premise retriever, and insight-plus-proof. Keep task, model, checker, context, and budget matched. The measured question is whether *presenting the strategy record causes incremental solved-target utility*, not whether successful tasks happen to have nearby proofs.

3. **From “positive transfer only” to calibrated negative transfer.** The system should predict incompatible transfers and explain which prerequisite fails (object type, domain, inequality direction, unavailable premise, quantifier scope, dependency, or precondition). Publish precision/recall and coverage-risk curves; a system that rejects every candidate must fail positive-control recall. Confidence must be calibrated against adjudicated labels, not just similarity scores.

4. **From compilation to an evidence ladder.** Store separate states for parse/typecheck, exact statement checked, axiom/dependency policy, statement correspondence, dependent successors, mutation detection, and independent review. No higher state is inferred from a lower state. Every claim should link to content-addressed receipts and exact toolchain versions.

5. **From single-split scores to adversarial generalization.** The family split is a useful start, not a contamination guarantee. Add normalized statement/proof fingerprints, cross-family dependency and near-duplicate edges, source-to-target leakage reports, exact retrieval logs, and a later or access-controlled external challenge set. Retain unknown exposure as unknown.

6. **From raw solve rate to useful verified work.** Measure verified solves per fixed dollar / wall-clock hour, median and tail latency, tokens per verified solve, downstream regression, proof portability/maintenance, and abstention quality. Do not trade a lower correctness rate for high raw throughput.

7. **From explanatory annotation to causal intervention.** A strategy description may correlate with proof success without causing it. The secondary experiment therefore compares a full typed insight to a high-similarity but adjudicated-incompatible decoy, then removes preconditions, failure boundaries, or evidence anchors one component at a time. Mechanically derive every intervention from the same frozen record, keep task/model/checker/budgets fixed, match input tokens, and compare paired exact-target semantic-gate outcomes. If context-token matching is not exact, downgrade causal claims to exploratory. The new transfer-selectivity schema and Rust analyzer now separate gold adjudications from system decisions, reject reviewer/component inconsistencies, and compute false-accept/false-reject rates, compatible recall, abstention, selective risk, and component-clustered intervals. A distinct compatibility-probability field enables Brier/ECE/reliability reporting without equating an accept/reject decision to a calibrated probability; probability coverage and cluster-bootstrap intervals are emitted, but calibration remains descriptive until a decision threshold is separately preregistered. This is evaluator implementation, not evidence that any component has yet improved proof performance.

8. **From trusting benchmark rows to auditing benchmark validity.** A Lean kernel proves a formal statement, not that it faithfully encodes the intended mathematical problem or that the harness resists vacuous/loophole solutions. The June 2026 audit *Faults in Our Formal Benchmarking* reports 4,833 findings across 10,318 problems in five benchmark families, including 398 mechanically certified issues such as counterexamples, vacuity, and unsound axioms ([paper and audit artifacts](https://paperswithlean.com/paper/2606.29493.html)). This means benchmark qualification must itself be an auditable result: run static defect checks, inspect assumptions and axioms, test plausible trivializations, map the informal claim to the exact Lean statement, and publish every excluded/unknown item. A passing proof must not erase a defective target specification.

9. **From random held-out splits to two distinct contamination-resistant challenges.** Add (a) temporal repository replay where the exact theorem statement and proof availability are evidenced at target time and the solution appears only later in source history, and (b) proof-verified rewriting where a mathematically equivalent task variation is re-formalized and kernel checked. RePro (August 2026) reports that its *retained* rewritten GSM8K/MATH instances reached 100% well-definedness, feasibility, and reference-answer correctness in its study ([paper/code](https://paperswithlean.com/paper/2609.00062.html)); this motivates verified rewrites but does **not** establish absence of training contamination or directly validate research-math theorem transfer. Report performance on original versus rewritten forms, semantic equivalence review, and unknown exposure separately. Keep the temporal and rewritten challenge sets disjoint from development and calibration.

10. **From domain-narrow score chasing to structured transfer coverage.** TheoremBench shows that premise exposure and subtheorem difficulty can alter reported theorem-level success ([paper](https://arxiv.org/abs/2606.09450)); VeriSoftBench reports a gap between curated dependency context and raw repository context on 500 obligations across 23 projects ([benchmark](https://utopia-group.github.io/VeriSoftBench/)); CAM-Bench targets 1,000 Lean problems in computational and applied mathematics ([paper](https://arxiv.org/abs/2605.17255)). Report results by domain, dependency depth, target difficulty, source-target family distance, and context regime. PITB should establish value on mathematical transfer itself, then demonstrate portability across repository-scale verification and applied mathematics—not pool these tasks into an incomparable headline score.

11. **From single-system insight extraction to portable proof-strategy contracts.** ITPEval evaluates statement and proof translation across Lean 4, Rocq, Isabelle, and HOL Light using 1,560 source files and 6,848 theorems. Its reported statement-translation pass@1 peaks at 29.1% and proof-translation pass@1 at 10.5%; a separate definitional-equality check finds only 54.0% equivalence among one translated subset that had already passed Lean checking ([ITPEval](https://arxiv.org/abs/2607.19407)). PITB should not claim cross-system portability now, but its insight record can eventually serve as an intermediate representation: preserve mathematical preconditions, proof-state transformation and failure boundaries, then separately compile or map the strategy into each prover's native context. The acceptance criterion must be target-kernel verification plus statement correspondence in each system, not textual or syntactic similarity. Start with the Rust analyzer's current Lean-only evidence format and treat cross-prover realization as a later external track.

12. **From isolated theorem success to theory-scale compositionality.** LCS-Bench reports 327 textbook items, over 4,076 Lean declarations and 85K-plus lines, with state-of-the-art model performance of 20.1% on its auto-formalization tasks ([LCS-Bench](https://arxiv.org/abs/2606.26525)). This suggests a second long-horizon challenge for strategy transfer: can insights be composed across a dependency graph while definitions and local assumptions remain stable? Do not compare its auto-formalization score with PITB's exact-target proof success. A future pilot should measure obligation closure, definition drift, dependency reuse, regression and proof maintenance over a multi-theorem unit.

## A measurement metric the field often underreports: temporal resolution coverage

Before any proof model is run, publish a **task-resolution funnel** alongside solve rates:
1. raw source rows,
2. safe canonical-path joins,
3. exact declaration-name matches,
4. proof-transition matches (dataset proof appears inside the named declaration at the completion commit and is absent from every direct parent),
5. exact statement stability at the pre-state and completion commit,
6. buildable/replayable target states,
7. eligible scored targets.

These are separate evidence gates, not an aggregate quality score. A benchmark that starts with only its easiest-to-resolve rows can make proof accuracy appear high even if its task mapping is weak. PITB's new temporal dataset auditor maintains path-only counts as a baseline diagnostic and adds a second, declaration-scoped theorem/proof-transition join; it only calls a join stable when statement text matches at both historical points and the proof was absent at target time. That is a **textual provenance gate**, not a semantic equivalence oracle: it deliberately leaves mismatched or ambiguous cases unresolved instead of promoting them into the denominator.

For external reporting, give both the raw corpus coverage and the eligible-task solve rate. Also report exclusions and unknowns by reason. If the strict identity join resolves only a small fraction of the corpus, increase the annotated sample or improve source mapping before making comparisons; never remove ambiguous rows from the published denominator.

## Proposed benchmark scorecard

The current preregistered primary gate is unchanged unless another explicit pre-outcome amendment is committed.

| Measure | PITB acceptance target | Why it matters |
|---|---|---|
| Primary exact-target utility | F improves semantic-gate success by at least **+10 percentage points** over the strongest validation-selected baseline, with the family-clustered 95% CI wholly above zero | Evidence of practically meaningful incremental transfer |
| Independent sample floor | At least **20 eligible independent test-family clusters** even to consider a confirmatory claim; plus simulation-based power planning for the actual family-size distribution | Prevents a tiny pilot from masquerading as confirmation; 20 is a project-specific floor, not a power guarantee |
| Completeness of test accounting | Every frozen test family represented by eligible, excluded, or unknown rows; 0 silent family omissions | Prevents denominator shrinkage |
| Exact-target / semantic validity | Report both checker pass and full semantic-gate pass; require zero disallowed axioms, statement-mismatch regressions, or unreviewed “verified” claims | Compilation is not the outcome |
| Negative controls | **100% rejection** of the frozen finite incompatibility/mutation suite | Guards known failure classes; does not prove universal correctness |
| Positive-control behavior | Publish precision/recall and false-rejection rate; proposed initial target **≥90% recall** on expert-labelled compatible positives while meeting the negative gate | Avoids trivial reject-everything systems |
| Reproducibility | Two identical runs over the pinned source snapshot yield byte-identical index/split artifacts; every scored run binds protocol, split, inputs, checker outputs, and configuration hashes | Reconstructable evidence |
| Efficiency | Tokens, time, cost and model calls per attempted target and per verified solve, arm by arm, under matched task budgets | Shows whether the method is useful, not merely more expensive |
| Insight fidelity | Two blinded annotators; report Krippendorff's alpha, target **≥0.80** for confirmatory subjective labels | Tests whether the explanation corresponds to actual proof structure |
| External replication | Repeat the registered contrast on at least one separately sourced, contamination-audited challenge suite; use a second independent suite before making broad SOTA claims | Separates an internal result from generalization |
| Negative-transfer calibration | Publish transfer precision/recall, calibration and coverage-risk; include cost of false acceptance explicitly | Mathematical analogies can be damaging when directions or assumptions differ |

The ≥90% positive-control recall target is now included in the protocol's **secondary transfer-selectivity track** by a pre-outcome amendment dated 2026-10-21; that track also requires at least 20 independent pair-graph components in each of the compatible and incompatible classes before a confirmatory transfer-selectivity claim. These are acceptance targets, not measured results. The external replication milestones remain proposed and still require a separately frozen challenge-set design. Hash the current protocol bytes into any future run config; do not reuse a digest from an earlier protocol revision.

## Baseline matrix and honest SOTA language

- **A–E:** no retrieval, BM25, typed syntax, dependency graph, raw proof retrieval.
- **F:** typed insight record only (primary treatment).
- **G:** insight plus proof context (complementarity analysis).
- **H:** HDC/embedding retrieval (ablation; no assumed benefit).
- **I:** strong premise retrieval, preferably the pinned LeanSearch v2 implementation on targets it supports, or a named same-corpus dependency-aware/global premise retriever where it does not.

Choose the strongest comparator from A–E and I on validation semantic-gate success only. Evaluate G and H as separate secondary ablations; do not swap them into the primary contrast after seeing test results. For external leaderboard context, rerun published baselines on the same target statements, checker, token/context cap, and cost policy where possible. Otherwise call them *related work*, not comparable scores.

A result can establish “improves over the strongest registered baseline on this frozen dataset” without establishing “state of the art.” The latter needs an apples-to-apples comparison, an external challenge set, contamination caveats, and independent reproduction.

## Execution order (do not wait for the full corpus experiment)

### P0 — make the measurement system trustworthy

1. Pass Rust fmt, isolated crate tests, strict Clippy, schema positives/negatives, and deterministic source-index reproduction.
2. Independently compare the generated family split and source fingerprints against the pinned manifest.
3. Finish the eligibility/audit schema for exact Lean declaration, toolchain, dependencies/axioms, statement correspondence, successor tests, and mutation receipts.
4. Check the run analyzer's fail-closed guards with synthetic fixtures only; label these as implementation tests, not mathematical evidence.

### P1 — build a small, auditable training/validation corpus

1. Manually inspect training-family artifacts only. Preserve the split and mark missing/unclear formalizations as excluded or unknown with reasons.
2. The initial training-only library now includes family 025's greedy-residue decomposition and family 026's eventual-inclusion / positive-linear-count density-transport strategy. Both are static source annotations only: no transfer has been attempted, and their license status remains `needs_review` until confirmed.
2. Require two reviewers for the first strategy records. Record disagreement rather than silently adjudicating it away.
3. Add simple direct negative-transfer pairs (such as an upper-bound insight offered for a lower-bound goal) and compatible positive controls before any model comparison.
4. Select the retriever, prompt/configuration, checker, task budget, validation baseline and statistical seed using validation families only.

### P2 — run a low-cost pilot without test tuning

1. Exercise the full input-to-receipt pipeline on training/validation examples and synthetic controls.
2. On the held-out test partition, run the frozen arms once under identical caps, retaining every attempted/not-attempted/tool-unavailable/timeout/excluded state.
3. Produce a paired per-family analysis report with all raw denominators, confidence interval, cost and time.
4. If there are fewer than 20 eligible family clusters or simulated power is inadequate, publish an underpowered pilot and collect an independent set; do not relax thresholds.

### P3 — establish generalization

1. Replicate with a separately selected challenge set, audited for cross-family theorem equivalence and retrieval contamination.
2. Include dependency-rich repository-scale formal verification, not only isolated Mathlib statements.
3. Add an informal research-proof track with blind grading of correctness, exposition and attribution; never mix those scores with Lean exact-target success.
4. Only after comparable baseline reruns and an independent replication should the project make a broad SOTA claim.


## 21 October 2026 update: formal infrastructure and the critic are separate frontiers

Two useful benchmark lines expand the comparison surface beyond isolated proof completion.

**AutoformBench — infrastructure construction, not just proof search.** Released 13 October 2026, it contains 341 Lean 4 formalization tasks from 24 advanced-mathematics sources, with human-formalized and audited targets; the solver must also construct missing definitions, intermediate lemmas and proof infrastructure ([benchmark report](https://ai.meta.com/research/publications/autoformbench-a-lean-formalization-benchmark-for-advanced-mathematics/)). This exposes a limitation of any transfer score that assumes the target's formal scaffolding already exists. PITB should keep its current exact-target task definition intact, then add a separately reported challenge stratum where a solver must build the formal context before the target proof can be checked.

For that stratum, measure the full chain: intended-statement fidelity, definition/helper-lemma construction, final target proof, dependent-successor tests, and resource cost. Track whether the retrieved insight reduces the missing-infrastructure gap or merely helps after that gap has been manually removed. Count incomplete scaffolding as a failure/partial outcome, not as a proof success. Audit provenance, licensing and task eligibility before adapting external artifacts; don't assume that a benchmark page alone grants permission to republish its data.

**FormalRewardBench — proof judgment, not theorem solving.** Its 250 Lean preference pairs include five curated error-injection strategies; the paper reports 59.8% for its best frontier LLM evaluator and 24.4% for specialized theorem-prover models in the studied setting ([paper](https://arxiv.org/abs/2605.10141)). These scores cannot be compared numerically with PITB. The key methodological lesson is to evaluate the *critic* independently: can it recognize when an insight's preconditions fail or a seemingly minor theorem mutation invalidates the move, even when the text remains plausible?

### Updated benchmark architecture

Use a benchmark ladder, not a single blended leaderboard:

| Track | Capability under test | Required evidence |
|---|---|---|
| A. Retrieval/selectivity | Retrieve compatible strategies and reject hard negatives | Dual-reviewed labels, leakage-safe components, false-accept/false-reject/abstention and calibration |
| B. Exact-target transfer | Does the strategy improve proof completion on the same target? | Paired arms, target-side kernel receipt, statement correspondence, dependent successors and mutations |
| C. Infrastructure construction | Can it create missing definitions and supporting lemmas? | Vetted informal target, audited formal statement, complete helper-declaration trace, final proof and successor tests |
| D. Proof criticism | Can it detect a strategically invalid move or a broken precondition? | Matched minimal counterfactual pairs, independent labels, error-class recall and false acceptance |
| E. Repository / portability | Does it work in real dependency-rich projects or another prover? | Pinned project context, dependency closure, native checker, maintenance/regression tests and per-system semantic correspondence |

Keep all five scores separate. AutoformBench informs Track C; FormalRewardBench informs the adversarial-pair methodology for Track D; neither is an apples-to-apples SOTA baseline for Track B. The strongest comparison remains a rerun of current retrieval/prover baselines on a common target set under matched context, checker and budget.

**Immediate experiment after the integrity gate passes:** select a small, independently reviewed pilot from an eligible advanced-mathematics or repository task source; run full insight, surface-matched incompatible decoy, precondition ablation and no-insight control on identical targets; retain every missing-context, timeout and construction failure. First validate the complete receipt pipeline and token/accounting controls. Do not promote a pilot into a confirmatory or SOTA claim until component counts, power, leakage audit and the frozen analysis gate are satisfied.

This is a new evaluation design and research hypothesis, not a measured PITB gain. No transfer result, AutoformBench score reproduction, or SOTA claim is implied.


## 22 October 2026 update: verified-progress telemetry, without proxy-score gaming

Hard-mode theorem proving and learning-to-rank proof steps point to a missing measurement layer. DAP separates answer discovery from formal proving ([ACL 2026](https://aclanthology.org/2026.acl-long.3/)); OptProver's preference objective penalizes valid but non-progressing steps ([ICML 2026](https://proceedings.mlr.press/v306/li26ek.html)). Neither directly establishes that a proof-insight retrieval system benefits from intermediate-state metrics, but both motivate testing the gap between *doing formal work* and *making verified progress*.

PITB now has a draft `proof-progress-trace.schema.json`, a synthetic trace, and CI-side relational checks. Trace records distinguish obligation instance IDs from formula digests, retain event-level resource counters and receipts, and represent splits/refinements separately from receipt-backed discharges. The checker rejects orphan obligations, duplicate closures, non-monotonic counters, over-budget events, and a claimed verified success without kernel/semantic-gate/successor receipts. A synthetic case splits one root into two child goals and then verifies both leaves; the split itself is not counted as a closure.

### How to use this without corrupting the benchmark

- Keep exact-target semantic-gate success as the primary endpoint.
- Use trace diagnostics to investigate *why* a strategy helped or stalled: receipt-backed leaf closures per token, time/tokens to first verified closure, unresolved leaf fraction and no-closure event distribution.
- Keep speculative search branches separate. A future branch-aware trace must distinguish tentative closures from closures on the retained proof trajectory.
- Validate telemetry against the proof assistant/checker boundary; do not trust self-reported model narration about solved goals.
- Do not claim telemetry makes a model better. It is instrumentation and evaluation design until a matched intervention shows a repeatable improvement in verified results.

This creates a potential research contribution beyond another retrieval index: a common, content-addressed record of strategy use, proof-state lineage, verified intermediate progress and terminal semantic outcome. It can enable causal analysis and better credit assignment while refusing to use intermediate activity as a substitute for correctness.


## 22 October 2026 update: audited counterfactuals and current prover frontier

### Why strategy applicability should be tested counterfactually

Pythagoras-Prover introduced Augmented Lean Formalisation (ALF), generating statement variants and a contamination-sensitive MiniF2F-ALF benchmark; the paper reports accuracy drops for all evaluated models on that set ([paper](https://arxiv.org/abs/2606.12594)). This is evidence that benchmark variation matters, but generated variants are not by themselves proof that a particular strategy's applicability changed correctly. PITB's next step is narrower and more auditable: pair an independently reviewed compatible base target with a single declared counterfactual edit, and require adjudicators to verify whether that specific edit preserves or breaks the insight's prerequisites.

The new `counterfactual-precondition-suite.schema.json` formalizes base/variant statement digests, mutation-fragment hashes, mutation receipts, source insight digests, dual adjudications, and stable join IDs for both eventual system outcome rows. Its seven-pair fixture covers a semantics-preserving control plus changed precondition, inequality direction, object type, quantifier scope, premise, and predicate. The fixture is clearly synthetic; it is not a mathematical dataset or evidence about an actual model. The current workflow validates 13 negative controls covering duplicate IDs, statement-digest collisions, faulty mutation descriptions, reviewer failures, family-component/split leaks, and unknown-label misuse.

For a real study, measure **paired contract adherence**: the system must handle the compatible base correctly *and* reject or abstain on the individually verified incompatible counterfactual. Alongside that, report positive recall, false acceptance by mutation type, false rejection, and stability on semantics-preserving variants. Do not assume that reversing an inequality or removing an assumption makes every theorem incompatible; reviewers must establish that each actual mutation crosses the selected insight's applicability boundary. Run matched model/checker/budget conditions, preserve raw output rows and receipts, and keep unresolved judgments unknown.

### Industry frontier: compare the right task

OProver is an important open-weight whole-proof reference. Its paper/released repository reports OProver-32B Pass@32 of 93.3 on MiniF2F, 58.2 on ProverBench, and 11.3 on PutnamBench; it uses verified-proof retrieval, iterative compiler-feedback repair, and a train-time loop incorporating verified proof and repair trajectories ([paper](https://arxiv.org/abs/2605.17283); [official implementation](https://github.com/multimodal-art-projection/OProver)). Those numbers are whole-proof Pass@32 scores; they are **not** transfer-selectivity or counterfactual sensitivity results.

Pythagoras-Prover reports 93.0% MiniF2F-Test at 32B, and also studies generated statement variants with MiniF2F-ALF ([paper](https://arxiv.org/abs/2606.12594)). OptProver targets Olympiad-to-undergraduate optimization transfer and penalizes valid but non-progressing steps ([ICML 2026 paper](https://proceedings.mlr.press/v306/li26ek.html)). Process-Verified RL explicitly uses Lean elaboration for tactic-level verified feedback during training ([ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/554a5982e26d0129f4038ad7a68fb3aa-Abstract-Conference.html)). ProofOptimizer optimizes a different objective: it reports proof-length reductions of 87% on MiniF2F, 57% on PutnamBench and 50% on Seed-Prover's IMO 2025 proofs ([ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/60917970c4257e50aff60d1d7bde2edf-Abstract-Conference.html)).

These are distinct strengths, not one leaderboard. PITB should not claim that an insight-transfer metric beats OProver's Pass@32. A credible comparison plan is:

- Keep matched-task exact-target verified success as PITB's primary endpoint; reproduce relevant whole-proof baselines on the same eligible targets if a suitable adapter and checker are available.
- Compare retrieval/strategy intervention arms at equal target, context, tokens, retries and cost; select the strongest baseline on validation data only.
- Evaluate the counterfactual suite independently for positive recall, false-accept by critical mutation, no-op stability and paired contract adherence.
- Only then test whether the intervention helps under missing formal infrastructure, dependency-rich repositories, or hard-mode witness discovery—each on its own denominator.

The potential contribution is not another claimed best percentage. It is a **falsifiable strategy applicability test** that connects preconditions to controlled statement changes and kernel-checked target outcomes. This could complement large whole-proof systems and proof-state feedback methods if an actual, replicated experiment demonstrates incremental value. No such value is measured yet.


## 22 October 2026 update: executable paired counterfactual analysis

The counterfactual suite has advanced from a dataset schema to a Rust outcome-join analyzer. It reads base/variant IDs from each contrast and joins them to already-recorded transfer-selectivity outcomes. It refuses missing or duplicate row links, source/target/statement drift, changed component IDs, adjudication status/label disagreement, reviewer receipt mismatches and actual resource use beyond the declared cap. The report includes per-mutation false acceptance and sensitivity, paired contract adherence, no-op decision stability, Brier score, probability shift, and component-clustered percentile intervals; its confirmatory decision remains explicitly unevaluated.

This addresses a gap left by statement-mutation benchmarks: robustness should be evaluated on matched pairs with an explicit, reviewed applicability hypothesis, not only as aggregate pass-rate degradation on automatically generated variants. The seven pairs/fourteen outcome rows checked into CI are synthetic fixtures; their high scores verify the analysis wiring, not model behavior. The first real experiment should use a small reviewed set, with two experts verifying every changed premise and target statement, and retain uncertainty or ambiguity as unknown.

A second current-industry comparison underscores why task separation is important. Pythagoras-Prover reports MiniF2F-Test 89.8% at Pass@32 and 93.0% at Pass@2048 for the 32B model, along with a MiniF2F-ALF statement-variation benchmark. These are whole-proof results under the authors' protocol, not counterfactual transfer metrics ([paper](https://arxiv.org/abs/2606.12594), [project results](https://pythagoras-lm.github.io/)). RLMEval's 613 theorems across six real Lean projects report a best pass rate of 10.3%, exposing a large gap between curated benchmarks and research-level projects ([ACL Anthology](https://aclanthology.org/2025.findings-emnlp.581/)). PITB should therefore retain separate strata for competition-style exact-target proofs, counterfactual applicability, constructive infrastructure building, and repository-scale work.

The next credible advancement is to have the real prover/checker emit matched base and counterfactual outcomes plus receipts, then compare insight treatment with lexical, typed-syntax, dependency and raw-proof retrieval at equal budgets. Until that experiment and an external replication pass, no claim of improved transfer or state-of-the-art performance is justified.


## 22 October 2026 update: retrieval must be a strong, adaptive baseline

The newest paired analyzer work makes a stronger experimental design possible, but its value depends on a fair baseline. **Do not compare typed-insight transfer only against no retrieval.** TheoremBench evaluates nearly one hundred classical theorems both with only the main goal and with extracted supporting subtheorems; its authors report that explicit premises improve performance, while its token-efficiency and theorem-coverage measures expose uneven progress across theorem structure ([paper](https://arxiv.org/abs/2606.09450)). ProofFusion's Coq study reports an average 6.89% increase in proved theorems across four provers and 17.50% more solves among previously unprovable tasks using adaptive retrieval in its own 10,782-theorem, 26-project setup ([FSE 2026 paper](https://conf.researchr.org/details/fse-2026/fse-2026-research-papers/57/ProofFusion-Improving-Neural-Theorem-Proving-via-Adaptive-Retrieval-Augmented-Reasoning)). Neither result is numerically comparable to PITB, but both make it indefensible to use weak retrieval as the only comparator.

The next paired-counterfactual experiment should freeze at least these arms on the *same* target pairs:

- **No insight / no retrieval** as a floor.
- **Strong adaptive premise or dependency retrieval** from a frozen corpus, selected and tuned on validation only.
- **Raw proof retrieval** to isolate whether gains are just exposure to more proof text.
- **Typed insight only** to test the structured strategy abstraction.
- **Typed insight + strong retrieval** to measure complementarity.
- **Surface-matched incompatible decoy** to test whether preconditions prevent over-transfer.

All arms must have equal caps and retain model/config/prompt hashes, but per-target contents and the exact retrieved items must also be persisted. Use the same target statement and checker where the contrast allows it. On a counterfactual statement variant, the base/variant pair is the unit of semantic intervention; the target statements necessarily differ by the reviewed mutation, while the model, prompt template, configuration, checker policy and resource budget remain matched.

The new outcome analyzer now binds each side to distinct target artifact paths, artifact hashes and statement hashes, plus the source insight's own artifact digest. It also requires matching model/config/prompt/budget across each pair and scans the entire suite for split/component leakage—even when an invocation scores only one partition. This prevents a common reproducibility failure: two claimed variants silently using the same artifact identity or different inference budgets. The synthetic fixtures and their idealized scores verify only this execution contract.

### Better benchmark claim hierarchy

1. **Integrity claim:** the frozen contrast, source artifacts, reviewer judgments, outcome rows and exact run configs join with no provenance or leakage violations.
2. **Behavioral claim:** the model accepts compatible bases, rejects/abstains on reviewed incompatible changes, and remains stable on semantics-preserving variants.
3. **Utility claim:** typed insights improve semantic-gate-verified target solves over the strongest validation-selected baseline, with matched budgets and family/component-clustered uncertainty.
4. **Generalization claim:** the paired result repeats on a held-out artifact bundle and dependency-rich/repository tasks with independent semantic review.

Each level needs its own receipts; passing an earlier level never implies the next one. This gives PITB a plausible research niche—measuring **applicability, not topical resemblance**—without pretending the current synthetic suite proves that insight transfer works. No measured transfer gain or SOTA result is available yet.


## 22 October 2026 update: global recovery and cost-aware proof routing

The field is moving beyond “sample more proofs and repair compiler errors” toward *global proof coordination* and metareasoning about what action to buy next. Three recent papers make this a concrete baseline requirement:

- **CoCo-Prover** uses an AND/OR proof hypergraph joined to a cross-lemma dependency graph, then routes bounded specialist actions according to expected success and cost. In its own Lean program-verification experiments, it reports a better success–cost frontier and up to 30.9% lower cost than its strongest baseline ([arXiv:2610.09681](https://arxiv.org/abs/2610.09681)).
- **Maestro** diagnoses failures and can correct false lemmas, retract an unsound subtree, and replan globally, instead of treating every failure as a local request for another proof attempt. The preprint reports solving 669/672 PutnamBench, 59/60 Lean-IMO-Bench and 6/6 IMO 2026 in its evaluation setup ([arXiv:2610.25920](https://arxiv.org/abs/2610.25920)).
- **mini-Lean-agent** shows why a minimal generic coding-agent scaffold and an independent trusted evaluator should be included as a baseline; the authors report 7/12 Putnam 2025 solves at a $2/problem cap for one model setup, and 12/12 averaging $1.28 for another ([arXiv:2610.17331](https://arxiv.org/abs/2610.17331)).

Those numbers use different models, task sets and resource policies; they are not directly comparable to PITB's counterfactual sensitivity or transfer score. The implication is methodological: PITB should isolate the incremental value of the **typed insight** from raw proof retrieval, the general-purpose agent, local repair, global failure coordination and cost-aware routing.

The proof-progress trace schema now has optional events for failure diagnosis, lemma invalidation, subtree retraction, global replanning, action routing and automation dispatch. Failure classes distinguish insufficient guidance, false lemma/specification, decomposition error, missing dependency, tooling failure, budget exhaustion and unknown; routing events require a decision receipt and record pre-action success/cost estimates. This allows a later live collector to ask whether the diagnosis was right, whether retraction/replanning recovered a verified proof, and whether the routing decision was economically justified. This is schema/CI instrumentation only—no current prover emits these events yet.

### New experiment: one task set, five orchestration arms

On the same frozen targets, checker and total budget, compare a minimal shell agent, a fixed local-repair loop, typed-insight local repair, global failure coordinator, and cost-aware coordinator. The primary endpoint remains exact-target semantic-gate success. Report the success–cost curve at several frozen spending caps, cost at fixed solve rates, verification receipt rate, recovery rate after diagnosed false-lemma/decomposition failures, and calibration of pre-action estimates. On paired counterfactuals, hold prompt template, model/config, checker policy, cache/retry rules and resource caps constant, and only vary the target by the reviewed semantic mutation.

A proposed additional engineering target is at least 25% lower cost at fixed semantic-gate solve rate relative to the strongest validated local-repair baseline, without increasing false acceptance. That target is aspirational and informed by cost-aware prover work, not a PITB result. Preserve the current +10pp paired utility gate as the separate primary endpoint. This gives Symthaea a potentially distinctive research question: not just whether a proof strategy transfers, but whether a coordinator knows *when its preconditions fail, when to abandon it, and what verified next action is worth the cost*.


## 22 October 2026 update: paired outcome analysis now includes actual resource costs

The counterfactual analyzer now requires measured runtime usage on both sides of each pair and reports calls, tokens, wall time and micro-USD separately for the compatible base and counterfactual target. It adds component-clustered intervals for cost/token/time ratios and signed variant-minus-base deltas, and the report schema distinguishes bounded rates, nonnegative means and signed paired differences. CI exercises missing-usage and paired-resource assertions as well as provenance/reviewer mutations.

This is an important guardrail, not a result. The synthetic fixture deliberately gives its adjudicated base and variant rows equal costs and matched inference budgets, so its cost ratio of 1 and zero delta only verify the reporting path. The analyzer's paired costs are not a full success–cost frontier, because the base and counterfactual are different reviewed targets. To compare with CoCo-Prover-like resource-routing work, PITB needs multiple preregistered spending caps per method and must report (i) verified semantic-gate solves at each cap, (ii) cost at fixed verified-solve rates, and (iii) uncertainty under target-family clustering. The primary proof-success and transfer-applicability scores remain separate.


## 22 October 2026 update: a preregistered multi-budget frontier, not a cost-ratio claim

A single matched counterfactual pair provides resource diagnostics but cannot establish which theorem-proving system has a better success–cost frontier: its base and counterfactual statements are different targets. To compare the global-recovery / cost-aware route ideas with industry systems, PITB now has a frozen `success-cost-frontier-plan.schema.json` and synthetic plan fixture. It requires a complete Cartesian target × method × budget schedule, at least three strictly increasing caps for calls/tokens/wall time/micro-USD, stable arm settings, unique run IDs and target-component consistency. Its synthetic validation fixture has two targets × three method arms × three budgets = 18 planned executions; it is not an executed experiment.

This puts a common design in place for comparing a minimal shell agent, fixed local recovery, typed-insight local recovery, global failure coordinator and cost-aware coordinator on exactly the same targets. Real outcomes should then yield per-arm verified semantic-gate solve curves at fixed spending caps and cost at fixed solve rates, with component-clustered uncertainty. Use the full preregistered matrix and retain timeouts, missing outcomes and tool failures; never call the top-performing budget tier a frontier using only one point. The primary +10pp verified-utility criterion remains separate from the aspirational 25% cost reduction target.

CI validates eight negative controls against the synthetic plan: missing/duplicate Cartesian cell, undeclared arm, target-component drift, non-increasing cap, duplicate run ID, too few distinct method classes and unsafe target path. These controls certify the schedule validator only. No method has been run across this matrix and no cost-efficiency gain is measured.

## 22 October 2026 update: separate reformulation robustness from semantic sensitivity

A 16 October submission, *Evaluating the Robustness of Proof Autoformalization in Lean 4*, separates global changes (style/paraphrase while meaning stays the same) from local counterfactual changes (a value, symbol or proof step changes and the output must reflect that edit). It reports variation under global paraphrase and weak fidelity to local changes in evaluated models ([paper](https://arxiv.org/abs/2606.14867)). MathAdv spans 13 undergraduate/graduate mathematics domains and includes expert-crafted transformations that test robustness to equivalent presentation ([paper](https://arxiv.org/abs/2608.25449)). These are not PITB results, but they expose a gap in a counterfactual suite containing only syntax-level no-ops and local semantic mutations.

PITB now distinguishes `no_semantic_change` (tracked fragment unchanged) from `equivalent_reformulation` (surface fragment changes, intended semantics remain equivalent after independent review). The synthetic suite adds one matched pair and outcome rows; the report exposes stability by mutation class rather than pooling the reformulation into one aggregate. All adjudications in the added pair are placeholders, so this is a contract test, not a theorem-prover robustness score.

Benchmark comparison should also include dependency-aware retrieval and cost-quality routing, not only proof-generation pass@k. LeanGraphSearch reports up to +12% recall over LeanSearch v2 on MathlibQR and downstream pass-rate gains of up to 10% over LeanSearch v2 in its reported setting ([paper](https://arxiv.org/abs/2610.22339)). An October 2026 cost-quality study reports 25.8% average cost reduction while preserving performance on a PutnamBench subset through cost-aware routing between continuing search and restarting with a new decomposition ([study](https://www.sri.inf.ethz.ch/publications/rognvaldsson2026optimizingcostqualitytradeoffagentic)). These are distinct task metrics. PITB should rerun the strongest feasible dependency-aware retrieval and resource-routing baselines on the same pinned targets and spending caps, while keeping transfer-applicability results separate from whole-proof SOTA scores.

## Research references

- Kim, Han & Hwang, [Benchmarking Testing in Automated Theorem Proving](https://aclanthology.org/2026.acl-industry.150/), ACL 2026 Industry Track.
- Wu, George & Anandkumar, [ITPEval: Benchmarking Formal Translation Across Interactive Theorem Provers](https://arxiv.org/abs/2607.19407), July 2026.
- Feng et al., [Theory-Scale Auto-Formalization of Logics for Computer Science (LCS-Bench)](https://arxiv.org/abs/2606.26525), June 2026.
- Zhang et al., [Beyond Compilation: Evaluating Faithful Natural-Language-to-Lean Statement Formalization](https://arxiv.org/abs/2606.31002), June 2026.
- Gao et al., [LeanSearch v2: Global Premise Retrieval for Lean 4 Theorem Proving](https://arxiv.org/abs/2605.13137), May 2026; [code and benchmarks](https://github.com/frenzymath/LeanSearch-v2).
- Pham et al., [TheoremBench](https://arxiv.org/abs/2606.09450), June 2026.
- Xin et al., [VeriSoftBench](https://arxiv.org/abs/2602.18307); [repository](https://github.com/utopia-group/VeriSoftBench).
- [First Proof Batch 3 official results](https://1stproof.org/third-batch-press-release.html), 14 October 2026.
- [Frozen OpenAI math source snapshot](https://github.com/openai/math/tree/1581d4cbdd69a0faf34b2089614fe2a814316a6a).

## 22 October 2026 update: Lean-calibrated counterfactual qualification

The benchmark now has a small, isolated Lean 4.34.1 calibration package in `data/benchmarks/proof-insight-transfer-v0/lean-pilot/`, alongside a case manifest that binds each pedagogical contrast to source statement fragments and named evidence declarations. The pinned workflow builds the examples, invokes Nanoda with `sorry` disallowed, and emits a content-addressed receipt covering the commit, statement-fragment digests, Lean/Lake versions, compiled artifact, source files and build log.

The calibration deliberately distinguishes three different formal situations: alpha-renamed binder expressions proven equal by reflexivity; a surface-different proposition pair with an explicit kernel-checked `↔` proof; and a premise-removal mutation whose unconditional variant is refuted by a checked counterexample. This is a useful implementation step because the relation label must be supported by a proof artifact, not inferred solely from a mutation name or source hash.

This calibration is **not** a model transfer experiment. The case manifest marks model outcomes `not_run`; no compatible-transfer recall, mutation sensitivity, exact-target utility gain or SOTA improvement follows from it. Promotion to a scored pair requires the exact-head Lean/Nanoda receipt, independent contract adjudication, linkage to actual solver outcome rows and matched budgets.

The external frontier reinforces this design choice. T² evaluates the semantic behavior of formal statements via dependent successor theorems ([ACL 2026](https://aclanthology.org/2026.acl-industry.150/)); the formal benchmark audit documents defects not detected by a kernel check alone ([Faults in Our Formal Benchmarking](https://arxiv.org/abs/2606.29493)); and recent work explicitly trains formal counterexample generation as a complement to proving ([Learning to Disprove](https://arxiv.org/abs/2603.19514)). PITB should compare itself against those capabilities as distinct tasks, not claim their results as its own. Its proposed research contribution remains causal strategy applicability: whether a typed strategy's declared prerequisites predict and govern safe transfer, and whether doing so improves exact-target, semantically qualified proof utility at a matched budget.

**Next qualification gate:** pass the exact-head Lean pilot workflow and validate its uploaded receipt; then use the pilot to test the manifest-to-outcome data path with model outcomes still unscored. Only after independent review and a powered, family-disjoint dataset exists should PITB report confirmatory mutation sensitivity or paired proof-utility results.

A follow-on beyond statement mutation is now specified in `counterfactual-mutation-adequacy.md`: replay the same Lean target into paired local proof states, then perform one legal context intervention (for example, hide a required local hypothesis) while preserving the exact theorem statement, goal type and environment. This would test whether a strategy selector tracks the actual prerequisite rather than merely reacting to statement text. It should be a separate applicability/action-selection track; a failed proof attempt in a reduced context is not evidence that the theorem is false. The state-intervention schema and synthetic CI fixture are implemented, and a small Lean elaboration example now traces an unchanged goal before/after `clear h`; a general state replayer, adjudicated dataset and model outcomes are not yet implemented or measured.


## 22 October 2026 update: fixed-theorem proof-state intervention contract

PITB now has a separate `proof-state-intervention-suite.schema.json` and four-pair synthetic fixture. It preserves the exact target theorem statement, elaborated goal type, source commit, Lean toolchain and environment digest while describing one legal local-context intervention: clear a required hypothesis, alpha-rename a hypothesis, remove an independently adjudicated irrelevant declaration, or reorder independent declarations. The schema binds base/variant state snapshots and replay receipt digests, transition operation receipts, dual-review action-policy labels, unique outcome IDs, and train/validation/test component identity.

The dedicated CI guard is designed to reject theorem/goal/environment drift, malformed single-declaration transitions, an action policy inconsistent with the changed prerequisite, reviewer disagreement represented as consensus, duplicate outcome joins, and split leakage. The checked-in fixture is synthetic and all outcome links are `not_run`. No actual proof-state replay or model behavior has been measured.

The motivation follows Lean's own metaprogramming model, where a goal metavariable has an associated local context and target type, and the tactic APIs expose both for inspection/manipulation ([Lean 4 metaprogramming guide](https://leanprover-community.github.io/lean4-metaprogramming-book/main/04_metam.html), [tactics and local-context guide](https://leanprover-community.github.io/lean4-metaprogramming-book/main/09_tactics.html)). The research question is now sharper than statement mutation alone: given the same target and goal, does an insight-aware planner withdraw a move when its prerequisite declaration is absent, remain stable under alpha-renaming, and avoid needless abstention when irrelevant facts change? A real dataset still requires an instrumented Lean-side snapshot/operation-replay harness, independent applicability adjudication, and linked measured outcomes.


## 24 October 2026 update: automate leakage triage without inventing ground truth

The new family-leakage audit and title-candidate generator are in the PR branch. The analyzer verifies the frozen split and reports cross-split components from a separately reviewed relationship inventory; the candidate generator uses title-token overlap only and is schema-constrained to label every candidate `title_similarity_only` and `unreviewed`. CI is configured to generate the candidate report twice and byte-compare it, then validate the report shape and count reconciliation. No candidate is automatically promoted to a dependency edge, and no family is excluded from the frozen split by the lexical heuristic.

An independent in-session implementation of the same title-token thresholds over the pinned catalogue surfaced 85 candidate pairs, 33 across splits and 10 across train/test. This result is an exploratory lexical screen, not a Rust binary execution, and not evidence that any pair is mathematically dependent or that leakage occurred. It shows why expert review should be prioritized by an explainable queue, while hard leakage decisions remain evidence-based.

### Current external comparison remains task-specific

- **Open whole-proof search:** Goedel-Prover-V2 reports 90.4% MiniF2F pass@32 for its 32B self-correction setup; Pythagoras-Prover reports 93.0% MiniF2F-Test pass@2048 and 85.04% MiniF2F-ALF pass@32 in its unified evaluation. These are not the same inference budget and must not be ranked as if they were. See the [Goedel-Prover-V2 ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html) and [Pythagoras-Prover results](https://pythagoras-lm.github.io/).
- **Advanced formalization:** AutoformBench contains 341 tasks from 24 advanced-mathematics sources and requires agents to construct missing definitions, helper lemmas and formal infrastructure. Its current source-present leaderboard reports 281/341 (82.4%) for GPT-6 Astra. This is a formalization-and-infrastructure score, not isolated theorem Pass@N or insight-transfer utility. See [AutoformBench](https://github.com/facebookresearch/autoformbench).
- **Real project work:** SorryDB's January 2026 `SorryDB_2601` snapshot contains 1,000 Lean proof obligations with repository, commit, Lean version and source location preserved; the larger project continuously discovers open tasks from real repositories. This makes it a useful external operational track, but a rolling task stream is a different denominator from a frozen competition benchmark. See [SorryDB](https://sorrydb.org/index.html).
- **Semantic validity:** T² and the ICML benchmark audit remain essential complementary comparators: successor-theorem testing and corpus/harness auditing detect failure modes that kernel acceptance or title similarity cannot. See [T²](https://aclanthology.org/2026.acl-industry.150/) and [Faults in Our Formal Benchmarking](https://proceedings.mlr.press/v306/ammanamanchi26a.html).

### Revised decision rule

Keep four results strictly separate: (1) whole-proof Pass@N under a pinned Lean/Mathlib and identical search budgets, (2) advanced-formalization completion with statement fidelity and infrastructure cost, (3) real-repository proof obligations on a pinned or prospective snapshot, and (4) causal PITB transfer utility/selectivity under matched budgets. The PITB-specific win condition stays the preregistered +10 percentage-point semantically qualified utility gain with a family/component-clustered 95% interval above zero; additionally require the complete reviewed cross-family inventory, test-family accounting, and independent receipt review before any confirmatory claim. Do not claim SOTA from schema completeness, title-candidate counts, or successful index generation.

Current validation boundary: the new Rust binaries and updated workflow are committed to the research branch, but exact-head GitHub status/run queries return no check records and the authoring environment lacks Rust tooling. No compile, test, or new model-score pass is claimed.
### 24 October 2026 update: keep title and artifact-path signals separate

A second candidate-discovery binary now scans the `linked_paths` for each of the 372 pinned families. This expands the review queue beyond the theorem title while preserving source paths and shared-token evidence. Its output is independently schema-validated and repeated in CI for byte-identical reproducibility. The path channel remains explicitly unreviewed and is not combined with the title-score channel or the leakage edge inventory.

The next measurement should be candidate recall/precision from a preregistered random sample, stratified by signal source and split direction, adjudicated by two reviewers with source artifacts visible. Report inter-reviewer agreement and adjudication rate; estimate candidate yield and time per confirmed hard relationship. Until that study exists, a larger candidate count is a review workload signal—not higher recall, more leakage, or a benchmark score.

## 24 October 2026 update: make the power gate executable before test outcomes

The pre-registered protocol's power-planning requirement now has a deterministic Rust implementation, `plan_pitb_power`, and a strict Draft 2020-12 report schema. The planner uses the proposed/observed connected-component size vector, a beta-binomial shared-family-difficulty sensitivity model, a configured baseline and expected gain, and an explicit ICC. It simulates the probability of observing the registered minimum gain plus a positive lower 95% cluster-level interval bound, and reports Monte Carlo uncertainty and probability-clipping effects.

This closes a tooling gap, **not** the statistical qualification gate. No PITB family sizes have been supplied to the planner because the real cross-family relationship/quarantine inventory is still incomplete. The planner's t-interval is approximate, ICC is a sensitivity input rather than an estimate, and the model must be checked by an independent statistician before confirmatory use. Repeat over a predeclared parameter grid only after the eligible component distribution is frozen; preserve the full grid and configuration digests. The PR workflow now contains a repeat-run byte comparison, report-schema validation and an impossible-power negative control, but the latest exact-head workflow has not been observed passing. No Rust compile, unit-test, power, transfer-utility or SOTA pass is claimed here.


## 24 October 2026 follow-up: power analysis must model the paired task and heterogeneous family effect

A second audit of the new Rust power planner identified a modeling limitation: the initial generator gave control and treatment a shared family baseline probability but then drew their outcomes independently within each target and applied the same gain in every family. That is not a sufficiently general model for paired proof-system comparisons: both arms operate on the same theorem tasks, and a transfer intervention can help some mathematical families while harming others.

The planner now simulates bivariate Bernoulli potential outcomes on the same target using a configurable positive-association fraction, and gives each component a configurable normally distributed treatment-effect deviation. The report labels the association parameter as a fraction of the Frechet upper bound on joint success—not as a Pearson correlation—and reports the generated marginal gain after probability clipping. CI is configured to validate two scenarios, including heterogeneous family effects and conditionally independent paired outcomes.

This is a model-sensitivity capability, not evidence that the selected assumptions are realistic. There is no empirical estimate of the ICC, paired potential-outcome association or family-level treatment-effect variance. For confirmatory power assessment, use a preregistered multidimensional sensitivity grid, the actual eligible component sizes, and independent statistical review; consider a second generator and analysis method as a cross-check. The real cross-family relationship inventory remains incomplete, exact-head workflow has not been observed passing, and no compile/test/power/utility/SOTA claim is made.

Methodological basis: simulation-based planning is appropriate when binary outcomes, cluster-size variation and small cluster counts make simple design-effect approximations unreliable; published methodological reviews emphasize sensitivity to ICC and cluster-size distribution ([cluster-trial design guidance](https://academic.oup.com/ije/article/52/5/1648/7172835); [comparison of power methods for clustered binary outcomes](https://pubmed.ncbi.nlm.nih.gov/17238238/)). These sources motivate the sensitivity work, but do not validate PITB's particular generative model.


See also [PITB's exploratory power-planning sensitivity illustration](pitb-power-planning-sensitivity-illustration-2026-10-24.md). It shows under synthetic assumptions that the 20-component floor may have low probability of meeting the combined +10-point observed-effect and positive-interval gate. The finding is not a measured PITB result or final sample-size prescription; it motivates sensitivity planning on the real eligible component sizes once the relationship audit is complete.


### 24 October correction: align the planned interval to the registered paired estimator

The power-planning implementation has since been corrected to use the paired component-percentile-bootstrap interval from PITB's success-cost frontier analysis, not the t-interval approximation described in an earlier section above. The bootstrap replicate count is an explicit, schema-bound input. The linked sensitivity note's tables were refreshed under that component-bootstrap rule using a separate exploratory Monte Carlo implementation; they are not Rust output or measured PITB performance. The next design gate remains the real component-size distribution after the independently reviewed relationship/quarantine audit, followed by a preregistered grid and independent statistical review.


### 28 October update: stress-test inference under heterogeneous effects

Added the [weak-null calibration audit](../../blob/research/proof-insight-transfer-v0/data/benchmarks/proof-insight-transfer-v0/weak-null-calibration-audit-2026-10-28.md). In one exploratory synthetic family with zero target-weighted average effect but mixed component-level effects, the simple cluster-paired t interval rejected too often under severe component-size imbalance (4.95% at a 2.5% reference rate), while the component-percentile bootstrap had a 3.15% rejection rate; an unqualified restricted wild-cluster comparator reached 6.10%. These figures include Monte Carlo uncertainty and are not project performance.

The result does not select a universal interval. Inference remains unqualified until PITB's actual eligible component distribution and estimand are frozen, interval availability is reported separately, a broader preregistered calibration grid is run, and an independent statistician reviews the final design. The current head still has no measured transfer-utility result; indexing, synthetic schemas, and calibration simulations must not be reported as a SOTA win.


## 28 October 2026 frontier refresh: fresh research tasks, Hard Mode, and subgoal utility

The current benchmark frontier is broader than whole-proof completion. PITB should adopt these as **separate comparator strata** and not pool headline rates whose target units and compute budgets differ.

- **Fresh research-level theoretical CS — TCSAlgBench.** The September 2026 release describes 398 theorem-level challenges from 138 STOC/COLT 2026 papers, a refreshable pipeline, and matched comparisons between direct inference, prover-verifier discussion, and agent workflows. Its reported verifier-accepted coverage is 23.6% for GPT-5.6 Sol max after ten-round discussion in the model comparison, and 25.4% for an agentic-planning workflow in a separate agent comparison. This is particularly relevant to exposure control because a benchmark can add new paper-derived challenges over time; it is not a substitute for PITB's paired transfer experiment. ([paper](https://arxiv.org/abs/2609.35606))
- **Subgoal completion — FormalML.** The ICLR 2026 benchmark contains 4,937 Lean 4 proof obligations in machine-learning theory, spanning optimization and probability inequalities, with complex research contexts and premise retrieval. PITB should track whether an insight closes a previously unsolved obligation, changes the number of generated obligations, and reduces tokens/cost to a receipt-backed closure; raw subgoal counts must not substitute for the terminal exact-target endpoint. ([ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/55d4e6e1a0557d3d9e3140966c1db533-Abstract-Conference.html))
- **Advanced algebra — FATE.** FATE-H and FATE-X each contain 100 formal algebra problems at different difficulty tiers; the paper reports 3% pass@64 on FATE-H and 0% on FATE-X for the best evaluated system. These figures are task-specific, but they are a useful reminder that stronger models and contest success do not imply robust research-level formalization. ([ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/11c0318fc7b0315fa9542449b1c9ba32-Abstract-Conference.html))
- **Answer discovery plus proving — Discover and Prove (DAP).** The ACL 2026 work explicitly separates Hard Mode tasks, where the answer or witness is not inserted into the formal statement, from the easier proof-only setup. It reports 36 PutnamBench theorems formally proved in Hard Mode and improves CombiBench solved count from 7 to 10 at pass@16. PITB's constructive-witness track should therefore separately score answer/witness accuracy, formal proof success, and equivalence between the discovered answer and the checked theorem. ([ACL Anthology](https://aclanthology.org/2026.acl-long.3/))
- **Proof structure and cost — TheoremBench and Minimal Agent.** TheoremBench's main and premised variants expose related subtheorem structures and token-efficiency differences; the ICML 2026 Minimal Agent study reports competitive results from a simpler iterative system at a fraction of the cost of larger architectures. PITB should compare terminal verified success at matched spend and report the observed success/cost Pareto frontier; do not assume a larger or more complex agent is better. ([TheoremBench](https://arxiv.org/abs/2606.09450); [Minimal Agent](https://proceedings.mlr.press/v306/requena26a.html))
- **Infrastructure construction — AutoformBench.** Its 341 advanced-mathematics tasks require the agent to build missing definitions, helper lemmas, and formal infrastructure, not merely discharge an isolated goal. Keep this as a distinct extension to test whether transferred insights reduce the missing-infrastructure gap. ([Meta AI research page](https://ai.meta.com/research/publications/autoformbench-a-lean-formalization-benchmark-for-advanced-mathematics/))

### Strategic implication for PITB

Do not compete by publishing an unqualified global theorem-proving pass rate. Instead, establish the more specific, falsifiable result that a typed insight adds value over the strongest retriever on the same hidden targets, same prover loop, same context and spend; quantify false-acceptance on adversarial transfer pairs; test whether the benefit survives new families and fresh repository tasks; and report verified solve/cost Pareto points. The project's headline claim should remain the causal transfer effect, while Hard Mode discovery, subgoal completion, infrastructure construction, research-level fresh tasks, and semantic audit remain separate tracks.

All external percentages above are the cited papers' own results. They are not reproduced by PITB and are not numerically comparable to its family-clustered causal endpoint. No PITB transfer score or SOTA claim is established by this literature review.