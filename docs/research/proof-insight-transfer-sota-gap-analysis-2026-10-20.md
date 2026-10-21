# Proof Insight Transfer: state-of-the-art gap analysis and next experiments

**Date:** 2026-10-20  
**Status:** literature-informed research plan; the figures below are published reference results from their stated setups, not PITB results. **Symthaea has no measured insight-transfer result yet.**

## Executive decision

The differentiated research question is not “can a retriever find a similar proof?” Strong premise retrieval already helps Lean proving. It is:

> Can a proof-derived, typed strategy—with explicit preconditions, state/obligation change, evidence, and predicted failure boundary—cause more **verified exact-target successes** on a previously unseen mathematical family than the strongest matched retrieval baseline?

The most important missing comparator was a strong global-premise retriever. PITB now preregisters arm I alongside A–E for validation-only baseline selection. The primary outcome remains F (typed insight) minus the strongest preregistered validation-selected baseline, with semantic-gate success as the endpoint and no claim until all gates pass. This is a pre-outcome amendment; re-hash the current protocol bytes into any future run manifest.

## What the field already does well—and where PITB must be stricter

### Retrieval is a serious baseline, not a straw man

LeanSearch v2 reports **nDCG@10 = 0.62 vs. 0.53** for the next-best system on its single-query retrieval benchmark. On its 69-query research-Mathlib premise-group benchmark, its reasoning mode recovered **46.1%** of ground-truth premise groups within ten candidates, compared with **38.0%** for strong reasoning retrieval and **9.3%** for the listed premise-selection baselines. Its fixed-prover downstream experiment reported **20%** proof success with LeanSearch v2, **16%** with the next-best retriever, and **4%** without retrieval ([paper](https://arxiv.org/abs/2605.13137); [reproduction artifacts](https://github.com/frenzymath/LeanSearch-v2)).

Those numbers are **not** comparable to PITB's eventual success rate: the task sets, model loop, corpora, budget, and outcome units differ. They do show why a weak lexical-only baseline would be inadequate. Arm I must pin the retriever implementation, source/index snapshot, candidate budget, and model/service version. If the published LeanSearch v2 system cannot serve a particular eligible target, PITB must identify the locally reproducible same-corpus substitute and not label it as a reproduction of the paper.

### Compilation alone is not a sufficient correctness endpoint

The ACL 2026 T² study covers **2,206 problems** from five real Lean 4 repositories with an average of 41 dependent successor theorems per problem. In its published setup, Claude Sonnet 4.5 reached **38.9% Testing Accuracy** even with natural-language proof and successor theorems supplied as context ([ACL Anthology](https://aclanthology.org/2026.acl-industry.150/); [code/data](https://github.com/ldilab/T2)). This motivates separating kernel/checker acceptance from statement correspondence, dependency/axiom audit, dependent-successor checks, and frozen mutation controls.

Successor checks are not magically semantic oracles: failures can result from unrelated build problems, and a dependent theorem may not exist for a target. PITB should retain exact receipts, distinguish relevant semantic regression from unrelated failure, and record a defensible not-applicable decision rather than inventing proxy tests.

### Formal-math coverage and proof efficiency are separate axes

TheoremBench uses nearly **100 classical theorems**, with both a main-target version and a premised version that exposes related supporting subtheorems. It reports theorem-level coverage and token efficiency, highlighting that performance on easy subtheorems and lengthy tactic traces can obscure weak full-theorem progress ([paper](https://arxiv.org/abs/2606.09450)). PITB should therefore report target-level exact success, per-family coverage, tokens/time/cost per verified solve, and complete-family coverage—not only a pooled pass percentage.

### Generalization beyond competition-style mathematics matters

VeriSoftBench contains **500 Lean 4 proof obligations across 23 real-world repositories**. Its authors report poor transfer from Mathlib-style provers to repository-centric software verification and improved performance when context is limited to curated dependency closures rather than the full repository ([paper](https://arxiv.org/abs/2602.18307); [repository and evaluation instructions](https://github.com/utopia-group/VeriSoftBench)). PITB should eventually test whether insight records help under dependency-rich context, not just on the source corpus where the insight was extracted.

### Informal research-mathematics value has different costs and scoring

First Proof Batch 3, published **14 October 2026**, tested five systems on ten research problems for 24 hours per system without human intervention; 30 expert reviewers graded correctness, exposition, and attribution. ProofCouncil was graded correct on **9/10**, while the direct GPT-6 Astra entry reached **7/10**. The reported costs were **$1,888** and **$36**, respectively, and adequate citations were judged on **7/10** ProofCouncil outputs versus **4/10** direct Astra outputs ([official press release](https://1stproof.org/third-batch-press-release.html); [benchmark page](https://1stproof.org/third-batch.html)). This is a distinct, human-reviewed informal-proof setting, not a Lean pass-rate baseline. It does establish that abstention, citation provenance, exposition, and resource cost deserve explicit measures.

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
