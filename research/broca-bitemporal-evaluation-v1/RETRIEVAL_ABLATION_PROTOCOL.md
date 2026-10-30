# Broca paired retrieval and stopping evaluation protocol

**Protocol version:** BROCA-RETRIEVAL-ABLATION-V1  
**Status:** experiment specification; no Broca results are claimed.  
**Purpose:** identify when extra retrieval improves a justified answer/action, not merely when it returns more evidence.

## Research question

For a fixed query, corpus snapshot, user/scope, valid-time/known-time coordinates and reader, does an additional retrieval action improve the final supported answer or lifecycle action enough to justify its measured cost and risk?

The unit of comparison is the same query under paired retrieval policies. Do not compare unrelated workloads, generators, prompts or context budgets.

## Treatment arms

| Arm | Retrieval policy | Hypothesis tested |
|---|---|---|
| A | No retrieval / parametric-only | Counterfactual baseline for marginal effect |
| B | Always BM25 top-k | Fixed retrieval baseline |
| C | BM25 score threshold | Whether lexical score is enough to decide |
| D | Confidence-only trigger (CONFRAG-like) | Whether reader confidence reduces calls without lifecycle awareness |
| E | Claim–evidence coverage stopping (HALT-like) | Whether all expected claims have sufficient admissible evidence |
| F | Lifecycle/proof coverage plus calibrated VOI | Whether lifecycle admissibility and conservative marginal utility improve the frontier |
| G | Always hybrid lexical+dense+rerank (where available) | Upper-effort retrieval baseline, with the full incremental cost counted |

For each query, all arms receive the same immutable corpus and EvidenceIndex snapshot digest, principal/scope, valid_at, known_at, source revisions, reader model/version, system and query prompts, tokenizer, packet renderer, proof verifier and final evaluator. Policies may differ only in retrieval/stopping behavior.

## Paired execution requirements

1. Freeze benchmark revision, raw dataset digest, split manifest, code commit, model weights, prompts, tokenizer, renderer and utility-schema digests before evaluation.
2. Use separate calibration, validation and sealed test splits. Group by source document / conversation / user before splitting; no near-duplicate query or source material may cross splits.
3. For calibration data, execute both retrieval and no-retrieval potential outcomes for each query with the same reader/evaluator seed where deterministic replay is possible. Store both outcomes and all measured costs. Do not infer a counterfactual from reader confidence.
4. Execute each online policy with the same maximum total budget. Record realized calls, evidence tokens, query tokens, packet tokens, estimator tokens, wall-clock latency, GPU/CPU time, cache state and serving cost.
5. Preserve exact per-query action and proof receipts. Opaque proof capability is retained only in the in-process test harness; persisted receipt data is not treated as answer authority.
6. Run at least five fixed seeds where stochastic model behavior is involved. Report paired per-query deltas and group-level uncertainty intervals.
7. Have answer and action labels scored with a deterministic gold protocol or blinded independent annotators. LLM-as-judge may be reported as secondary analysis, not the only judge.

## Outcome definitions

Each query records independent, non-substitutable metrics:

- **Evidence retrieval:** strict recall-all@k, fractional recall@k, any-hit@k, source revision recall, multi-session completeness, stale-source hit rate, cross-scope candidate rate.
- **Evidence admissibility:** percentage of retrieved evidence accepted after scope, source identity, valid-time, known-time, retraction, authorization and contradiction checks; invalid admission count.
- **Proof quality:** complete required-claim coverage, proof verifier acceptance, independent source-family count, invalid-certificate rate, packet-token cost and search-limit rate.
- **Answer/action quality:** LifeEM where applicable, exact answer correctness, action exact match, unsupported-assertion rate, citation/proof completeness, stale-warning precision/recall, conflict-explanation precision/recall, answerable abstention rate and unanswerable false-answer rate.
- **Retrieval effect:** paired correctness delta, paired lifecycle-loss delta, frequency of benefit/neutrality/harm, probability calibration, Brier score, ECE, utility-bound coverage.
- **Cost:** calls/query, retrieval result tokens, estimator computation/tokens, total prompt/completion tokens, p50/p95/p99 latency, cost/correct supported answer and cost per successful required-claim proof.

Strict recall and answer accuracy must never be reported as if they were the same metric. LifeEM must not be numerically pooled with ordinary LoCoMo or LongMemEval exact-match results.

## Paired inference and calibration

- Primary quality delta: per-query paired difference in exact supported-answer correctness or action score between each adaptive policy and its matched baseline.
- Cluster bootstrap: resample independent source-document / conversation / user groups, not individual questions, to account for within-source dependence. Report 95% intervals and effect size.
- Binary paired correctness: also report discordant pairs and exact McNemar test; do not rely on an unpaired accuracy comparison.
- Multiple policy arms: pre-register one primary contrast (F vs E) and control familywise error for exploratory arm comparisons.
- Utility calibration: freeze estimator on training data, calibrate residuals on held-out groups, then test coverage and realized budget transfer on a sealed test split. The runtime conformal routine requires 29 independent calibration groups for its pinned 90% joint marginal bound.
- Thresholds are frozen before sealed evaluation. Report target budget and realized usage by domain, split, question type and seed.

## Retrieval harm taxonomy

Count a retrieval call as harmful when it causes one or more of the following relative to the paired no-call condition: a previously supported answer becomes unsupported; a current source is displaced by stale material; contradictory evidence causes a false answer; cross-scope evidence appears; proof coverage falls; answer/action correctness decreases; or reader context grows without adding an admissible premise and causes a measured error/cost penalty.

Report both the percentage of calls that are harmful and the magnitude of the downstream loss. A high-recall result can still have a high harm rate.

## Release gates

These are prospective targets, not current results:

- **Soundness:** zero false proof certificates and zero ANSWER actions when any required claim is stale-only, conflicted, missing, unauthorized, provenance-invalid, cross-scope or not independently verified in deterministic conformance tests.
- **Completeness:** prove every required claim on the answerable subset; report retrieval recall separately to expose retrieval misses.
- **Budget:** zero hard call/token/cost overshoot under actual accounting.
- **Calibration:** achieve the stated held-out grouped-conformal bound coverage on independent test groups; report coverage by domain and group size.
- **Efficiency:** adaptive policy F should improve cost/latency at statistically non-inferior LifeEM and supported-answer correctness versus both always-retrieve and coverage-only baselines. The non-inferiority margin must be pre-registered.
- **Harm:** no statistically significant increase in retrieval harm, stale-use errors, cross-scope leakage, false answer or unsupported assertions.
- **Reproducibility:** all data manifests, per-query receipts, evaluation code, random seeds, model/tokenizer/renderer identifiers and policy versions are pinned.

## Known limitations

The current implementation supplies a deterministic VOI gate and grouped calibration procedure. It does not yet train the point estimator, orchestrate all treatment arms, run the paired benchmark, calculate statistical intervals, or measure the targets above. The conformal guarantee relies on exchangeable independent calibration groups; distribution shift, split leakage or correlated groups weaken the guarantee. The utility schema must define correctness gain and lifecycle-risk reduction as disjoint components to prevent double counting.

## Versioned calibration artifact

The calibration fixture [retrieval-value-calibration-fixtures-v1.json](retrieval-value-calibration-fixtures-v1.json) contains 29 synthetic independent groups with paired predicted/observed utility components and expected conformal residual offsets. The Rust calibration test reads this artifact directly. It is a deterministic conformance fixture, not an empirical calibration dataset and not evidence that real-world coverage has been achieved. The module `memory_retrieval_value_benchmark.rs` consumes matched with/without-retrieval outcomes, derives disjoint component deltas, and reports 2,000-replicate deterministic cluster-bootstrap 95% intervals over independent source/user/conversation groups. It also exports paired outcomes into calibration observations for a frozen point estimator.

## Relevant external comparisons

- HALT: claim–evidence coverage stopping, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.343/
- CONFRAG: confidence-triggered RAG, EMNLP 2026 Industry Track: https://aclanthology.org/2026.emnlp-industry.171/
- Active RAG: marginal correctness, calibration, realized budgets, retrieval harm and total cost: https://arxiv.org/abs/2607.24010
- Zhao (27 October 2026): adaptive hybrid retrieval achieved similar evidence Hit@10 on a small held-out QASPER sample, but no reliable paired-bootstrap utility advantage was established: https://arxiv.org/abs/2610.37358
- AgentAbstain: paired should-act / should-abstain tasks: https://agentabstain.github.io/

