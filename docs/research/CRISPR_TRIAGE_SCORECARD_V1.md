# CRISPR Off-Target Triage — Scorecard and Value-of-Information Protocol v1

**Status:** proposal / not executed  
**As-of date:** 2026-10-16  
**Parent:** [CRISPR Cost-Aware Evidence Bench v1](../CRISPR_COST_AWARE_BENCHMARK_V1.md)  
**Primary task:** CRISPR-OT-001, retrospective, read-only computational evaluation  
**Authority ceiling:** research benchmarking only; no guide-selection advice, experimental instructions, clinical recommendation, or safety certification.

## Executive decision

Do not make Symthaea compete to be the next standalone off-target predictor. The frontier already includes RNA-pretrained sequence models such as [CCLMoff (Communications Biology, 2025)](https://www.nature.com/articles/s42003-025-08275-6) and hybrid prior/data-learning methods such as [CrisprPr (Briefings in Bioinformatics, 2026)](https://academic.oup.com/bib/article/27/2/bbag140/8559621). The stronger, falsifiable bet is to test Symthaea as a **decision and evidence layer over specialist tools**: maintain candidate-universe coverage, expose disagreement and missing evidence, calibrate or abstain where justified, and measure the review burden needed to recover known findings.

The question is not “Which model has the highest AUROC?” It is:

> Under a fixed, lawful, fully specified evidence universe and review budget, does a Symthaea-added policy surface more independently established positive sites, with no hidden increase in missed positives, worse calibration, or loss of generalization?

This reframing responds to two published limitations. Recent work discusses limited generalization to unseen guides and inconsistent negative construction. A 2026 systematic benchmark of 14 tools shows why candidate nomination, ranking on a constructed candidate universe, and experimental detection must be evaluated separately: measured positives added to a ranking universe cannot prove that the original search tool would have nominated them. See [Du et al., 2025](https://doi.org/10.1038/s42003-025-08275-6), [CrisprPr, 2026](https://academic.oup.com/bib/article/27/2/bbag140/8559621), and [Kaufmann et al., 2026](https://doi.org/10.1177/10430342261466407).

## 1. Paradigm shifts that change the experiment

### A. From score prediction to evidence-efficient decisions

Compare policies at the same candidate universe and review budget. A high-ranked list is useful only if it retrieves independently established findings with fewer review slots and makes blind spots visible. Candidate slots are a review-burden proxy—not measured lab dollars or a medicine-price estimate.

### B. From one universe to two separate denominators

Maintain two linked but non-interchangeable tracks:

1. **Native nomination:** Did a candidate-generation/search method find a measured positive before positive seeding?
2. **Conditional ranking:** Given one frozen candidate universe, where did measured positives rank?

Publish the pre-seeding universe and post-seeding ranking universe separately. Results after positive seeding must never be presented as candidate-discovery recall.

### C. From binary labels to an evidence-state model

Every candidate must be explicitly **positive**, **assay-defined negative**, or **unknown/unassayed**. “Not reported,” “not nominated,” and “not detected by a limited assay” are not automatically true negatives. Record the assay, detection scope, threshold, source cohort and provenance for every outcome.

The 0.1% indel threshold in the Kaufmann benchmark is an **operational definition for reproducing that study**, not a universal biological or clinical boundary.

### D. From random rows to biological and measurement generalization

Hold out whole guide/Cas9 groups where possible; keep duplicates, near-identical guides, source-study derivatives and reused genomic sites together across splits. Add leave-one-assay-family-out and independent-cohort evaluation when lawful data are available. A row-level random split is a diagnostic only, not proof of generalization.

CrisprPr's published evaluation excluded duplicated or at-least-90%-similar guides relative to its training set and filtered guides used to construct biological priors—an explicit reminder that leakage controls must cover labels and priors. Require a provenance-aware leakage audit, not just a random seed.

### E. From consensus to calibrated disagreement and abstention

Do not treat model agreement as truth. Record per-model ranks/scores, missing outputs, model version, domain and training-data knowledge when available. Test whether disagreement or uncertainty helps prioritize review or detect out-of-domain examples. Permit an explicit **insufficient evidence** state. Abstention must be scored as abstention, not silently discarded or converted into a low-risk result.

### F. From static leaderboard to review-policy ablation

Compare four predeclared policies: best single specialist, simple rank fusion, specialist rank alone, and Symthaea-added evidence triage. Ablate the Symthaea components individually (candidate coverage, disagreement, uncertainty/abstention, provenance checks). A win over a weak baseline does not establish value over the best fair specialist.

### G. From reported benchmark to sealed prediction and independent replay

Freeze data-source and transformation digests, split IDs, model/provider/configuration, exclusions, predictions and metric implementation before the evaluator reveals holdout labels. A verifier must reconstruct counts and metrics without relying on model narration. Include deliberate corruptions and leakage fixtures that must fail closed.

### H. From closed-world labels to partial identification

Published off-target benchmarks note that unsequenced high-scoring candidates may be undiscovered positives, not ordinary false positives. Treat this as an ascertainment problem: the observed label table is not automatically the complete biological truth for the declared search space. See [Kaufmann et al., 2026](https://doi.org/10.1177/10430342261466407). This complements earlier work on predictive uncertainty in off-target activity, including [crispAI](https://doi.org/10.1093/nar/gkae759).

For each fixed ranked universe, report the usual metric against the measured-positive reference **and** worst/best-case label-assignment bounds whenever unknown/unassayed candidates exist. These bounds are not posterior probabilities or confidence intervals. They vary unknown labels only inside the declared candidate universe; they cannot bound sites omitted from every nomination method.

Let:
- P = number of eligible measured-positive reference sites;
- pK = measured positives among the top K;
- U = number of unknown/unassayed candidates in the fixed candidate universe;
- uK = unknown/unassayed candidates among the top K.

Then, where P > 0 and K slots are available:

- observed known-label Recall@K = pK / P;
- worst-case conditional recall bound = pK / (P + U - uK);
- best-case conditional recall bound = (pK + uK) / (P + uK);
- precision bounds = [pK / K, (pK + uK) / K].

The lower recall case treats selected unknowns as negative and all unselected unknowns as positive; the upper case does the reverse. Report the observed estimate and bounds together. Never report an unknown as a negative merely to obtain a point estimate. These are **label-uncertainty bounds**, distinct from group-aware statistical uncertainty intervals.

### I. From price claims to causal cost accounting

Report computational resources and review slots first. Progress to human review minutes only when measured. Report per-qualified-finding cost only when numerator and denominator are observed end-to-end. Do not convert fewer ranked candidates into claims of lower experimental costs, clinical costs, or treatment prices without prospective accounting.

## 2. Required benchmark tracks

| Track | Question | Primary measures | Cannot establish |
|---|---|---|---|
| T0 Source replay | Did we reconstruct the cited cohort and transformations? | exact counts, digest verification, exclusions, output census | model improvement |
| T1 Native nomination | Which measured positive sites were in a tool's original search output? | positive coverage, missed-positive count, per-guide coverage | ranking quality after seeding |
| T2 Conditional ranking | Given one declared universe, how well are measured positives ranked? | macro Recall@K, Precision@K, PR-AUC when labels support it, false-negative census | completeness of original nomination |
| T3 Budgeted review | How much review is needed for an agreed level of measured-positive recall? | recall at fixed slots; slots needed to match baseline recall; precision trade-off | actual dollar savings |
| T4 Calibration and abstention | Are scores usable as probabilities, and does the system know when it lacks evidence? | Brier/log loss where labels support them, reliability plots, risk–coverage curve, abstention rate | clinical safety |
| T5 Shift and robustness | Does the effect persist on unseen groups and independent data? | leave-guide/Cas9-group-out delta, leave-assay-family-out delta, independent-cohort delta, group-level uncertainty | universal generalization from one small cohort |
| T6 Incrementality | Does Symthaea add value over the same specialist predictions? | paired delta at equal budget; ablations; runtime/resource delta | benefit attributable to an upstream model unless ablated |
| T7 Evidence/replay integrity | Can an independent evaluator detect tampering or leakage? | deterministic replay, exact manifest, negative controls, digest/count agreement | biological truth from cryptography alone |
| T8 Partial-label identification | How much do unresolved labels change metric interpretation? | observed-label estimate, worst/best-case recall and precision bounds, unknown-label census | sites omitted from every nomination method; truth outside the fixed candidate universe |

Tracks are reported separately. A composite “CRISPR score” that hides failed nomination, calibration, or generalization is prohibited.

## 3. Metric definitions and denominators

A **benchmark episode** is a pinned combination of source cohort, assay context, guide/Cas9 group, reference genome/version, candidate universe, label policy, split and prediction snapshot. Every metric must identify its episode(s).

- **Native positive coverage** = unique measured positive sites present in native tool output / unique measured positive sites eligible in the reference cohort. Count a miss when the native generator never nominated the site. Publish numerator and denominator.
- **Recall@K** = measured positives among the first K candidates / all eligible measured positives in the declared ranking universe. State explicitly whether positives were added to that universe after the native search phase.
- **Precision@K** = measured positives among the first K candidates / K reviewed candidates, only when outcomes for the selected candidates are actually known under a declared label policy. Do not treat unassayed candidates as negatives to manufacture this value.
- **Slots@R** = smallest K at which a policy meets a preregistered recall R on the same eligible positive reference and same candidate universe. If it never reaches R, report **not reached**, not a fabricated K.
- **Fixed-budget delta** = Symthaea policy metric minus the strongest fair baseline metric at the same fixed K on identical episodes. The primary comparison is paired by guide/Cas9 group.
- **Calibration:** report Brier score, log loss and reliability diagrams only for labels whose semantics support probability evaluation. PR-AUC is a ranking metric, not calibration. If the assay/reference labels are incomplete or negatives are indefensible, mark calibration **not estimable** and explain why.
- **Risk–coverage:** for each predeclared abstention rule, report fraction of cases assigned a usable score (coverage), error/false-negative behaviour on the evaluated labelled set, and group distribution. Abstention is neither a positive nor a negative.
- **Uncertainty interval:** resample the highest independent unit supported by the study—normally guide/Cas9 group, not individual rows—using a paired method for the policy difference. With only 26 source groups, show every group result and state that interval precision/generalization is limited. Do not use row-level bootstrap to create artificial sample size.
- **Resource accounting:** report candidate slots and, when measured, wall-clock time, CPU/GPU model and utilization, peak memory, and human review minutes. Keep measured quantities separate from projected costs.

- **Partial-label bounds:** If unknown candidates exist within the fixed ranking universe, report the known-label metric and worst/best-case assignments to those unknown labels. For P > 0, measured positives pK in the top K, U total unknown candidates, and uK unknown candidates in the top K, conditional recall bounds are [pK / (P + U - uK), (pK + uK) / (P + uK)]. Precision bounds are [pK/K, (pK+uK)/K] when K candidates are available and K > 0. These are not confidence intervals and do not cover positives absent from the declared candidate universe.

Report macro (equal-group-weighted) and micro (pooled) summaries together. Include per-guide/Cas9 results and all invalid/omitted cases. Do not let a large group conceal a failure on a small group.

## 4. Predeclared targets

These are **project engineering targets, not published standards or clinical acceptance criteria**.

### Primary success: one of two alternatives

1. **Equal budget:** at least **+5 absolute percentage points** in macro Recall@K versus the strongest fair baseline on predeclared held-out guide/Cas9 groups at the same K; or
2. **Equal achieved recall:** at least **20% fewer candidate review slots** to reach the baseline's preregistered recall target on the same episode set.

For either alternative:

- endpoint, K or recall target, exclusion policy, comparator, split, interval procedure and analysis are frozen before holdout reveal;
- report precision, false negatives and per-group deltas alongside the headline;
- the paired group-aware interval must exclude no improvement for a “demonstrated improvement” claim;
- if the interval includes no improvement or the eligible sample is inadequate, report **null/inconclusive**, not success;
- cross-dataset robustness requires at least one separate, independently sourced cohort with explicit permission for the intended use.

### Qualification targets that are binary gates

- **100%** of used artifacts have source identity, terms/permission status, retrieval/version information and verified SHA-256 digests recorded.
- **100%** of candidate outcomes use explicit positive / assay-defined-negative / unknown semantics; unknowns are never silently imputed as true negatives.
- **100%** of reported tables identify the candidate universe, seed status, group unit, denominator and missingness policy.
- **100%** of deliberate corruption/leakage fixtures are rejected by the independent verifier.
- Exact deterministic replay of the metric layer on the same pinned inputs produces the same canonical output and manifest digest.
- A prediction-only or schema-valid receipt can never promote an evidence state to independently observed, assessed or replicated.

- **100%** of metrics affected by unknown/unassayed candidates report partial-identification bounds or an explicit not-estimable state; no unknown label is silently coerced into a negative.
Any failed binary gate blocks performance claims until remediated.

## 5. Fair comparator matrix

All policies receive the same permitted inputs and are tested against the same frozen episodes.

| ID | Policy | Why it is included |
|---|---|---|
| B0 | Deterministic simple/rule baseline | establishes whether complexity adds value |
| B1 | Best qualified single specialist tool available under the dataset's use terms | primary performance comparator |
| B2 | Simple rank fusion of available specialists | tests whether fusion alone explains gains |
| B3 | Specialist output without Symthaea triage | isolates the orchestration effect |
| S1 | Symthaea layer with provenance/candidate coverage only | measures baseline integration value |
| S2 | S1 + model disagreement/uncertainty | tests uncertainty contribution |
| S3 | S2 + explicit abstention policy | tests whether refusing overconfident decisions is useful |
| S4 | Full predeclared Symthaea policy | candidate system under test |
| N0 | Random order, repeated with fixed seeds | sanity reference, not a deployment candidate |

Freeze tool versions and individual results before combining them. If a specialist tool cannot be reproduced lawfully or technically, report it as unavailable; do not substitute hand-entered or invented outputs.

## 6. Anti-leakage and negative-control suite

The evaluator must fail closed on each of the following:

- exact cohort digest changes after preregistration;
- the metric implementation consumes post-seeding candidates in the native nomination denominator;
- a guide/Cas9 group, duplicate site or known derivative crosses a prohibited split boundary;
- no-bulge and bulge-enabled cohort counts or labels are mixed;
- source rows are lost, duplicated, silently relabelled or reordered in a way that changes identity;
- unknown/unassayed outcomes are coerced into negatives;
- denominator or positive threshold changes after holdout reveal;
- predictions use holdout labels or prior construction exposes test labels;
- external cohort use terms do not allow the proposed use;
- stale commit, skipped workflow, queued job or missing artifact is reported as a pass;
- a digest receipt exists but the referenced bytes are unavailable or not verified;
- the full results census omits predictions, failed examples or exclusions.

Negative controls should be deterministic fixtures and independently testable. “The model says it did not leak” is not an acceptable verifier.

## 7. Source replay anchor and limits

The current first proving vertical is the 2026 Kaufmann et al. study and its [public companion repository](https://github.com/maren-ha/OffTargetPredictionBenchmark), with the repository's [Zenodo record](https://zenodo.org/records/20627722). The parent contract records the published cohort anchors and exact reproduction-repository commit observed during research.

Do not treat these as reproduced project results. At protocol-authorship time, exact artifacts and digests have **not** been verified in this project; the cohort has not been replayed; publication metrics have not been recomputed; and no Symthaea improvement has been demonstrated. The first stage is a source/rights and artifact-identity audit, not model execution.

The [CCLMoff study](https://www.nature.com/articles/s42003-025-08275-6) is a useful literature comparator because it combines 13 genome-wide detection technologies across 21 publications and reports leave-one-dataset-out evaluation, but its reported metrics are not project-reproduced here. [CrisprPr](https://academic.oup.com/bib/article/27/2/bbag140/8559621) is a useful comparator for prior-knowledge integration and leakage controls. A literature result is not a substitute for pinned inputs and replay.

**Restricted data rule:** the UNCOVERseq study's published use restrictions prohibit commercial predictive-model development and related commercial uses without prior written permission. This project does not ingest or use that study's restricted data for model development unless the exact intended use is authorized in writing. Published article metrics may be cited as context with their study-specific scope; they are not training data or universally transferable performance guarantees.

## 8. Ordered execution plan

1. **Source/rights freeze:** inventory artifacts, terms, versions and hashes. If permission is unclear, block ingestion and choose a separately permitted cohort.
2. **Independent cohort replay:** reconstruct identities, counts, exclusions, native search universe and post-seeding universe; publish a reconciliation table.
3. **Metric verifier first:** implement/validate the deterministic scorer with synthetic fixtures for nomination vs ranking, unknown labels, duplicates, missing rows and changed digests. Keep production implementation Rust-first and dependency-minimal.
4. **Specialist baselines:** reproduce at least one permitted specialist and a simple baseline before running a Symthaea variant.
5. **Preregistered ablation:** freeze candidate policies, groups, endpoints and resource budget; never tune against the test fold.
6. **Independent replay:** an evaluator not responsible for generating predictions verifies all outputs and negative-control failures.
7. **External cohort:** only after the first cohort's gates pass and a second cohort has explicit use rights.
8. **Prospective follow-on:** a future authorized research collaboration may test whether ranking changes actual review outcomes. That is a separate gate; retrospective enrichment does not demonstrate prospective laboratory savings.

Do not wait on CI to write contracts, audit provenance, review code, implement deterministic fixtures or inspect local artifacts. But do not call any gate passed without exact, reproducible evidence.

## 9. Mycelix evidence projection

Compose with [Mycelix DESCI-BIO-002 (#3331)](https://github.com/Luminous-Dynamics/mycelix/issues/3331). Mycelix stores permissions/provenance, attribution, collaboration and links to versioned evidence. It is not a parallel biological truth ledger.

Use distinct records for source contract; preregistered prediction; computational result; authorized raw observation; independent assessment; and replication. Each projection states its evidence class, source identifiers/digests, responsible actor, permissions, timestamp and limitations. Reputation, consensus, signature validation or model agreement cannot upgrade an evidence class.

## 10. Reporting language

Allowed before execution: “proposed target,” “published source reports,” “not reproduced,” “not tested,” “inconclusive,” and “candidate review slots.”

Allowed only after the corresponding gate passes: “reproduced,” “improved under the preregistered split,” or “generalized to the named independent cohort.”

Not allowed from this benchmark alone: “CRISPR-safe,” “clinically safer,” “treatment price reduced,” “lab costs reduced,” “validated guide,” or “new therapy discovered.”

The credible breakthrough would be evidence that a general-purpose reasoning/evidence system improves a well-defined specialist workflow in a leak-resistant, independent, cost-accounted benchmark—even if the result is negative. The protocol is designed to reveal that answer rather than assume it.


## 11. Synthetic contract fixtures

The machine-readable companion contract contains four small fixtures intended to become deterministic unit tests for the future Rust scorer/verifier:

1. **F1 — nomination vs seeded ranking.** Four eligible positives, two in the native nominations: expected native coverage is 2/4 = 0.5. Later seeding the two missing positives into a ranking universe allows Recall@5 to reach 1.0, but native nomination coverage remains 0.5.
2. **F2 — unknown is not negative.** If an unknown/unassayed candidate is inside the review set, precision is not estimable under the declared label semantics; it may not be silently counted as a true negative.
3. **F3 — digest mismatch.** An observed artifact digest that differs from the frozen manifest blocks replay and blocks metric claims.
4. **F4 — group leakage.** A guide/Cas9 group appearing in both train and held-out partitions invalidates the declared group holdout and any associated generalization claim.

5. **F5 — partial-label sensitivity bounds.** Two measured positives and two unknown candidates with K=2 produce observed known-label recall 0.5, conditional recall bounds [1/3, 2/3], and precision bounds [0.5, 1.0]. The bounds vary unknown labels only inside this exact universe.

The dependency-free Rust metric kernel is now committed at [scripts/crispr_triage_verifier.rs](../../scripts/crispr_triage_verifier.rs). It implements native nomination coverage, Recall@K, point Precision@K with unknown-label abstention, slots-at-recall, partial-label bounds, SHA-256-claim format/match checks, and group-leakage validation. It includes unit tests for fixtures F1–F5 plus invalid targets and duplicate identities.

Run the unit tests with an installed Rust toolchain:

```sh
rustc --edition=2024 --test scripts/crispr_triage_verifier.rs -o /tmp/crispr-triage-tests
/tmp/crispr-triage-tests
```

The SHA-256 helper validates two already-computed digest claims; it does not hash artifact bytes. The production orchestrator must compute hashes from retrieved bytes using a trusted implementation.
