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

For a **paired policy comparison**, do not infer superiority merely from overlap/non-overlap of each policy's separate bounds. Both policies must have the same candidate IDs and label mapping, and an unknown site must receive the same latent label in both. At fixed K, let d = known-positive hits for A minus known-positive hits for B; a = unknown IDs selected by A but not B; b = unknown IDs selected by B but not A; n = all remaining unknown IDs in the universe (selected by both or neither). With P measured positives:

- lower numerator = d - b; lower denominator = P + b, plus n only if the lower numerator is positive;
- upper numerator = d + a; upper denominator = P + a, plus n only if the upper numerator is negative;
- paired label-sensitivity bounds = [lower numerator / lower denominator, upper numerator / upper denominator].

This is a sharp worst/best-case bound for Recall@K(A) - Recall@K(B) under unrestricted binary assignments to unknown labels inside the shared universe. A positive lower bound supports label-robust superiority *within that universe*; a bound spanning zero means observed-label improvement is not robust to all unresolved-label assignments. Neither outcome addresses positive sites omitted from every nomination method.

### I. From a universal score to a context-stratified Pareto frontier

Recent work makes clear that “off-target activity,” “specificity,” and “risk” change meaning across contexts and scales; therapeutic interpretation may depend on cell/tissue context and functional consequences, not just a molecular edit score. See [Stewart et al., *Nature Genetics* (2026)](https://doi.org/10.1038/s41588-025-02428-3) and [On-target and off-target activities of CRISPR therapeutics across scales, *Trends in Biotechnology* (2026)](https://doi.org/10.1016/j.tibtech.2026.07.015). Uncertainty-aware prediction itself is already an established line of work (for example, [crispAI, *Nucleic Acids Research* (2024)](https://doi.org/10.1093/nar/gkae759)); Symthaea must demonstrate an incremental value rather than relabel existing ideas.

The evaluation should **not** collapse molecular nomination, conditional ranking, editing magnitude, genomic/functional consequence, cell/tissue context, delivery, and treatment-access economics into one universal “CRISPR safety” score. These are distinct endpoints with different evidence requirements. This first vertical measures only computational candidate-triage performance; it cannot infer clinical harm from an off-target rank.

Instead, report a **Pareto frontier** across a preregistered vector of supported objectives, potentially including:
- maximize macro Recall@K at an exact review budget;
- minimize candidate review slots at a fixed achieved recall;
- maximize worst-guide/Cas9-group recall;
- minimize the paired label-sensitivity interval width, while still reporting the interval's position;
- minimize calibration loss only when probability and label semantics are valid;
- minimize the upper unknown-aware selective error-rate bound at fixed, matched coverage;
- maximize accepted-label completeness at a fixed coverage tier, while separately reporting unknown-label bounds;
- minimize measured runtime/compute.

The frontier keeps nondominated policies: no other policy is at least as good on every declared objective and strictly better on one. Different nondominated policies may represent legitimate trade-offs. Do not invent universal weights to force one winner, and do not convert missing/not-estimable values to zero. A policy enters a given frontier only when all of that frontier's preregistered metrics are available and comparable; report excluded policies and reasons separately.

**Frontier acceptance gate:** 100% of compared vectors use the same objective definitions, units, direction, held-out episodes and metric availability rules. The output includes every nondominated policy and its full vector. Frontier membership is not a benchmark win by itself: any claim of improvement still requires the primary predeclared outcome, uncertainty interval and ablations to pass.

### J. From price claims to causal cost accounting

Report computational resources and review slots first. Progress to human review minutes only when measured. Report per-qualified-finding cost only when numerator and denominator are observed end-to-end. Do not convert fewer ranked candidates into claims of lower experimental costs, clinical costs, or treatment prices without prospective accounting.

### K. From a single confidence number to selective, context-aware prediction

This is not a claim that uncertainty modelling is new. Özden and Minary's crispAI already models off-target cleavage uncertainty with a zero-inflated negative-binomial count model and reports calibrated uncertainty estimates ([NAR 2024](https://doi.org/10.1093/nar/gkae759)). The proposed contribution here is the **evaluation and evidence contract** around selective policies: matched risk–coverage, visible unknown-label bounds, per-context reporting, and incremental comparison against qualified uncertainty-aware specialists. The July 2026 14-tool benchmark also found substantial score overlap and unassayed candidate scores overlapping observed positives, reinforcing that calibration claims must disclose the labelled denominator and unknowns ([Kaufmann et al. 2026](https://doi.org/10.1177/10430342261466407)).

Probability and willingness to act are separate quantities. A calibrated probability may still be inappropriate to use in every context; a high-confidence but out-of-distribution case may need abstention. Freeze the confidence/abstention policy before holdout evaluation, keep exact ties at a threshold together, and report coverage alongside selective error.

On label-eligible cases, report Brier score, log loss (without silently clipping exactly wrong 0/1 predictions), fixed-bin reliability summaries and calibration-bin counts. Unknown/unassayed labels are excluded from those proper-scoring-rule denominators but counted explicitly; known-label calibration is conditional on ascertainment and is not a full-cohort guarantee. For every risk–coverage point, report observed-label risk and sharp error-rate bounds that assign each accepted unknown outcome either a correct or incorrect prediction. Those bounds are label-assignment bounds, not confidence intervals. The curve includes a zero-coverage origin; risk, bounds and accepted-label completeness are not estimable at zero accepted cases and must be null, not zero.

Produce the same curve and calibration summary separately for every preregistered context (at minimum the guide/Cas9 group when supported). A pooled curve may be reported as a supplement, never as a substitute for per-context results. Contexts with no observed labels have calibration marked not estimable, while their coverage and unknown-label census remain visible.

### L. From static candidate ranking to decision-critical measurement allocation

The next question is not only which policy ranks better, but **which unknown outcome should be measured next to determine whether the policy comparison changes?** For a frozen shared candidate universe and top-K comparison, an unknown candidate that is in exactly one policy's top K can directly change the paired positive-hit-count numerator (hits in A minus hits in B) if it proves positive. Measuring such a label resolves one unit of that specific ambiguity. This is a narrow structural value-of-measurement objective, not an expected-value-of-information estimate.

The initial planner ranks these decision-differential unknowns by declared positive cost units, selecting the cheapest first to maximize the number measured under a fixed budget when each has equal unit value. It explicitly counts unknowns selected by both policies or by neither; those may matter to denominator uncertainty, absolute recall bounds, or other questions, but they do not directly change this numerator component. The planner therefore reports residual ambiguity and must never claim it has resolved the entire paired sensitivity interval.

Benchmark the allocation against random, highest-confidence/score-first, and top-ranked-unknown-first allocation at identical cost budgets and on identical preregistered episodes. Report measurement spend, differential unknown labels resolved, unresolved count, context coverage, and full paired sensitivity bounds before/after actual permitted outcomes become available. Do not use post-hoc outcome knowledge to select labels for the planner; do not treat cost units as laboratory-dollar savings.

A related generalization warning applies to conformal or nominally calibrated uncertainty: distribution-free marginal coverage depends on exchangeability, and distribution shift can invalidate its guarantee. This project must evaluate coverage and selective risk by held-out guide/Cas9 and assay context instead of assuming in-distribution calibration transfers ([coverage under shift, 2026](https://arxiv.org/abs/2602.14913); [calibration failure diagnostics, UAI 2026](https://proceedings.mlr.press/v337/lee26g.html)). These are evaluation motivations, not evidence that conformal methods or any Symthaea policy has been tested here.

Research task: [CRISPR-OT-004 #7589 — allocate measurements to resolve decision-critical unknown labels](https://github.com/Luminous-Dynamics/symthaea/issues/7589).

### M. From point-estimate Pareto winners to interval-aware robust frontiers

A point-estimate frontier can overstate decisiveness when group-aware metric intervals, partial-label bounds, or cost ranges overlap. Retain intervals in the objective vectors and distinguish three relations:

- **Robust/definite dominance:** A's worst-case bound is no worse than B's best-case bound for every preregistered objective, and strictly better on at least one.
- **Possible-only dominance:** at least one assignment within the stated objective intervals permits dominance, but the bounds do not establish it for all assignments.
- **Unresolved or incompatible trade-off:** neither strict dominance direction follows from the intervals.

The resulting **robust frontier** removes only policies definitely dominated under these conservative bounds. Policies may remain on it because they are genuinely competitive or because the evidence is too imprecise to decide. Report robust dominance pairs and possible-only pairs separately; do not misread possible dominance as a win.

This is an established optimization concept, not a novel algorithmic claim. Recent interval-valued multi-objective research supports keeping uncertainty ranges instead of collapsing them to midpoints ([Information Sciences, 2026](https://doi.org/10.1016/j.ins.2026.123217)). For uncertainty under covariate shift, KMM-CP (UAI 2026) highlights the importance of support overlap and shift assumptions when making coverage claims ([paper](https://proceedings.mlr.press/v337/laghuvarapu26a.html)). Neither source validates this implementation or the CRISPR benchmark.

The Rust utility consumes intervals; it does not compute them. The caller must document their source and should use group-aware, paired, and where necessary simultaneous intervals across policies/objectives. Coordinate-wise intervals do not model covariance, and this conservative rule may leave a large frontier when evidence is weak. That is a meaningful “insufficient evidence to decide” result, not a defect to hide through arbitrary weights.

Research task: [CRISPR-OT-005 #7602 — interval-aware robust Pareto frontier](https://github.com/Luminous-Dynamics/symthaea/issues/7602).

### N. From source-cohort performance to bounded target-population transportability

A pooled source-cohort improvement is not automatically an improvement in a target population with different proportions of guides, assays, cell contexts, or editor families. Group shifts can alter policy utility even when every source row and metric is computed correctly. Target-group prevalence may itself be uncertain.

The transportability utility accepts context-specific paired Recall@K delta intervals plus preregistered lower/upper target-prevalence bounds. It computes the minimum and maximum mixture delta over all target weight vectors satisfying those bounds and summing to one, and emits the witness weight vector for each endpoint. This is exact for the declared bounded-simplex linear objective while allowing each context's delta to vary anywhere inside its interval; treating group deltas and weights as independent can be conservative where actual dependence is known but not represented.

- **Robustly positive:** the entire target-mixture interval is strictly above zero.
- **Robustly non-positive:** the entire interval is at or below zero.
- **Inconclusive:** the interval includes positive and non-positive values.

Every target-relevant context must be supported by a permitted metric episode. Unsupported contexts fail closed; they are never assigned a zero effect. Intervals must identify the group unit, evaluation episodes, paired/group-aware method, source/target population, and joint-coverage limitations. This is not a generic shift detector or a clinical transportability proof.

The research motivation is established rather than novel: group-weighted conformal prediction analyzes how group proportions affect coverage under covariate shift ([Bhattacharyya & Barber, 2026](https://doi.org/10.1214/26-EJS2506)); KMM-CP highlights support-overlap and selective-correction limits under shift ([UAI 2026](https://proceedings.mlr.press/v337/laghuvarapu26a.html)); and CS-WCP explicitly considers uncertain group proportions in a preprint ([2026 preprint](https://arxiv.org/abs/2609.27955)). This project does not claim novelty for shift-aware or mixture-robust evaluation. The proposed integration is the target-mixture envelope for paired CRISPR benchmark policy deltas, connected to the existing partial-label and interval-frontier contracts.

Research task: [CRISPR-OT-006 #7615 — robust target-mixture transportability](https://github.com/Luminous-Dynamics/symthaea/issues/7615).

### O. From shift envelopes to a robustness radius

The bounded transportability envelope tells whether any allowed target mixture can change the sign of a policy gain. The next useful question is: **how far must the group composition move from the observed source cohort before the nominal gain disappears?**

The shift-radius utility reports the minimum total-variation (TV) distance between source and target context weights that reaches a non-positive weighted effect, respecting the declared target-prevalence bounds. TV is half the L1 distance: a radius of 0.05 means five percentage points of population mass must be reallocated.

It calculates two different quantities:
- **Nominal radius:** use each context's preregistered point-estimate delta.
- **Conservative radius:** use each context's supplied lower effect bound.

A zero nominal radius means the source point estimate is already non-positive. A zero conservative radius means robust positivity is absent already at the source mix, even if the nominal estimate is positive. If no feasible target mixture can erase a positive effect within the declared bounds, the result is not erasable within those bounds. These are sensitivity outcomes, not statistical confidence intervals.

The implementation is [scripts/crispr_shift_robustness_radius.rs](../../scripts/crispr_shift_robustness_radius.rs). The exact optimization transfers mass from higher-effect to lower-effect contexts until the weighted effect reaches zero or the declared target-mixture capacities are exhausted. It validates witness weights and checks that TV distance equals half the L1 change. All group-effect values and target-prevalence bounds remain supplied inputs with independent provenance requirements.

This is an established robustness idea rather than a novelty claim. Ma and Glynn analyze the minimum distribution shift needed to violate a performance tolerance ([Operations Research](https://doi.org/10.1287/opre.2022.0658)); the 2026 Iterative Robust Satisficing work explicitly minimizes fragility under distribution shift ([PMLR 306](https://proceedings.mlr.press/v306/agirman26a.html)). This CRISPR-specific integration exposes the sign-change radius for paired Recall@K deltas alongside the nominal and conservative transportability envelope.

Research task: [CRISPR-OT-007 #7623 — minimum population-shift robustness radius](https://github.com/Luminous-Dynamics/symthaea/issues/7623).

## 2. Required benchmark tracks

| Track | Question | Primary measures | Cannot establish |
|---|---|---|---|
| T0 Source replay | Did we reconstruct the cited cohort and transformations? | exact counts, digest verification, exclusions, output census | model improvement |
| T1 Native nomination | Which measured positive sites were in a tool's original search output? | positive coverage, missed-positive count, per-guide coverage | ranking quality after seeding |
| T2 Conditional ranking | Given one declared universe, how well are measured positives ranked? | macro Recall@K, Precision@K, PR-AUC when labels support it, false-negative census | completeness of original nomination |
| T3 Budgeted review | How much review is needed for an agreed level of measured-positive recall? | recall at fixed slots; slots needed to match baseline recall; precision trade-off | actual dollar savings |
| T4 Calibration and abstention | Are probabilities reliable, and does the policy know when to abstain? | Brier/log loss, fixed-bin reliability, risk–coverage curve, observed-label risk and unknown-label risk bounds, per-context abstention | clinical safety or unobserved biological outcomes |
| T5 Shift and robustness | Does the effect persist on unseen groups and independent data? | leave-guide/Cas9-group-out delta, leave-assay-family-out delta, independent-cohort delta, group-level uncertainty | universal generalization from one small cohort |
| T6 Incrementality | Does Symthaea add value over the same specialist predictions? | paired delta at equal budget; paired partial-label bounds; ablations; runtime/resource delta | benefit attributable to an upstream model unless ablated |
| T7 Evidence/replay integrity | Can an independent evaluator detect tampering or leakage? | deterministic replay, exact manifest, negative controls, digest/count agreement | biological truth from cryptography alone |
| T8 Partial-label identification | How much do unresolved labels change metric interpretation? | observed-label estimate, worst/best-case recall and precision bounds, unknown-label census | sites omitted from every nomination method; truth outside the fixed candidate universe |
| T9 Multi-objective frontier | Which policies offer genuine trade-offs without arbitrary score weights? | complete objective vectors, dominated-policy count, Pareto membership, excluded-policy reasons | a universal safety score or proof of biological/clinical improvement |
| T10 Selective prediction | What error/uncertainty trade-off occurs as the policy abstains, and where does calibration fail? | Brier/log loss on measured labels, fixed-bin reliability, full risk–coverage curve, unknown-aware error bounds, per-context label completeness | clinical safety or outcomes missing from the evidence universe |
| T11 Decision-critical measurement allocation | Which unknown outcomes most directly resolve disagreement in a preregistered top-K policy comparison? | differential unknown labels resolved at fixed declared cost, residual numerator-ambiguity count, context coverage, before/after paired label-sensitivity bounds | full expected value of information, total uncertainty resolved, experimental savings, biological safety |
| T12 Interval-aware robust frontier | Which policy dominance claims survive interval uncertainty in every objective? | robust frontier, definitely dominated policies, robust dominance pairs, possible-only dominance pairs, point-frontier disagreements | intervals generated, simultaneous coverage, biological/clinical safety |
| T13 Target-mixture transportability | Does policy improvement survive every preregistered plausible target-context mixture? | lower/upper target-mixture delta, endpoint witness weights, robust-positive/non-positive/inconclusive classification, source-versus-target sign change | unsupported contexts, target-population truth beyond declared bounds, clinical safety |
| T14 Shift robustness radius | How much feasible context-composition change erases a nominal or conservative policy gain? | nominal and lower-bound TV radius, sign-change witness, source effect and target bounds | real-world shift size, unseen contexts, clinical safety |

Tracks are reported separately. A composite “CRISPR score” that hides failed nomination, calibration, or generalization is prohibited.

## 3. Metric definitions and denominators

A **benchmark episode** is a pinned combination of source cohort, assay context, guide/Cas9 group, reference genome/version, candidate universe, label policy, split and prediction snapshot. Every metric must identify its episode(s).

- **Native positive coverage** = unique measured positive sites present in native tool output / unique measured positive sites eligible in the reference cohort. Count a miss when the native generator never nominated the site. Publish numerator and denominator.
- **Recall@K** = measured positives among the first K candidates / all eligible measured positives in the declared ranking universe. State explicitly whether positives were added to that universe after the native search phase. If fewer than K candidates exist, report **not estimable at budget K** rather than silently evaluating a smaller budget. K=0 is not an impact result.
- **Precision@K** = measured positives among the first K candidates / K reviewed candidates, only when outcomes for the selected candidates are actually known under a declared label policy. Do not treat unassayed candidates as negatives to manufacture this value.
- **Slots@R** = smallest K at which a policy meets a preregistered recall R on the same eligible positive reference and same candidate universe. If it never reaches R, report **not reached**, not a fabricated K.
- **Fixed-budget delta** = Symthaea policy metric minus the strongest fair baseline metric at the same fixed K on identical episodes. The primary comparison is paired by guide/Cas9 group.
- **Calibration:** on measured outcomes with defensible positive/negative semantics, report Brier score, log loss, fixed-bin reliability data and bin counts. Do not silently clip overconfident errors at probabilities 0 or 1. Report the unknown-label count next to all loss metrics; these scores are conditional on label ascertainment. ECE may be included but is not a stand-alone calibration certificate. If no valid labels exist, calibration is **not estimable**.
- **Selective risk–coverage:** accept higher-confidence cases first under a frozen policy; cases tied exactly on confidence enter together. At each point report accepted count, coverage of the entire declared universe, observed-label coverage, label completeness among accepted cases, observed-label risk, and error-rate bounds that allow every accepted unknown label to be either correct or incorrect. The observed-label error rate is conditional on ascertainment; the bound is not a statistical confidence interval.
- **Context stratification:** repeat calibration and the complete risk–coverage curve per preregistered context. Never use a pooled aggregate to conceal a context with no labels, high unknown fraction, or high observed error.
- **Uncertainty interval:** resample the highest independent unit supported by the study—normally guide/Cas9 group, not individual rows—using a paired method for the policy difference. With only 26 source groups, show every group result and state that interval precision/generalization is limited. Do not use row-level bootstrap to create artificial sample size.
- **Resource accounting:** report candidate slots and, when measured, wall-clock time, CPU/GPU model and utilization, peak memory, and human review minutes. Keep measured quantities separate from projected costs.

- **Partial-label bounds:** If unknown candidates exist within the fixed ranking universe, report the known-label metric and worst/best-case assignments to those unknown labels. For P > 0, measured positives pK in the top K, U total unknown candidates, and uK unknown candidates in the top K, conditional recall bounds are [pK / (P + U - uK), (pK + uK) / (P + uK)]. Precision bounds are [pK/K, (pK+uK)/K] when K candidates are available and K > 0. These are not confidence intervals and do not cover positives absent from the declared candidate universe.

Report macro (equal-group-weighted) and micro (pooled) summaries together. Include per-guide/Cas9 results and all invalid/omitted cases. Do not let a large group conceal a failure on a small group.

## 4. Predeclared targets

These are **project engineering targets, not published standards or clinical acceptance criteria**.

### Primary success: one of two alternatives

1. **Equal budget:** at least **+5 absolute percentage points** in macro Recall@K versus the strongest fair baseline on predeclared held-out guide/Cas9 groups at the same K; or
2. **Equal achieved recall:** at least **20% fewer candidate review slots** to reach the baseline's preregistered recall target on the same episode set.

### Secondary selective-prediction targets

These are engineering aspirations for eligible, measured-label cohorts—not clinical standards:

- Always publish risk–coverage points at every unique preregistered confidence tier and the full curve; do not select a favorable single threshold after seeing results.
- Compare Brier score and log loss against the strongest fair calibrated baseline on exactly the same held-out, label-eligible cases; an aspirational target is at least a **10% relative reduction in Brier score** without worse log loss, with group-paired uncertainty reported. If the interval does not exclude no improvement, call the result inconclusive.
- At predeclared coverage levels (50%, 80%, and 90%, where attainable without splitting confidence ties), report both observed-label risk and unknown-label error bounds. Compare only matched coverage policies and identical context episodes; an unestimable point is not replaced with the nearest available threshold.
- Report each context's calibration, coverage and label completeness separately. No target here implies that an off-target prediction estimates clinical harm or establishes editing safety.

For either primary alternative:

- endpoint, K or recall target, exclusion policy, comparator, split, interval procedure and analysis are frozen before holdout reveal;
- report precision, false negatives and per-group deltas alongside the headline;
- where unknown candidates exist, report the paired label-sensitivity bound for the policy delta; call the gain label-robust within the fixed universe only when its worst-case lower bound is above zero;
- the paired group-aware statistical interval must exclude no improvement for a “demonstrated improvement” claim;
- if the interval includes no improvement or the eligible sample is inadequate, report **null/inconclusive**, not success;
- cross-dataset robustness requires at least one separate, independently sourced cohort with explicit permission for the intended use.

### Qualification targets that are binary gates

- **100%** of used artifacts have source identity, terms/permission status, retrieval/version information and verified SHA-256 digests recorded.
- **100%** of candidate outcomes use explicit positive / assay-defined-negative / unknown semantics; unknowns are never silently imputed as true negatives.
- **100%** of reported tables identify the candidate universe, seed status, group unit, denominator and missingness policy.
- Every fixed-budget comparison has at least K candidates for every policy, or that policy's metric is marked not estimable; no shorter list is silently treated as the same budget.
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

**Artifact-rights audit finding (2026-10-16):** the linked [Zenodo record](https://zenodo.org/records/20627722) identifies `offtarget_prediction_benchmark_zenodo_artifacts_v1.zip` (190.7 MB) and exposes MD5 `611cb28ab0516e471c5dc59ea44a330a`. In the retrieved record view, the Rights/License section displayed no license value. This does not prove the dataset is unlicensed, but permission is unresolved. Do not download, ingest, or use the archive for model/benchmark development until the record metadata or rights holder explicitly confirms a license/permission covering the intended use. The reproduction repository's MIT code declaration is not a data license. The GitHub-tracked compact truth table's data rights also require explicit checking.

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

The machine-readable companion contract contains thirteen synthetic fixtures intended to become deterministic unit tests for the Rust scorer/verifier:

1. **F1 — nomination vs seeded ranking.** Four eligible positives, two in the native nominations: expected native coverage is 2/4 = 0.5. Later seeding the two missing positives into a ranking universe allows Recall@5 to reach 1.0, but native nomination coverage remains 0.5.
2. **F2 — unknown is not negative.** If an unknown/unassayed candidate is inside the review set, precision is not estimable under the declared label semantics; it may not be silently counted as a true negative.
3. **F3 — digest mismatch.** An observed artifact digest that differs from the frozen manifest blocks replay and blocks metric claims.
4. **F4 — group leakage.** A guide/Cas9 group appearing in both train and held-out partitions invalidates the declared group holdout and any associated generalization claim.

5. **F5 — partial-label sensitivity bounds.** Two measured positives and two unknown candidates with K=2 produce observed known-label recall 0.5, conditional recall bounds [1/3, 2/3], and precision bounds [0.5, 1.0]. The bounds vary unknown labels only inside this exact universe.

6. **F6 — paired policy delta.** Two policies have the same five candidate IDs and labels, two measured positives, and three unknowns. At K=2 each finds one measured positive; they each uniquely select one different unknown, and one unknown is neutral. The observed known-label delta is 0, but the sharp paired label-sensitivity bounds are [-1/3, +1/3]. This is a comparison bound, not a confidence interval.

8. **F8 — Pareto frontier without scalarization.** Compare four synthetic policies on macro recall (maximize), review slots (minimize), and worst-group recall (maximize). Three policies have real trade-offs and remain on the frontier; a fourth is worse than at least one policy on every objective and is removed. No weighted sum or universal “safety score” is produced.

The second dependency-free Rust helper, [scripts/crispr_decision_frontier.rs](../../scripts/crispr_decision_frontier.rs), implements objective-direction-aware dominance and sorted Pareto membership, rejecting duplicate IDs/objective names, non-finite or incomplete vectors. Its 10 unit tests are authored but have not been compiled or executed because `rustc` is unavailable in this authoring environment.

Run its tests when a Rust toolchain is available:

```sh
rustc --edition=2024 --test scripts/crispr_decision_frontier.rs -o /tmp/crispr-frontier-tests
/tmp/crispr-frontier-tests
```

The follow-on research task is [CRISPR-OT-002: context-stratified Pareto frontier for evidence triage (#7571)](https://github.com/Luminous-Dynamics/symthaea/issues/7571).

7. **F7 — underfilled fixed budget.** A ranking contains one candidate, but the declared review budget is K=2. Recall@2 and its partial-label sensitivity metric must return `not_estimable_at_budget_K`; neither metric may silently score the one-candidate list as a two-slot comparison.

The dependency-free Rust metric kernel is now committed at [scripts/crispr_triage_verifier.rs](../../scripts/crispr_triage_verifier.rs). It implements native nomination coverage, Recall@K, point Precision@K with unknown-label abstention, slots-at-recall, individual and paired partial-label bounds, SHA-256-claim format/match checks, and group-leakage validation. It includes unit tests for fixtures F1–F7 plus underfilled fixed budgets, invalid targets, duplicate identities/ranks, reference-label contradictions, and mismatched candidate universes.

Run the unit tests with an installed Rust toolchain:

```sh
rustc --edition=2024 --test scripts/crispr_triage_verifier.rs -o /tmp/crispr-triage-tests
/tmp/crispr-triage-tests
```

The SHA-256 helper validates two already-computed digest claims; it does not hash artifact bytes. The production orchestrator must compute hashes from retrieved bytes using a trusted implementation.

**Independent formula check (2026-10-16):** a separate deterministic checker compared the paired closed-form sensitivity bounds with brute-force assignments of all unknown labels across 1,763,520 paired-ranking cases; no bound mismatches were found. This verifies the formula design against the enumerated cases, not the Rust implementation. Rust compilation/unit tests and any biological cohort replay remain unexecuted.


9. **F9 — calibration, abstention and unknown-aware risk.** The synthetic calibration slice contains two measured outcomes (p=0.9 positive and p=0.2 assay-defined negative) plus one unknown (p=0.8). Expected Brier score is 0.025 and five-bin expected calibration error is 0.15, with the unknown explicitly counted and excluded from the loss denominator. In a separate three-case risk slice, one of two observed outcomes is wrong and one accepted outcome is unknown; the full-coverage observed-label risk is 0.5 and the unknown-aware error-rate bounds are [1/3, 2/3]. Exact confidence ties enter as a tier, the full curve starts at zero coverage (undefined risk/bounds are null), and a context with no measured labels has no calibration estimate.

Cross-domain contract fixture: [SELECTIVE_RISK_CROSS_DOMAIN_V1.json](../../research/selective-risk/fixtures/SELECTIVE_RISK_CROSS_DOMAIN_V1.json) defines the expected zero-coverage origin, inclusive tied-score tiers, unknown-label identification bounds and denominator semantics shared with Broca retrieval research. Each adapter must supply its own predeclared failure event. The fixture is arithmetic-checked, but neither Rust implementation has run it yet.

The new dependency-free Rust implementation is [scripts/crispr_selective_risk.rs](../../scripts/crispr_selective_risk.rs). It provides fixed-bin calibration summaries, Brier/log loss, deterministic risk–coverage curves, unknown-label error bounds, and context-stratified reports. Fifteen unit tests are authored but not compiled or executed in the current authoring environment because rustc is unavailable. When a Rust toolchain is available, run:

    rustc --edition=2024 --test scripts/crispr_selective_risk.rs -o /tmp/crispr-selective-tests
    /tmp/crispr-selective-tests

Tracking issue: [CRISPR-OT-003 #7580 — calibration-first selective prediction](https://github.com/Luminous-Dynamics/symthaea/issues/7580).

These code and fixture additions are methodological tooling only. No data cohort, biological model, or Symthaea improvement has been evaluated.


10. **F10 — decision-critical measurement allocation.** On a shared six-candidate universe with K=3, two unknown outcomes are differentially selected (one top-K only in policy A with cost 2, and one top-K only in policy B with cost 1). Another unknown is selected by both and one by neither. With cost budget 2, the plan selects the cheaper B-only outcome, spends 1 unit, leaves 1 unit unused, and reduces the unresolved direct paired-numerator ambiguity count from 2 to 1. An independent exhaustive oracle enumerated 21,844 cost vectors (1–7 candidates, cost units 1–4) across 240,284 budget comparisons (budgets 0–10); cheapest-first matched the maximum-cardinality subset found by exhaustive enumeration in every case (zero mismatches). This validates the narrow greedy cardinality rule, not the Rust implementation. It does not claim to resolve common-label denominator uncertainty or the full paired sensitivity bounds.

The dependency-free implementation is [scripts/crispr_measurement_priority.rs](../../scripts/crispr_measurement_priority.rs). It computes top-K membership from complete one-based rankings for both policies on the shared universe; it does not accept caller-supplied disagreement flags. The 19 authored tests cover cost-bounded allocation, policy-exclusive membership, ties, labels, cardinality, duplicate/gapped rankings, invalid identities, and zero cost. They have not been compiled or executed because `rustc` is unavailable in the authoring environment. Run:

    rustc --edition=2024 --test scripts/crispr_measurement_priority.rs -o /tmp/crispr-measurement-tests
    /tmp/crispr-measurement-tests

This allocation method is deliberately narrow and deterministic. Its proposed 20% improvement over random allocation at equal cost is an evaluation target, not a current result; compare score-first and top-ranked-unknown baselines as well. No assay or biological data have been run.


11. **F11 — robust frontier under objective intervals.** Compare three synthetic policies on macro recall (maximize) and review slots (minimize). `robust` has recall [0.90, 0.92] and slots [90, 100]; `uncertain` has recall [0.80, 0.93] and slots [95, 130]; `dominated` has recall [0.70, 0.75] and slots [120, 140]. The robust policy definitely dominates `dominated`, but its intervals overlap the trade-off policy's in ways that prevent definite dominance. Expected robust frontier: [`robust`, `uncertain`]; definitely dominated: [`dominated`]. Possible-only dominance is reported separately. These are synthetic ranges, not statistical confidence intervals or biological results.

The dependency-free Rust utility is [scripts/crispr_interval_frontier.rs](../../scripts/crispr_interval_frontier.rs). Fifteen unit tests are authored but not compiled or executed because `rustc` is unavailable. It rejects duplicate identifiers/objective names, dimensional mismatches, non-finite bounds and reversed intervals. Its relation semantics are documented in source: robust dominance requires A's worst bound to beat B's best bound on all objectives; possible dominance is only a feasible relation under coordinate-wise interval uncertainty. The module does not derive intervals or encode covariance, pairing, or multiple-comparison correction.

    rustc --edition=2024 --test scripts/crispr_interval_frontier.rs -o /tmp/crispr-interval-tests
    /tmp/crispr-interval-tests

Independent F11 arithmetic matches the expected robust frontier and all three possible-only dominance pairs. A separate endpoint oracle checked 5,184 two-objective interval-pair vectors across all four maximize/minimize direction combinations, and 82,944 endpoint assignments; possible- and robust-dominance relations had zero mismatches. This validates only the mathematical relation definitions for those enumerated intervals, not the Rust source or any statistical/biological claim.

The next acceptance check is independent F11 evaluation plus point-frontier vs interval-frontier disagreement reporting on exactly the same metric episodes. No biological cohort or Symthaea improvement has been evaluated.


12. **F12 — bounded target-context mixture.** Context A has paired Recall@K delta interval [0.08, 0.12] and target prevalence [0.2, 0.5]; B has [0.02, 0.07] and [0.3, 0.6]; C has [-0.25, -0.15] and [0.1, 0.4]. Feasible weight vectors must sum to one. A representative permitted nominal mix (A=0.35, B=0.45, C=0.20) evaluated at interval midpoints produces +0.01525; however the bounded-mixture envelope is [-0.076, +0.073], therefore the target-mixture conclusion is **inconclusive**. Lower-bound witness is A=0.2, B=0.4, C=0.4; upper-bound witness is A=0.5, B=0.4, C=0.1. This demonstrates how a positive point estimate can fail transportability robustness.

The dependency-free implementation is [scripts/crispr_transportability.rs](../../scripts/crispr_transportability.rs). Sixteen Rust unit tests are authored but not compiled or executed because `rustc` is unavailable. It rejects unsupported target contexts and infeasible prevalence bounds; emits both witness mixtures; and never fills an unsupported context with zero. The interval bounds are provided inputs, not estimated by this utility.

    rustc --edition=2024 --test scripts/crispr_transportability.rs -o /tmp/crispr-transport-tests
    /tmp/crispr-transport-tests

An independent grid oracle checked all 12 feasible tenth-step prevalence vectors for F12 and all 8 endpoint combinations of the three context delta intervals (96 assignments total). A broader deterministic oracle checked 2,000 feasible three-context fixtures, 10,856 feasible prevalence vectors, and 86,848 endpoint assignments on 0.1 grids, with zero mismatches against exhaustive enumeration. It reproduced the exact lower bound -0.076 at [0.2, 0.4, 0.4] and upper bound +0.073 at [0.5, 0.4, 0.1]. This is a direct synthetic calculation, not Rust execution or biological evidence.


13. **F13 — population-shift robustness radius.** Use F12's source weights A=0.35, B=0.45, C=0.20; nominal context deltas A=0.10, B=0.045, C=-0.20; lower bounds A=0.08, B=0.02, C=-0.25; and target bounds A=[0.2,0.5], B=[0.3,0.6], C=[0.1,0.4]. The nominal source-mixture effect is +0.01525. The minimum nominal TV shift to reach zero is 0.0508333333, with witness A=0.2991666667, B=0.45, C=0.2508333333. The conservative source mixture is already -0.013, so the lower-bound radius is 0. This exposes the difference between a point estimate with a nonzero robustness radius and a conservative conclusion that is not positive even before any shift.

Thirteen Rust tests are authored in the new module; they are not compiled/executed because `rustc` is unavailable. An independent vertex-enumeration oracle compared the greedy radius with exact feasible-polytope vertices on 2,500 deterministic three-context fixtures, for both nominal and lower-bound effects (5,000 comparisons). There were zero mismatches, including 4,041 erasable and 959 non-erasable cases. The oracle included source-weight breakpoints for the TV objective. This checks the mathematical optimizer against the tested finite-dimensional formulation, not the Rust source or biological truth.

    rustc --edition=2024 --test scripts/crispr_shift_robustness_radius.rs -o /tmp/crispr-shift-radius-tests
    /tmp/crispr-shift-radius-tests


### P. From convenience verification to probability-sampled audit

Unknown-label bounds remain essential, but they do not make the *measured subset* representative when outcomes are preferentially verified for candidates already nominated or highly ranked by a tool. Diagnostic-test literature calls an analogous failure partial-verification or work-up bias: the score being evaluated can influence who receives the reference test ([overview and correction strategies](https://pubmed.ncbi.nlm.nih.gov/35124189/)). The 2026 CRISPR benchmark separately describes genome-wide nomination tools and pair scorers, and reconstructs a union of nominated sites; ranking-only evaluation on that constructed universe cannot establish full nomination coverage ([Kaufmann et al., 2026](https://doi.org/10.1177/10430342261466407)).

The next experimental-design paradigm is a **probability-sampled verification audit** over a frozen, complete candidate frame. A design may deliberately oversample score bands or model disagreements, but each candidate must have a known, nonzero inclusion probability and the selection mechanism must not use the unobserved outcome. Freeze the full ranking and K before revealing newly measured outcomes. Then use inverse-inclusion-probability (Horvitz–Thompson) sums to estimate the total positive count and the positive count within top K; report design-weighted ratio estimates for recall and precision alongside the unweighted audit-subset metrics.

This estimates performance *within the declared frame*, not sites absent from every tool's candidate universe. A convenience sample with unknown inclusion probabilities, any zero-probability stratum, a changed rank after outcomes are seen, or an unverified ranking census must fail closed. The kernel reports effective sample size as a diagnostic but computes **no confidence intervals**; any real-cohort comparison still needs a design-appropriate uncertainty procedure, guide/Cas9-group reporting, permitted sources, and independent replay. It does not mark unassayed sites as negatives.

Tracking task: [CRISPR-OT-008 #7633 — probability-sampled verification to correct ascertainment bias](https://github.com/Luminous-Dynamics/symthaea/issues/7633).

**F14 synthetic contract check (2026-10-16):** four audited outcomes are drawn from a declared ten-candidate ranking, with unequal inclusion probabilities. The hand-computed inverse-probability sums estimate 10 candidates, six positives overall, four candidates in top K=4, and two positives in top K. The weighted ratio recall is 1/3, while naïve audit-subset recall is 0.5. This is a synthetic arithmetic demonstration—not a Rust test, a real sampling audit, a biological result, or evidence of Symthaea improvement.

Dependency-free Rust module: [scripts/crispr_verification_bias.rs](../../scripts/crispr_verification_bias.rs). The 21 authored unit tests are not compiled or executed in this authoring environment because `rustc` is unavailable. The kernel validates declared design fields and sampled row identity/propensity structure, but cannot prove the sample was truly randomized or that the caller's full-frame manifest is true.

    rustc --edition=2024 --test scripts/crispr_verification_bias.rs -o /tmp/crispr-verification-bias-tests
    /tmp/crispr-verification-bias-tests


**Independent estimator oracle (2026-10-16):** exhaustive design-expectation checks covered 2,048 deterministic finite-population fixtures and 128,512 non-empty sample subsets, using unequal candidate inclusion probabilities and conditional Bernoulli selection designs. The four Horvitz–Thompson count totals (candidate count, population positives, top-K candidates, and top-K positives) matched their true finite-population totals in expectation; maximum absolute numerical error was below 6e-15. This validates the count-estimator equations on the enumerated designs. It does not establish that a real sample followed its declared design, validate ratio-estimator unbiasedness, generate confidence intervals, compile Rust, or demonstrate biological improvement.


**Resolved-label inclusion guard:** the kernel now uses an explicit `AuditOutcome` enum. A sampled row marked `UnknownUnassayed` is rejected; it cannot silently become an assay-negative. The inclusion probability must refer to entry into the final resolved, analyzable-label sample, not merely an audit invitation. Any nonresponse/assay-failure mechanism must be justified by the independent audit design; the kernel does not estimate response propensities.
