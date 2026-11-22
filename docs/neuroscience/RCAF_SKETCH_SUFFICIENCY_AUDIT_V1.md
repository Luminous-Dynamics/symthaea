# RCAF Conditional-Score Sketch Sufficiency Audit v1

**Status:** executable synthetic diagnostic of conditional null-score distributions within a predeclared covariate sketch.  
**Base:** weighted continuous-shift calibration and estimator-response outputs.  
**Purpose:** test whether covariate compression leaves a measurable change in the conditional null-score law inside the compressed sketch cells.

## 1. The research question

A sketch maps the original acquisition covariates X to a compressed representation Z=T(X). The intended benefit is to reduce the cost of importance weighting by making the source and target covariate distributions more similar after compression.

The risk is that the sketch discards covariate information that still changes the nuisance-only score distribution. If motion burden, scanner drift or physiology changes the score conditional on Z, then matching or weighting only on Z can conceal residual shift.

RCAF therefore adds a held-out score-distribution audit after a sketch has been selected. The module compares the source and target empirical null-score CDFs **within each predeclared sketch cell**, separately for each of the three RCAF estimator statistics.

## 2. What is measured

For estimator e and sketch cell g, let F_source(g,e) and F_target(g,e) be the conditional null-score distributions in that cell. The empirical diagnostic is the two-sample Kolmogorov–Smirnov distance:

`D_hat(g,e) = sup_t |Fhat_source(t|g,e) - Fhat_target(t|g,e)|`.

The module then adds simultaneous Dvoretzky–Kiefer–Wolfowitz radii for the source and target empirical CDFs:

`r(n) = sqrt(log(4 * G * M / delta) / (2*n))`

where G is the number of predeclared sketch cells, M=3 is the number of estimator score channels, and delta is the family-wise confidence error.

The reported envelope is:

`L_hat(g,e) = min(1, D_hat(g,e) + r(n_source,g) + r(n_target,g))`.

The factor (4GM) accounts for two-sided DKW tails across both domains, every declared cell and all three estimators. Under independent or exchangeable observations within each source/target cell, a union bound gives simultaneous coverage at least (1-delta) for the empirical conditional CDF deviations.

This is an auditable **empirical conditional score-CDF discrepancy diagnostic**. It is not the theoretical sketched-calibration leakage `Delta_T` from the cited paper, and it does not bound every aspect of the full conditional response law. It audits three scalar RCAF null-score distributions only.

## 3. What the code does

The Rust module is `research/rcaf/src/sketch_sufficiency_audit.rs`.

- Source and target observations carry a stable replicate ID, a predeclared sketch-cell ID, and three null scores in pinned RCAF estimator order.
- Source and target splits and replicate IDs must be distinct.
- Every source sketch cell must belong to the predeclared support.
- Unseen target sketch cells are retained as explicit `abstain_unseen_sketch_cell` records.
- Expected cells with missing source support, missing target support or too few observations receive no leakage bound.
- The report includes empirical KS distances, source/target DKW radii, per-cell upper envelopes and the worst expected-cell result for each estimator.
- An overall worst-cell bound is emitted only when every expected sketch cell is audited.

The implementation does not learn a sketch, estimate a density ratio, choose cells after seeing target scores or claim arbitrary continuous-shift validity.

## 4. Required statistical discipline

The DKW result assumes independent or exchangeable observations within each domain and sketch cell. Repeated time windows from one participant are not independent replicates. Real neuroimaging use must split or aggregate at participant/session/site cluster level and account for cluster dependence.

The cell definition must be selected before target outcomes are inspected. An overly fine sketch produces under-resolved cells and abstention; an overly coarse sketch can hide residual conditional score shift. The report exposes both failure modes rather than silently merging or borrowing neighbouring cells.

The audit's simultaneous bound covers the three declared RCAF score CDFs over the expected cells. It does not automatically cover post-hoc selection among other metrics, alternative sketch definitions, multiple atlases or different preprocessing pipelines. Those choices require a pre-registered family or an additional multiplicity correction.

Unseen sketch cells do not receive a borrowed bound. Missing target or source support is not reported as zero discrepancy.

## 5. Synthetic three-cell integration fixture

The CLI smoke example compares source and target null-score distributions across:
- `low_motion`;
- `medium_motion`;
- `high_motion`.

The first cell has identical source and target score distributions. The other cells have increasing synthetic conditional score shifts. Ten target observations from an unseen `new_scanner` sketch cell are deliberately included and must abstain.

There are 99 source and 99 target observations in each declared cell. The fixture validates:
- all three estimator score channels;
- the simultaneous DKW radius across 3 cells × 2 domains × 3 estimators;
- monotone empirical KS response to the constructed shift;
- an upper envelope no smaller than the empirical KS distance;
- explicit abstention for the unseen target cell;
- no overall bound when any expected cell is missing or under-resolved.

This is a deterministic software fixture, not evidence that the sketch is sufficient for a real acquisition population.

## 6. How this fits weighted conformal calibration

The weighted continuous-shift module handles known or estimated density-ratio weights in the full covariate space. A sketch can potentially lower the chi-square cost of weighting, but that benefit is scientifically useful only if the discarded covariate variation does not substantially alter the conditional score law within the sketch representation.

The correct comparison is therefore not only:
- full-covariate versus sketched importance-weight ESS;
- chi-square divergence and maximum normalized weight;
- finite-threshold rate and test-point mass abstention.

It must also include:
- per-cell source/target conditional score KS;
- simultaneous DKW leakage envelopes;
- under-resolved and unseen-cell abstention;
- stability across alternative, pre-registered sketches;
- participant/site-held-out validation.

A sketch should be preferred only when it reduces weighting cost **without unacceptable conditional score leakage**. This module measures one necessary diagnostic; it does not by itself certify sketch sufficiency.

## 7. Late-2026 research anchors

- Mehrdad Pournaderi (2026), *Sketched Calibration for Conformal Prediction under Covariate Shift*, arXiv:2610.09208 (6 October 2026). Studies calibration sketches, the chi-square cost of covariate shift, and a conditional-score leakage term for discarded covariates. https://arxiv.org/abs/2610.09208
- Mehrdad Pournaderi (2026), *Weighted conformal prediction under covariate shift: process limits, estimated weights and efficiency*, arXiv:2611.20052 (15 November 2026). Studies weighted conformal process limits, likelihood-ratio estimation error and second-moment / chi-square effects. https://arxiv.org/abs/2611.20052
- Aabesh Bhattacharyya and Rina Foygel Barber (2026), *Group-weighted conformal prediction*, Electronic Journal of Statistics 20(1):1171–1199, DOI 10.1214/26-EJS2506. https://doi.org/10.1214/26-EJS2506

## 8. Claim limits

This v1 audits empirical conditional CDF differences for three scalar RCAF null scores inside predeclared sketch cells. It does not estimate the theoretical `Delta_T`, prove that a sketch preserves the full conditional response law, establish arbitrary covariate-shift validity, or demonstrate empirical scanner/site or biological sufficiency. An unseen or under-resolved sketch cell abstains rather than receiving a fabricated leakage bound.
