# Multi-list ascertainment and unseen-error sensitivity protocol V1

**Status:** research protocol only; no multiple-systems estimator or natural-data result is implemented.
**Tracking:** [SCI-CONNECTOMICS-009 / #9248](https://github.com/Luminous-Dynamics/symthaea/issues/9248).
**Design-based anchor:** [SCI-CONNECTOMICS-006 / #9204](https://github.com/Luminous-Dynamics/symthaea/issues/9204).

## 1. Why candidate recall still has a denominator problem

The #9204 probability-sampled audit improves on evaluation only within the proposal pool: it samples eligible spatial/neuron PSUs independently and asks experts to enumerate errors. But its total-error denominator still assumes that the exhaustive-search protocol finds all eligible errors in each sampled PSU. Exhaustiveness is an operational attestation, not direct proof that no error escaped every reviewer and detector.

This protocol introduces multiple-systems estimation (MSE), also called capture–recapture, as a sensitivity analysis for the all-lists-missed population. It does not replace the probability sample, and its estimates are never labelled observed ground truth.

## 2. Independent discovery lists

On each probability-sampled PSU, run several discovery lists independently before any list shares locations or candidate identities:

1. Frozen current candidate generator.
2. A separately developed EM/image model or specialist.
3. A morphology / mesh / skeleton detector with separately versioned rules.
4. Blinded expert search team A.
5. Blinded expert search team B or a different, genuinely independent review protocol.

Five lists are preferred, with at least four meaningfully distinct ascertainment mechanisms as the target. A list is not independent merely because it has a different name. Record model ancestry, shared training sets, image inputs, feature extractors, rule sources, reviewer instructions, referrals, and whether prior lists influenced later search.

All discovery lists must be frozen before any outcome is seen. Record the capture result before cross-list linkage and adjudication. Search teams must not receive model candidate locations or one another's discoveries.

## 3. Error-event ontology and identity linkage

Define the unit being counted (a unique proofreading error event, not a crop, candidate row, or reviewer action), the closed acquisition/revision window, PSU boundary, and family vocabulary before collection.

After independent discovery, link observations that refer to the same underlying error. Preserve exact, probabilistic, and ambiguous linkage separately. Independent adjudicators should be blind to list identity, candidate rank, model confidence, and treatment assignment.

False positive proposals are retained as rejected discoveries for list-quality reporting but are not counted as confirmed population events. Indeterminate adjudications remain a distinct state and enter prespecified sensitivity analyses; they must not be silently treated as correct or incorrect.

## 4. Capture-history data

For each confirmed error event, record a binary vector identifying which lists found it. With five lists, an observed history such as 10110 means lists 1, 3, and 4 found the same event. Every nonzero history pattern must be retained in the contingency table.

The all-zero pattern is not observed. MSE estimates this missing cell under an identifying assumption. The estimate is therefore conditional on the model, linkage, closed-population definition, and assumptions about detection heterogeneity and list dependence.

## 5. Estimation strategy and refusal rules

Preregister a model set rather than selecting a model after observing which result looks most plausible:

- Independence benchmark, where scientifically defensible.
- Log-linear models with explicitly selected pairwise list interactions.
- Stratified or covariate-adjusted models for error family, morphology/compartment, image quality, depth, region, species, and source dataset.
- Latent detection-heterogeneity models and Bayesian sensitivity models with explicit priors and prior predictive checks.

Each model must state its identifying restriction for the all-zero cell. An observed-cell goodness-of-fit statistic or AIC/BIC ranking alone cannot validate that restriction. Report estimates across all defensible preregistered models and their sensitivity range.

Return NOT_IDENTIFIABLE or NOT_ESTIMABLE when plausible dependence assumptions, sparse overlaps, linkage uncertainty, or priors produce materially different unseen counts. Do not promote a point estimate merely because an optimizer converged. Do not silently assume independence among reviewers or models that share data, architectures, labels, or instructions.

The primary #9204 design-based Horvitz–Thompson estimates remain the anchor. If MSE quantities are combined across probability-sampled PSUs, use a preregistered design-consistent estimator that propagates within-PSU MSE uncertainty and between-PSU sampling variance. Do not pool all events as independent observations.

## 6. Controls required before biological interpretation

- Enumerated populations with known total errors and known all-zero cells.
- Independent, positively dependent, and negatively dependent list pairs.
- Shared image inputs, shared model backbones, shared training labels, reviewer referral, and other realistic dependence structures.
- Heterogeneous detection probabilities by error family, branch size, compartment, morphology, and image quality.
- Sparse overlaps, empty capture-history cells, and lists with no detections in rare families.
- False-positive proposals and adjudication disagreement.
- Exact, probabilistic, false-positive, and missed-identity linkages.
- Non-closed populations, segmentation revisions during collection, duplicate PSUs, and overlapping spatial frames.
- Semi-synthetic, blinded corruption challenges with known inserted errors, reported separately from natural-error estimates.
- Repeated-seed simulation measuring bias, interval coverage, convergence, and false-ESTIMABLE rates.

## 7. Required report

Report list definitions and lineage; every observed capture-history count; list sizes and intersections; adjudication and linkage uncertainty; model equations and identifying restrictions; priors; per-family support; estimated all-zero population by model; model sensitivity range; uncertainty intervals; PSU/stratum design uncertainty; audit burden; and one of ESTIMABLE, SENSITIVITY_ONLY, NOT_IDENTIFIABLE, or NOT_ESTIMABLE.

Keep these quantities separate:

- Observed union of independently confirmed errors.
- Design-weighted error total from #9204.
- Candidate-generation recall conditional on the ascertainment protocol.
- MSE sensitivity estimate of errors missed by every list.
- Queue-level review yield and causal workflow effects.

## 8. Benchmark objective

The benchmark should reward a system that detects known errors, exposes disagreement, and correctly abstains when unseen-error estimates are not identifiable. Proposed tests should target calibrated uncertainty and refusal under model misspecification—not merely low error on synthetic populations generated from the same assumptions as the estimator.

No natural-data unseen-error estimate, corrected candidate recall, or biological result is claimed by this protocol.

## References

- Bird & King, capture–recapture theory and log-linear extensions for dependence and heterogeneity: https://pubmed.ncbi.nlm.nih.gov/7485050/
- Hook & Regal, practical issues in Multiple Systems Estimation: https://pmc.ncbi.nlm.nih.gov/articles/PMC6055983/
- The central role of identifying assumptions in population-size estimation, Biometrics (2024): https://academic.oup.com/biometrics/article/80/1/ujad028/7624336
- Capture–recapture log-linear model pitfalls and non-identifiability: https://pmc.ncbi.nlm.nih.gov/articles/PMC11741671/