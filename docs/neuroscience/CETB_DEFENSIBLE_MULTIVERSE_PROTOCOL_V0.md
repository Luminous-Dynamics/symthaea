# CETB defensible multiverse protocol V0

**Status:** Proposed analysis-governance protocol.  
**Date:** 2026-11-21

## 1. Purpose

CETB should measure uncertainty introduced by defensible analytic choices without turning the analysis into an unrestricted search for a favourable result.

A multiverse is not every possible permutation of every setting. It is the set of analysis pipelines that can each be justified independently for the same scientific question, subject to explicit compatibility rules. Equivalence between options is a separate judgement from their individual defensibility.

This follows the 2026 multidisciplinary multiverse guidance (Short et al., DOI 10.1177/25152459261434881): identify defensible options first, assess compatibility, and only then decide which alternatives are equivalent for a declared criterion.

## 2. Decision stages

Each analysis manifest should declare the stages where defensible choices exist:

1. **Measurement / construct:** operational endpoint, region definitions, sensor/reference scheme, connectivity statistic family.
2. **Preprocessing:** filtering, motion handling, censoring, nuisance regression, global-signal choice, source reconstruction.
3. **Estimation:** correlation/precision/phase/connectivity estimator, sign semantics, regularization, segment duration and repeat placement.
4. **Model / inference:** paired estimand, participant-level test, hierarchical model, bootstrap and multiplicity policy.
5. **Decision / reporting:** calibration, abstention, threshold and claim ceiling.

Not all stages apply to every dataset. Record not-applicable stages with a rationale rather than silently omitting them.

## 3. Option-level defensibility record

Every option must be described before outcomes are inspected:
- stable option ID and decision stage;
- exact implementation/version or parameter value;
- independent rationale (theoretical, methodological, prior empirical validation or established convention);
- source citations or explicit reason no external source applies;
- expected construct and estimand;
- known limitations and failure modes;
- whether it is eligible for the primary lane, sensitivity lane or exploratory-only lane.

An option is not defensible merely because it is easy to compute or produces a favourable score.

## 4. Compatibility before enumeration

Some options cannot be combined coherently. Examples:
- a signed statistic cannot be passed into an algorithm that assumes non-negative weights without a declared transform;
- a directed estimator cannot be silently symmetrized if directionality is the estimand;
- an atlas-specific feature may not be compared across incompatible region definitions;
- a preprocessing branch requiring a field absent from the dataset is not a valid pipeline;
- a participant-level inference cannot treat segments as independent subjects.

Compatibility constraints must be encoded before outcomes are viewed. The enumerator must distinguish forbidden combinations, allowed combinations, and combinations requiring a separately named estimand or analysis lane.

## 5. Equivalence is not defensibility

Two options may each be defensible yet target different constructs or estimands. Assess equivalence separately against a predeclared criterion:
- construct equivalence;
- numerical scale equivalence;
- estimand equivalence;
- expected measurement reliability;
- expected power / precision;
- decision consequences.

Use the statuses equivalent, non-equivalent or uncertain, with a written criterion and rationale. Do not average results from non-equivalent estimands into a single score.

## 6. Pipeline generation and result reporting

Generate every compatible pipeline in the frozen manifest, or use a declared deterministic sample when the space is too large. If sampled, publish the sampling method and seed.

For every generated pipeline report:
- pipeline ID and exact option IDs;
- input, code, configuration, environment and output digests;
- completion or failure status and failure reason;
- primary estimand and effect estimate;
- participant-level uncertainty;
- quality, exclusion and missingness counts;
- calibration and negative-control results.

Report the distribution across the multiverse, not only the best pipeline:
- median and quantiles of the effect estimate;
- fraction of pipelines with the same effect direction;
- range and between-pipeline dispersion;
- proportion of valid/failed pipelines;
- which decision nodes explain the largest variation;
- whether the primary conclusion is stable across all or most defensible pipelines.

Do not select the pipeline with the highest accuracy, lowest p-value or strongest theory-support after viewing confirmatory outcomes. Any prediction-model selection must still occur within nested training folds.

## 7. Primary lane, sensitivity lane and exploration

Keep separate:
- **Primary lane:** one preregistered, independently justified pipeline tied to the primary estimand.
- **Sensitivity multiverse:** alternative defensible specifications that target the same estimand or have explicit equivalence status.
- **Exploratory lane:** non-equivalent estimands, new feature families, post-hoc variants and hypotheses generated after outcome access.

The exploratory lane can generate hypotheses, but its results cannot be promoted to confirmatory evidence without a new untouched evaluation.

## 8. CETB implementation requirements

The machine-readable contract in research/cetb/multiverse/multiverse.schema.json requires:
- frozen decision nodes and option rationales;
- explicit compatibility rules;
- separately stated equivalence assessments;
- an outcome-access declaration;
- a reproducible pipeline-generation seed and mode;
- fixed reporting fields and claim ceilings.

The accompanying synthetic fixture is an example contract only. It is not a data analysis.

## 9. Sources and limits

- Short et al. (2026), “Multicurious: A Multidisciplinary Guide to Multiverse Analysis,” Advances in Methods and Practices in Psychological Science, DOI 10.1177/25152459261434881.
- Ubaida et al. (2026), Diagnostics, DOI 10.3390/diagnostics16203312, provides a direct example of why preprocessing, model selection, thresholds, fusion and site correction must be audited together.
- This is a proposed CETB protocol. It is not a regulatory standard or a claim that every pipeline variant is scientifically equivalent.
