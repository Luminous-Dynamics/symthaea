# Relational Harmonics — Held-Out Prediction Qualification

**Status:** research contract / non-production  
**Lane:** Relational Harmonics  
**Purpose:** test whether relational features add out-of-sample predictive information

## 1. Qualification question

The next claim boundary is predictive rather than ontological:

> Does a relational feature set predict an independently observed future interaction outcome better than isolated-agent features, synchrony alone, or an explicitly observed common-driver baseline on data that occur later in time?

The answer is only allowed to be made from held-out observations.

This lane does **not** establish:

- consciousness;
- phenomenal experience;
- love, wisdom, or happiness;
- causal influence;
- relationship quality;
- physical resonance between agents.

## 2. Feature families

The evaluator compares seven fixed model families:

| Family | Inputs |
|---|---|
| PersistenceBaseline | last training outcome carried forward unchanged |
| IsolatedAgents | agent A + agent B state summaries |
| CommonDriver | supplied shared-context signal |
| SynchronyOnly | current alignment |
| NonRelationalContext | isolated agents + common driver + synchrony |
| RelationalAugmented | all non-relational context + directional and turn-taking relational channels |
| RelationalProfile | relational channels without isolated-agent/common-driver context |

The critical nested comparison is **RelationalAugmented vs NonRelationalContext**. This asks whether directional and turn-taking relational information improves prediction after isolated-agent state, shared context, and synchrony have already been given to the competing model.

The richer relational family must earn its additional degrees of freedom by reducing held-out error.

No adaptive feature selection occurs inside the evaluator.

## 3. Temporal safety boundary

Each observation has:

- feature_time: latest time at which predictor features are available;
- outcome_time: time at which the target becomes independently observable;
- future_outcome: the target value.

The constructor rejects outcome_time <= feature_time.

The holdout additionally requires:

max(train outcome_time) < min(test feature_time)

This is stronger than a simple row split. It prevents a training example's future label horizon from overlapping the held-out predictor interval.

A configurable gap is retained between train and test feature windows.

The design follows time-series evaluation guidance that training must use only observations available before the forecast target; random IID cross-validation is inappropriate when observations are temporally dependent.

## 4. Deterministic model

The initial qualification model is a small linear predictor with:

- explicit intercept;
- fixed, caller-supplied ridge coefficient;
- deterministic Gaussian elimination with pivoting;
- no hyperparameter search;
- no random seed;
- no test-set tuning.

The output reports mean absolute error and mean squared error on the held-out segment.

A positive relative MSE improvement means the relational model has lower held-out error than the selected baseline. A negative value means the baseline performs better.

The evaluator does not convert improvement into a significance claim.

The persistence baseline is intentionally non-parametric: it uses only the most recent training target. A relational model that cannot beat this baseline has not demonstrated useful predictive value merely by exploiting temporal persistence.

## 5. Multiple prediction null families

### 5.1 CircularShift

Relational channels are circularly shifted together independently inside the train and test partitions.

Properties preserved:

- marginal channel values;
- within-channel temporal ordering up to the circular boundary;
- model dimensionality.

Property intentionally destroyed:

- original partner-specific time alignment.

The shift is performed separately inside train and test, so surrogate construction cannot pull future test values into the training feature interval.

### 5.2 FeatureDecoupling

The four relational channels are shifted by distinct deterministic offsets inside each partition.

Properties preserved:

- each channel's values;
- approximate within-channel temporal structure;
- identical model capacity.

Property intentionally destroyed:

- coherent temporal alignment among the relational channels themselves.

This tests whether an apparent relational advantage depends on the channels working as a coherent bundle rather than merely adding four predictive degrees of freedom.

## 6. Interpretation boundary

The null output reports:

- requested surrogate count;
- realized surrogate count;
- observed relational-model MSE;
- minimum surrogate MSE;
- exceedance count;
- exceedance fraction.

The exceedance fraction is the fraction of surrogates that perform at least as well as the observed relational model under the same holdout. It is **not** named a p-value and must not be used as one.

A future statistical qualification lane may replace this deterministic finite family with a prespecified inferential procedure, including an appropriate surrogate design, confidence/error control, and multiplicity handling.

## 7. Required independent target design

The API deliberately does not manufacture the future outcome.

The caller must provide an independently observed target whose semantic collection window is after the predictor window.

A synthetic fixture may construct the target from known coefficients solely to verify implementation correctness. Such a fixture is a test of the evaluator, not evidence about real relational dynamics.

For scientific qualification, targets should be chosen before model fitting and should not be derived from the same relational score being claimed as predictive evidence.

## 8. Recommended empirical protocol

The minimum useful experiment should use repeated held-out temporal segments rather than one favorable split.

For each segment:

1. freeze the predictor definitions;
2. define the outcome horizon before looking at test results;
3. fit all fixed feature families, including the persistence baseline, using training observations only;
4. evaluate on the contiguous held-out future segment;
5. record MAE and MSE for all families;
6. repeat the same evaluation for each null family;
7. preserve all split definitions and source data hashes.

The qualification artifact should retain:

- split boundaries;
- outcome horizon;
- feature definitions;
- model parameters;
- observed scores;
- surrogate scores;
- software commit SHA;
- source-data identifiers/hashes.

The implementation now includes a rolling-origin evaluator with fixed-width training and test windows and a fixed forward step. The per-origin summaries remain available, while the evaluator also reports mean MSE across origins.

The rolling-origin qualification path applies three prediction null families at every origin:

- CircularShift;
- FeatureDecoupling;
- IncrementalRelationalShift.

IncrementalRelationalShift is the most targeted null for the nested comparison: isolated-agent state, common driver, and synchrony remain fixed while only the added directional/turn-taking relational channels are shifted.

## 9. Repeated rolling-origin evaluation

A single blocked holdout is an implementation qualification, not a scientific result.

The production research protocol therefore uses repeated forward-only origins with fixed:

- training-window length;
- test-window length;
- gap;
- ridge coefficient;
- forecast horizon semantics;
- feature definitions.

Each origin creates a new contiguous future-held-out segment. No future origin is allowed to become training data for an earlier origin.

The evaluator reports the per-origin scores as well as mean MSE across origins. This keeps the repeated evaluation auditable rather than hiding heterogeneity inside one aggregate.

The mean is descriptive only. It is not a substitute for an inferential procedure that accounts for dependence between overlapping rolling windows.

The design is deliberately descriptive at this stage. Predictive-error differences across strongly dependent origins should not automatically be converted into a classical IID significance test. Recent work shows that strong dependence can materially distort predictive-accuracy inference, so the first qualification target is repeatability and effect stability rather than a convenient p-value. [Coroneo & Iacone, 2025](https://doi.org/10.1016/j.ijforecast.2024.11.003)

## 10. What would count as meaningful evidence

A strong result would require more than:

RelationalAugmented MSE < NonRelationalContext MSE

and more than a single favorable origin.

The stronger pattern is:

- RelationalAugmented beats the nested NonRelationalContext model;
- RelationalAugmented also beats persistence;
- the improvement repeats across multiple future-held-out origins;
- the improvement is not explained by the common-driver baseline;
- the improvement survives the nested non-relational comparison rather than merely beating synchrony;
- the improvement degrades under partner circular-shift nulls;
- the improvement also degrades under feature-decoupling nulls;
- the incremental relational channels degrade under IncrementalRelationalShift while the non-relational context stays fixed;
- the direction and approximate magnitude are stable under prespecified horizons and fixed regularization choices;
- no single favorable origin is responsible for the result.

A useful negative result is:

RelationalAugmented ~= NonRelationalContext

because that would show that the additional relational channels do not yet add predictive information beyond obvious context and synchrony.

A failure of any of these is informative and should narrow the claim rather than trigger threshold tuning.

## 11. Literature boundary

Time-series cross-validation guidance recommends evaluating future observations using training information that precedes them in time and warns against IID shuffling when observations are autocorrelated:

- Hyndman & Athanasopoulos, Forecasting: Principles and Practice, Time Series Cross-Validation:
  https://otexts.com/fpptr/tscv.html
- scikit-learn, Time Series Cross-Validation:
  https://scikit-learn.org/stable/modules/cross_validation.html

Recent interpersonal-synchrony work reinforces the need for this conservative design. Gordon & Bartsch (2026) review the heterogeneous empirical correlates of interpersonal physiological synchrony and explicitly describe its psychological meaning as ambiguous:

- Gordon, I. & Bartsch, R. P. (2026), Nature Reviews Psychology 5, 201–215:
  https://doi.org/10.1038/s44159-026-00535-4

Transfer-entropy work also supports retaining a finite-sample qualification boundary. Kirkley (2025) describes positive bias in sparse finite data and the difficulty of statistical significance for conventional finite-data estimators:

- Kirkley, A. (2025), Physical Review E 112, L052304:
  https://doi.org/10.1103/tcss-5hn3

Surrogate approaches are established for directional information analysis, but the surrogate construction must match the sampling design and the null hypothesis:

- Schreiber (2000), Physical Review Letters 85, 461–464:
  https://doi.org/10.1103/PhysRevLett.85.461

## 12. Stop conditions

Do not:

- tune a threshold against the held-out segment;
- call lower prediction error proof of causality;
- treat null exceedance fractions as formal p-values;
- treat synthetic fixture success as empirical validation;
- select only the best split or best rolling origin;
- merge several feature families into one authoritative relational score;
- infer love, wisdom, happiness, or consciousness directly from predictive performance.

The qualification standard is:

**predictive advantage + temporal isolation + independent target + null discrimination + reproducibility**

before any stronger interpretation is considered.
