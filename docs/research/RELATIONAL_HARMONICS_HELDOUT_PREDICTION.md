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

The evaluator compares four fixed model families:

| Family | Inputs |
|---|---|
| IsolatedAgents | agent A + agent B state summaries |
| CommonDriver | supplied shared-context signal |
| SynchronyOnly | current alignment |
| RelationalProfile | alignment + A->B proxy + B->A proxy + turn-taking |

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
3. fit all four models using training observations only;
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

A later rolling-origin implementation should expand this from one blocked holdout to repeated forward-only evaluation.

## 9. What would count as meaningful evidence

A strong result would require more than:

RelationalProfile MSE < SynchronyOnly MSE

The stronger pattern is:

- relational profile beats synchrony-only;
- relational profile beats isolated-agent features;
- relational profile beats the common-driver baseline;
- the improvement repeats across held-out temporal segments;
- the improvement degrades under partner circular-shift nulls;
- the improvement also degrades under feature-decoupling nulls;
- results remain stable under prespecified changes to the temporal horizon and model regularization.

A failure of any of these is informative and should narrow the claim rather than trigger threshold tuning.

## 10. Literature boundary

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

## 11. Stop conditions

Do not:

- tune a threshold against the held-out segment;
- call lower prediction error proof of causality;
- treat null exceedance fractions as formal p-values;
- treat synthetic fixture success as empirical validation;
- select only the best split;
- merge several feature families into one authoritative relational score;
- infer love, wisdom, happiness, or consciousness directly from predictive performance.

The qualification standard is:

**predictive advantage + temporal isolation + independent target + null discrimination + reproducibility**

before any stronger interpretation is considered.
