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
- train-only feature standardization;
- fixed transforms carried unchanged into the held-out window;
- no hyperparameter search;
- no random seed;
- no test-set tuning.

The output reports mean absolute error and mean squared error on the held-out segment.

A positive relative MSE improvement means the relational model has lower held-out error than the selected baseline. A negative value means the baseline performs better.

The evaluator does not convert improvement into a significance claim.

The persistence baseline is intentionally non-parametric: it uses only the most recent training target. A relational model that cannot beat this baseline has not demonstrated useful predictive value merely by exploiting temporal persistence.

For fitted linear models, each training window supplies its own feature mean and scale. Constant training features use unit scale after centering. The held-out window is transformed with those frozen training statistics; no test-window statistics enter preprocessing. Ridge regularization is applied after this standardization so the fixed lambda is not silently reweighted by raw feature units.

## 5. Multiple prediction null families

### 5.1 CircularShift

Relational channels are circularly shifted together independently inside the train and test partitions.

Properties preserved:

- marginal channel values;
- within-channel temporal ordering up to the circular boundary;
- model dimensionality.

Property intentionally destroyed:

- original partner-specific time alignment.

The shift is performed separately inside train and test, so surrogate construction cannot pull future test values into the training feature interval. Every shifted channel is normalized to a non-zero circular offset, so an offset that would otherwise wrap exactly to the identity cannot silently produce an unchanged surrogate. The realized surrogate count is capped by the available non-zero circular shifts.

### 5.2 FeatureDecoupling

The four relational channels are shifted by channel-specific deterministic non-zero offsets inside each partition. The evaluator requires at least five observations in a partition for this null, because four channels need four distinct non-zero circular offsets; undersized partitions fail closed instead of silently weakening the null.

Properties preserved:

- each channel's values;
- approximate within-channel temporal structure;
- identical model capacity.

Property intentionally destroyed:

- coherent temporal alignment among the relational channels themselves.

This tests whether an apparent relational advantage depends on the channels working as a coherent bundle rather than merely adding four predictive degrees of freedom.

### 5.3 IncrementalRelationalShift

Synchrony and all non-relational context are held fixed while only the directional and turn-taking relational channels are shifted inside each train and test partition. The three added channels are shifted together by the same non-zero circular offset, preserving their within-block temporal alignment while moving that entire incremental block away from its original placement.

This is the targeted null for the critical RelationalAugmented vs NonRelationalContext comparison. It asks whether any incremental gain depends on the temporal organization of the added relational block rather than merely on the presence of extra model capacity.

All three null families retain their exact deterministic shift schedule, per-surrogate MSE vector, configuration, and input commitment. The trace is replayable against the supplied source samples. The in-memory qualification bundle additionally binds the requested surrogate count and all three null traces to one top-level holdout configuration, the observed RelationalAugmented MSE, and their required interpretation statuses (observed = Measured, null = Proxy).

These deterministic circular shifts are deliberately treated as **calibration diagnostics, not general-purpose significance tests**. Surrogate validity depends on the null hypothesis and the temporal properties of the data; circular shifting is not universally valid for nonstationary series. It also creates an artificial adjacency between the final and first observations, which can distort dynamics when those endpoints are not naturally adjacent. A truncated time-shift construction avoids that wraparound by sacrificing the affected boundary observations, but it has its own stationarity and finite-sample conditions and therefore belongs in a future separately qualified null family rather than being silently substituted here. Classical surrogate-data work stresses that a surrogate result is meaningful only relative to a specified null hypothesis, and modern time-series work likewise shows that naive permutation assumptions can fail under temporal dependence. citeturn919765search0turn919765search9turn740736search0turn740736search3

## 6. Interpretation boundary

The null output reports:

- requested surrogate count;
- realized surrogate count;
- observed relational-model MSE;
- exact deterministic surrogate shifts;
- complete per-surrogate MSE vector;
- minimum surrogate MSE;
- exceedance count;
- exceedance fraction;
- replay input commitment.

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
4. fit any preprocessing parameters on that training window only and freeze them before transforming the held-out segment;
5. evaluate on the contiguous held-out future segment;
6. record MAE and MSE for all families;
7. repeat the same evaluation for each null family;
8. preserve all split definitions, preprocessing parameters, and source data hashes.

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

IncrementalRelationalShift is the targeted nested null for the critical RelationalAugmented vs NonRelationalContext comparison: synchrony and non-relational context remain fixed while only the added relational channels are shifted.

The in-memory single and rolling qualification bundles now retain a top-level BLAKE3 commitment over the exact source sequence plus their complete configuration. Every bundle also carries caller-attested protocol, source-data, and software provenance and a second versioned BLAKE3 qualification-identity commitment over that provenance, the input commitment, configuration, and surrogate count. This prevents a valid-but-different provenance record from being silently substituted into an otherwise identical numerical result. Each null trace carries two distinct commitments: a null-local replay commitment over its exact source slice, holdout configuration, null family, feature family, and requested surrogate count; and a parent qualification commitment that binds the retained trace to the compound bundle's top-level source/configuration identity. The separation preserves standalone null replay even after a trace is embedded in a rolling bundle, while the compound validator still requires every retained null trace to carry the identical parent commitment. The rolling bundle additionally retains the requested surrogate count at every origin and the realized origin_starts schedule. This provenance structure follows the general W3C PROV distinction between the entities used, the activity that generated the result, and the agents responsible for that activity: https://www.w3.org/TR/prov-primer/.


## 9. Evidence packet and provenance

The evaluator can emit a validated JSON evidence packet rather than only a summary score.

The packet carries:

- a caller-attested protocol identifier;
- the exact source-data SHA-256 digest;
- the exact 40-character software commit SHA;
- split configuration and observed horizon bounds;
- the score for every feature family;
- per-family held-out feature timestamps and outcome timestamps;
- independently retained held-out outcomes and predictions;
- held-out feature rows for every fitted feature family;
- the persistence baseline forecast for the persistence family;
- the complete held-out evaluation configuration, including train/test/gap sizes and ridge coefficient;
- fitted linear-model coefficients;
- training-window feature means and scales.

The packet verifier also binds the complete top-level holdout configuration to every record. It rejects mismatched train/test/gap sizes or ridge settings before accepting the packet. It then recomputes each fitted-model held-out prediction from the retained feature row, fitted coefficients, and frozen training means/scales, and recomputes MAE/MSE from the resulting predictions and observed outcomes. A replay verifier can additionally rerun the complete evaluation against supplied samples and reject any mismatch in training data, preprocessing, fitted coefficients, predictions, scores, or the computed input commitment. The compound observed-plus-null qualification bundle has the same fail-closed replay boundary: it can be replayed against the exact source sequence, configuration, and surrogate count and must reproduce the complete bundle rather than merely agreeing internally. For the persistence family it binds every retained prediction to the recorded constant baseline forecast. It therefore rejects tampered predictions even when the reported loss is also modified. Standalone prediction traces additionally reject sub-minimal training/test window declarations, keeping the public trace validator aligned with the evaluator's minimum split contract. The verifier also rejects non-finite values, wrong feature dimensions, missing feature families, inconsistent timestamps, mismatched provenance, and rolling-origin child packets that disagree with the parent configuration. Evaluator entry points also revalidate the constructor-level sample invariants, so callers cannot bypass finite-value, relational-range, or strictly-future checks by constructing the public sample fields directly.

The provenance fields are intentionally caller-supplied. The evaluator must not invent a dataset hash or software identity. A packet with absent or malformed provenance is therefore invalid for empirical qualification.

JSON schemas are versioned in the emitted document:

- `relational-prediction-evidence/v4` for one held-out segment;
- `relational-prediction-rolling-evidence/v4` for the repeated-origin bundle;
- `relational-prediction-null-evidence/v2` for a null calibration trace, with separate local replay and parent-qualification commitments.

For rolling-origin bundles, the packet also retains `origin_starts`, the exact source-sample start index realized for each child segment. The parent packet also commits to the complete source sequence used to derive those starts, so replay can distinguish a schedule mismatch from a different input sequence. Validation recomputes these from `first_origin` and `step_samples`, so the declared schedule and retained child packets cannot silently diverge.

The evidence packet is intentionally not a self-contained copy of the training dataset. It retains enough of the held-out computation to independently reconstruct test predictions and verify the reported losses, while the caller-attested source-data SHA-256 remains the commitment to the underlying source artifact. Full coefficient-training replay still requires access to the exact source data identified by that digest.

Serialization is an evidence transport mechanism, not an inference procedure. A valid packet proves that the recorded computation is internally self-consistent; it does not prove that the source data are scientifically appropriate, that the target is truly independent, or that the measured predictive difference is causal.

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

The evaluator reports the per-origin scores as well as mean MSE across origins. It also exposes the full per-origin relative MSE-improvement vector for the critical RelationalAugmented vs NonRelationalContext comparison, together with median improvement, worst-origin improvement, and counts of origins beating the nested baseline and persistence. These are descriptive diagnostics, not pass/fail criteria. This keeps the repeated evaluation auditable rather than hiding heterogeneity inside one aggregate.

The mean is descriptive only. It is not a substitute for an inferential procedure that accounts for dependence between repeated rolling estimates.

For the critical RelationalAugmented versus NonRelationalContext comparison, the replayable evidence API also exposes the target-level squared-loss differential for every held-out target: non-relational-context squared error minus relational-augmented squared error. Positive values therefore favor the relationally augmented forecast. Each differential retains its origin index, sample index, feature/outcome timestamps, and observed target, while the corresponding forecast losses remain traceable to the retained predictions. This preserves the actual loss process that a later dependence-aware nested-forecast procedure should analyze, rather than reconstructing it from an aggregate MSE after the fact. The current protocol deliberately makes held-out test windows disjoint, but training windows may still overlap across origins, so the origin-level scores are not assumed IID.
### Loss-process dependence characterization

Before formal inference, the retained target-level differentials are now characterizable as an ordered loss process without selecting a significance procedure. For a single held-out window, the descriptive profile records:

- mean and variance of the loss differential;
- autocovariance and autocorrelation vectors through a caller-supplied maximum lag;
- first non-positive autocorrelation lag and maximum absolute non-zero-lag autocorrelation;
- a deterministic Bartlett long-run variance estimate truncated at the declared lag;
- an effective-sample-size diagnostic when the long-run variance estimate is finite and positive.

The autocovariance convention is explicitly 1/n, and the Bartlett weights are 1 - k/(L+1) for lag k and declared maximum lag L. These are reporting conventions, not claims that the estimator is the correct inferential variance estimator for the eventual scientific analysis. The lag bandwidth must therefore be treated as part of the analysis manifest when a real run is performed.

Rolling-origin evidence is characterized at two levels rather than by concatenating windows. Each disjoint held-out target window gets its own serial-dependence profile. Separately, the ordered vector of origin-level mean loss differentials receives an across-origin dependence profile. This distinction is important because concatenating disjoint windows would manufacture a false adjacency between the last target of one origin and the first target of another. It also keeps within-window temporal dependence conceptually separate from dependence induced by overlapping training sets across rolling origins.

The resulting artifact is intentionally descriptive. A large positive lag-1 autocorrelation, a low effective sample size, or a large long-run variance does not establish or refute predictive superiority. Instead, it identifies conditions that must be respected by the later inferential procedure. Recent forecast-evaluation work recommends accounting for dependence in the loss differential and shows that strong serial dependence can materially distort ordinary equal-accuracy tests. citeturn859792search1turn859792search4

The dependence profile returned through an evidence packet also retains that packet's exact evaluator BLAKE3 commitment. This prevents a detached dependence report from being mistaken for a characterization of a different source/evaluation run. Free-standing profiles may still be computed for exploratory diagnostics, but those are intentionally unbound and should not be promoted into qualification evidence.

The implementation therefore stops at dependence characterization → method selection rather than jumping directly to a p-value. For nested RelationalAugmented versus NonRelationalContext forecasts, dedicated nested-model procedures remain the relevant candidate class; the eventual choice must also commit to forecast horizon, dependence estimator, small-sample treatment, bootstrap/self-normalization strategy if used, and multiplicity rules before examining the real test results. citeturn859792search7turn859792search5


For the critical RelationalAugmented versus NonRelationalContext comparison, the two models are nested. A future inferential lane therefore must use a method appropriate to nested forecast comparisons rather than applying an unmodified two-model IID or Diebold-Mariano-style test. The dependence structure of the retained loss differentials must be characterized first, and any bootstrap or self-normalized procedure must be prespecified together with its horizon, origin, and multiplicity rules. The current implementation deliberately stops before this inferential step. Literature on nested forecast comparison provides dedicated procedures for this setting, while recent work shows that strong dependence can materially distort ordinary predictive-accuracy tests. citeturn626983search2turn626983search9

Recent 2026 forecasting work also warns that apparent model preference can be driven by temporal instability and recommends rolling-origin evaluation to reduce sensitivity to an unusually favorable testing segment. That supports retaining the complete origin-level loss/improvement vector rather than reporting only its mean. [Liu et al. (2026)](https://doi.org/10.1016/j.orl.2026.107468)

A separate 2026 literature describes rolling-origin forecast instability as changes in forecasts for the same target caused by later forecast origins. The present protocol intentionally does not estimate that quantity because its test windows are disjoint and each target is scored once. A future instability study would therefore be a separate protocol requiring repeated forecasts of shared targets rather than quietly changing this qualification definition. [Caljon et al. (2026)](https://doi.org/10.1016/j.ijforecast.2025.07.002)

The design is deliberately descriptive at this stage. Predictive-error differences across strongly dependent origins should not automatically be converted into a classical IID significance test. Recent forecast-evaluation work explicitly develops procedures that account for autocorrelation and overlapping forecast windows, reinforcing the need to preserve loss differentials and their dependence structure before choosing formal inference. [Grant, Mrazik & Satchell, 2026](https://doi.org/10.1002/for.70150)

A 2026 Journal of Applied Econometrics contribution develops robust forecast-accuracy tests for nested regressions and highlights the nonstandard behavior of common nested forecast tests under practical dependence and estimation conditions, reinforcing the decision to defer formal inference until the empirical loss process is characterized. [Morico et al. (2026)](https://doi.org/10.1002/jae.70056)

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

A September 2026 methodological review further frames synchrony inference around specificity, generality, and sensitivity, and recommends explicit controls for common inputs/shared influences plus model-class and timescale commitments:

- Danyluck et al. (2026), Psychophysiology 63(9), e70404:
  https://doi.org/10.1111/psyp.70404

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
