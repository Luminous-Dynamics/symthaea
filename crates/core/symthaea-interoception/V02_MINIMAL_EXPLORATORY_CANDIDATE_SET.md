# Affective Emergence v0.2 — Minimal Exploratory Candidate Set

Status: **normative design-only / blocked on Native Interoception v0.1 qualification**

This contract defines the smallest initial exploratory candidate set intended to answer the first observational-affect questions without turning the factor space into an unrestricted metric search.

The purpose is not to select an emotionally intuitive scalar. The purpose is to force a small number of competing explanations to disagree under prospectively designed scenarios.

## 1. Principle

Prefer the smallest candidate set that can separately test:

- current regulatory burden;
- realized regulatory change;
- one-step expectation error;
- revision of the same future support;
- rolling-horizon turnover;
- urgency / breach imminence;
- cumulative exposure;
- confidence/precision;
- the legacy v0.1 precision×importance aggregate;
- external prefix-history information beyond current state;
- simple nuisance explanations such as current drive magnitude.

A candidate that cannot be distinguished from a simpler baseline under the locked scenario set does not advance merely because its formula is more sophisticated.

## 2. Initial finite set

The first exploratory set should contain exactly the following candidate roles unless a new design-freeze identity explicitly supersedes this contract.

The identifiers below are design IDs; the later machine-readable `ObservationalCandidateDefinitionManifest` remains authoritative and binds the exact formula, preprocessing, evaluator-isolation, fixtures, and source implementation.

### E00 — constant/null baseline

`e00_constant_zero_v1`

Purpose:

- prove the analysis does not manufacture apparent structure from ranking, normalization, missingness, or cohort aggregation;
- provide a deliberately weak floor.

Expected form: constant zero / typed available value under every valid cut point.

This candidate has no affect interpretation.

### E01 — current viability burden

`e01_r0_w1_a3_t0_h0_viability_mean_v1`

Coordinate:

- relation: `R0CurrentBurden`;
- weighting: `W1ViabilityWeightOnly`;
- aggregation: `A3WeightedMeanAllDeclared`;
- temporal: `T0Instantaneous`;
- forecast: `None`;
- information: `OfflinePrefixCausal`;
- history: `H0CurrentNativeStateOnly`.

Purpose: simplest normative current-state burden baseline without epistemic precision multiplying severity.

### E02 — current drive magnitude nuisance baseline

`e02_current_drive_magnitude_v1`

Purpose:

- test whether candidate structure is merely stimulus/load magnitude;
- remain intentionally simpler than native-state or forecast quantities.

Use a prospectively fixed norm/reduction over the current declared drive only. It must not inspect semantic condition identity or future protocol.

### E03 — realized viability change

`e03_r1_w1_a3_t0_h1_realized_change_v1`

Coordinate:

- relation: `R1RealizedChange`;
- weighting: `W1ViabilityWeightOnly`;
- aggregation: `A3WeightedMeanAllDeclared`;
- temporal: `T0Instantaneous`;
- forecast: `None`;
- information: `OfflinePrefixCausal`;
- history: `H1ReplayedPrefixHistory` because the immediately prior realized point is required.

Purpose: ask whether actual regulatory improvement/worsening explains candidate structure without invoking prediction.

### E04 — one-step forecast residual

`e04_r2_w1_a3_t0_h1_one_step_residual_v1`

Coordinate:

- relation: `R2OneStepForecastResidual`;
- weighting: `W1ViabilityWeightOnly`;
- aggregation: `A3WeightedMeanAllDeclared`;
- temporal: `T0Instantaneous`;
- forecast: one prospectively chosen native prefix-causal forecast policy;
- information: `OfflinePrefixCausal`;
- history: `H1ReplayedPrefixHistory`.

Purpose: separate better/worse-than-expected realization from actual improvement.

### E05 — aligned overlapping-future revision

`e05_r3_w1_a3_t1_h1_overlap_revision_v1`

Coordinate:

- relation: `R3OverlappingFutureRevision`;
- weighting: `W1ViabilityWeightOnly`;
- aggregation: `A3WeightedMeanAllDeclared`;
- temporal: prospectively fixed aligned weighted mean over shared absolute future support;
- forecast: same policy family used by E04 unless the candidate manifest explicitly declares a comparison policy;
- information: `OfflinePrefixCausal`;
- history: `H1ReplayedPrefixHistory`.

Purpose: ask whether the predicted future itself improved/worsened, separated from the realized one-step error and from rolling-window turnover.

### E06 — rolling-horizon change control

`e06_r4_w1_a3_t1_h1_rolling_mean_change_v1`

Coordinate:

- relation: `R4RollingHorizonChange`;
- weighting: `W1ViabilityWeightOnly`;
- aggregation: `A3WeightedMeanAllDeclared`;
- temporal: `T1DiscountedMean`;
- forecast: same locked forecast family as E05;
- information: `OfflinePrefixCausal`;
- history: `H1ReplayedPrefixHistory`.

Purpose: deliberately retain the finite-horizon boundary-turnover quantity as a competing explanation for E05. E06 is not described as forecast revision.

### E07 — breach-imminence urgency

`e07_u1_w0_a2_t8_h0_breach_latency_v1`

Coordinate:

- relation: `U1Urgency`;
- weighting: `W0RawChannel` or an equivalently prospectively fixed non-confidence severity basis;
- aggregation: `A2PeakDeviation` / declared worst-threat projection;
- temporal: `T8FirstBreachLatency`;
- forecast: prospectively locked prefix-causal policy;
- information: `OfflinePrefixCausal`;
- history: `H0CurrentNativeStateOnly` when the forecast can be constructed from current native state/current allowed inputs only.

Purpose: test whether an apparently arousal/urgency-like signal is explained by simple breach imminence rather than a broader affect construct.

Typed no-breach is not zero latency.

### E08 — cumulative viability exposure

`e08_r0_w1_a4_t2_h1_cumulative_exposure_v1`

Coordinate:

- relation: declared burden/exposure relation;
- weighting: `W1ViabilityWeightOnly`;
- aggregation: `A4WeightedSumAllDeclared`;
- temporal: `T2DiscountedCumulativeExposure`;
- forecast/history source: explicitly frozen;
- information: `OfflinePrefixCausal`;
- history: `H1ReplayedPrefixHistory` when exposure uses the observed prior prefix, or H0 only if computed solely over a current prospective trajectory. The exact manifest must choose one and cannot switch post hoc.

Purpose: test accumulated duration/intensity separately from instantaneous burden and normalized mean burden.

### E09 — explicit confidence baseline

`e09_r0_w3_a3_t0_h0_confidence_mean_v1`

Coordinate:

- relation: descriptive current confidence/precision;
- weighting: `W3ConfidenceOnly`;
- aggregation: declared fixed-set mean;
- temporal: `T0Instantaneous`;
- forecast: `None`;
- information: `OfflinePrefixCausal`;
- history: `H0CurrentNativeStateOnly`.

Purpose: test whether a result attributed to viability is actually tracking confidence/precision.

This is not a burden candidate.

### E10 — legacy v0.1 burden baseline

`e10_r0_w2_a3_t0_h0_legacy_weighted_burden_v1`

Coordinate:

- relation: `R0CurrentBurden`;
- weighting: `W2LegacyPrecisionTimesImportance`;
- aggregation: `A3WeightedMeanAllDeclared`;
- temporal: `T0Instantaneous`;
- information: `OfflinePrefixCausal`;
- history: `H0CurrentNativeStateOnly`.

Purpose: compare the frozen v0.1 aggregate against the importance-only E01 interpretation rather than silently choosing one.

### E11 — history-added-information candidate

`e11_h1_history_summary_v1`

Purpose:

- test whether prospectively specified same-scenario prefix history adds information beyond the H0 current-state candidates;
- serve as the smallest explicit H1-vs-H0 discriminator.

Its exact sufficient statistic must be frozen prospectively. Prefer one simple history summary, such as cumulative prior viability exposure or repeated-breach exposure, rather than a learned/high-capacity history model.

It must be evaluated on matched-current-native-state / different-history scenarios.

A successful E11 supports only **external historical information gain**, not native memory or mood.

## 3. Why this is enough for the first exploration

The set is intentionally small but covers the primary alternative explanations:

- E01: current condition;
- E02: current stimulus/load;
- E03: actual improvement;
- E04: better/worse than expected;
- E05: changed future outlook;
- E06: rolling-window artifact/control;
- E07: urgency;
- E08: cumulative exposure;
- E09: confidence;
- E10: legacy precision×importance hypothesis;
- E11: external history beyond current state;
- E00: null floor.

Do not add another candidate merely because it is available in the factor space. Addition requires a written discrimination obligation that cannot be answered by the current set and a new candidate-set identity.

## 4. Forecast-policy discipline

Do not duplicate E04/E05/E06/E07 across every forecast policy in the first exploratory set.

Choose one primary prefix-causal forecast policy prospectively for the minimal set, based on the simplest policy consistent with the scientific question and available native information.

Use alternate prefix-causal forecast policies in targeted sensitivity/disagreement diagnostics rather than multiplying the primary candidate registry.

Oracle policy remains diagnostic only.

## 5. Preprocessing discipline

Preferred initial preprocessing for E00–E11 is `None` or purely structural fixed transformation.

If any candidate requires fitted preprocessing:

- the candidate definition must bind its `PreprocessingManifest`;
- fitting may use only prospectively identified discovery/calibration or external-reference artifacts;
- confirmatory/holdout data cannot contribute fitted parameters;
- the candidate-set digest changes if preprocessing changes.

For initial exploration, avoid fitting unless needed to answer a specific discrimination question.

## 6. Evaluator isolation

All E00–E11 evaluations use the `NoneAcrossEvaluationCoordinates` persistent-state class from `V02_OBSERVATORY_STATE_LIFECYCLE.md`.

H1 candidates may read same-scenario history directly from the immutable prefix, but may not carry mutable state from a previous scenario, candidate, cut point, batch, or process invocation.

## 7. Required pairwise discrimination obligations

The initial scenario/cut-point matrix must include at least one prospective discriminator for:

- E03 vs E01 — change vs current burden;
- E03 vs E02 — realized change vs drive magnitude;
- E04 vs E03 — forecast residual vs actual change;
- E05 vs E04 — future revision vs one-step residual;
- E05 vs E06 — aligned future revision vs horizon turnover;
- E07 vs E01 — urgency vs current burden;
- E08 vs E01 — cumulative exposure vs instantaneous burden;
- E09 vs E01 — confidence vs viability burden;
- E10 vs E01 — legacy precision×importance vs viability-only burden;
- E11 vs E01/E03 — external history vs current-state/recent-change explanations;
- every non-null candidate vs E00;
- forecast/history candidates vs E02 nuisance load.

If any required pair lacks a discriminator, the exploratory design is incomplete.

## 8. Candidate reduction after exploration

The exploratory phase does not have to produce a winner.

Prospectively valid outcomes include:

- one candidate supported beyond all required baselines;
- several candidates in one observational equivalence class;
- different candidates explaining different scenario families;
- simpler baseline dominates;
- no candidate clears the minimum discrimination margin;
- weighting/temporal/history ambiguity remains unresolved;
- no unique candidate.

Use a prospective parsimony rule: when two candidates are empirically equivalent under all registered discriminators, retain the simpler/lower-information candidate as the preferred explanation and preserve the equivalence result.

## 9. No affect interpretation during candidate selection

Candidate ranking and reduction must use only the preregistered numerical/discrimination criteria.

Do not select based on which time series:

- looks most emotional;
- resembles human valence intuitively;
- produces pleasing plots;
- has the most dramatic excursions;
- correlates best with semantic scenario names after unblinding.

Interpretation comes after blinded candidate evidence freezes.

## 10. Machine-readable manifest

A future `ExploratoryCandidateSetManifest` should bind:

- schema/version;
- exact design-contract-registry digest;
- ordered candidate-definition digests for E00–E11;
- candidate role IDs;
- required pairwise discrimination obligations;
- primary forecast-policy choice;
- preprocessing policy summary;
- evaluator-isolation manifest digest;
- scenario-discrimination manifest digest;
- selection/parsimony rule digest;
- canonical SHA-256.

Validation must reject:

- missing required role;
- duplicate role;
- extra unregistered candidate;
- candidate-definition digest mismatch;
- absent required discriminator;
- oracle/retrospective candidate in a primary role;
- confirmatory-fitted preprocessing;
- cross-evaluation mutable-state authority.

## 11. Claim boundary

This finite set is designed to determine whether any regulatory observable adds reproducible information beyond simple state, load, change, confidence, urgency, history, exposure, and legacy-weighting explanations.

Even a uniquely successful exploratory candidate is not evidence of emotion, subjective valence, feeling, mood, suffering, sentience, or consciousness. It is only a candidate for later independently frozen confirmatory testing.