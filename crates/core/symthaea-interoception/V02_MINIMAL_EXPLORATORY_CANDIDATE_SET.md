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
- absolute prospective cumulative exposure;
- confidence/precision;
- the legacy v0.1 precision×importance aggregate;
- external prefix-history information beyond current state;
- simple nuisance explanations such as current drive magnitude.

A candidate that cannot be distinguished from a simpler baseline under the locked scenario set does not advance merely because its formula is more sophisticated.

## 2. Initial finite set

The first exploratory set contains exactly the following candidate roles unless a new design-freeze identity explicitly supersedes this contract.

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

Coordinate:

- relation: `BaselineReactive`;
- weighting: not applicable;
- aggregation: fixed Euclidean norm of the eight-channel current drive-rate vector;
- temporal: `T0Instantaneous`;
- forecast: `None`;
- information: `OfflinePrefixCausal`;
- history: `H0CurrentNativeStateOnly`.

Purpose:

- test whether candidate structure is merely stimulus/load magnitude;
- remain intentionally simpler than native-state or forecast quantities.

Use a prospectively fixed Euclidean norm over the eight-channel current `InteroceptiveDrive` rate vector after exact conversion to the locked v0.2 accumulation representation. No fitted normalization is permitted in the first lineage.

It must not inspect semantic condition identity or future protocol.

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

`e04_r2_w1_a3_t0_h1_drive_persistence_residual_v1`

Coordinate:

- relation: `R2OneStepForecastResidual`;
- weighting: `W1ViabilityWeightOnly`;
- aggregation: `A3WeightedMeanAllDeclared`;
- temporal: `T0Instantaneous`;
- forecast: `ObservedDrivePersistence`;
- information: `OfflinePrefixCausal`;
- history: `H1ReplayedPrefixHistory`.

Purpose: separate better/worse-than-expected realization from actual improvement.

At `t-1`, the forecast assumes the drive observed at `t-1` persists according to the frozen native dynamics-aware forecast semantics. It does not use the actual future drive at `t`.

### E05 — aligned overlapping-future revision

`e05_r3_w1_a3_t1_h1_drive_persistence_revision_v1`

Coordinate:

- relation: `R3OverlappingFutureRevision`;
- weighting: `W1ViabilityWeightOnly`;
- aggregation: `A3WeightedMeanAllDeclared`;
- temporal: prospectively fixed aligned weighted mean over shared absolute future support;
- forecast: `ObservedDrivePersistence`;
- information: `OfflinePrefixCausal`;
- history: `H1ReplayedPrefixHistory`.

Purpose: ask whether the predicted future itself improved/worsened, separated from the realized one-step error and from rolling-window turnover.

The forecast at each cut point may use only the drive already observed at that cut point; it cannot use the later realized drive schedule.

### E06 — rolling-horizon change control

`e06_r4_w1_a3_t1_h1_drive_persistence_rolling_change_v1`

Coordinate:

- relation: `R4RollingHorizonChange`;
- weighting: `W1ViabilityWeightOnly`;
- aggregation: `A3WeightedMeanAllDeclared`;
- temporal: `T1DiscountedMean`;
- forecast: `ObservedDrivePersistence`;
- information: `OfflinePrefixCausal`;
- history: `H1ReplayedPrefixHistory`.

Purpose: deliberately retain the finite-horizon boundary-turnover quantity as a competing explanation for E05. E06 is not described as forecast revision.

### E07 — breach-imminence urgency

`e07_u1_w0_a2_t8_h0_drive_persistence_breach_latency_v1`

Coordinate:

- relation: `U1Urgency`;
- weighting: `W0RawChannel`;
- aggregation: `A2PeakDeviation` / worst projected channel;
- temporal: `T8FirstBreachLatency`;
- forecast: `ObservedDrivePersistence`;
- information: `OfflinePrefixCausal`;
- history: `H0CurrentNativeStateOnly` because the forecast uses only current state/configuration and the drive observed at the cut point.

Purpose: test whether an apparently urgency-like signal is explained by simple breach imminence rather than a broader affect construct.

The value is a typed latency to the first projected viability breach under the locked forecast. `NoProjectedBreachWithinHorizon` is a distinct typed state, not zero or infinity substituted ad hoc.

### E08 — absolute prospective cumulative viability exposure

`e08_p0_w1_a3_t2_h0_drive_persistence_cumulative_v1`

Coordinate:

- relation: `P0ProspectiveBurden`;
- weighting: `W1ViabilityWeightOnly`;
- aggregation: `A3WeightedMeanAllDeclared` per projected step over the same fixed eight-channel set as E01;
- temporal: `T2DiscountedCumulativeExposure` — discounted **sum**, not normalized mean;
- forecast: `ObservedDrivePersistence`;
- information: `OfflinePrefixCausal`;
- history: `H0CurrentNativeStateOnly`.

Purpose: test whether absolute projected duration/intensity under current conditions explains candidate structure beyond instantaneous burden and breach latency.

E08 uses the same v0.1 dynamics-aware constant-drive trajectory family/horizon/discount convention as the forecast-bearing candidates, but does not normalize cumulative weighted burden by total discount weight. It is therefore distinct from the legacy v0.1 normalized `discounted_debt`/mean quantity.

### E09 — explicit confidence nuisance baseline

`e09_baseline_confidence_w3_a3_t0_h0_v1`

Coordinate:

- relation: `BaselineNuisance`;
- weighting: `W3ConfidenceOnly`;
- aggregation: fixed eight-channel arithmetic mean;
- temporal: `T0Instantaneous`;
- forecast: `None`;
- information: `OfflinePrefixCausal`;
- history: `H0CurrentNativeStateOnly`.

Purpose: test whether a result attributed to viability is actually tracking confidence/precision.

This is explicitly **not** a burden relation. Its baseline relation class prevents `W3ConfidenceOnly` from silently inheriting burden semantics.

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

### E11 — fixed trailing-history nuisance baseline

`e11_baseline_history_w1_a3_t3_h1_trailing16_v1`

Coordinate:

- relation: `BaselineNuisance`;
- weighting: `W1ViabilityWeightOnly`;
- aggregation: `A3WeightedMeanAllDeclared` at each realized past point;
- temporal: `T3UndiscountedCumulativeExposure` over exactly the **16 completed realized states immediately preceding `t`**, excluding the current state at `t`;
- forecast: `None`;
- information: `OfflinePrefixCausal`;
- history: `H1ReplayedPrefixHistory`.

Purpose:

- test whether a simple amount-of-recent-regulatory-history explanation adds information beyond H0 current burden and E03 one-step change;
- provide the smallest explicit H1-vs-H0 history baseline without a learned model or arbitrary post-hoc memory kernel.

E11 is `Unavailable(InsufficientHistory)` until 16 completed prior realized states exist. The window is fixed at 16 because it matches the frozen v0.1 default allostatic horizon length, giving a prospective structural symmetry rather than tuning the window to results.

E11 must be evaluated on matched-current-native-state / different-history scenarios.

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
- E08: absolute prospective cumulative exposure;
- E09: confidence;
- E10: legacy precision×importance hypothesis;
- E11: simple recent external history beyond current state;
- E00: null floor.

Do not add another candidate merely because it is available in the factor space. Addition requires a written discrimination obligation that cannot be answered by the current set and a new candidate-set identity.

## 4. Primary forecast-policy decision

For the first exploratory lineage, the primary prefix-causal forecast policy for E04, E05, E06, E07, and E08 is `ObservedDrivePersistence`.

Rationale:

- it uses only information already observed at the cut point;
- it asks a simple local counterfactual: what happens if the current observed load continues?;
- it maps naturally to the frozen v0.1 dynamics-aware constant-drive rollout;
- it avoids true-future schedule leakage;
- it gives E02 current drive magnitude a strong nuisance baseline against which to test whether forecast simulation adds anything beyond the load itself.

The policy identity includes the exact v0.1 dynamics-aware rollout semantics, horizon, discount, and timestep requirements.

`NativeZeroInputRecovery` and `KinematicVelocity` remain prospectively named sensitivity/diagnostic policies, not duplicate primary candidates. They may be used in locked forecast-policy disagreement scenarios and robustness reports, but they do not multiply the E00–E11 primary exploratory registry.

`OracleDiagnostic` remains an upper-bound/diagnostic authority only and can never replace the primary policy.

Changing the primary policy requires a new candidate-set/design identity before exploratory outputs are inspected.

## 5. Preprocessing discipline

Initial preprocessing for E00–E11 is `None` except exact type conversion / structural arithmetic explicitly included in each candidate formula.

No z-scoring, cohort normalization, fitted clipping, learned threshold, adaptive smoothing, or fitted calibration is permitted in the minimal first exploratory candidate set.

A later need for fitted preprocessing requires a new candidate definition and candidate-set/design identity before the affected exploratory outputs are inspected.

## 6. Evaluator isolation

All E00–E11 evaluations use the `NoneAcrossEvaluationCoordinates` persistent-state class from `V02_OBSERVATORY_STATE_LIFECYCLE.md`.

H1 candidates may read same-scenario history directly from the immutable prefix, but may not carry mutable state from a previous scenario, candidate, cut point, batch, or process invocation.

## 7. Required pairwise discrimination obligations

The initial scenario/cut-point matrix must include at least one prospective discriminator for:

- E03 vs E01;
- E03 vs E02;
- E04 vs E03;
- E04 vs E02;
- E05 vs E04;
- E05 vs E06;
- E05 vs E02;
- E07 vs E01;
- E07 vs E02;
- E08 vs E01;
- E08 vs E07;
- E08 vs E02;
- E09 vs E01;
- E10 vs E01;
- E11 vs E01;
- E11 vs E03;
- every non-null candidate vs E00.

If any required pair lacks a discriminator, the exploratory design is incomplete.

## 8. Candidate reduction after exploration

The exploratory phase does not have to produce a winner.

Prospectively valid outcomes include one candidate supported beyond all required baselines, multiple observational equivalence classes, different target-specific candidates, a simpler baseline dominating, unresolved ambiguity, no unique candidate, or a complete null.

Use a prospective parsimony rule: when two candidates are empirically equivalent under all registered discriminators, retain the simpler/lower-information candidate as the preferred explanation and preserve the equivalence result.

## 9. No affect interpretation during candidate selection

Candidate ranking and reduction must use only the preregistered numerical/discrimination criteria.

Do not select based on which time series looks most emotional, resembles human valence intuitively, produces pleasing plots, has dramatic excursions, or correlates best with semantic scenario names after unblinding.

Interpretation comes after blinded candidate evidence freezes.

## 10. Machine-readable manifest

A future `ExploratoryCandidateSetManifest` should bind:

- schema/version;
- exact design-contract-registry digest;
- ordered candidate-definition digests for E00–E11;
- candidate role IDs;
- required pairwise discrimination obligations;
- primary forecast policy = `ObservedDrivePersistence`;
- sensitivity/diagnostic forecast policies = `NativeZeroInputRecovery`, `KinematicVelocity`;
- preprocessing policy = no fitted preprocessing;
- evaluator persistent-state policy = `NoneAcrossEvaluationCoordinates`;
- scenario-discrimination manifest digest;
- functional-evaluation/promotion contract digest;
- selection/parsimony rule digest;
- canonical SHA-256.

Validation must reject missing/duplicate/extra candidate roles, candidate-definition mismatch, E11 window drift, missing discriminators, primary forecast-policy substitution, oracle/retrospective primary authority, fitted/adaptive preprocessing, or cross-evaluation mutable-state authority.

## 11. Claim boundary

This finite set is designed to determine whether any regulatory observable adds reproducible information beyond simple state, load, change, confidence, urgency, recent history, prospective exposure, and legacy-weighting explanations.

Even a uniquely successful exploratory candidate is not evidence of emotion, subjective valence, feeling, mood, suffering, sentience, or consciousness. It is only a candidate for later independently frozen confirmatory testing.