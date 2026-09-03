# Affective Emergence v0.2 — Candidate Factor-Space Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract prevents observational candidates from hiding multiple scientific choices inside one formula or name.

A v0.2 candidate is identified by an explicit coordinate across orthogonal design axes. The coordinate is part of the candidate manifest and therefore part of prospective evidence identity.

## 1. Principle

Do not let a candidate called “regulatory improvement” silently choose:

- what regulatory relationship is measured;
- how channel burden is weighted;
- how channels are projected/reduced into a scalar or vector;
- how burden is aggregated over time;
- which forecast policy is used;
- what information class is permitted;
- whether the candidate uses only current native state or externally replayed history.

Represent these as explicit factors.

## 2. Candidate coordinate

A future `CandidateCoordinate` should bind at least:

- `relation_basis`;
- `weighting_basis`;
- `channel_aggregation_basis`;
- `temporal_aggregation`;
- `forecast_policy` when applicable;
- `information_class`;
- `history_access_basis`;
- `availability_rule`;
- `numeric_contract_version`.

The coordinate is not a substitute for the exact formula manifest. It is a typed index into the scientific design space; the exact candidate definition still binds formula, indices, normalization, numerical rules, fixtures, and implementation identity.

## 3. Relation basis

Initial neutral relation classes:

- `R0CurrentBurden` — descriptive current realized condition;
- `P0ProspectiveBurden` — absolute projected regulatory burden/exposure over a declared prefix-causal forecast trajectory, without subtracting another time point;
- `R1RealizedChange` — change from prior realized burden to current realized burden;
- `R2OneStepForecastResidual` — prior forecast for current point minus realized current burden;
- `R3OverlappingFutureRevision` — revision of predictions for the same absolute future support;
- `R4RollingHorizonChange` — change in finite-horizon aggregate, explicitly allowing horizon turnover;
- `U1Urgency` — breach imminence / breadth / peak / rate family;
- `BaselineReactive` — simple current-state, drive, or stimulus baseline;
- `BaselineNuisance` — prospectively declared nuisance/history control.

`P0ProspectiveBurden` exists because the frozen v0.1 allostatic layer already computes absolute prospective burden. It must not be forced into R0 current-state semantics or R4 change semantics merely to fit the enum.

A later new relation class is a design/evidence change, not a free-form label.

## 4. Weighting basis

From `V02_WEIGHTING_DECOMPOSITION.md`:

- `W0RawChannel` — no normative/confidence weighting beyond the declared channel projection;
- `W1ViabilityWeightOnly` — current importance/preference weighting without precision;
- `W2LegacyPrecisionTimesImportance` — exact v0.1 weighting hypothesis;
- `W3ConfidenceOnly` — explicit precision/confidence observable, not burden;
- `WFutureQualified` — reserved only for a separately frozen future weighting contract.

`W3ConfidenceOnly` cannot be combined with formulas that claim to be a burden aggregate unless the candidate definition explicitly defines the resulting cross-quantity relation. Confidence is not silently reclassified as normative burden.

## 5. Cross-channel aggregation basis

From `V02_CHANNEL_AGGREGATION_CONTRACT.md`:

- `A0FullVector` — preserve all declared channel values;
- `A1SingleChannel(channel)` — one prospectively named channel;
- `A2PeakDeviation` — worst declared channel deviation;
- `A3WeightedMeanAllDeclared` — weighted mean over a prospectively fixed channel set;
- `A4WeightedSumAllDeclared` — weighted sum over that fixed set;
- `A5BreachBreadth` — count/breadth under a declared boundary rule;
- `A6ThreatenedSubsetDiagnostic` — diagnostic subset reduction with prospectively legal prefix-causal membership;
- `AFutureQualified` — reserved for later separately frozen aggregation semantics.

The aggregation basis fixes denominator/projection semantics independently of weighting. `W1 × A3` and `W1 × A4` are different candidates even with the same channel weights.

Healthy-channel dilution under `A3` is a declared sensitivity, not hidden recovery. `A4`, `A2`, `A1`, and `A0` have different invariances and must not be treated as aliases.

## 6. Temporal aggregation

From `V02_ALLOSTATIC_EXPOSURE_DECOMPOSITION.md`:

- `T0Instantaneous`;
- `T1DiscountedMean`;
- `T2DiscountedCumulativeExposure`;
- `T3UndiscountedCumulativeExposure`;
- `T4Peak`;
- `T5Terminal`;
- `T6PreferredRangeExposure`;
- `T7ViabilityBreachExposure`;
- `T8FirstBreachLatency`;
- `T9RecoveryExposure`.

Not every relation supports every temporal class. Invalid combinations must fail manifest validation rather than being interpreted ad hoc.

`P0ProspectiveBurden` is the natural relation for T1/T2/T4/T5/T6/T7/T8 summaries over one current prefix-causal forecast trajectory when no inter-time subtraction is intended.

## 7. Forecast policy

Initial prefix-causal forecast policy classes remain separate:

- `None` — no forecast required;
- `NativeZeroInputRecovery`;
- `ObservedDrivePersistence`;
- `KinematicVelocity`;
- later separately qualified predictive policy.

Oracle future knowledge is not a forecast-policy variant in this enum. It belongs to `OracleDiagnostic` information authority and cannot enter the primary prefix-causal registry.

The initial minimal exploratory set freezes `ObservedDrivePersistence` as the primary policy for its forecast-bearing roles; zero-input recovery and kinematic velocity remain targeted sensitivity/diagnostic policies rather than duplicate primary candidate families.

## 8. Information class

Use explicit authority/execution classes:

- `OfflinePrefixCausal` — primary initial v0.2 evidence mode over immutable prefix replay;
- `OnlinePrefixCausalShadow` — later real-time shadow-equivalence mode;
- `RetrospectiveDiagnostic`;
- `OracleDiagnostic`.

Changing information class changes candidate identity even if the numerical expression is otherwise identical.

## 9. History access basis

From `V02_HISTORY_STATE_SUFFICIENCY.md`:

- `H0CurrentNativeStateOnly` — current validated native state/configuration plus other explicitly current allowed inputs only;
- `H1ReplayedPrefixHistory` — immutable execution-prefix history is consumed by the external observatory;
- `H2NativePersistedMemory` — reserved for a future separately qualified native memory/state mechanism; unavailable to initial v0.2;
- `H3RetrospectiveHistory` — realized post-cut-point history; diagnostic only;
- `H4OracleFuture` — future information; oracle diagnostic only.

`H1` is still prefix-causal when it uses only history through the cut point, but it supports a different claim from `H0`. It shows that prior trace history helps an external observable, not that the native regulator itself stores that history.

Changing history window, reset semantics, forgetting factor, or sufficient-statistic definition changes candidate identity.

## 10. Compatibility matrix

A future candidate-registry validator should enforce an explicit compatibility table.

Examples:

- `R0CurrentBurden × T8FirstBreachLatency` is invalid; use a prospective relation such as `P0ProspectiveBurden`/urgency with a declared forecast;
- `P0ProspectiveBurden` requires a prefix-causal forecast policy and one declared current-origin forecast trajectory;
- `P0ProspectiveBurden × T2DiscountedCumulativeExposure` is valid for absolute projected exposure when horizon/discount are frozen;
- `R1RealizedChange × T0Instantaneous` is valid if the two realized cut points are explicitly defined;
- `R3OverlappingFutureRevision` requires a forecast policy and overlapping absolute future support;
- `T9RecoveryExposure` requires a declared perturbation/recovery window and cutoff semantics;
- `W3ConfidenceOnly × T2CumulativeExposure` may be valid as confidence exposure, but it must be interpreted as confidence exposure, not burden;
- `A0FullVector` cannot carry a scalar denominator policy;
- `A1SingleChannel` cannot normalize across unrelated channels;
- `A6ThreatenedSubsetDiagnostic` is invalid when subset membership depends on future, post-run exclusion, semantic arm identity, or unblinded interpretation;
- `A5BreachBreadth` cannot claim continuous severity without an additional declared input;
- `H0CurrentNativeStateOnly` cannot require an arbitrary earlier trace window;
- `H1ReplayedPrefixHistory` must declare the exact allowed history window/aggregation rule;
- `H2NativePersistedMemory` is invalid in initial v0.2;
- `H3RetrospectiveHistory` requires retrospective information authority;
- `H4OracleFuture` requires oracle information authority;
- `OracleDiagnostic` candidates cannot be marked primary endogenous candidates;
- `OnlinePrefixCausalShadow` cannot become confirmatory-primary until its offline equivalence gate is qualified.

The compatibility table itself is versioned evidence-critical design.

## 11. Candidate ID rule

Human-readable IDs should expose enough of the coordinate to prevent accidental confusion, while the canonical manifest remains authoritative.

Example forms:

- `r0_w1_a3_t0_h0_viability_mean_v1`
- `p0_w1_a3_t2_h0_drive_persistence_cumulative_v1`
- `r1_w1_a3_t0_h1_realized_change_v1`
- `r3_w1_a3_t1_h1_drive_persistence_revision_v1`
- `r4_w2_a3_t1_h1_legacy_rolling_mean_change_v1`
- `u1_w0_a2_t8_h0_first_breach_latency_v1`
- `r0_w3_a3_t0_h0_confidence_mean_v1`

Do not encode interpretation-bearing terms such as `valence`, `fear`, `sadness`, `mood`, or `pain` into v0.2 candidate IDs.

## 12. Candidate-set generation is closed, not Cartesian by default

The factor axes define a design space, but v0.2 must not automatically compute every possible Cartesian combination.

The first exploratory lineage is governed by `V02_MINIMAL_EXPLORATORY_CANDIDATE_SET.md`, which freezes a finite E00–E11 role set and required pairwise discrimination obligations.

Reasons:

- avoid combinatorial fishing;
- avoid post-hoc selection from hundreds of near-duplicate formulas;
- make compute/evidence accounting bounded;
- preserve meaningful null results.

The factor coordinate helps explain why each chosen candidate exists; it does not authorize unbounded candidate generation.

## 13. Identifiability requirement

Before exploratory/confirmatory interpretation, apply `V02_IDENTIFIABILITY_AND_DISCRIMINATION.md`:

- every required primary-vs-baseline comparison must have at least one registered discriminator;
- candidate equivalence classes must be preserved rather than broken by interpretive preference;
- the scenario/cut-point design matrix should isolate weighting, cross-channel aggregation, temporal aggregation, relation, forecast-policy, and history-access axes where possible;
- H1 history-sensitive candidates intended to add information beyond current state must be tested against H0 current-state baselines on matched-current-state histories;
- `InsufficientDiscrimination` is a valid design outcome.

Candidate complexity cannot substitute for identifiability.

## 14. Exploratory-to-confirmatory promotion

Exploratory work may compare only the locked finite candidate set.

Before confirmatory work:

- compute the locked candidate discrimination/equivalence report;
- classify history-sensitive candidate gains as external-history gains unless H2 native memory exists in a later lineage;
- choose the primary candidate by a prospectively declared exploratory selection/parsimony rule;
- freeze its exact candidate definition and coordinate;
- freeze required nuisance/baseline candidates;
- freeze the holdout scenario cohort and analysis plan;
- create a new confirmatory root.

A different coordinate that looks better after confirmatory unblinding motivates a new confirmatory lineage; it cannot replace the original primary candidate in place.

If the selected primary remains observationally equivalent to a simpler baseline under all registered discriminators, a claim of superiority over that baseline is not confirmatory-identifiable.

## 15. Interaction tests

The exploratory scenario program should contain factorial discriminators that isolate axes rather than only testing full candidate outputs.

At minimum:

- weighting-only manipulations with temporal profile and aggregation basis held fixed;
- healthy-channel dilution/denominator manipulations with deviated-channel state held fixed;
- concentration/distribution manipulations separating mean, sum, peak, vector, and breadth;
- temporal-profile manipulations with weighting/aggregation inputs held fixed;
- same-current-native-state / different-history manipulations separating H1 from H0;
- restart-equivalence tests showing native future equality from matched complete state/config + identical future inputs;
- forecast-policy disagreement with realized prefix held as comparable as possible;
- relation-basis crossed-sign cases for R1/R2/R3/R4;
- absolute prospective-burden cases separating P0 from R0 current burden and U1 urgency;
- information-class suffix-mutation tests proving prefix-causal payload invariance.

This supports mechanistic attribution when candidates disagree.

## 16. Evidence identity

The canonical candidate manifest should bind:

- the complete `CandidateCoordinate`;
- exact compatibility-contract version;
- exact minimal exploratory candidate-set identity when applicable;
- exact weighting-decomposition contract digest/version;
- exact channel-aggregation contract digest/version;
- exact allostatic-exposure contract digest/version;
- exact history/state-sufficiency contract digest/version;
- exact temporal-alignment contract digest/version;
- exact execution/information contract digest/version;
- exact preprocessing/evaluator-isolation identities;
- exact formula/fixtures/implementation identity.

Changing any coordinate field changes candidate identity even if an accidental numerical coincidence leaves one fixture unchanged.

## 17. Claim boundary

A factorized candidate space makes the experiment easier to audit and helps explain whether observed structure comes from current state, absolute prospective burden, realized change, forecast error/revision, weighting, cross-channel reduction, temporal integration, forecast assumptions, information authority, or external history access.

It does not establish that any coordinate corresponds to emotion, valence, mood, suffering, sentience, consciousness, or native memory.