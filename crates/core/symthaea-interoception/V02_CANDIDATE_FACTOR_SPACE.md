# Affective Emergence v0.2 — Candidate Factor-Space Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract prevents observational candidates from hiding multiple scientific choices inside one formula or name.

A v0.2 candidate is identified by an explicit coordinate across orthogonal design axes. The coordinate is part of the candidate manifest and therefore part of prospective evidence identity.

## 1. Principle

Do not let a candidate called “regulatory improvement” silently choose:

- what regulatory relationship is measured;
- how channel burden is weighted;
- how burden is aggregated over time;
- which forecast policy is used;
- what information class is permitted.

Represent these as explicit factors.

## 2. Candidate coordinate

A future `CandidateCoordinate` should bind at least:

- `relation_basis`;
- `weighting_basis`;
- `temporal_aggregation`;
- `forecast_policy` when applicable;
- `information_class`;
- `channel_projection` when applicable;
- `availability_rule`;
- `numeric_contract_version`.

The coordinate is not a substitute for the exact formula manifest. It is a typed index into the scientific design space; the exact candidate definition still binds formula, indices, normalization, numerical rules, fixtures, and implementation identity.

## 3. Relation basis

Initial neutral relation classes:

- `R0CurrentBurden` — descriptive current condition;
- `R1RealizedChange` — change from prior realized burden to current realized burden;
- `R2OneStepForecastResidual` — prior forecast for current point minus realized current burden;
- `R3OverlappingFutureRevision` — revision of predictions for the same absolute future support;
- `R4RollingHorizonChange` — change in finite-horizon aggregate, explicitly allowing horizon turnover;
- `U1Urgency` — breach imminence / breadth / peak / rate family;
- `BaselineReactive` — simple current-state, drive, or stimulus baseline;
- `BaselineNuisance` — prospectively declared nuisance control.

A later new relation class is a design/evidence change, not a free-form label.

## 4. Weighting basis

From `V02_WEIGHTING_DECOMPOSITION.md`:

- `W0RawChannel` — no cross-channel aggregate;
- `W1ViabilityWeightOnly` — current importance/preference weighting without precision;
- `W2LegacyPrecisionTimesImportance` — exact v0.1 aggregate hypothesis;
- `W3ConfidenceOnly` — explicit precision/confidence observable, not burden;
- `WFutureQualified` — reserved only for a separately frozen future weighting contract.

`W3ConfidenceOnly` cannot be combined with formulas that claim to be a burden aggregate unless the candidate definition explicitly defines the resulting cross-quantity relation. Confidence is not silently reclassified as normative burden.

## 5. Temporal aggregation

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

## 6. Forecast policy

Initial prefix-causal forecast policy classes remain separate:

- `None` — no forecast required;
- `NativeZeroInputRecovery`;
- `ObservedDrivePersistence`;
- `KinematicVelocity`;
- later separately qualified predictive policy.

Oracle future knowledge is not a forecast-policy variant in this enum. It belongs to `OracleDiagnostic` information authority and cannot enter the primary prefix-causal registry.

## 7. Information class

Use explicit authority classes:

- `OfflinePrefixCausal` — primary initial v0.2 evidence mode over immutable prefix replay;
- `OnlinePrefixCausalShadow` — later real-time shadow-equivalence mode;
- `RetrospectiveDiagnostic`;
- `OracleDiagnostic`.

Changing information class changes candidate identity even if the numerical expression is otherwise identical.

## 8. Channel projection

Candidates that consume vector-valued burden should declare how channels are projected:

- `FullVector`;
- `SingleChannel { channel }`;
- `PeakChannel`;
- `ViabilityWeightedAggregate`;
- `LegacyWeightedAggregate`;
- `BreachBreadth`;
- another prospectively typed projection.

Projection must be compatible with weighting basis. For example, `FullVector` should not claim an aggregate weighting that was never applied.

## 9. Compatibility matrix

A future candidate-registry validator should enforce an explicit compatibility table.

Examples:

- `R0CurrentBurden × T8FirstBreachLatency` is invalid without a forecast/trajectory relation;
- `R1RealizedChange × T0Instantaneous` is valid if the two realized cut points are explicitly defined;
- `R3OverlappingFutureRevision` requires a forecast policy and overlapping absolute future support;
- `T9RecoveryExposure` requires a declared perturbation/recovery window and cutoff semantics;
- `W3ConfidenceOnly × T2CumulativeExposure` may be valid as cumulative low/high-confidence exposure, but it must be named and interpreted as confidence exposure, not burden;
- `OracleDiagnostic` candidates cannot be marked primary endogenous candidates;
- `OnlinePrefixCausalShadow` cannot become confirmatory-primary until its offline equivalence gate is qualified.

The compatibility table itself is versioned evidence-critical design.

## 10. Candidate ID rule

Human-readable IDs should expose the coordinate enough to prevent accidental confusion, while the canonical manifest remains authoritative.

Example forms:

- `r1_w1_t0_viability_change_v1`
- `r4_w2_t1_legacy_rolling_mean_change_v1`
- `r4_w1_t2_viability_cumulative_change_v1`
- `u1_w0_t8_first_breach_latency_v1`
- `r0_w3_t0_confidence_v1`

Do not encode interpretation-bearing terms such as `valence`, `fear`, `sadness`, `mood`, or `pain` into v0.2 candidate IDs.

## 11. Candidate-set generation is closed, not Cartesian by default

The factor axes define a design space, but v0.2 must not automatically compute every possible Cartesian combination.

Before exploratory execution, freeze an explicit finite `ExploratoryCandidateSetManifest` containing the exact candidate-definition digests eligible for comparison.

Reasons:

- avoid combinatorial fishing;
- avoid post-hoc selection from hundreds of near-duplicate formulas;
- make compute/evidence accounting bounded;
- preserve meaningful null results.

The factor coordinate helps explain why each chosen candidate exists; it does not authorize unbounded candidate generation.

## 12. Exploratory-to-confirmatory promotion

Exploratory work may compare the locked finite candidate set.

Before confirmatory work:

- choose the primary candidate by a prospectively declared exploratory selection rule;
- freeze its exact candidate definition and coordinate;
- freeze required nuisance/baseline candidates;
- freeze the holdout scenario cohort and analysis plan;
- create a new confirmatory root.

A different coordinate that looks better after confirmatory unblinding motivates a new confirmatory lineage; it cannot replace the original primary candidate in place.

## 13. Interaction tests

The exploratory scenario program should contain factorial discriminators that isolate axes rather than only testing full candidate outputs.

At minimum:

- weighting-only manipulations with temporal profile held fixed;
- temporal-profile manipulations with weighting inputs held fixed;
- forecast-policy disagreement with realized prefix held as comparable as possible;
- relation-basis crossed-sign cases for R1/R2/R3/R4;
- information-class suffix-mutation tests proving prefix-causal payload invariance;
- channel-projection cases where peak-channel and aggregate conclusions differ.

This supports mechanistic attribution when candidates disagree.

## 14. Evidence identity

The canonical candidate manifest should bind:

- the complete `CandidateCoordinate`;
- exact compatibility-contract version;
- exact weighting-decomposition contract digest/version;
- exact allostatic-exposure contract digest/version;
- exact temporal-alignment contract digest/version;
- exact execution/information contract digest/version;
- exact formula/fixtures/implementation identity.

Changing any coordinate field changes candidate identity even if an accidental numerical coincidence leaves one fixture unchanged.

## 15. Claim boundary

A factorized candidate space makes the experiment easier to audit and helps explain whether observed structure comes from relation semantics, weighting, temporal integration, forecast assumptions, or information authority.

It does not establish that any coordinate corresponds to emotion, valence, mood, suffering, sentience, or consciousness.
