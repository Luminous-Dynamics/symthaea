# Affective Emergence v0.2 — Candidate Definition Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract defines how an observational regulatory candidate becomes an immutable research object. It exists to prevent formula drift, hidden weighting/temporal changes, information-class changes, or post-result reinterpretation from being treated as the same candidate.

## Principle

A candidate is not just a function name. Its scientific identity includes every choice that can change the emitted value or its interpretation.

Changing any identity-bearing field creates a new candidate definition and therefore a new prospective evidence identity.

Candidate identity is now explicitly factorized by `V02_CANDIDATE_FACTOR_SPACE.md`. The factor coordinate does not replace the exact formula manifest; both are required.

## Proposed candidate manifest

A later `ObservationalCandidateDefinitionManifest` should bind at minimum:

- schema version;
- stable candidate ID;
- candidate-definition version;
- complete `CandidateCoordinate` / factor-space version;
- neutral relation family (`current_burden`, `realized_change`, `forecast_residual`, `overlap_revision`, `rolling_horizon_change`, `regulatory_urgency`, or explicitly declared baseline);
- weighting basis;
- temporal aggregation basis;
- execution/information class;
- forecast-information/policy class when applicable;
- channel projection when applicable;
- exact mathematical/sign convention;
- required input fields;
- forbidden input classes;
- forecast horizon;
- discount factor;
- temporal-alignment rule;
- overlap weighting kernel;
- normalization rule;
- numerical precision/accumulation rule;
- undefined/missing-value semantics;
- minimum time index at which the candidate is defined;
- unit/domain convention;
- source implementation identity;
- compatibility-contract version;
- reference-vector fixture digest;
- v0.1 model-semantics version;
- v0.1 snapshot/execution schema identities used by the candidate;
- design-contract-registry identity or the frozen design identity that transitively binds it.

The validated canonical form should have a deterministic SHA-256.

## Neutral naming rule

Candidate IDs and causal APIs must remain interpretation-neutral during v0.2.

Preferred examples expose enough of the factor coordinate to make accidental substitution visible:

- `r0_w1_t0_viability_burden_v1`
- `r1_w1_t0_viability_change_v1`
- `r2_w2_t0_legacy_one_step_residual_v1`
- `r3_w1_t1_overlap_mean_revision_v1`
- `r4_w1_t2_viability_cumulative_change_v1`
- `u1_w0_t8_first_breach_latency_v1`
- `r0_w3_t0_confidence_v1`

Avoid `valence`, `fear`, `joy`, `sadness`, `mood`, `pain`, `arousal`, `dominance`, or similar interpretation-bearing identifiers in the v0.2 metric contract.

Interpretive labels may be discussed only in later reports and must never alter the numerical candidate definition.

## Execution / information classes

### OfflinePrefixCausal

Primary initial v0.2 evidence class.

Candidate computation occurs after the native execution trace is complete and frozen, but receives only a validated `ObservationPrefixView(t)` constructed from information available through cut point `t`.

It must pass:

- prefix-equivalence;
- future-suffix mutation invariance;
- candidate-payload/full-trace-provenance separation;
- source/dependency authority audits.

The full source-trace digest is **not** an allowed candidate input because it changes when the unseen suffix changes. It belongs only to the outer evidence envelope.

### OnlinePrefixCausalShadow

Later engineering-validation class.

A live/co-resident computation may be compared with the qualified offline payload only after it proves:

- no-observer/native trace equivalence;
- candidate-order independence;
- exact offline/online candidate-payload equivalence under the locked contract.

It is not the initial primary scientific evidence class.

### RetrospectiveDiagnostic

May use realized information after `t`, but the artifact must state the latest time required to compute the value. Retrospective candidates cannot be described as information available through `t`.

### OracleDiagnostic

May intentionally use true future experimental schedules or realized future information as an upper-bound/diagnostic control. Oracle metrics can never be selected as the primary endogenous prefix-causal candidate.

A change between these classes is a new candidate identity even if the numerical formula text appears unchanged.

## Factor-coordinate contract

Every candidate has a coordinate binding at least:

- relation basis;
- weighting basis;
- temporal aggregation;
- forecast policy;
- execution/information class;
- channel projection;
- availability rule;
- numeric-contract version.

The compatibility matrix is evidence-critical. Invalid coordinates fail candidate-manifest validation rather than being interpreted ad hoc.

Examples:

- R3 requires forecast trajectories with overlapping absolute future support;
- T8 first-breach latency requires a prospective trajectory, not a current-state-only relation;
- W3 confidence-only candidates must not be labeled as burden unless an explicit cross-quantity formula defines that relation;
- OracleDiagnostic cannot be confirmatory-primary;
- OnlinePrefixCausalShadow cannot replace OfflinePrefixCausal in the initial lineage without a new design identity.

## Exact formula contract

Natural-language descriptions are insufficient for confirmatory work.

Before a candidate enters confirmatory evidence, lock a formula specification that fixes:

- operand order;
- sign convention;
- temporal indices;
- relation basis;
- weighting basis;
- whether quantities are instantaneous, cumulative, normalized means, sums, peaks, terminal values, durations, or latencies;
- weighting and normalization constants;
- forecast policy/horizon/discount;
- boundary handling;
- tolerance/equality rules;
- handling of absent forecast breaches (`first_breach_step = None`);
- conversion rules between v0.1 `f32` native quantities and any v0.2 higher-precision derived arithmetic.

The implementation must have reference fixtures whose expected outputs are independently hand/computer-derived and whose fixture digest is bound into the candidate manifest.

## Floating-point discipline

v0.2 should not let floating-point implementation details silently define scientific outcomes.

Recommended contract:

- retain exact v0.1 native values as inputs without modifying native execution;
- use deterministic `f64` accumulation for v0.2 derived weighted sums/means/exposures unless a stronger reason is documented;
- do not enable fast-math or architecture-dependent approximate reductions in qualified evidence code;
- define comparison tolerances prospectively;
- distinguish exact-replay/payload equality from scientific equivalence tolerances;
- record non-finite derived values as hard validation failures, never clamp them into apparently valid observations.

If changing numeric representation changes a confirmatory threshold result, treat that as a sensitivity/failure finding rather than selecting the preferred representation after unblinding.

## Undefined is not zero

Candidates have different temporal requirements.

Examples:

- R1 requires both `t-1` and `t`;
- R2 requires a forecast made at `t-1` and the realized state at `t`;
- R3 requires two forecasts with at least one shared absolute future point;
- R4 requires two aggregate rolling forecasts;
- T8 requires a defined breach or an explicit typed no-breach state;
- T9 requires a prospectively declared perturbation/recovery window and cutoff.

When a candidate is not defined, emit an explicit typed unavailable state/reason. Do not silently substitute zero, carry forward the previous value, or drop the row.

Missingness itself must be deterministic and auditable.

## Information dependency declaration

Each candidate should declare an explicit dependency set. Proposed categories include:

- `CurrentNativeState`;
- `PriorNativeState`;
- `ObservedDriveHistory`;
- `ExecutedInterventionHistory`;
- `NativeDynamicsConfig`;
- `PrefixForecastTrajectory`;
- `PrefixDigest`;
- `RealizedFutureState`;
- `FutureProtocolSchedule`;
- `FullSourceTraceDigest`;
- `SemanticArmIdentity`.

Qualified `OfflinePrefixCausal` / `OnlinePrefixCausalShadow` candidates must reject dependency manifests containing:

- `RealizedFutureState`;
- `FutureProtocolSchedule`;
- `FullSourceTraceDigest`;
- `SemanticArmIdentity`.

The runtime API should be shaped so forbidden dependencies are unavailable by type, not merely discouraged by documentation.

## Payload vs evidence envelope

The candidate's prefix-causal computation emits a `CandidatePayload` containing only allowed prefix-derived identity and result information.

The outer `CandidateEvidenceEnvelope` binds the payload to the full source execution trace, study/evidence root, toolchain, and storage artifacts.

For two traces with identical allowed prefix but different unseen suffix:

- candidate payloads must be byte-identical under the canonical contract;
- outer evidence envelopes may differ because full-trace provenance differs.

A candidate manifest or implementation that makes payload identity depend on full-trace provenance fails prefix causality.

## Candidate equivalence and non-equivalence

Two candidate manifests are the same scientific candidate only when their canonical identity fields are identical.

The following are automatically new candidate identities:

- changing factor coordinate;
- changing sign;
- changing horizon or discount;
- changing zero-input to current-drive-persistence forecast policy;
- changing overlap weighting;
- changing weighting basis;
- changing temporal aggregation;
- changing channel projection;
- changing normalization;
- changing OfflinePrefixCausal to OnlinePrefixCausalShadow, RetrospectiveDiagnostic, or OracleDiagnostic;
- changing treatment of undefined values;
- changing temporal alignment;
- changing the v0.1 semantic lineage;
- changing source implementation in a way that alters reference outputs.

A pure source refactor may preserve candidate-definition version only when the factor coordinate, all reference fixtures, canonical definition fields, and qualified outputs remain identical.

## Finite candidate-set rule

The factor axes define a design space but do not authorize an unrestricted Cartesian search.

Before exploratory execution, freeze an explicit finite `ExploratoryCandidateSetManifest` listing every eligible candidate-definition digest.

After exploratory evaluation, a separately frozen confirmatory candidate/baseline set is required.

This prevents candidate proliferation from becoming post-hoc metric fishing.

## Candidate comparison states

Confirmatory comparison should not force a winner.

A preregistered comparison may resolve to states such as:

- `SupportedBeyondBaselines`;
- `EquivalentToBaseline`;
- `FailsDirectionalGate`;
- `FailsNeutralityGate`;
- `FailsPrefixCausality`;
- `FailsSensitivityRegion`;
- `WeightingAmbiguous`;
- `TemporalAggregationAmbiguous`;
- `NumericallyUnstable`;
- `Indeterminate`;
- `NoUniqueWinner`.

These are valid scientific outcomes and must not trigger retrospective metric redefinition inside the same confirmatory lineage.

## Primary/secondary promotion rule

Exploratory work may choose a primary candidate from the prospectively frozen exploratory candidate set under a declared selection rule.

Once confirmatory study identity is locked:

- the primary candidate cannot be replaced after observing confirmatory results;
- its factor coordinate cannot be changed;
- secondary candidates remain secondary;
- a secondary candidate that looks better may motivate a new confirmatory lineage, not promotion within the current one;
- the exact baseline set and candidate-ranking rule remain frozen.

## Required implementation gates

Before any v0.2 candidate is eligible for confirmatory use, tests should require:

1. canonical candidate digest is stable under serialization round trip;
2. every identity-bearing field and factor coordinate changes the digest when altered;
3. invalid factor combinations are rejected by the compatibility validator;
4. reference fixtures reproduce exactly or within the prospectively declared numerical tolerance;
5. OfflinePrefixCausal candidates pass prefix-equivalence and future-mutation invariance;
6. full source-trace provenance cannot enter candidate computation;
7. RetrospectiveDiagnostic/OracleDiagnostic candidates cannot enter the prefix-causal qualified API by construction;
8. OnlinePrefixCausalShadow cannot be treated as equivalent to offline evidence without its explicit equivalence gates;
9. undefined cases remain explicitly unavailable rather than zero-filled;
10. derived values remain finite on the declared valid scenario region;
11. candidate computation does not mutate or alter the underlying v0.1 execution artifact;
12. semantic emotion labels are absent from the metric-definition API;
13. candidate payloads bind prefix/candidate identity while outer envelopes bind full source/study/toolchain provenance;
14. candidate set membership is finite and prospectively locked.

## Claim boundary

This contract can establish that a numerical regulatory observable and all of its relation, weighting, temporal, forecast, information, and projection choices were fixed prospectively, computed reproducibly, and compared fairly.

It cannot establish that the observable is emotion, subjective valence, feeling, mood, suffering, sentience, or consciousness.
