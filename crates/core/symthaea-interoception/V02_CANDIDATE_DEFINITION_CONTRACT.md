# Affective Emergence v0.2 — Candidate Definition Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract defines how an observational regulatory candidate becomes an immutable research object. It exists to prevent formula drift, hidden normalization changes, information-class changes, or post-result reinterpretation from being treated as the same candidate.

## Principle

A candidate is not just a function name. Its scientific identity includes every choice that can change the emitted value or its interpretation.

Changing any identity-bearing field creates a new candidate definition and therefore a new prospective evidence identity.

## Proposed candidate manifest

A later `ObservationalCandidateDefinitionManifest` should bind at minimum:

- schema version;
- stable candidate ID;
- candidate-definition version;
- neutral family name (`realized_change`, `forecast_residual`, `overlap_revision`, `rolling_debt_change`, `regulatory_urgency`, or explicitly declared baseline); 
- temporal availability: `OnlinePrefixCausal`, `Retrospective`, or `OracleDiagnostic`;
- forecast-information class when applicable;
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
- reference-vector fixture digest;
- v0.1 model-semantics version;
- v0.1 snapshot/execution schema identities used by the candidate.

The validated canonical form should have a deterministic SHA-256.

## Neutral naming rule

Candidate IDs and causal APIs must remain interpretation-neutral during v0.2.

Preferred examples:

- `r1_realized_change_v1`
- `r2_one_step_residual_v1`
- `r3_overlap_revision_v1`
- `r4_rolling_debt_change_v1`
- `u1_breach_imminence_v1`

Avoid `valence`, `fear`, `joy`, `sadness`, `arousal`, `dominance`, or similar interpretation-bearing identifiers in the metric contract.

Interpretive labels may be discussed only in later reports and must never alter the numerical candidate definition.

## Temporal-availability classes

### OnlinePrefixCausal

May use only the information permitted by `V02_INFORMATION_FIREWALL.md` through time `t` plus prospectively locked policy parameters.

It must pass prefix-equivalence and future-mutation invariance tests.

### Retrospective

May use realized information after `t`, but the artifact must state the latest time required to compute the value. Retrospective candidates cannot be described as available to the agent at `t` and cannot be used in the v0.2 no-feedback online pathway.

### OracleDiagnostic

May intentionally use true future experimental schedules or realized future information as an upper-bound/diagnostic control. Oracle metrics can never be selected as the primary endogenous candidate.

A change between these classes is a new candidate identity even if the numerical formula text appears unchanged.

## Exact formula contract

Natural-language descriptions are insufficient for confirmatory work.

Before a candidate enters confirmatory evidence, lock a formula specification that fixes:

- operand order;
- sign convention;
- temporal indices;
- whether quantities are instantaneous, cumulative, normalized means, or sums;
- weighting and normalization constants;
- boundary handling;
- tolerance/equality rules;
- handling of absent forecast breaches (`first_breach_step = None`);
- conversion rules between v0.1 `f32` native quantities and any v0.2 higher-precision derived arithmetic.

The implementation must have reference fixtures whose expected outputs are independently hand/computer-derived and whose fixture digest is bound into the candidate manifest.

## Floating-point discipline

v0.2 should not let floating-point implementation details silently define scientific outcomes.

Recommended contract:

- retain exact v0.1 native values as inputs without modifying native execution;
- use deterministic `f64` accumulation for v0.2 derived weighted sums/means unless a stronger reason is documented;
- do not enable fast-math or architecture-dependent approximate reductions in qualified evidence code;
- define comparison tolerances prospectively;
- distinguish exact-replay equality from scientific equivalence tolerances;
- record non-finite derived values as hard validation failures, never clamp them into apparently valid observations.

If changing numeric representation changes a confirmatory threshold result, treat that as a sensitivity/failure finding rather than selecting the preferred representation after unblinding.

## Undefined is not zero

Candidates have different temporal requirements.

Examples:

- R1 requires both `t-1` and `t`;
- R2 requires a forecast made at `t-1` and the realized state at `t`;
- R3 requires two forecasts with at least one shared absolute future point;
- R4 requires two aggregate rolling forecasts.

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
- `RealizedFutureState`;
- `FutureProtocolSchedule`;
- `SemanticArmIdentity`.

Qualified online candidates must reject dependency manifests containing `RealizedFutureState`, `FutureProtocolSchedule`, or `SemanticArmIdentity`.

The runtime API should be shaped so forbidden dependencies are unavailable by type, not merely discouraged by documentation.

## Candidate equivalence and non-equivalence

Two candidate manifests are the same scientific candidate only when their canonical identity fields are identical.

The following are automatically new candidate identities:

- changing sign;
- changing horizon or discount;
- changing zero-input to current-drive-persistence forecast policy;
- changing overlap weighting;
- changing normalization;
- changing online to retrospective availability;
- changing treatment of undefined values;
- changing temporal alignment;
- changing the v0.1 semantic lineage;
- changing source implementation in a way that alters reference outputs.

A pure source refactor may preserve candidate-definition version only when all reference fixtures, canonical definition fields, and qualified outputs remain identical.

## Candidate comparison states

Confirmatory comparison should not force a winner.

A preregistered comparison may resolve to states such as:

- `SupportedBeyondBaselines`;
- `EquivalentToBaseline`;
- `FailsDirectionalGate`;
- `FailsNeutralityGate`;
- `FailsPrefixCausality`;
- `FailsSensitivityRegion`;
- `NumericallyUnstable`;
- `Indeterminate`;
- `NoUniqueWinner`.

`NoUniqueWinner` and `Indeterminate` are valid scientific outcomes and must not trigger retrospective metric redefinition inside the same confirmatory lineage.

## Primary/secondary promotion rule

Exploratory work may choose a primary candidate from a preregistered exploratory candidate set.

Once confirmatory study identity is locked:

- the primary candidate cannot be replaced after observing confirmatory results;
- secondary candidates remain secondary;
- a secondary candidate that looks better may motivate a new confirmatory lineage, not promotion within the current one;
- the exact baseline set and candidate-ranking rule remain frozen.

## Required implementation gates

Before any v0.2 candidate is eligible for confirmatory use, tests should require:

1. canonical candidate digest is stable under serialization round trip;
2. every identity-bearing field changes the digest when altered;
3. reference fixtures reproduce exactly or within the prospectively declared numerical tolerance;
4. online candidates pass prefix-equivalence and future-mutation invariance;
5. retrospective/oracle candidates cannot enter the online qualified API by construction;
6. undefined cases remain explicitly unavailable rather than zero-filled;
7. derived values remain finite on the declared valid scenario region;
8. candidate computation does not mutate or alter the underlying v0.1 execution artifact;
9. semantic emotion labels are absent from the metric-definition API;
10. candidate artifacts bind exact source, v0.1 lineage, study, execution, and candidate-definition digests.

## Claim boundary

This contract can establish that a numerical regulatory observable was fixed prospectively, computed reproducibly, and compared fairly.

It cannot establish that the observable is emotion, subjective valence, feeling, sentience, or consciousness.