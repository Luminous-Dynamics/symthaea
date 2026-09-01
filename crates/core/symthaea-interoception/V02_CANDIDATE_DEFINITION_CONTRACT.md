# Affective Emergence v0.2 — Candidate Definition Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract defines how an observational regulatory candidate becomes an immutable research object. It prevents formula drift, hidden weighting/aggregation/temporal changes, preprocessing leakage, evaluator-state dependence, information-class changes, or post-result reinterpretation from being treated as the same candidate.

## 1. Principle

A candidate is not just a function name or mathematical expression.

Its scientific identity includes every prospectively chosen element that can change:

- what information it is allowed to use;
- what value it emits;
- when the value is available;
- how the value is scaled/preprocessed;
- whether process/evaluator history can affect the value;
- how the value is interpreted relative to simpler baselines.

Changing an identity-bearing field creates a new candidate/evidence identity unless an explicit output-preserving equivalence proof applies.

The factor coordinate from `V02_CANDIDATE_FACTOR_SPACE.md` is necessary but not sufficient: exact formula, preprocessing, evaluator isolation, fixtures, and source identity are also required.

## 2. Proposed ObservationalCandidateDefinitionManifest

A future canonical manifest should bind at minimum:

- schema/version;
- stable candidate ID and definition version;
- complete `CandidateCoordinate` / factor-space version;
- relation basis;
- weighting basis;
- cross-channel aggregation basis;
- temporal aggregation basis;
- forecast policy/information basis where applicable;
- execution/information class;
- history-access basis;
- exact mathematical/sign convention;
- required input fields and forbidden input classes;
- forecast horizon/discount where applicable;
- temporal-alignment and overlap rules;
- normalization/scaling semantics;
- exact preprocessing-manifest digest or explicit `None`;
- calibration-cohort/fitted-parameter digests where applicable;
- evaluator-isolation-manifest digest;
- allowed evaluator persistent-state/cache class;
- numerical precision/accumulation rules;
- undefined/missing/out-of-range semantics;
- minimum defined time index;
- unit/domain convention;
- source implementation identity;
- compatibility-contract version;
- reference fixture digest;
- v0.1 model-semantics/snapshot/execution identities;
- frozen design/registry identity;
- canonical SHA-256.

Two definitions with identical mathematics but different value-changing preprocessing or evaluator-state policy are different scientific candidates.

## 3. Neutral naming

Candidate IDs remain interpretation-neutral during v0.2.

Preferred forms expose the factor coordinate, for example:

- `r0_w1_a3_t0_h0_viability_mean_v1`
- `r1_w1_a4_t0_h0_viability_sum_change_v1`
- `r2_w2_a3_t0_h0_legacy_residual_v1`
- `r3_w1_a3_t1_h1_overlap_revision_v1`
- `r4_w1_a4_t2_h1_cumulative_change_v1`
- `u1_w0_a2_t8_h0_breach_latency_v1`.

Avoid interpretation-bearing IDs such as `valence`, `fear`, `joy`, `sadness`, `pain`, `mood`, `arousal`, or `dominance`.

## 4. Execution / information classes

### OfflinePrefixCausal

Primary initial scientific class.

Native execution completes and freezes first. Candidate computation then receives only a validated `ObservationPrefixView(t)` built from information available through `t`.

It must pass:

- prefix equivalence;
- unseen-future/suffix mutation invariance;
- payload/full-trace-provenance separation;
- source/dependency authority audits;
- evaluator isolation/order invariance;
- preprocessing holdout-leakage gates.

The full source-trace digest is forbidden from candidate computation and belongs only to the outer evidence envelope.

### OnlinePrefixCausalShadow

Later engineering-validation class requiring exact offline/online payload equivalence and no-observer/native-trace equivalence. It is not the initial primary scientific evidence class.

### RetrospectiveDiagnostic

May use information realized after `t`; artifact must declare latest required time. It cannot be described as information available to the system at `t`.

### OracleDiagnostic

May intentionally consume true future schedule/realized future information as a diagnostic upper bound. It can never be promoted as the primary endogenous prefix-causal candidate.

Changing class changes candidate identity.

## 5. History-access basis

Candidate identity must distinguish at least:

- `H0CurrentNativeStateOnly`;
- `H1ReplayedPrefixHistory`;
- future separately qualified `H2NativePersistedMemory`;
- retrospective/oracle history diagnostics.

H1 means the **external observatory** uses prior events from the same immutable prefix. It does not establish that the native regulator stores or experiences that history.

A rolling window, cumulative accumulator, repeated-breach counter, or history-aware forecast in the observatory remains external history-derived unless a separately qualified native state carries an equivalent sufficient statistic.

## 6. Factor-coordinate contract

Every candidate coordinate binds at least:

- relation basis;
- weighting basis;
- cross-channel aggregation basis;
- temporal aggregation;
- forecast policy;
- execution/information class;
- history-access basis;
- availability rule;
- numeric-contract version.

Invalid coordinates fail manifest validation.

Examples:

- R3 requires overlapping forecast support;
- T8 first-breach latency requires a prospective trajectory;
- W3 confidence cannot silently become normative burden;
- A0 full vector cannot carry a scalar denominator;
- OracleDiagnostic cannot be confirmatory-primary;
- H2 is unavailable until a separate native-memory lineage exists.

## 7. Exact formula contract

Natural-language descriptions are insufficient.

Before confirmatory use, lock:

- operand order/sign;
- temporal indices;
- relation basis;
- weighting basis;
- channel aggregation/denominator;
- temporal aggregation/support;
- forecast policy/horizon/discount;
- boundary handling;
- equality/tolerance rules;
- absent-breach handling;
- numeric representation/conversion rules.

Reference fixtures must independently reproduce expected outputs and their digest is part of candidate identity.

## 8. Preprocessing/calibration identity

Every value-changing transform is part of candidate identity under `V02_CALIBRATION_AND_PREPROCESSING.md`.

The manifest binds an exact preprocessing definition or explicit `None`.

Examples of identity-bearing transforms include:

- z-score/min-max/reference scaling;
- clipping/saturation;
- smoothing/windowing;
- baseline subtraction;
- fitted thresholds;
- unit conversions that alter numerical representation;
- imputation/missingness transforms.

Fitted parameters must come only from prospectively identified discovery/calibration or independent external-reference artifacts and must be frozen before confirmatory execution.

Confirmatory values may not refit or adapt preprocessing.

Same formula + different fitted preprocessing parameters = different candidate identity.

## 9. Evaluator state-lifecycle identity

`V02_OBSERVATORY_STATE_LIFECYCLE.md` governs implementation-state authority.

Initial primary evaluation should behave as scenario-local deterministic computation from the candidate definition plus allowed prefix.

Candidate identity/evidence binds:

- evaluator-isolation manifest;
- allowed persistent-state class;
- reset lifecycle;
- cache policy;
- concurrency/order contract.

H1 permits within-scenario prefix history; it does not permit state carried from another scenario or arm.

Cross-scenario mutable state, adaptive cohort statistics, candidate-order dependence, or cache-key access to forbidden provenance are integrity failures.

## 10. Floating-point discipline

- retain exact v0.1 native values as inputs;
- use deterministic `f64` derived accumulation unless prospectively justified otherwise;
- no fast-math/architecture-dependent approximate reductions in qualified evidence;
- define scientific comparison tolerances prospectively;
- distinguish exact replay equality from scientific equivalence;
- non-finite derived values fail validation rather than being clamped into validity.

If numeric representation changes a confirmatory result, preserve that as sensitivity/failure evidence.

## 11. Undefined is not zero

Different candidates require different support.

Examples:

- R1 requires two realized cut points;
- R2 requires prior forecast + current realization;
- R3 requires overlapping forecast support;
- R4 requires two finite-horizon aggregates;
- T8 requires a breach or explicit no-breach state;
- T9 requires a declared perturbation/recovery window.

Undefined candidates emit typed unavailable states. Do not substitute zero, carry-forward, or silently drop rows.

## 12. Information dependency declaration

Candidate definitions declare dependencies such as:

- current/prior native state;
- observed drive history;
- executed intervention history;
- native dynamics config;
- prefix forecast trajectory;
- prefix digest;
- preprocessing constants;
- scenario-local evaluator state derived from the same prefix;
- realized future state;
- future protocol schedule;
- full source-trace digest;
- semantic arm identity.

Qualified prefix-causal candidates reject dependencies on realized future state, future schedule, full source-trace digest, semantic arm identity, post-run exclusion outcome, or other candidate results.

## 13. Payload vs evidence envelope

`CandidatePayload` contains only allowed prefix-derived identity/result information.

`CandidateEvidenceEnvelope` binds the payload to full execution provenance, study/evidence root, toolchain, storage artifacts, and outer source-trace identity.

For identical allowed prefixes with divergent unseen suffixes:

- candidate payloads must be identical;
- evidence envelopes may differ.

Suffix-sensitive identity entering the payload is a prefix-causality failure.

## 14. Candidate equivalence and non-equivalence

Automatically new identities include changes to:

- factor coordinate;
- formula/sign/indices;
- horizon/discount/forecast policy;
- weighting/aggregation/temporal integration;
- history-access or information class;
- preprocessing transform or fitted parameters;
- evaluator persistent-state/cache authority;
- normalization;
- undefined/out-of-range handling;
- native semantic lineage;
- implementation changes altering reference outputs.

A source refactor may preserve identity only when all canonical fields, fixtures, preprocessing parameters, evaluator-isolation semantics, and qualified outputs remain identical.

## 15. Finite candidate-set discipline

The factor space is not an unrestricted search space.

Before exploratory execution, freeze a finite `ExploratoryCandidateSetManifest` containing every eligible candidate-definition digest.

Exploratory preprocessing variants must also be finite and prospective.

Before confirmation, freeze one primary candidate plus required baselines/sensitivity candidates under a declared selection rule.

## 16. Identifiability and parsimony

Every primary-vs-baseline superiority claim requires a registered discriminator under `V02_IDENTIFIABILITY_AND_DISCRIMINATION.md`.

Candidate fingerprints may collapse definitions into observational equivalence classes.

If a complex candidate is indistinguishable from a simpler baseline under all locked discriminators, the correct outcome is `EquivalentToBaseline` or `InsufficientDiscrimination`, not interpretive promotion.

H1 candidates claiming history information beyond H0 require matched-current-state history discriminators.

## 17. Candidate comparison states

Valid states include:

- `SupportedBeyondBaselines`;
- `EquivalentToBaseline`;
- `FailsDirectionalGate`;
- `FailsNeutralityGate`;
- `FailsPrefixCausality`;
- `FailsSensitivityRegion`;
- `WeightingAmbiguous`;
- `AggregationAmbiguous`;
- `TemporalAggregationAmbiguous`;
- `PreprocessingSensitive`;
- `CalibrationSensitive`;
- `HistoryInformationNotIdentified`;
- `NumericallyUnstable`;
- `Indeterminate`;
- `NoUniqueWinner`.

These outcomes do not authorize retrospective candidate redefinition.

## 18. Required implementation gates

Before confirmatory eligibility, mechanically require at least:

1. stable canonical candidate digest;
2. every identity-bearing field changes the digest;
3. invalid factor combinations rejected;
4. reference fixtures reproduce;
5. prefix/suffix invariance passes;
6. full source-trace provenance excluded from computation;
7. diagnostic authority classes cannot enter primary API;
8. undefined states remain typed;
9. derived values remain finite on valid domain;
10. native execution artifact is immutable;
11. semantic emotion labels absent from candidate API;
12. finite candidate-set membership frozen;
13. preprocessing parameters reproduce from their locked calibration source;
14. confirmatory holdout cannot change preprocessing parameters;
15. evaluator output is invariant to candidate/scenario order, cache warmth, batch size, and allowed concurrency;
16. incremental H1 evaluation equals from-scratch prefix computation;
17. H1-vs-H0 discrimination is present for history-information claims;
18. candidate/evidence envelope binds exact frozen design and runtime identities.

## 19. Claim boundary

This contract can establish that a regulatory observable—including its information, history, weighting, aggregation, temporal, preprocessing, and evaluator-state semantics—was fixed prospectively and computed reproducibly.

It cannot establish emotion, subjective valence, native mood, native memory, suffering, sentience, or consciousness.