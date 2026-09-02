# Affective Emergence v0.2 — Minimal Malicious Fixture Matrix

Status: **normative design-only / blocked on Native Interoception v0.1 qualification**

This contract defines a compact suite of intentionally wrong observatory implementations and evidence artifacts. The suite exists to demonstrate that the validation architecture can detect known violations before it is trusted to classify unknown candidate behavior.

A validation suite that only passes correct implementations is incomplete evidence.

## 1. Principle

Prefer a small set of malicious fixtures with high coverage over dozens of redundant negative tests.

Each fixture must have:

- stable fixture ID;
- exact malicious behavior;
- expected failing gate(s);
- expected failure class;
- a positive control proving the corresponding valid implementation passes;
- deterministic artifact/fixture digest.

A malicious fixture that unexpectedly passes is an `IntegrityFailure` of the observatory validation system.

## 2. Initial fixture set

The initial suite should contain exactly the following malicious roles unless a new design-freeze identity supersedes this contract.

### M01 — future suffix reader

Behavior:

- candidate value at cut point `t` directly reads one or more fields from the realized suffix after `t`.

Expected detection:

- future-suffix mutation invariance;
- exact-prefix/divergent-future twin equality;
- capability/source dependency audit.

Expected status: `IntegrityFailure::FutureInformationLeak`.

Positive control: same formula implemented only over `ObservationPrefixView(t)`.

### M02 — suffix-sensitive provenance reader

Behavior:

- candidate does not read future state directly but includes the full source-trace digest or another suffix-sensitive identity in its computation/payload.

Expected detection:

- prefix payload identity differs across suffix-mutated twins;
- full-trace-provenance dependency audit rejects the candidate.

Expected status: `IntegrityFailure::SuffixSensitiveProvenanceLeak`.

Positive control: full source-trace digest appears only in the outer evidence envelope.

### M03 — semantic arm reader

Behavior:

- candidate or primary analysis inspects semantic arm/scenario identity, semantic mapping, or semantic canary content.

Expected detection:

- mapping permutation invariance;
- semantic-label canary scan;
- dependency/capability audit.

Expected status: `IntegrityFailure::SemanticLeak`.

Positive control: opaque blind codes only until explicit unblinding.

### M04 — confirmatory cohort fitter

Behavior:

- candidate uses confirmatory/holdout cases to fit z-score parameters, min/max bounds, clipping thresholds, smoothing hyperparameters, or another value-changing preprocessing parameter.

Expected detection:

- preprocessing-manifest source-cohort validation;
- add/remove/reorder unrelated holdout-case invariance;
- calibration/holdout content-overlap audit.

Expected status: `IntegrityFailure::ConfirmatoryPreprocessingLeak`.

Positive control: same transform frozen from structural constants or prospectively identified discovery/calibration data.

### M05 — cross-scenario state carrier

Behavior:

- evaluator keeps mutable running state from scenario A and uses it when evaluating scenario B.

Expected detection:

- A→B vs B→A order permutation;
- cold-process vs warm-process equality;
- isolated-single-scenario vs batch equality.

Expected status: `IntegrityFailure::CrossScenarioStateLeak`.

Positive control: each evaluation coordinate is recomputed from immutable inputs with `NoneAcrossEvaluationCoordinates` state authority.

### M06 — cache-key poisoning

Behavior:

- evaluator cache key omits an identity-bearing field such as candidate digest, prefix digest, cut point, preprocessing digest, or forecast-policy identity, causing an artifact from one coordinate to be reused for another.

Expected detection:

- cache disabled vs enabled equality;
- cache miss vs hit equality;
- candidate-order permutation;
- duplicate-evaluation removal;
- explicit cache-key contract audit.

Expected status: `IntegrityFailure::CacheIdentityViolation`.

Positive control: content-addressed cache key binds all identity-bearing inputs or cache is disabled in qualified computation.

### M07 — unavailable-as-zero implementation

Behavior:

- R1/R2/R3/latency/recovery or another temporally undefined candidate silently returns `0` rather than a typed unavailable/no-breach state.

Expected detection:

- undefined-value fixture;
- availability-state equality checks;
- reference-vector validation.

Expected status: `IntegrityFailure::InvalidUndefinedSemantics` or candidate disqualification under the locked taxonomy.

Positive control: exact typed unavailable/no-breach representation.

### M08 — off-by-one temporal aligner

Behavior:

- R2 or R3 compares mismatched absolute times, e.g. prior forecast `h` against current forecast `h` rather than prior `h+1` against current `h` for shared support.

Expected detection:

- deterministic self-consistency fixture;
- explicit temporal-shift fixture;
- overlapping-support identity check.

Expected status: `IntegrityFailure::TemporalAlignmentError`.

Positive control: shared absolute future points align exactly under self-consistent forecasts.

### M09 — rolling change masquerading as revision

Behavior:

- implementation reports R4 finite-horizon aggregate change but labels/registers it as R3 aligned future revision.

Expected detection:

- rolling-horizon boundary-only diagnostic where overlapping forecasts are unchanged but entering/dropped boundary terms differ;
- candidate-coordinate/formula compatibility validation.

Expected status: `IntegrityFailure::RelationSubstitution`.

Positive control: R3 remains neutral while R4 changes in the boundary-only fixture.

### M10 — precision/importance conflation masquerading as viability-only

Behavior:

- candidate registered as `W1ViabilityWeightOnly` actually multiplies deviation by precision×importance or otherwise lets confidence alter intrinsic viability severity.

Expected detection:

- burden-vs-precision discriminator scenarios;
- exact formula/reference fixtures;
- candidate-definition digest/source audit.

Expected status: `IntegrityFailure::WeightingSubstitution`.

Positive control: W1 changes only with viability weighting inputs while W2 reproduces the legacy combined aggregate.

### M11 — aggregation/denominator substitution

Behavior:

- candidate registered as weighted sum, peak, single channel, or fixed-set weighted mean silently uses another aggregation rule or changes the denominator/channel set adaptively.

Expected detection:

- healthy-channel dilution fixture;
- matched mean/different sum;
- fixed peak/different background;
- channel permutation;
- exact aggregation fixture.

Expected status: `IntegrityFailure::AggregationSubstitution`.

Positive control: each A0–A5 aggregation obeys its declared invariances.

### M12 — oracle masquerading as prefix-causal forecast

Behavior:

- candidate or forecast policy consumes the true future protocol schedule/realized future while declaring `OfflinePrefixCausal` authority.

Expected detection:

- capability/source dependency audit;
- future schedule redaction equivalence;
- suffix mutation;
- oracle namespace/type separation.

Expected status: `IntegrityFailure::OracleAuthorityEscalation`.

Positive control: true-future information exists only in `OracleDiagnostic` artifacts excluded from primary candidate ranking.

### M13 — native-trace mutating observer

Behavior:

- observatory changes native state, drive, interventions, transition ordering, or shared memory while measuring.

For the initial offline-replay architecture this should be structurally impossible; still retain a malicious integration fixture for later shadow/live mode.

Expected detection:

- observer/no-observer native trace equality;
- capability boundary audit;
- immutable input type/source audit.

Expected status: `IntegrityFailure::ObserverFeedback`.

Positive control: initial offline evaluation begins only after the native trace is complete/frozen.

### M14 — frozen-artifact substitution

Behavior:

- candidate value, availability state, scenario, preprocessing parameters, exclusion receipt, or blinded comparison is changed after its upstream digest is frozen while downstream evidence tries to retain the old lineage identity.

Expected detection:

- digest/root mismatch;
- artifact dependency graph verification;
- blinded-artifact recomputation/replay;
- scenario/exclusion accounting.

Expected status: `IntegrityFailure::ArtifactSubstitution`.

Positive control: complete realized evidence package reproduces every upstream digest before semantic evaluation.

## 3. Coverage map

The initial fixture suite must cover at least these contract families:

| Contract family | Required malicious fixtures |
| --- | --- |
| future/prefix information | M01, M02, M12 |
| blinding/semantics | M03 |
| preprocessing/holdout isolation | M04 |
| evaluator state/cache isolation | M05, M06 |
| undefined/numeric semantics | M07 |
| temporal relation/alignment | M08, M09 |
| weighting/aggregation semantics | M10, M11 |
| observer causal isolation | M13 |
| provenance/evidence lineage | M14 |

A future contract added to the normative registry should either map to an existing fixture or add a new malicious fixture role before design freeze.

## 4. Mutation strength

Each fixture should be strong enough that a naive implementation would plausibly pass ordinary unit tests.

Examples:

- M01 reads only one future bit/threshold rather than replacing the whole computation;
- M02 leaks only through a digest-derived branch;
- M04 changes normalization slightly using holdout statistics;
- M05 carries a small running offset across scenarios;
- M06 collides only when two coordinates share a partial key;
- M08 is a one-step shift that looks numerically plausible;
- M10 uses the frozen legacy weighting and therefore may look scientifically reasonable;
- M14 changes one value after freezing while keeping all human-readable labels unchanged.

The purpose is to test whether gates catch subtle violations, not cartoonishly broken code.

## 5. Positive-control pairing

Every malicious fixture must have one nearest valid control differing only in the violating behavior.

Validation evidence should therefore report paired states:

- malicious fixture detected;
- valid control accepted.

A gate that rejects both may be over-restrictive. A gate that accepts both is ineffective.

## 6. Gate specificity

Prefer that each fixture has one primary expected gate and a small number of secondary corroborating gates.

Do not rely on an unrelated later failure to catch a violation.

For example:

- M04 should fail the preprocessing/holdout firewall before candidate ranking;
- M05 should fail evaluator-isolation before any H1 history interpretation;
- M09 should fail relation compatibility before semantic hypothesis evaluation;
- M14 should fail dependency/root validation before unblinding.

This helps identify the first broken evidence edge during reproduction.

## 7. Deterministic malicious-fixture manifest

A future `MaliciousFixtureSuiteManifest` should bind:

- schema/version;
- design-contract-registry digest;
- ordered fixture IDs M01–M14;
- source implementation/fixture digests;
- primary/secondary expected gate IDs;
- expected failure class;
- paired valid-control fixture digest;
- exact scenario/prefix inputs;
- execution environment/toolchain identity as needed;
- canonical SHA-256.

The realized `MaliciousFixtureValidationReport` should record for every fixture:

- fixture digest;
- whether malicious case was rejected;
- first gate that rejected it;
- observed failure class;
- whether valid control passed;
- artifact/log digest;
- overall suite status.

## 8. Promotion rule

No exploratory candidate results are scientifically interpretable until all integrity-blocking malicious fixtures expected for the implemented contract surface are detected and their paired valid controls pass.

Candidate-specific scientific failures are allowed and preserved. Validation-system inability to detect a known malicious implementation is not.

## 9. Minimality rule

Do not expand the suite for every conceivable coding bug.

Add a new malicious fixture only when:

1. a new normative contract creates a materially new authority/integrity boundary not represented by M01–M14; or
2. a real defect escapes the current suite and cannot be modeled by strengthening an existing fixture.

When a real escaped defect is found, add a regression fixture and record which prior gate failed to detect it.

## 10. Claim boundary

Passing this malicious-fixture suite shows that the implemented validation architecture detects a prospectively specified set of known information, preprocessing, evaluator-state, temporal, weighting, aggregation, causal, and provenance violations.

It does not prove absence of all bugs or establish that any surviving regulatory candidate is emotion, subjective valence, feeling, mood, suffering, sentience, or consciousness.