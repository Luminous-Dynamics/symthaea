# Scientific Uncertainty-Bearing Observation v1

**Status:** architecture contract only; non-authorizing; non-qualifying.

**Parent:** SCI-007 Scientific Claim / Falsifier Contract v1.

## 1. Purpose

SCI-007 makes falsifier evaluation depend on exact observations/evidence and an explicit uncertainty-handling policy. SCI-008 defines the missing generic observation/uncertainty boundary needed to make that honest.

Symthaea already has strong domain precedents:

- Economic Science separates `reported point != exact realized state` and allows exactness only through an explicit `ExactByConstruction` basis;
- Matter distinguishes tree-ensemble dispersion proxies from calibrated predictive uncertainty and can explicitly label blind structural coverage `NoExchangeabilityGuarantee`;
- Physical Agency already treats threshold-straddling intervals as indeterminate rather than silently successful;
- multiple evidence stacks keep measurement operationalization, sampling, provenance, execution, and scientific authority separate.

SCI-008 extracts only the common mechanics. It does not invent one universal statistical uncertainty model.

---

## 2. Core theorem

The shared kernel must preserve:

```text
scientific quantity / construct
    != measurement specification
    != observation event
    != reported value
    != uncertainty representation
    != uncertainty-method validity
    != calibration
    != sampling representativeness
    != scientific disposition
    != truth
```

and:

```text
point estimate != exact state
small uncertainty != exactness
zero-looking uncertainty != exactness
interval != calibrated coverage
model dispersion != posterior uncertainty
nominal coverage != realized coverage
precision != accuracy
accuracy != source trust
uncertainty != scientific authority
```

No universal `confidence: f64` is introduced.

---

## 3. Observation envelope vs domain value semantics

SCI-008 should not force every domain into one numeric representation.

The common envelope can bind:

```text
ScientificObservationV1 {
    observation_id
    target_quantity_or_construct_id
    measurement_specification_id
    value_payload_identity
    value_schema_profile_id
    unit_or_domain_semantics_id
    observation_time_semantics
    sample / subject / case identity
    provenance references
    execution / instrument references
    uncertainty_representation
    quality / applicability state
    limitations
}
```

The domain owns the exact value vocabulary:

```text
scalar
vector
tensor
categorical state
count
interval-censored value
ordinal value
field / image / spectrum
symbolic value
probability distribution
```

The common kernel owns the binding mechanics, not the scientific meaning of those payloads.

---

## 4. Measurement class is not uncertainty class

The measurement relation should remain visible independently of uncertainty.

Illustrative classes include:

```text
DirectMeasurement
DerivedMeasurement
ProxyMeasurement
LatentEstimate
ModelDerivedEstimate
SimulationOutput
ExactConstructedValue
```

These are not automatically an evidence ranking.

A direct measurement may have large uncertainty or poor calibration. A derived measurement may be very precise. A latent estimate may be scientifically appropriate for one use and inadmissible for another.

Thus:

```text
measurement class != scientific authority
```

---

## 5. Exactness

Exactness must be explicit rather than inferred from numerical width.

A generic exact observation should require an exactness basis such as:

```text
ExactByConstruction {
    basis_identity
    construction / counting semantics
    implementation or source identity where relevant
}
```

Examples may include:

- exact ledger/accounting atoms under a validated arithmetic contract;
- exact count of retained records under a frozen dataset manifest;
- exact symbolic value derived by definition;
- exact enumerated state where the domain can genuinely justify exactness.

The shared kernel must never implement:

```text
abs(error) < epsilon -> exact
standard_error == 0 -> exact
interval_width == 0 -> exact
```

A zero-width uncertainty representation may be invalid or may reflect a degenerate model/reporting convention. It does not itself mint exactness.

---

## 6. Uncertainty representation taxonomy

SCI-008 should preserve the declared uncertainty *kind* and its method identity rather than coercing all forms into one Gaussian standard deviation.

Illustrative classes:

```text
ExactByConstruction
InstrumentResolution
StandardError
ConfidenceInterval
CredibleInterval
ReplicateDistribution
EmpiricalQuantileInterval
ConformalInterval
RevisionRange
SamplingUncertainty
MeasurementProcessUncertainty
ModelDerivedUncertainty
EnsembleDispersionProxy
BoundedUnknown
Unknown
```

These classes are intentionally heterogeneous.

The shared kernel must not assume automatic conversion among them.

---

## 7. Interval semantics

An interval should retain enough information to know what it means.

Illustrative fields:

```text
lower bound
upper bound
bound inclusion semantics
coverage / credibility level if applicable
method identity
uncertainty source
reference population / predictive target
one-sided vs two-sided semantics
calibration scope
OOD / applicability state
```

The interval type must not claim more than the method establishes.

Examples:

```text
95% confidence interval != 95% probability that fixed parameter lies inside
95% credible interval != frequentist coverage guarantee
conformal target coverage != guaranteed OOD structural-frontier coverage
revision range != probability interval
instrument resolution != sampling error
```

---

## 8. Calibration is separate

An uncertainty object can exist without qualified calibration evidence.

SCI-008 should preserve:

```text
uncertainty representation
    != calibration procedure
    != calibration execution
    != calibration evidence
    != in-scope coverage assessment
    != OOD coverage guarantee
```

Illustrative calibration states may include:

```text
CalibrationNotAssessed
DeclaredMethodOnly
CalibratedWithinDeclaredReferenceScope
EmpiricalCoverageMeasured
CoverageTargetMetWithinDeclaredScope
NoExchangeabilityGuarantee
CalibrationInvalidated
```

The exact vocabulary should remain profile/domain-owned, but the separation is generic.

Matter's explicit `NoExchangeabilityGuarantee` is a strong conformance precedent.

---

## 9. OOD and applicability

Uncertainty must retain whether the evaluated point lies within the scope where its uncertainty semantics were established.

Possible state:

```text
InDeclaredDomain
OutOfDeclaredDomain
DomainUnknown
BoundaryCase
```

An OOD marker is not itself a numerical penalty or confidence deduction.

It is a separate epistemic coordinate that downstream policy may use to refuse, attenuate, or require additional evidence.

No generic rule such as:

```text
OOD -> confidence -= 0.2
```

belongs in SCI-008.

---

## 10. Dependency and correlated uncertainty

Scientific observations frequently share error sources.

Examples:

```text
same instrument calibration
same survey frame
same batch effect
same normalization transform
same external reference dataset
same model fit
same random-effect estimate
same weather station / sensor drift
same annotator pool
```

Therefore:

```text
multiple uncertain observations != independent errors
```

SCI-008 should permit uncertainty/evidence dependencies to reference SCI-006 dependency identities.

Where a domain has a qualified covariance/correlation model, the observation may bind an exact covariance artifact/model identity.

The shared kernel must not assume zero covariance merely because observations have different IDs.

---

## 11. Replicates

Replicate measurements should retain replicate identity and topology rather than immediately collapsing to mean ± standard error.

A future replicate-aware observation may bind:

```text
replicate observation ids
aggregation method identity
within-replicate / between-replicate structure
exclusion policy
missing replicate policy
```

The raw replicate distribution remains scientifically useful even when an aggregate is computed.

Thus:

```text
aggregate summary != original replicate evidence
```

---

## 12. Censoring, truncation, and detection limits

Measurement status must remain distinct from ordinary uncertainty.

Illustrative observation states:

```text
ObservedValue
BelowDetectionLimit
AboveQuantificationLimit
LeftCensored
RightCensored
IntervalCensored
TruncatedByDesign
Missing
InvalidMeasurement
```

A censored value should not be silently replaced by the detection limit and then treated as an exact observation unless a domain-specific declared method explicitly performs that transformation and retains the transformation lineage.

---

## 13. Missingness

Missingness should be evidence-bearing rather than erased.

An observation/evaluation pipeline should be able to distinguish reasons such as:

```text
NotCollected
InstrumentFailure
UnavailableSource
ProtocolExclusion
ParticipantDropout
BelowDetection
DataCorruption
Redacted / inaccessible
Unknown
```

SCI-004 owns prospective missing-data policy. SCI-008 owns the actual observation-side missingness state.

A missing value is not automatically zero, normal, negative evidence, or evidence that a falsifier did not trigger.

---

## 14. Value precision and identity

Scientific identity should not depend on incidental display formatting.

SCI-002 remains the identity substrate.

A domain may use exact representations such as:

```text
integer atoms
fixed decimal
rational
IEEE-754 bit identity
canonical tensor bytes
canonical symbolic form
```

but SCI-008 must not globally require one numeric encoding.

Required theorem:

```text
same printed decimal != necessarily same exact numeric artifact
```

and:

```text
same exact bits != necessarily same scientific meaning without unit/schema/context
```

---

## 15. Transformation lineage

Any uncertainty-affecting transform should remain explicit lineage.

Examples:

```text
background subtraction
calibration transform
normalization
log transform
imputation
resampling
smoothing
feature extraction
derivative estimation
model-based correction
unit conversion
```

A transformed value receives a new artifact/semantic identity or an explicit transformation receipt. Its uncertainty may also change.

The system must not simply copy the source uncertainty field forward unless the transformation contract justifies that propagation.

---

## 16. Propagated uncertainty

Generic uncertainty propagation is not one formula.

Different cases may require:

```text
exact interval arithmetic
linear covariance propagation
Monte Carlo propagation
bootstrap
posterior transformation
symbolic error propagation
domain-specific bound calculus
unknown / not established
```

The propagation method must have its own identity and dependencies.

No common kernel shortcut should silently apply root-sum-of-squares or Gaussian assumptions.

---

## 17. Observation vs model prediction

Predictions and observations may share value schemas but remain different scientific object classes.

```text
model prediction != observed measurement
simulation output != physical observation
posterior estimate != direct observation
```

SCI-009 may compare predicted-observation distributions against future observations, but those objects must retain different lineage and authority.

---

## 18. Falsifier integration

SCI-007 falsifier evaluation should consume uncertainty-bearing evidence under an exact handling policy.

Examples:

```text
interval fully violates claimed bound -> maybe Triggered under declared rule
interval straddles boundary          -> Inconclusive
required interval absent            -> NotEvaluable
measurement invalid                 -> MeasurementInvalid
OOD uncertainty scope               -> domain policy decides applicability
```

The falsifier evaluator must not discard uncertainty and test only the reported point unless the preregistered policy explicitly permits point-only evidence.

---

## 19. Causal inference boundary

Observation uncertainty is not causal identification uncertainty by itself.

A causal estimate may carry uncertainty, but the uncertainty interval does not establish:

```text
exchangeability
instrument validity
parallel trends
no interference
positivity
correct identification strategy
```

These remain causal/assumption evidence coordinates.

Thus:

```text
precise causal estimate != identified causal effect
```

---

## 20. Source/provenance boundary

A perfectly calibrated uncertainty model says nothing by itself about whether the source is authentic, correctly attributed, or untampered.

Likewise a cryptographically verified artifact does not establish the validity of its uncertainty method.

SCI-002/SCI-003/provenance layers remain independent.

---

## 21. Suggested first Rust implementation

Start with an additive non-authorizing envelope:

```text
ScientificObservationRefV1
ObservationMeasurementClassV1
ObservationAvailabilityStateV1
UncertaintyRepresentationV1
UncertaintyScopeStateV1
```

The first tranche should **not** attempt universal numerical calculations.

Then add narrow adapters for existing strong domain types, for example:

- Economic `MeasuredValue` exact-vs-estimated semantics;
- Matter model dispersion / conformal interval semantics;
- Physical Agency interval-bound outcome evidence.

Adapters must preserve stronger local semantics and cannot transfer qualification into SCI-008.

---

## 22. Conjecture Engine migration direction

The Conjecture Engine currently often consumes simple numeric sequence pairs.

A future additive path could evolve toward:

```text
ObservedSequencePointV2 {
    x / condition identity
    observed value artifact
    uncertainty representation
    measurement method identity
    sample identity
    provenance/dependency references
}
```

The existing `(x, y)` API can remain as an explicitly simple/exact adapter only where the caller declares the values exact-by-construction or intentionally uncertainty-free for synthetic/mathematical benchmarks.

Real scientific data should not be silently forced through that adapter.

---

## 23. Interaction with SCI-009 experiment design

Expected information gain requires predicted observation distributions and measurement noise/uncertainty, not merely point predictions.

SCI-008 therefore precedes SCI-009.

A future planner may need:

```text
p(y | H, E)
measurement model
observation noise
missingness / censoring model
cost / duration / risk
```

A point-prediction variance baseline may remain useful but must not be called full Bayesian information gain.

---

## 24. Deliberate non-claims

SCI-008 does not:

- define one universal value representation;
- declare all uncertainty probabilistic;
- establish statistical validity of reported intervals;
- establish calibration automatically;
- infer representativeness;
- assume independence/covariance zero;
- define universal uncertainty propagation;
- prove causal identification;
- verify source provenance;
- decide falsifier outcomes;
- assign scientific truth/confidence;
- grant recommendation, safety, governance, or action authority.

---

## 25. Review boundary

Review SCI-008 only on:

> Does this contract preserve enough value, measurement, uncertainty-kind, calibration, applicability/OOD, dependency, replicate, censoring, missingness, and transformation information to stop point estimates from masquerading as exact truth—without forcing heterogeneous scientific uncertainty into one Gaussian/confidence scalar?
