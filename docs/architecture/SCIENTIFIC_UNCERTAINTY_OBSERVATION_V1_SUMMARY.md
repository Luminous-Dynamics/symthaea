# SCI-008 — Scientific Uncertainty-Bearing Observation v1 — Summary

SCI-008 defines the observation/uncertainty substrate required by SCI-007 falsifiers and later experiment design.

## Central boundary

```text
quantity / construct
    != measurement specification
    != observation
    != reported value
    != uncertainty representation
    != calibration
    != scientific authority
```

The key prohibitions are:

```text
point estimate != exact state
small uncertainty != exactness
interval != calibrated coverage
model dispersion != posterior uncertainty
nominal coverage != realized coverage
```

## Exactness

Only an explicit exactness basis may support exact-by-construction semantics.

The shared kernel must never infer exactness from small/zero error bars, display precision, or zero-width intervals.

## Heterogeneous uncertainty remains heterogeneous

SCI-008 permits explicit classes such as:

```text
ExactByConstruction
InstrumentResolution
StandardError
ConfidenceInterval
CredibleInterval
ReplicateDistribution
ConformalInterval
RevisionRange
ModelDerivedUncertainty
EnsembleDispersionProxy
BoundedUnknown
Unknown
```

No automatic conversion to one Gaussian standard deviation is implied.

## Calibration and OOD

Uncertainty representation and calibration are separate.

The architecture supports scoped states such as:

```text
CalibrationNotAssessed
CalibratedWithinDeclaredReferenceScope
EmpiricalCoverageMeasured
NoExchangeabilityGuarantee
CalibrationInvalidated
```

OOD status is an independent coordinate, not a scalar confidence penalty.

## Dependency / correlation

Observations may share instruments, calibration, sampling frames, transforms, models, batches, or other uncertainty sources.

SCI-008 therefore reuses SCI-006 dependency identities and never assumes independent error merely because two observations have different IDs.

## Missingness / censoring / replicates

The observation record can preserve:

```text
BelowDetectionLimit
Left/Right/IntervalCensored
TruncatedByDesign
Missing
InvalidMeasurement
```

as well as raw replicate identities and aggregation lineage.

Missing/censored observations cannot silently become zero or falsifier `NotTriggered`.

## Falsifier integration

SCI-007 can evaluate under an explicit uncertainty policy:

```text
interval fully violates bound -> potentially Triggered
interval straddles bound       -> Inconclusive
required uncertainty absent    -> NotEvaluable
measurement invalid            -> MeasurementInvalid
```

Point-only evaluation is allowed only when the prospectively frozen domain policy explicitly permits it.

## First implementation slice

Keep it additive and non-calculating:

```text
ScientificObservationRefV1
ObservationMeasurementClassV1
ObservationAvailabilityStateV1
UncertaintyRepresentationV1
UncertaintyScopeStateV1
```

Then adapt existing strong Economics, Matter, and Physical Agency uncertainty types without weakening them.

## Conjecture Engine direction

Real scientific sequences should eventually be able to carry value/uncertainty/method/sample/provenance/dependency identities rather than only `(x,y)` points.

The old tuple path can remain for explicitly simple mathematical/synthetic/exact uses.

## Dependency chain

```text
SCI-007 falsifier specification/outcomes
    -> SCI-008 uncertainty-bearing observation
    -> SCI-009 experiment design
```

SCI-008's purpose is not to calculate a universal uncertainty number. It is to ensure uncertainty cannot disappear before scientific reasoning consumes it.
