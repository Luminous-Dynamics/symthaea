# symthaea-domain-awareness

Evidence-first physical-domain awareness primitives for Symthaea.

The crate provides a shared vocabulary for air, surface, subsurface, land, space,
and cross-domain sensing without coupling observation to physical authority.

## Core pipeline

```text
ObservationEnvelope
    -> Track
    -> IdentityHypothesis[] / BehaviorHypothesis[]
    -> EpistemicState
    -> RiskAssessment
    -> SpatialSafetyKernel / OperationalDesignDomain
    -> downstream independent safety + authority boundary
```

## Non-negotiable invariants

1. **Observation is not identity.** Sensor detections, classifier outputs, human
   reports, Remote-ID-like assertions, AIS/VDES-like assertions, and other
   cooperative identity data remain evidence.
2. **Identity is not intent.** The core crate deliberately defines no `Target`
   or `Hostile` primitive.
3. **Uncertainty is retained.** `Unknown`, `InsufficientEvidence`,
   `ConflictingEvidence`, and `OutOfDistribution` are first-class states.
4. **Degradation cannot increase confidence.** Sensor-health trust caps are
   monotonic and are enforced when health changes.
5. **Correlated evidence is not independent evidence.** Multiple derived feeds
   from the same physical source count as one physical fault domain.
6. **Freshness is explicit.** Observed time, received time, clock source,
   clock uncertainty, and maximum valid age travel with the evidence.
7. **Risk is not authority.** `RiskAssessment::review_priority` exists only to
   prioritize protective review; downstream actuation requires an independent
   safety and authorization layer.
8. **Spatial rules are time-bounded evidence.** Protected, authorized, keep-out,
   corridor, emergency, and uncertain volumes carry validity windows, authority
   provenance, and evidence references.
9. **Missing spatial evidence fails closed.** `SpatialSafetyKernel` returns
   `Restricted` or `Incomplete`; it never manufactures a safe volume.
10. **Leaving the ODD can only reduce capability.** Environmental, sensing,
    navigation, communications, operator, and model-assurance degradation cannot
    expand the operational envelope.

## Spatial safety

`spatial` provides a pure 4-D (3-D space + time) evaluator. The initial geometry
is deliberately conservative and simple (`AxisAlignedBounds`), with explicit
coordinate frames and temporal validity. It emits assessments only and produces
no actuator commands.

## Operational Design Domain

`operational_domain` defines the evidence required for an operation to remain
inside its reviewed design envelope. Current conditions include visibility,
wind, optional sea state, navigation quality, required sensor modalities,
communications, operator availability, and model assurance.

Model assurance is represented as `Aligned`, `Restricted`, `Unsafe`, or
`Incomplete`, making it straightforward to bridge the helicopter digital-twin
divergence monitor later without silently retuning or trusting a diverged model.

## Intended integrations

Future adapters can normalize ROS2 sensors, radar/EO/IR/RF/acoustic observations,
cooperative aviation and maritime identity, weather, AUV/sonar observations, and
human reports into `ObservationEnvelope` without changing the core epistemic
contract.

The existing `symthaea-hal`, geofence/safety kernels, digital-twin divergence
monitoring, formal-safety crate, and Mycelix evidence plane should remain
separate layers. Hard-real-time actuation must not depend on distributed
consensus.

## Verification

```bash
cargo test -p symthaea-domain-awareness
```
