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
