# HUM-OBS-001A — Humanoid Capability Observation Envelope

Status: source-design candidate
Parent issue: #4780
Authority: observation/evidence bookkeeping only; **no safety or actuation authority**

## Purpose

Create one machine-readable envelope for internal and external humanoid capability runs while preventing benchmark scores from being confused with qualification, safety, or production authority.

## Core rule

```text
benchmark result
!= qualification
!= safety
!= deployment authority
```

The envelope therefore carries raw observations and explicit failure events. It deliberately does not expose an aggregate `humanoid_score`.

## Bound identities

Every observation binds:

- run identity;
- capability domain;
- execution substrate;
- exact source/model/morphology/sensor-actuator subject identity;
- internal protocol or external benchmark identity;
- exact internal case-set or external task-set identity;
- environment identity;
- authority profile identity;
- evidence profile identity;
- run disposition;
- measurements with provenance;
- explicit failure events.

An internal protocol name/version is not enough to identify what was actually tested. Internal benchmark identities therefore require a `case_set_id` just as external benchmark identities require a `task_set_id`.

## Capability domains

V1 includes locomotion, balance/recovery, dexterous manipulation, mobile manipulation, household tasks, perception/sensing, language-conditioned tasks, long-horizon composition, HRI, physical assistance, somatic contact, power/endurance, fault recovery, and conversational/social capability.

These categories are descriptive. Passing one domain never promotes another.

## Substrate separation

V1 keeps deterministic simulation, external benchmark simulation, instrumented fixtures, HIL, controlled physical testing, non-contact human factors, and human-contact studies distinct.

A simulation result cannot be relabeled as a physical result.

## Measurement provenance

Measurements distinguish:

- measured;
- derived;
- benchmark-native;
- simulation-only.

This field is not a full epistemic lineage yet. Later observatory tranches may bind exact calibration and evidence roots.

## Failure visibility

A run marked `Failed` must carry at least one explicit failure event. Failure IDs are unique inside a run and cannot disappear behind successful aggregate metrics.

`Inconclusive` and `InfrastructureIndeterminate` remain valid dispositions and must not be rewritten as PASS.

## External benchmark import

`ExternalBenchmarkSimulation` requires an external benchmark identity containing benchmark ID, version, and exact task-set ID. Future adapters for BEHAVIOR-1K, HumanoidBench, ManiSkill, CALVIN, LIBERO, and other suites should preserve their native metric definitions rather than translating them into one universal score.

## Internal benchmark import

Internal benchmark identities contain protocol ID, protocol version, and exact case-set ID. This prevents results from different force levels, disturbance directions, environments, or case selections from collapsing into the same benchmark identity.

## Relationship to existing humanoid evidence

The existing deterministic push-recovery benchmark already demonstrates the desired discipline: explicit protocol, disturbance matrix, raw recovery metrics, and a separate certification layer. This envelope generalizes the observation grammar without changing that benchmark's semantics.

## Nonclaims

This tranche establishes no external benchmark compatibility, qualification PASS, safety compliance, human-contact authorization, medical capability, or production readiness.
