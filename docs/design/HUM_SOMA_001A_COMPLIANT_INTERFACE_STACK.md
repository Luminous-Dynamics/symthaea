# HUM-SOMA-001A — Compliant Human-Contact Interface Stack

Status: source-design candidate
Issue: #4725
Authority: mechanical/sensing architecture only; **no motor authority and no physical qualification**

## Purpose

Represent the layered architecture of a soft human-facing robot interface before adding comfort optimization, active shape control, or physical contact authority.

The stack is ordered from the human-facing surface inward toward the robot structure.

## Design rules

A profile explicitly binds:

- one profile identity;
- one robot contact-site identity;
- dry or wet contact environment;
- an ordered layer stack;
- explicit sensor modalities required by later qualification.

There is deliberately no `Default` production profile.

## Layer vocabulary

The v1 semantic roles are:

- replaceable contact liner;
- compliant surface;
- tactile sensor skin;
- variable-compliance layer;
- fluid barrier;
- actuator-isolation layer;
- structural backing.

These are semantic roles, not material selections. A production design may use multiple physical technologies to satisfy one role or one physical laminate to implement multiple roles, but stronger claims require separate evidence.

## Human-facing boundary

The outermost layer must be a compliant surface or replaceable contact liner. The rigid structural backing must be unique and innermost.

A wet-contact profile additionally requires:

- a replaceable human-facing liner;
- a fluid barrier;
- that barrier to sit outward of protected actuator/structural layers.

This is an architectural contamination/isolation rule only. It does not prove cleanability, biocompatibility, fluid compatibility, or sterilization efficacy.

## Sensor requirements

Each profile must explicitly state which sensor modalities later qualification requires. V1 can express normal force, shear force, contact area, surface temperature, humidity, deformation, and slip.

Declaring a modality is not evidence that the installed hardware measures it accurately. Sensor identity, calibration, drift, fatigue, freshness, and provenance belong to HUM-TACT-001B and later qualification work.

## Comfort boundary

This tranche intentionally does not define a comfort score or optimizer. Future comfort control may propose changes to compliance, geometry, temperature, or motion only inside independently valid consent, physical qualification, and terminal safety constraints.

Comfort optimization must never create consent or increase physical authority.

## Research implications

Recent soft-robotics work supports intrinsic mechanical compliance and adaptive soft actuation for human interfaces. Flexible multiaxial tactile sensing is promising for normal/shear/contact regulation, but cyclic loading, viscoelastic drift, fatigue, and interfacial degradation mean calibration health must be treated as evidence rather than assumed from sensor presence.

## Follow-ons

- HUM-SOMA-001B: mechanical material/compliance evidence contract;
- HUM-SOMA-001C: distributed tactile/contact-area observation semantics;
- HUM-SOMA-001D: bounded variable-compliance field proposal;
- HUM-SOMA-001E: distributed thermal-field semantics + independent cutoff;
- HUM-SOMA-001F: wet-interface/fluid-management evidence;
- HUM-SOMA-001G: liner/cassette hygiene lifecycle;
- HUM-SOMA-001H: acoustic/vibration isolation evidence;
- HUM-SOMA-001I: non-authoritative shape-morphing proposal field;
- HUM-SOMA-001J: private somatic preference model;
- HUM-SOMA-001L: instrumented compliant phantom qualification.

## Nonclaims

This source tranche establishes no material selection, biocompatibility, medical use, hygienic efficacy, safe force/pressure/temperature value, actuator command, human-contact authority, HIL qualification, human-trial authorization, or product-safety certification.
