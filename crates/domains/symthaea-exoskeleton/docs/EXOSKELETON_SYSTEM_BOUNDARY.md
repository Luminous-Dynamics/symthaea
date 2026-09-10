# Symthaea Exoskeleton System Boundary v0.1

## Purpose

This document defines the evidence and authority boundary between the physically grounded `symthaea-exoskeleton` robotics domain and fictional or speculative powered-armor mechanics implemented by downstream simulation consumers such as Symtropy.

## Core rule

`symthaea-exoskeleton` models a human-coupled powered wearable and may expose measured, simulated, estimated, or bounded engineering state such as joint kinematics, human/exoskeleton interaction torques, sensor readings, battery state, actuator limits, assistance state, safety state, and future power/thermal telemetry.

It does **not** establish that speculative personal force fields, energy shields, fictional materials, fictional weapons, or other game-world technologies are physically realizable.

A downstream simulator may compose those fictional systems around the exoskeleton representation, but their provenance and authority must remain distinct.

## Authority split

### Symthaea owns

- human-frame kinematics and dynamics represented by the exoskeleton simulators;
- actuator commands and bounded assistance behavior;
- modeled or observed sensor state and uncertainty;
- hard physical safety envelopes and fail-safe behavior;
- physically grounded power, efficiency, energy, and thermal state when modeled;
- wearer/frame calibration and fit state when modeled;
- evidence qualification for claims made by this robotics domain.

### Symtropy or another downstream world simulator owns

- fictional powered-armor archetypes;
- speculative field generators and energy shields;
- fictional armor/material behavior not backed by a qualified engineering model;
- game-world damage/effect semantics;
- inventory, manufacturing, logistics, faction availability, maintenance economy, and balancing;
- presentation, VFX, audio, HUD, and Muse interpretation.

## Composition rule

Downstream systems should treat Symthaea exoskeleton output as one component in a larger embodied-system composition:

`wearer -> human/frame coupling -> exoskeleton -> power/thermal bus -> equipment/protection -> world effects`

No downstream fictional capability may be written back as evidence that the physical exoskeleton possesses that capability.

## Safety hierarchy

Future controller work should preserve the following ordering:

1. hard actuator/joint/power/thermal/contact safety limits;
2. human consent, emergency-stop, and fail-safe state;
3. supervisory Symthaea/FEP/consciousness-derived policy;
4. adaptive assistance policy and task optimization.

Higher layers may reduce authority but must not override a lower hard safety limit.

## Game integration guidance

A Symtropy powered frame should reference or adapt qualified exoskeleton state, but should add fictional systems through explicit downstream components such as a generic protection stack or field generator. The preferred gameplay consequence chain is:

`effect -> protection -> condition -> capability`

not `shield_hp -> health_hp`.

## Evidence rule

Passing software tests for `symthaea-exoskeleton` is evidence about the tested software properties only. It is not certification of a physical medical, industrial, military, or protective device, and it is not evidence for speculative field technologies.

Any future engineering claim must name the exact model, configuration, evidence lineage, assumptions, and qualification status that support it.
