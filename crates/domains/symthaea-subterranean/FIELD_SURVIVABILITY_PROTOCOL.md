# Subterranean Field Survivability Protocol

**Date:** 2026-07-20  
**Scope:** Capability Campaign VIII, patches 64–76

## Purpose

This protocol defines how the subterranean platform remains truthful and bounded when sensors, actuators, power, cooling, or communications partially fail. It does not claim hardware certification, cryptographic sensor authentication, or real-world fault coverage. It defines deterministic in-crate authority and evidence contracts that external hardware and security adapters can satisfy.

## Authority order

From most authoritative to least authoritative:

1. Physical state integrity and critical redundant-sensor quorum.
2. Latched physical hazards and verified recovery planning.
3. Operator, watchdog, update, and recovery locks.
4. Actuator isolation and persistent mechanical health.
5. Power and thermal field envelope.
6. Communication-partition recovery and reconciliation hold.
7. Capability disposition and mission degradation.
8. Learned nominal control.

A lower layer may not restore authority removed by a higher layer.

## Redundant sensing

Up to three externally authenticated sensor paths may be declared. The crate independently enforces:

- bounded source IDs;
- monotonic per-source sequences;
- physical channel ranges;
- median fusion;
- source disagreement penalties;
- source isolation thresholds;
- two-source quorum on critical channels whenever redundancy is declared.

A fallback observation keeps numerical inputs bounded, but missing critical quorum remains a fail-closed sensor fault. Fallback values do not convert uncertainty into confidence.

## Actuator isolation

Long-term maintenance wear and short-horizon actuator response are distinct signals. The response supervisor compares energized commands with compatible state changes. Persistent mismatch latches isolation for only the affected actuator. Recovery plans, learned control, and field envelopes cannot command an isolated actuator.

Isolation clears only through explicit service authority. Ordinary compatible observations do not silently resurrect a latched actuator.

## Power and thermal envelope

The field envelope continuously restricts demand according to:

- battery reserve;
- cutter temperature;
- coolant health;
- mobility and cooling availability;
- maintenance state.

The envelope removes productive cutting before protected mobility. Thermal protection stops cutting and preserves cooling only when cooling hardware remains available. It never invents pump authority after physical failure.

## Capability dispositions

The composed capability profile has four states:

- `full_mission`: all productive capabilities remain available;
- `reduced_work`: safe operation remains possible, but productive work is suspended or derated;
- `return_only`: the platform retains enough mobility and sensing to withdraw;
- `hold_for_recovery`: safe autonomous motion cannot be justified.

Capability disposition can narrow mission intent but cannot override physical hazards, operator restrictions, or degraded-operation locks.

## Communication partitions

A lost mesh enters a bounded progression:

1. grace;
2. local autonomy;
3. return to mesh when feasible;
4. hold and beacon when return cannot be funded.

Reconnection does not immediately restore team authority. The platform enters `reconciling`, removes ordinary motion authority, and requires a bounded dwell of fresh, revision-consistent team state before declaring the mesh authoritative again.

## Checkpoint truth

Operational checkpoint schema v3 persists:

- redundant-source replay and reliability state;
- actuator isolation latches and health;
- power/thermal envelope state;
- partition and reconciliation state.

Restart must not forget stale sensor sequences, resurrect isolated actuators, or bypass reconciliation.

## Evidence

Every retained operational frame records:

- sensor-source count, quorum, disagreement, and reliability;
- isolated actuator count and affected capability families;
- power and thermal margins;
- field-envelope mode;
- capability disposition and work authority;
- partition duration, reconciliation dwell, revision gap, and team-authority status.

The bounded summary reports quorum-failure frames, actuator-isolation frames, derated operation, survival holds, partition operation, reconciliation, and extrema.

## Explicit non-claims

This crate does not by itself prove:

- physical independence of redundant sensors;
- sensor or peer cryptographic authenticity;
- diagnostic coverage for every actuator fault;
- certified battery or thermal models;
- safe behavior on unmodeled hardware;
- real-time performance on production computers;
- successful radio propagation through arbitrary geology.

Those claims require hardware-in-the-loop campaigns, authenticated adapters, calibrated plant models, and controlled-field evidence.
