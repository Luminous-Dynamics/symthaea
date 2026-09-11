# Symthaea Space Exosuit — SX-019 Suitport / Rover / Habitat Integration v0.1

## Status

Simulation architecture only. This document does **not** claim flight readiness,
human rating, connector qualification, pressure-boundary qualification, or
compatibility with any specific NASA/Axiom hardware interface.

## Goal

Treat the exosuit, rover, habitat, and suitport as an interoperable service
ecosystem while preserving one non-negotiable invariant:

> External infrastructure may replenish, cool, clean, diagnose, or shelter the
> suit, but failure or absence of that infrastructure must not consume or bypass
> the suit's protected independent survival reserve.

## Architecture

```text
local service discovery / physical presence
        |
        v
identity + freshness + per-capability admission
        |
        v
deterministic suitport sequence
Approach
 -> Capture
 -> SealVerify
 -> PressureBoundaryVerify
 -> UtilityNegotiate
 -> DustDecontamination
 -> ResourceTransfer
 -> Diagnostics
 -> DonDoffReady
        |
        v
pressure-boundary permission
```

Pressure-boundary permission is produced only by the deterministic state
machine. It is not a planner/AI command.

## Service capability model

Capabilities are advertised and admitted independently:

- mechanical capture;
- pressure-boundary interface;
- electrical charge;
- data/diagnostics;
- oxygen replenish;
- coolant/thermal service;
- PLSS regeneration;
- dust decontamination;
- tool/spare inventory;
- emergency shelter.

A node that can charge the suit is not assumed to provide oxygen. A node that
provides data is not assumed to be safe to open a pressure boundary against.

## Transactional resource transfer

Transfers use independent source- and suit-side measurements. If the meters
exceed the declared disagreement tolerance, the transfer is rejected before
state mutation. The smaller agreeing value is used for conservative accounting.

All mutations occur on cloned power/PLSS twins and commit only after all checks
succeed. Interrupted transfers are represented with an explicit completion
fraction and retain partial-transfer receipts rather than being promoted to a
full service event.

Routine service transfer is inbound-only. The model has no API that exports the
protected survival reserve to a rover or habitat.

## Dust / suitport coupling

Suitport decontamination readiness comes from the SX-017 dust twin's explicit
`SuitportInterface` state. Mechanical docking does not reset contamination.
Nominal boundary-opening progression therefore requires decontamination evidence
before resource transfer and diagnostics.

## Safe-haven integration

Safe-haven recommendations use:

- independent suit endurance;
- explicit reserve margin;
- conservative route-time upper bounds;
- local route confidence;
- identity verification;
- evidence freshness;
- shelter availability;
- radiation-shelter capability.

Fresh local evidence is sufficient. Cloud/network reachability is not required
for local safe-haven selection, which preserves useful operation during a comm
blackout.

## Current external reference direction

The design intentionally follows current interoperability trends rather than
inventing a Symthaea-only physical connector ecosystem:

- NASA's EVA and Human Surface Mobility Program treats spacesuits, Lunar Terrain
  Vehicles, pressurized rovers, tools, and EVA support systems as one operational
  surface-mobility architecture.
- NASA suitport work keeps the suit outside the pressurized cabin and targets
  reduced pre-EVA time, lower airlock consumables, and lower contamination
  transfer.
- NASA/Honeybee autonomous utility-connector work targets dust-tolerant reusable
  power/data/mechanical connections relevant to rovers and next-generation EVA
  suits.
- NASA's RADIAL connector work explicitly explores a common autonomous,
  dust-tolerant, genderless high-power/data/mechanical connector standard for
  lunar surface systems.

Reference URLs (research context only):

- https://www.nasa.gov/suits-and-rovers/
- https://techport.nasa.gov/projects/10728
- https://techport.nasa.gov/projects/8720
- https://techport.nasa.gov/projects/146924
- https://ntrs.nasa.gov/citations/20260000972

## Failure campaign

The first deterministic campaign includes:

- stale service identity;
- low identity confidence;
- service-node charging loss;
- capture loss;
- seal disagreement;
- pressure-sensor disagreement;
- unavailable decontamination;
- transfer-meter disagreement;
- interrupted transfer;
- unavailable safe haven;
- communication blackout with fresh local safe-haven evidence.

Containment properties include:

- service-admission failures do not imply physical authority;
- capture/seal/pressure/decon faults never produce pressure-boundary permission;
- contradictory meters roll back the resource transaction;
- interrupted service stays partial and explicitly receipted;
- inbound service cannot reduce protected survival energy;
- unavailable shelters are not invented;
- safe-haven selection can remain local/offline.

## Deferred hardware parameters

The software deliberately does not invent qualification values for:

- suitport capture geometry;
- sealing architecture;
- cabin/suit pressures;
- charging voltage/current;
- oxygen or coolant coupling pressures;
- connector pinout/protocol;
- decontamination technology/performance;
- emergency extraction hardware;
- certified meter tolerances;
- route-confidence or reserve thresholds.

All reference thresholds remain `Simulation` evidence until replaced by measured
or qualification evidence.

## Evidence ladder

```text
software simulation
 -> interface bench rig
 -> dusty connector testing
 -> pressure-boundary / suitport rig
 -> integrated rover/habitat mockup
 -> human-in-loop analog
 -> thermal-vacuum / regolith testing
 -> program qualification
```

Simulation evidence cannot satisfy the later stages merely because the software
passes its tests.
