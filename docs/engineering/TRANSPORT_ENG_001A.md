# TRANSPORT-ENG-001A — Multimodal transportation owner-coverage matrix

Parent: #6244  
Source issue: #6245  
Frozen base: `fbd7a754ea8389ca93f7680d93ed8b48553e6376`  
Schema: `transport-eng-001a-owner-matrix-v1`  
Canonical compact JSON SHA-256: `9b9efe660a2c85a5b799a9591d401c0a89526bfbe3eb82295ee9acf3fe94b9fb`

## Purpose

Freeze the first deterministic audit of transportation engineering ownership across materially different modes.

This artifact answers only:

- which modes already have dedicated engineering/runtime owners;
- which modes appear supportable by shared owners plus a bounded profile;
- which modes have a demonstrated missing domain-specific engineering theorem;
- which modes remain unresolved and therefore must not receive speculative roots.

It does **not** establish any vehicle capability, safety, certification, service capacity, economics, or physical-operation authority.

## Governing rule

```text
generic transport semantics
!= mode-specific physics
```

and:

```text
manufacturing owner exists
!= mode-specific dynamics/qualification owner exists

mode appears in a trade study
!= reusable engineering owner exists

provider/autopilot integration
!= complete vehicle engineering

simulation evidence
!= physical qualification
!= external regulatory approval
```

## Dispositions

- `ExistingDedicatedOwner`
- `ExistingSharedOwnersSufficient`
- `SharedOwnersNeedModeProfile`
- `DedicatedDomainRootLikelyMissing`
- `AuditUnresolved`

These are architecture dispositions, not maturity scores.

## Launch rule

A dedicated transport-domain root is justified only when the audit identifies a missing physics or qualification theorem that cannot be cleanly represented by existing owners plus a bounded profile.

## Audit result

| ID | Family | Disposition | Primary owner refs | Important open theorem |
|---|---|---|---|---|
| T01 | WarehouseMaterialHandlingAutonomousLogistics | `ExistingDedicatedOwner` | #1274, #1288, #6033, #6039, #5985 | physical qualification remains profile-specific; fleet/dispatch stays Mycelix |
| T02 | LowSpeedTerrestrialMobileRobotics | `ExistingDedicatedOwner` | #1274, #1288, #6033, #6039, #2022 | hardware-specific qualification; external public-space approval |
| T03 | RoadOffRoadWheeledVehicles | `DedicatedDomainRootLikelyMissing` | #5985, #1274, #1288, #1425, #2022 | tire-road/contact dynamics; braking/handling/stability; road-environment profile; crashworthiness/roadworthiness evidence mapping |
| T04 | TrackedHeavyTerrestrialVehicles | `SharedOwnersNeedModeProfile` | #6033, #1425, #5985, #2022 | soil-track interaction; load stability; heavy-duty braking/thermal profile |
| T05 | RailTramMetroFixedGuideway | `DedicatedDomainRootLikelyMissing` | #1425, #2022, #5985 | wheel-rail/guideway interaction; train braking/adhesion; signaling/separation interface; track/power collection qualification |
| T06 | CableRopewayConveyorElevatorFixedCorridor | `SharedOwnersNeedModeProfile` | #6039, #5760, #6032, #2022 | rope/cable traction and tension profile; station/terminal integration; evacuation/rescue evidence profile |
| T07 | RotorcraftHelicopter | `ExistingDedicatedOwner` | crate:symthaea-helicopter, #1274, #1288, #2022 | external airworthiness; hardware/HIL evidence depends exact profile |
| T08 | MultirotorEVTOL | `ExistingDedicatedOwner` | #1277, #1286, #1291, #1274, #1288 | eVTOL passenger/airworthiness profile; hardware-specific propulsion/structure qualification |
| T09 | FixedWingAircraft | `DedicatedDomainRootLikelyMissing` | #1274, #1288, #1291, #2022 | fixed-wing aerodynamics; flight dynamics; aeroelastic/loads integration; performance envelope/stall; airworthiness evidence mapping |
| T10 | MarineUnderwaterAUV | `ExistingDedicatedOwner` | #1259, crate:symthaea-maritime-core, crate:symthaea-auv | exact hydrodynamic/pressure-depth qualification by profile; physical sea-trial evidence |
| T11 | MarineSurfaceUSVVessel | `SharedOwnersNeedModeProfile` | #1259, crate:symthaea-maritime-core | USV adapter not yet equivalent to broad vessel engineering; hydrostatics/seakeeping for larger vessels unresolved |
| T12 | OrbitalSpacecraftCislunarTransport | `ExistingDedicatedOwner` | #1425, #1542, crate:symthaea-orbital | exact mission/vehicle-specific qualification; external launch/spaceflight authority |
| T13 | LanderDescentAscentVehicles | `SharedOwnersNeedModeProfile` | #1425, #1542, crate:symthaea-orbital | descent/ascent propulsion-coupled landing dynamics; surface interaction; abort/fallback profile |
| T14 | PlanetarySurfaceMobility | `SharedOwnersNeedModeProfile` | #1425, #2022, #6033 | terrain contact/slip/sinkage qualification; dust/thermal/localization degradation |
| T15 | IntermodalTerminalsDepotsDocksHubs | `ExistingSharedOwnersSufficient` | #6039, #5949, #6032, MYC:#3395 | mode-specific transfer fixtures; external facility/port/terminal approvals |
| T16 | LaunchAscentVehicles | `DedicatedDomainRootLikelyMissing` | crate:symthaea-orbital, #6173, #5681, #5074 | staged ascent dynamics; aerothermal trajectory coupling; propulsion/vehicle integration; range/flight-safety evidence mapping |
| T17 | LighterThanAir | `AuditUnresolved` | #1274, #2022, #6173 | buoyancy/aerostatics; envelope/structural coupling; wind/mooring operations |
| T18 | HovercraftAmphibious | `AuditUnresolved` | #5985, #1259, #6173 | cross-medium cushion/hydrodynamic/contact transition semantics |
| T19 | MaglevNovelGuidedHighSpeed | `AuditUnresolved` | #2022, #6173, #5670 | magnetic suspension/guidance; guideway-power coupling; high-speed dynamics |

## Confirmed root gaps from this generation

The initial issue/code audit found no dedicated reusable owner for:

1. road/off-road wheeled-vehicle dynamics and qualification;
2. rail/tram/metro/fixed-guideway engineering;
3. fixed-wing aircraft engineering;
4. launch/ascent vehicle engineering.

Those gaps are narrow: they do **not** authorize new copies of Universal Embodiment, generic controls, structural engineering, power, materials, manufacturing, evidence, service, operations, or authority semantics.

## Explicit non-root decision — surface maritime

Surface maritime is **not** promoted to a new root in this generation.

`#1259` already defines an adapter-first sequence over `symthaea-maritime-core`, with USV as the planned surface-vessel consumer. The correct next step is to exercise that adapter and then determine whether hydrostatics/seakeeping/large-vessel qualification requires a separate owner.

## Existing strong coverage

- Universal Embodiment / provider / safety / qualification: #1274, #1288 and related children.
- Helicopter: `crates/domains/symthaea-helicopter`, including explicit qualification/evidence artifacts.
- Multirotor / interactive simulation / flight-controller edge: #1277, #1286, #1291.
- Maritime substrate and AUV: #1259, `symthaea-maritime-core`, `symthaea-auv`.
- Orbital/cislunar: `symthaea-orbital`, #1425, #1542.
- Planetary mobility/heavy equipment/logistics: #1425, #6033.
- Generic kinodynamic reachability: #2022.
- Vehicle/material-handling manufacturing: #5985.
- Workcell/fleet composition: #6039.

## Search/evidence caution

The audit records that no dedicated open issue or indexed code owner was found for the four confirmed gaps. Search absence is an **audit observation**, not mathematical proof of repository-wide nonexistence. Any later discovery of a canonical owner should supersede the relevant gap row rather than create duplication.

## Claim ceiling

A PASS over this source can establish only deterministic representation of the declared owner-coverage audit and its bounded dispositions. It cannot establish that an existing owner is physically qualified, that a missing root will succeed, or that any transport system is safe, legal, certifiable, deployable, economically superior, or authorized to operate.
