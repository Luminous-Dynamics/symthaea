# Atelier Lightshell — Zone Transition Contract v0.1

## Purpose

Functionally graded materials create interfaces. The transition between two zones can alter compression response, stiffness, airflow, thermal behaviour, fatigue life and repairability. This contract defines how those transitions are represented and admitted without assuming a machine can manufacture them or that a simulated design will perform physically.

## Research evidence

Schmidt, Pietsch and Kyosev (2025) studied sectioned 3D spacer fabrics with sewn seams at 18, 24 and 43 mm spacing. The added seams changed cushion height and reduced compression-force absorption as seam count increased in their tested samples. They also demonstrate a specialized 3D blind-stitch technique that sews spacer fabrics without crushing the spacer layer. These are machine/material-specific findings, not universal seam-spacing targets. https://doi.org/10.1177/00405175241302802

Krier, Orlik and Pietsch (2025) combine experimental observations and yarn-level modelling for local spacer-fabric compression around seams. The paper states that the model applies in the linear-compression region before buckling and large bending. It must not be silently extrapolated to articulated joint movement. https://doi.org/10.2478/ama-2025-0009

Christidi, Louter and Popescu (2026) demonstrate functionally graded knitted textiles by mapping optimized force-density distributions to knit architectures and fabricating a physical prototype. The work highlights the need to refine the mapping between target and realized textile mechanics. https://doi.org/10.1007/s44223-025-00111-5

Stitch-programmed mechanics research shows that knitted material response can be changed by local stitch topology, rather than only changing yarn chemistry. This supports graded knit as a hypothesis to test, not an assumed manufacturing capability. https://doi.org/10.1038/s41467-024-46498-z

## Transition portfolio

| Strategy | Main opportunity | Required evidence family |
| --- | --- | --- |
| Graded knit | Continuous change in loop geometry/density | Local strain, stiffness jump, compression continuity |
| Intarsia | Integrated transition between knit structures/yarns | Strain concentration, stiffness jump, delamination |
| Spacer-safe sewn seam | Retrofit and repair; can preserve spacer thickness with suitable technique | Seam strength/fatigue, compression continuity |
| Bonded laminate | Smooth barrier or reinforcement transition | Peel strength, delamination, flex fatigue |
| Printed reinforcement | Localized compliant reinforcement | Peel, delamination, added mass |
| Replaceable overlap | Independent sensor or service cartridge | Retention cycles, repair time, installed mass |
| Spacer stitch | Local connection strategy in 3D spacer construction | Compression continuity, stiffness jump, strain concentration |

Every method is capability-dependent. Verify machine gauge, feeder/carrier control, needle selection, elastomer handling, sewing access, adhesive process, print adhesion and repair route. A process strategy declared in a document is not evidence the Atelier equipment can execute it.

## Contract semantics

A transition has two distinct zone endpoints, a strategy, process state, transition width, interface detail and process route. It has a strategy-specific minimum metric set. Proposed transition requirements may keep numeric limits and method references pending. `capability_checked` requires qualified machine/program evidence and the exact interface/process route. `frozen` additionally requires both endpoint construction revisions, numeric bounds, methods, acceptance rules and all transitions frozen.

The validator checks known zone/role/metric identities, strategy-specific metric completeness, canonical units and comparison directions, bounds, process scope and fail-closed evidence policy. It does not calculate strain fields, stiffness, compression response, manufacturability or physical performance.

## Physical experiment design

After machine capability review, create matched transitions joining identical parent-zone constructions and material lots. Compare graded knit, intarsia where supported, spacer-safe sewn seam, bonded interface, printed reinforcement and replaceable overlap. Include straight, curved and bent fixtures. Capture:

- full-field local strain (DIC where available) and strain concentration around the transition;
- local compression map, stiffness jump and compression continuity;
- seam/peel/retention strength and cycles to defined endpoint;
- flex-induced delamination, yarn pull-out, fray and migration;
- airflow and thermal discontinuity across adjacent zones;
- cleaning/aging retention and repair/replacement time;
- mass and bulk penalty.

Record specimen identity, process revision, conditioning, fixture, calibration, raw traces and uncertainty using the specimen-level coupon evidence contract (#12751).

## Benchmark objectives

Compare exact transition strategies against a matched conventional seam and a no-transition homogeneous control. Report peak interface strain concentration, stiffness discontinuity, compression transfer, seam/peel fatigue, permeability/thermal discontinuity, retention failure, mass and service/repair cost. Use hard constraints first; do not collapse the interface into one weighted score.

Targets such as 20% lower peak strain or 2x seam fatigue are not set until the machine, materials, baseline and measurement system are frozen. The first target is zero false admission of unknown or unsupported manufacturing processes.

## P0 source and fixtures

The Rust validator and JSON Schema admit a synthetic five-zone portfolio with three proposed transitions: impact spacer to articulation gaiter, ventilated region to passive service textile, and sensor insert to passive textile. All manufacturing capability is unknown. Authored tests cover duplicate identities, unknown/self zone endpoints, unsupported strategies, missing strategy metrics, units, bounds, capability prerequisites, process strategy scope, construction identity and unknown fields.

## Claim boundary

No seam-free, graded-knit, bonded or printed strategy is claimed superior or manufacturable until exact equipment/process evidence and matched physical measurements exist. Coupon interface qualification is not integrated robot qualification, safety certification, or authorization to manufacture.

## References

- Schmidt, Pietsch & Kyosev (2025), [Sectioned spacer fabrics by sewing seams](https://doi.org/10.1177/00405175241302802).
- Krier, Orlik & Pietsch (2025), [Yarn-level modelling of seam effects](https://doi.org/10.2478/ama-2025-0009).
- Christidi, Louter & Popescu (2026), [Functionally graded knitted textiles](https://doi.org/10.1007/s44223-025-00111-5).
- Singal et al. (2024), [Programming mechanics in knitted materials, stitch by stitch](https://doi.org/10.1038/s41467-024-46498-z).