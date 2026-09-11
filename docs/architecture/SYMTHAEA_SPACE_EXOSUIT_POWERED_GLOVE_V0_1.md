# Symthaea Space Exosuit — Powered Glove v0.1

Status: **simulation research only**

This note defines the SX-018 hand-system boundary. It does not claim flight readiness, human rating, superiority over NASA/xEVAS hardware, validated glove biomechanics, validated lunar-dust coefficients, or qualification-grade fail safety.

## Design objective

The powered glove is a human-assist device, not an autonomous gripper. The software should help answer whether bounded tendon assistance can reduce hand workload and fatigue while preserving useful tactility and a mechanically backdrivable zero-power state.

The defining invariant is:

> Loss of power, electronics, trusted sensing, drive health, pressure-integrity margin, or the declared release path removes powered hand authority rather than increasing grip.

A blocked passive-release path is `ServiceRequired`, not an excuse to keep powered force enabled.

## Mode model

- `Transparent`: passive glove load only.
- `FineManipulation`: deliberately low assist authority to preserve fine-control/tactility margin.
- `GripAssist`: moderate bounded tendon assistance.
- `HoldAssist`: higher bounded assistance for sustained gripping.
- `ExerciseResistance`: powered resistance for hand loading/training; it is subject to the same power and safety authority as assistance.

Exercise resistance is not allowed to survive a powered-authority failure.

## Modeled couplings

The digital twin currently composes:

1. pressure differential -> closing resistance;
2. passive garment/tendon breakaway load;
3. requested contact force;
4. per-digit powered assistance or exercise resistance;
5. per-digit drive health and sensor confidence;
6. glove-wide electrical power ceiling;
7. estimated wearer force and fatigue;
8. pressure/abrasion/sensing/mode effects on a tactility proxy;
9. pressure-integrity and release-path gating.

The tactility and fatigue models are trade-study proxies, not physiological validation models.

## Benchmark contract

Every declared glove benchmark records:

- protocol identifier;
- wearer-force impulse;
- peak wearer force;
- electrical energy;
- ending mean fatigue;
- minimum tactility proxy;
- degraded-step count;
- mandatory zero-power backdrive probe.

A candidate cannot make a meaningful "better glove" claim from reduced hand force alone. Energy, fine-control/tactility, durability, failure behavior, pressure, task protocol, and uncertainty must remain visible.

## Glove-specific dust ontology

Generic suit `Joint` and `OuterTextile` dust state is not silently reused for the hand. SX-018 defines four glove-specific subzones:

- palm;
- finger joints;
- tendon paths;
- wrist/pressure seal.

The submodel reports retained dust, abrasion health, sensor-confidence multiplier, drive-health multiplier, wrist-seal risk, and explicit EDS electrical demand. It may derate a powered-glove trade-study state but never overwrite a more severe independent fault.

Reference EDS placement and all degradation coefficients are simulation assumptions. They are specifically designed to be replaced by glove-specific regolith, abrasion, bend-cycle, seal, thermal-vacuum, force-sensing, and human-in-loop measurements.

## External research motivation

NASA human-systems standards require suited mobility, dexterity, and tactility within acceptable workload/fatigue and injury limits. NASA pressure-garment testing also treats glove pressure, hand strength, fatigue, and dexterity as coupled quantities. NASA technology work on damage-tolerant glove palms, self-diagnosing/self-healing pressure-garment materials, high-performance EVA gloves, and earlier robotic grip-assist concepts motivates the benchmark dimensions here.

Those programs are references for questions and test structure, not evidence that the SX-018 implementation matches their hardware.

## Evidence ladder

Simulation results may justify better experiments, but may not promote themselves to qualification evidence.

Required progression:

`simulation -> benchtop tendon/glove rig -> pressurized glovebox -> abrasion/regolith/TVAC cycling -> instrumented human-in-loop protocol -> integrated suit testing -> program qualification`

At each transition, simulation coefficients should be replaced or bounded by the strongest available measured evidence while preserving provenance.

## Next coupling

The next high-value hand work after software verification is:

- connector/tool/fine-pinch task libraries;
- per-finger contact-force and joint-position sensing;
- manual emergency tendon disconnect/backdrive verification;
- puncture/pressure-loss detection localized to glove segments;
- thermal/cold-hand model;
- tool and rover/habitat interface ergonomics;
- glove maintenance/replacement-unit accounting;
- hardware-in-the-loop benchmark adapter.
