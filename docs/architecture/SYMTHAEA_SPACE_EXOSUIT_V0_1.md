# Symthaea Space Exosuit v0.1

Status: research architecture / simulation requirements. Not flight hardware, not human-rated, and not evidence of superiority over AxEMU or any other EVA suit.

## Objective

Develop a wearable-spacecraft architecture in which a conventional pressure/life-support survival layer remains independent of a lightweight powered-assist frame. Symthaea may predict intent, workload, hazards, maintenance needs, and useful assistance, but a deterministic local safety kernel constrains powered motion and the wearer remains the ultimate local authority.

The program earns claims by comparing against a declared baseline under the same protocol. Simulation guides design; it does not establish human-rated superiority.

## Authority boundary

```text
wearer intent / task plan
        |
        v
Symthaea proposal
        |
        v
CertifiedAssistEnvelope   <-- deterministic, no Phi requirement
        |
        v
existing MotorSafetyLevel / SafeFallback / local controller
        |
        v
powered frame

pressure garment / primary life support
        |
        +---- remains survivable independently of Symthaea assistance
```

Hard rules:

1. Pressure integrity, oxygen delivery, CO2 removal, and primary thermal survival are outside powered-assist authority.
2. Loss of Symthaea must not make the pressure garment unsafe.
3. Loss of powered assistance must leave a manually survivable/backdrivable state by design; this must eventually be demonstrated on hardware, not inferred from simulation.
4. Phi/consciousness metrics may be experimental advisory inputs to assistance generation, but are not the certified safety boundary for human-rated EVA.
5. Non-finite or malformed safety state fails closed for powered assistance.
6. Capability is not authorization, and simulation evidence is not qualification evidence.

## Benchmark gates

A future integrated `better than baseline` claim requires all gates below to pass against a named comparison suit/system under the same declared protocol and with qualification-grade evidence.

| Gate | Primary question | Example metric family |
|---|---|---|
| Mobility/workload | Does powered assistance reduce the cost of useful locomotion? | metabolic energy per standardized traverse; task time at matched workload |
| Dexterity | Can the wearer perform representative EVA manipulation better? | completion time/errors/force on tool and connector tasks |
| Fail-safe mobility | Can the wearer remain safe and mobile after assist faults? | fault-to-transparent latency; manual task completion after injected failures |
| Balance recovery | Does assistance reduce falls/stumbles without destabilizing the wearer? | perturbation recovery rate; peak corrective torque; false interventions |
| Personalization | Can assistance adapt across body geometries without violating fixed limits? | workload/dexterity improvement across declared participant range |
| Fault diagnostics | Are faults detected early and correctly? | detection latency, missed faults, false alarms |
| Serviceability | Can failed modules be replaced quickly and correctly? | mean replacement time, tool count, verification success |
| Dust robustness | Does mobility/performance remain acceptable after abrasive exposure? | torque/friction/seal-performance drift over exposure cycle |
| Thermal management | Can the system anticipate and reduce thermal excursions? | prediction error; time outside preferred envelope; energy consumed |
| Integrated EVA | Does the whole system improve useful work without hiding regressions? | standardized EVA task suite with workload, safety, energy, and completion metrics |

The benchmark API uses uncertainty-adjusted comparisons. A candidate only passes a required improvement margin when its adverse uncertainty bound still beats the baseline's favorable bound.

## Evidence ladder

```text
Simulation
   -> HardwareBench
   -> HumanInLoop
   -> Qualification
```

No automatic promotion is implied. Every transition needs a declared protocol, traceable configuration, and independent evidence appropriate to the next claim.

## v0.1 implementation

`crates/domains/symthaea-exoskeleton/src/space_exosuit.rs` provides:

- `SpaceEnvironmentState`: explicit gravity/vacuum/temperature/dust/radiation simulation inputs;
- `SuitSafetyState`: observation-only suit/physiology inputs exposed to powered assist;
- `CertifiedAssistEnvelope`: deterministic assist limits carrying an evidence level;
- `SpaceExosuitSafetyKernel`: clamps valid requests and removes powered assist on unsafe/malformed state;
- `BenchmarkMeasurement`: same-protocol candidate/baseline comparison with uncertainty;
- `superiority_claim_readiness`: refuses an integrated claim until every mandatory gate has qualification-level passing evidence.

The provided numerical envelope is marked `Simulation` and exists only for software tests. It is not a NASA/Axiom requirement set and must never be described as human-rating data.

## Next tranches

### SX-002 — reduced-gravity full-frame dynamics

Make the 20-DOF Symtropy full-frame simulator accept an explicit environment/gravity model. Preserve Earth regression behavior while adding lunar-gravity scenarios. Do not assume Earth gait torques transfer unchanged to the Moon.

### SX-003 — pressure-joint resistance and workload

Add an evidence-tagged pressure-garment joint-resistance model and metabolic/work estimator. Parameter uncertainty must remain visible. Gate 1 becomes executable only when the baseline and candidate use the same validated protocol/model.

### SX-004 — glove/tendon assist

Add hand/finger intent sensing and bounded assist as a separate capability from the full-frame motors. Optimize for reduced fatigue and preserved manual usability after power loss.

### SX-005 — balance/fall benchmark

Use lunar-gravity terrain perturbations and backpack/suit mass distribution to evaluate recovery assistance. False-positive interventions are first-class failures.

### SX-006 — suit digital twin

Track module identity, dust/thermal/radiation exposure, actuator efficiency, sensor drift, maintenance history, and prediction uncertainty. Integrate with the existing digital-twin/evidence architecture rather than adding a second asset-history system.

### SX-007 — PLSS interface boundary

Represent pressure, oxygen, CO2, thermal, power, and consumable state as read-only service observations to the assist layer. Any future PLSS controller remains a separately qualified deterministic system.

### SX-008 — hardware-in-the-loop

Only after simulation gates stabilize: motor/encoder/IMU bench rigs, failure injection, backdrivability measurements, thermal-vacuum-compatible components where appropriate, and eventually human-in-the-loop work under an approved safety program.

## External benchmark context

Current public sources are useful as comparison context, not as evidence that this project has matched them:

- NASA, `NASA Moon Mission Spacesuit Nears Milestone`, 2026-02-12: AxEMU technical/testing progress including human task and underwater surface-operation work.
- NASA Office of Inspector General, `NASA's Acquisition of Next-Generation Spacesuit Services`, IG-26-006, 2026-04-20: schedule, demonstration, qualification, and program-risk context for next-generation lunar/microgravity suits.
- NASA Technical Reports Server, ICES-2025-56, `Artemis Suit Material Project Overview`: lunar dust, durability, thermal, flexibility, and materials-development context.

Reference URLs:
- https://www.nasa.gov/humans-in-space/nasa-moon-mission-spacesuit-nears-milestone/
- https://oig.nasa.gov/audits/nasas-acquisition-of-next-generation-spacesuit-services/
- https://ntrs.nasa.gov/citations/20250003726

## Claim policy

Allowed today:

> `Symthaea Space Exosuit v0.1 defines a simulation and evidence program intended to test whether powered, predictive assistance can outperform a declared EVA baseline on specific gates.`

Not allowed today:

> `The Symthaea exosuit is better than AxEMU.`

That statement requires the integrated qualification evidence encoded by the benchmark gates and does not currently exist.
