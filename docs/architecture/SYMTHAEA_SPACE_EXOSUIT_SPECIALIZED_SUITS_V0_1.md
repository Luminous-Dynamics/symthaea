# Space Exosuit mission-specialized suit architecture v0.1

**Status:** Research architecture and software requirement-traceability contract only. It is not a pressure-vessel design, approved interface control document, hardware specification, flight-readiness claim, or spacesuit certification.

**Research track:** [SX-028 — mission-specialized suit profiles on a shared, verifiable platform contract](https://github.com/Luminous-Dynamics/symthaea/issues/7504).

**Implementation:** Draft branch `feat/space-exosuit-specialized-suit-profiles`, stacked on [SX-027 boundary projections PR #7501](https://github.com/Luminous-Dynamics/symthaea/pull/7501).

## Design decision: a family, not one universal suit

The platform should standardize evidence semantics, selected interfaces, service/diagnostic contracts, and safety invariants while permitting physical variants tuned to genuinely different mission environments and tasks. The Lunar Grounding Challenge announced by NASA in October 2026 also motivates an explicit electrostatic charge-management capability for relevant lunar surface profiles; it is an interface/evidence requirement to investigate, not a prescribed grounding device or validated solution. A common data model must never be mistaken for universal physical component compatibility.

This is supported by real differences between orbital EVA and surface suits. NASA's public suit material describes different mobility/design considerations for lunar surface suits, while NASA-STD-3001 Volume 2 explicitly includes incapacitated-crew rescue resources. NASA TechPort describes the suitport as a bulkhead, capture, and sealing interface designed to keep the suit outside the pressurized habitat and reduce contamination transfer. Active lunar dust research includes protective fabrics and cleaning/seal technologies. Those references motivate the following software profile taxonomy; they do **not** make this document's profile requirements authoritative NASA requirements.

Sources:
- NASA spacesuits overview: https://www.nasa.gov/humans-in-space/astronauts/spacesuits/
- NASA spacewalk suit construction/mobility overview: https://www.nasa.gov/centers-and-facilities/johnson/spacewalk-spacesuit-basics/
- NASA-STD-3001 Volume 2, §11 spacesuits and incapacitated crew rescue: https://www.nasa.gov/reference/11-0-spacesuits-vol-2/
- NASA TechPort, EVA suitport: https://techport.nasa.gov/projects/10728
- NASA TechPort, active dust mitigation fabric research: https://techport.nasa.gov/projects/125577
- NASA OIG, next-generation suit development and schedule risks (April 20, 2026): https://oig.nasa.gov/news/spacesuit-development-delays-could-impact-artemis-timeline-and-iss-operations/

## Initial profile family

| Profile | Distinctive capability set (initial hypotheses) | Key verification focus |
|---|---|---|
| Orbital EVA | Microgravity translation/restraint, tether and anchor interfaces, orbital worksite interfaces, high-dexterity glove interfaces | Tether loads, restraint ergonomics, tool handling, micrometeoroid/debris and thermal conditions, pressure-boundary integrity |
| Lunar surface EVA | Planetary locomotion, dust mitigation, electrostatic charge-management interface, abrasion/puncture resistance, dexterity, surface navigation interface | Dust adhesion/abrasion, charging in shadow/polar regions, vehicle-contact ESD, joint wear, kneeling and climbing, thermal cycles, geology/sample handling |
| Mars surface EVA | Surface mobility, dust/abrasion defenses, navigation, field repair, replaceable life-support modules, radiation-monitoring interface | Extended operations, maintainability, environmental compatibility, consumables, communication-delay procedures |
| Construction and heavy tooling | Worksite restraint, tether/anchor interfaces, load-bearing tool interfaces, dexterity and field repair | Tool reaction forces, entanglement, local strain, snagging, equipment changeover, safe manual release |
| Dust-intensive surface work | Locomotion, dust mitigation, electrostatic charge-management considerations, abrasion/puncture resistance, contamination containment and field repair | Seal/connector contamination, charge accumulation where relevant, garment cleaning, dust ingress paths, degradation detection and serviceability |
| Suitport and habitat servicing | Mechanical/pressure compatibility, contamination containment, dust mitigation, resource service and modular replacement interfaces | Seal evidence, positive capture, connector compatibility, transfer accounting, dust exclusion and offline failure handling |
| Crew rescue and recovery | Rescue handling, tether/anchor interfaces, protected emergency reserve and repair capability | Incapacitated crew transfer, rescuer workload/consumables, handling aids, communications and contingency endurance |
| Long-duration orbital maintenance | Microgravity translation, restraints, repairability, replaceable life-support modules, servicing and radiation-monitoring interfaces | Maintainability, fault isolation, repair inventory, spares logistics, system lifecycle and worksite ergonomics |

These are **starting points for simulation and requirements review**. Mars-specific items in particular remain hypotheses until tied to a declared mission scenario, engineering evidence and actual program requirements. The list is not an official standards matrix and does not claim any variant is safe for its named role.

## Scope boundary: EVA is one suit family, not every pressure suit

The eight profiles in this tranche are EVA configurations. They should not be treated as a universal taxonomy for every crew-worn survival system. Launch-and-entry survival suits, intravehicular emergency/escape suits, and EVA suits operate in different vehicle interfaces, duration assumptions and use cases. For example, a crew survival/launch-entry suit may depend on vehicle-fed interfaces and have no EVA-style portable life-support backpack; making the current EVA common-foundation list mandatory for every class would encode the wrong abstraction.

The next platform layer should therefore type the **suit family** separately from the **mission/task profile**:
- **Launch and entry survival:** vehicle-compatible breathing/pressure support, seat/restraint compatibility, emergency egress and post-landing survival interfaces.
- **Intravehicular emergency / escape:** cabin-depressurization contingency, vehicle emergency interfaces, restraint/egress, and constrained-duration survival.
- **EVA:** portable life support or an explicit umbilical variant, pressure-boundary integrity, thermal management, mobility appropriate to the environment, and contingency/rescue requirements.

These are research categories, not an assertion that all missions use the same pressure-suit type. Each family needs its own required safety foundation and evidence policy. Shared communications, sensors, data contracts, fit records or service connectors can be standardized where interface control and validation justify it, without sharing a physical subsystem merely for architectural neatness.

## Common foundation vs. specialized capability

The evaluator currently defines a small common foundation for every profile:
- pressure-boundary integrity;
- primary life-support interface;
- thermal management;
- independent emergency reserve;
- manual fallback;
- communications and telemetry; and
- fault containment/safe degradation.

Mission-specific capabilities are kept separate. Orbital translation hardware is not automatically a substitute for planetary locomotion; lunar dust controls do not establish suitport compatibility; a resource connector does not establish a pressure seal; and the presence of repair hardware does not establish crew-rescue capability.

Modularity is a *compatibility claim to verify*, not a default. Life-support packs, gloves, helmets, boots, suitports, thermal loops and power modules may be shared only where the specific interface geometry, sealing, loads, electrical/power properties, contamination behavior, fault containment and verification artifacts demonstrate interoperability.

## Software contract

`suit_profile.rs` defines:
- eight explicit mission profiles and ordered required-capability sets;
- capability claims keyed to implementation ID, evidence level, evidence reference, exact variant ID, and exact hardware revision;
- a caller-specified minimum evidence policy;
- deterministic findings for missing capability, evidence below policy and duplicate/conflicting claims;
- an evaluator status based only on the specified policy and declared artifacts.

The maturity sequence (Declared → Simulated → BenchTested → HumanInLoop → Qualified) is an ordering for triage only. Claims are scoped to one exact variant ID and hardware revision, so reusing them against another variant/revision fails closed. The evaluator does not auto-promote evidence or prove the artifact referenced is genuine. Even `MeetsDeclaredEvidencePolicy` is **not** certification, mission approval or proof of operational compatibility. The evaluator has no actuator or life-support command interface.

Run on a configured Rust workspace:

```sh
cargo test -p symthaea-exoskeleton suit_profile
```

Tests are authored for shared foundation invariants, profile differentiation, missing capability, insufficient evidence, evidence non-transfer, duplicate claims, malformed metadata and order-invariant output. These tests still need to execute on the exact PR head.

## Paradigm shifts required to make the platform genuinely better

1. **Optimize for mission outcomes, not suit feature count.** Compare task completion, crew effort, dexterity, time to egress, consumables, thermal burden, injury/exposure risk, repair time and graceful degradation under matched scenarios.
2. **Treat suit + habitat + rover + tools + servicing as one mission system.** The wearable remains independently survivable; external services add capability but never become an unacknowledged survival dependency.
3. **Make every claim traceable.** A capability maps to a particular implementation revision, evidence artifact, test configuration, limitations and verification owner. Evidence gaps remain visible.
4. **Design for maintainability and fleet diversity.** Prefer replaceable modules only where interface contracts and failure containment are verified. Avoid creating eight wholly independent architectures if genuine commonality can be demonstrated.
5. **Make human factors first-class.** Fit range, dexterity, fatigue, visibility, glove comfort, donning/doffing and rescue ergonomics need measured protocols and wearer participation. A software profile cannot substitute for these evaluations.
6. **Keep safety authority deterministic and separate.** Symthaea can explain, diagnose, compare options and recommend; it must not override independent pressure/life-support limits or manually verified interlocks.

## Evidence gates

Before using any profile for physical design decisions: connect each claim to mission-specific requirements; have independent reviewers validate the requirement mapping; define interface-control documents and test boundaries; create nominal and adversarial scenario corpora; verify pressure, thermal, mobility, gloves, dust, servicing and rescue subsystems; then use calibrated rigs and human-in-the-loop testing in an approved progression. Keep simulations, bench data, human-in-loop data and qualification evidence explicitly distinct.
