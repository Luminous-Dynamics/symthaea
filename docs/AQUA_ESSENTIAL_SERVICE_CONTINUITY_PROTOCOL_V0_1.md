# Luminous Aqua — Essential-Service Continuity Protocol v0.1

**Status:** software / product-state requirements and unoccupied validation protocol; no physical continuity test exists  
**Workstream:** [AQUA-ESSENTIAL-CONTINUITY-001 #12760](https://github.com/Luminous-Dynamics/symthaea/issues/12760)  
**Product anchor:** AQS-1 #12717  
**Related:** DHW cascade #12709, wet-path hygiene #12703, adaptive experience #12712, service compiler #12749

## 1. Paradigm shift: qualify graceful degradation, not just nominal efficiency

A bathroom is normally evaluated under stable water pressure, electricity, communications and source quality. Aqua should also be evaluated when one or more utilities become unavailable or unreliable.

The goal is not to keep water flowing at any cost. The goal is to maximize **qualified essential bathing service** while preserving safe limits, water-quality authority, user control and an explainable recovery path.

A correctly isolated fixture under unsafe or unknown conditions is safer than an apparently continuous but unqualified service. Continuity and safety must therefore be reported as separate dimensions.

This is a resilience benchmark, not an assumption that outages should be served by stored or alternate water. Any buffer tank, alternate source or non-potable use needs a distinct professional design, water-quality, stagnation, backflow, cleaning and jurisdictional review.

## 2. Scope and physical boundaries

First reference configuration: AQS-1 direct-flow shower + basin + dry-side service spine. No shower recirculation, greywater-to-bathing connection, pool, plunge, steam, sauna or fountain is introduced by this protocol.

Maintain separate, identified circuits for:
- qualified potable cold supply;
- qualified hot-water source and storage;
- each fixture branch;
- wastewater / drain;
- any separate non-potable WaterLoop system;
- any future pool / plunge / aquatic treatment system.

No software transition can create a physical cross-connection, qualify an alternate source or promote unknown water to bathing use. No non-potable, greywater, drain, pool or unqualified stored water may feed a bathing outlet.

## 3. Degradation states

Use a small, explicit state vocabulary. State names describe product availability; they do not replace local device control or safety analysis.

| State | Meaning | Permitted behavior |
|---|---|---|
| QUALIFIED_FULL | All required source, device, interlock, evidence and utility conditions are inside the admitted envelope | Normal locally controlled service |
| QUALIFIED_REDUCED | A prequalified reduced mode remains within its own documented pressure / temperature / flow / hygiene / electrical envelope | Only that exact reduced mode; no improvised setpoint or source substitution |
| MANUAL_ONLY | Optional digital/planner/network functions are unavailable, but independent local controls and protections remain admitted | Direct manual service inside the existing qualified device envelope |
| ISOLATED_STOP | A hard limit, leak, drain fault, water-quality condition, protection state or source condition makes the affected service inadmissible | Close/isolate affected branch using the qualified local design; no planner override |
| RECOVERY_PENDING | An outage or fault has ended but the affected system has not completed required inspection, flush, metrology and source-quality reconciliation | No automatic return to a higher service state |

Aqua's software may display an observed state and recommend a transition; the qualified local controller and physical protection implement the actual response. The model must distinguish "communications unavailable" from "physical protection unavailable."

## 4. Transition and authority contract

### Non-safety service dependencies

- Loss of internet, cloud, Symthaea or user profile should not remove basic manually controlled bathing service.
- Loss of a preference service should remove recommendations, not alter mechanical safety limits or stop a qualified manual mode.
- Loss of optional telemetry must be visible. It must not be reported as a zero value or treated as a current pass.

### Hard physical / evidence conditions

- Pressure outside the exact rated / admitted envelope: no continued service based on interpolation.
- Water-quality evidence unknown, stale, revoked or not applicable to intended use: no claim or permissive for that use; the responsible local process must apply its qualified isolation behavior.
- Temperature control or independent protection uncertain: no software-mediated continuation.
- Drain blockage, uncontrolled leak or electrical protection fault: use the professionally designed stop / isolation response.
- Sensor contradiction or stale measurement: treat the relevant state as unknown until reconciled.
- Loss of electrical power: only a prequalified passive/manual fallback may remain available. Do not assume a valve's failure position or safe temperature behavior without its exact design evidence.

### Recovery

After service interruption or protective stop:
1. preserve the event record and exact hardware/configuration identity;
2. establish that the utility/source has returned within the admitted envelope;
3. inspect the affected line, drain, valve, exchanger and any retained-volume boundary as applicable;
4. perform the exact qualified flush / cleaning / disinfection procedure where required;
5. obtain current source-quality and metrology evidence from their owners;
6. reconcile water/energy/residual state using Symtropy's canonical outputs;
7. require the locally authorized restart permissive;
8. return to service only at the highest state actually requalified.

A power restoration or network reconnection alone is never a requalification event.

## 5. Fault-injection matrix

Test unoccupied analytical / hardware-in-the-loop cases first. Use only pressures, temperatures, voltages and fault injection points within a professionally reviewed test envelope.

| Scenario | Injection phase | Expected assessment |
|---|---|---|
| Pressure sag within admitted range | warm-up / active / cleaning | Does qualified reduced mode remain inside its declared envelope? |
| Pressure below admitted minimum or above rated maximum | each phase | Local safe stop or approved isolation; no interpolation claim |
| Loss of pressure during active use | active | Safe local behavior; log interrupted service and restart requirements |
| Hot-water supply interruption | warm-up / active | No unsafe automatic compensation; only prequalified alternative behavior |
| Cold-water supply interruption | warm-up / active | Temperature and hydraulic protections retain authority |
| Mains power loss | warm-up / active / cleaning | Device enters exact documented failure state; manual fallback only if separately qualified |
| Controller reboot / process crash | each phase | No unsafe transient, stale command replay or unauthorized restart |
| Internet / cloud loss | active | Manual/local service remains available; planner becomes unavailable |
| Flow / pressure / temperature channel stale or disconnected | each phase | Affected evidence becomes unknown; no optimistic continuation |
| Conflicting sensor channels | active | Contradiction is surfaced and handled by qualified local logic |
| Water-quality record expired / revoked | pre-use / active | Use permissive follows Mycelix / competent authority; Aqua cannot waive it |
| Blocked inlet strainer / drain obstruction | active / post-use | Leak/overflow protection and isolation behave as designed |
| Restoration after interruption | post-fault | Recovery remains pending until required evidence and physical procedure complete |

Test cases must include negative controls: a planner requesting continuation after a stop condition, stale evidence replayed against a new installation digest, and network reconnection attempting to replay a prior command.

## 6. Continuity benchmark vector

Do not publish one blended "resilience score." Report at least:

- **Qualified service availability:** completed required tasks divided by attempted tasks under each declared disturbance profile.
- **Safe-state correctness:** whether each hard-stop case reaches its prescribed local safe state; false continuation is separately enumerated and must be zero in the defined software / test corpus.
- **Time to safe state:** measured from injected fault to independently observed local safe state.
- **Time to manual fallback:** measured only where a manual fallback is prequalified and available.
- **Recovery time and burden:** time, labour, flush / cleaning water, energy, consumables and inspection needed before requalification.
- **Unserved service:** interrupted / unavailable task count and duration, with the reason for stop.
- **Resource penalty:** additional water and purchased energy required for the disturbance and recovery.
- **Evidence integrity:** stale, missing, contradictory and revoked evidence correctly rejected.
- **User agency:** manual override and explicit choice remain available; no hidden automatic restart.
- **Maintenance impact:** fault-induced wear, replacement and downtime.

A stop that correctly protects users is not scored as a safety failure. Report the loss of service separately from the correctness of protection.

## 7. Comparator design

Compare AQS-1 to one declared conventional reference installation under the same injected utility events, task profile, installation assumptions and measurement protocol.

Also include component-level ablations:
- local manual controls without any planner;
- optional digital layer enabled versus disconnected;
- qualified passive fallback versus no fallback (where the latter is an analytical comparison only);
- any proposed storage / buffer concept as a separate study arm, not assumed part of the baseline.

Freeze exact model, component, circuit, firmware, installation and maintenance configuration. Vendor claims are not substitutes for injected fault tests.

## 8. Initial benchmark hypotheses

These are engineering objectives, not achieved results or regulatory limits:

1. Zero planner/cloud dependency for normal manual use and emergency stop.
2. Zero false continuation in the defined hard-stop negative-control corpus.
3. 100% explicit distinction between unknown / stale / revoked evidence and a valid current record in the software contract tests.
4. Every reduced service state has an exact qualified device envelope and separate acceptance evidence; unsupported reduced mode is rejected.
5. Every outage/fault path reaches either a documented qualified fallback or the documented isolated-stop state.
6. No automatic restoration to a higher state until source, physical and evidence recovery gates pass.
7. Recovery water, energy, labour and downtime are included in the service boundary.
8. No continuity gain is credited if achieved by bypassing safety, source-quality or metrology requirements.

The confirmatory test matrix, exact local response, and any response-time limits must be approved by the competent device / safety owners. This document invents no valve failure position, electrical limit, water-quality threshold or universal outage duration.

## 9. Storage and alternate-source decision gate

A buffer tank is not automatically a resilience improvement. Compare its benefit against:
- stagnation and residence-time behavior;
- source quality and treatment dependence;
- vessel and wetted-material compatibility;
- backflow / cross-connection risk;
- cleaning access and disinfectability;
- flushing and reject volume;
- pump and standby energy;
- failure modes and leak consequences;
- installation footprint, cost and maintenance;
- lawful intended use in the deployment jurisdiction.

The default AQS-1 design remains direct supply with no additional bathing-water buffer until a separate evidence package justifies one. Greywater and other non-potable supplies remain physically separate and use-class specific.

## 10. Evidence ownership and staged gates

- Aqua owns the product state vocabulary, task-level continuity outcomes and comparator context.
- Local device / mechanical / electrical authorities own protection, actuator failure behavior and restart design.
- Mycelix owns source-quality assessment, sensor identity, calibration, timebase, uncertainty and applicability.
- Symtropy owns canonical physical water / energy / material / residual closure and its existing hydraulic / thermal models.
- Wet-path hygiene #12703 owns its independent cleaning / exposure boundary.
- Symthaea service compiler #12749 remains advisory and cannot authorize or override a transition.

Stages:
- **C0 contract:** state machine, threat/fault model and negative-control corpus.
- **C1 deterministic replay:** synthetic event traces and state-transition tests, no connected hardware.
- **C2 unoccupied HIL / test cell:** only after the exact device and test fixture design are reviewed.
- **C3 professional qualification:** plumbing, electrical, water-quality, hygiene and accessibility review for the exact deployment.
- **C4 supervised occupied evaluation:** separate authorization and protocol, after earlier gates.

## 11. Claim ceiling

This protocol establishes requirements and test cases only. It does not demonstrate outage resilience, safe state transitions, water quality, a safe buffer tank, regulatory compliance or human suitability. It authorizes no construction, water-source connection, procurement or occupied/human-exposure operation.
