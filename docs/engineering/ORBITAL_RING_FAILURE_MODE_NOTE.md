# Orbital-ring failure-mode evidence note — Birch, Part II

**Status:** primary-source requirements extraction; not a validated fault model, quantitative safety case, or present-day design recommendation.  
**Sources reviewed:** scanned pages 123 and 125 of Paul Birch, “Orbital Ring Systems and Jacob's Ladders — II,” *Journal of the British Interplanetary Society*, vol. 36, pp. 115–128 (1983); [PDF scan](https://www.orionsarm.com/fm_store/OrbitalRings-II.pdf). The paper explicitly frames Part II as engineering/logistics/safety analysis built on the Part I theoretical concept.

## 1. Source findings that change the model plan

### A. Distinguish support loss from drive-power loss

In §5.2, p. 123, Birch calls skyhooks the most likely source of trouble. He discusses sudden changes in the magnetic field around lift coils, superconducting breakdown, winding protection against transients, and structural bracing against magnetic stresses. He also describes cryogenic failure as a loss of superconductivity. These are distinct initiating events and should not be collapsed into one generic “power failure”.

### B. Sudden change in skyhook load can disturb the orbital rings

In §5.6, p. 125, Birch describes a sudden decrease in the load carried by a skyhook. The orbital rings cannot immediately follow their original paths; the source says they may overshoot and miss the next skyhooks, allowing the rings to tear apart. The source then describes internal ring tension as a possible mitigating effect, but concludes that additional protection is generally needed.

The design response described in the same section is an inter-ring coupling/rider system. Riders use superconducting magnetic coupling to hold the rings together, resist vertical displacement, and couple through spring-loaded and damped sections. They are distributed at intervals intended to be close enough to protect the ORS against sudden load changes. This is a specific mechanical containment/control function, not merely a reliability improvement to record in a checklist.

**Engineering implication:** a candidate model must not assume that an orbiting ring plus independent skyhooks stays coupled under an arbitrary support-load step. It must either model the rider/inter-ring coupling and its limits or explicitly analyze a different architecture. A simulation with the coupling omitted cannot support a claim that this fault is survivable.

### C. Redundancy must be a physical topology with a bounded response

The discussion in §5.2 describes multiple independent lift-coil units and several skyhooks supporting critical loads. It states that load can remain within design limits when enough lift units remain operational, while faulty units are monitored and replaced. These are qualitative design claims, not proof that the proposed redundancy works under correlated failures, load redistribution, faults during a transient, or loss of a whole skyhook.

The model therefore needs explicit support topology, which ring segments and skyhooks carry each load, the coupling distance, allowable displacement, actuation/communication delay, and a defined fallback state. A global “redundancy = true” flag is insufficient.

### D. A ladder-collapse model is not an orbital-ring-breakup model

Also on p. 123, the source treats the collapse of a Jacob's Ladder after its skyhook is lost, describing a falling cable, lateral deflection due to the Coriolis effect, and a guard band. That is a payload-access-cable hazard with a different state, path, and consequence set from the p. 125 orbital-ring response to a sudden support-load change. Keep these as separate scenarios and safety claims.

### E. Keep partial-ring power/energy figures scoped to that architecture

In the partial orbital ring system (PORS) discussion around p. 120, Birch gives an illustrative PORS power-dissipation estimate of about \(2.1\times10^8\,\mathrm{W}\) and total stored kinetic-plus-potential energy of about \(9.8\times10^{12}\,\mathrm{J}\). These numbers belong to the paper's specified partial-ring example; they are not transferable to the continuous-skyhook ORS in Part I's Table 8. The fact that both are called an “orbital ring” is not a valid reason to combine their energy or power budgets.

## 2. Required fault scenarios for the executable workstream

| Scenario ID | Initiating event | Minimum state to model | Required output / stopping rule |
|---|---|---|---|
| F-SKYHOOK-LOAD-STEP | Sudden reduction/removal of a skyhook's supported load | Initial orbit/ring state; affected support; load vector and rate of change; rider topology and stiffness/damping; ring tension; support-to-support spacing and delays | Peak ring offset, support miss distance, coupling force, transient energy/momentum residuals, and whether the next support is engaged. Mark inconclusive if coupling/geometry is unspecified. |
| F-LIFT-COIL-LOSS | One lift coil/actuator unavailable | Coil topology, remaining lift capacity, saturation, current decay and adjacent load redistribution | Local support margin and transient peaks; no “redundant” pass from static capacity alone. |
| F-CRYOGENIC-QUENCH | Local or common-mode superconducting quench | Heat load, quench onset/propagation assumptions, shield/coil geometry, current and magnetic-field response, affected length, protection/interruption behavior | Quench extent and time-to-limit, force interruption history, loads on remaining supports, thermal/electrical ledger; no assumed instantaneous energy disappearance. |
| F-RIDER-COUPLING-LOSS | One inter-ring rider/coupler loses force or breaks away | Rider spacing, coupling force law, spring/damper, vertical displacement, neighbor state, common-mode faults | Relative ring displacement, stress/force concentration, remaining restraint margin, fail-closed claim status if force law is absent. |
| F-LADDER-COLLAPSE | A Jacob's Ladder loses its skyhook | Cable mass/tension, geometry, payload occupancy, Coriolis frame, ground path/guard band | Ladder trajectory and risk envelope; report separately from orbital-ring structural integrity. |
| F-DEBRIS-SEGMENT | Local impact damages a ring, lift coil, or rider | Impactor distribution/velocity, exposed cross-section, shield response, damaged length, detection and isolation topology | Damage extent, affected support span, isolation time, and recovery feasibility with model uncertainty. |

These scenario names are a proposed workstream contract. They are not claims that the listed transients have been simulated.

## 3. Evidence contract for a load-step experiment

1. Freeze one architecture, geometry, inertial state, support and rider locations, force laws, damping model, and fault-injection time in a canonical manifest.
2. Independently check the no-fault equilibrium and small perturbation response before applying a finite step.
3. Inject a bounded step and at least one ramped load-removal case. Sweep support-loss severity, affected angular span, rider spacing/stiffness/damping, delay, and uncertainty; retain failed/divergent runs in the denominator.
4. Track ring position and velocity, support/rider forces, coupling saturation, and total energy/vector-momentum residuals. A low final residual is not proof the transient stayed inside limits.
5. Compare at least one simplified case to a separate derivation/implementation before upgrading beyond T0/T1. Do not borrow historical safety margins or failure probabilities as validation data.

## 4. Explicit limits

The paper's claims are theoretical engineering analysis from 1983. They identify mechanisms and possible mitigations, but the scanned sections do not provide modern component qualification, a validated transient model, calibrated input distributions, correlated-fault probabilities, or experimental evidence that the complete system survives the described event. The use of shields, redundancy, or damped couplers must remain a hypothesis until the chosen implementation is modeled, independently checked, and eventually tested.

**Next gate:** specify a continuous-skyhook support-loss test geometry and the inter-ring rider force/kinematic contract. If those fields are not available, keep F-SKYHOOK-LOAD-STEP blocked rather than letting a nominal orbit propagator imply safety.
