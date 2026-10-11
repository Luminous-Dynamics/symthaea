# F-SKYHOOK-LOAD-STEP input and evidence ledger

**Purpose:** close inputs by evidence, not by placeholder completion. This ledger defines what each of the manifest's 23 fields means, the unit/representation contract, and the evidence needed before independent review. It does not itself select a feasible orbital-ring design.

Primary failure-mechanism source: Birch, *Orbital Ring Systems and Jacob's Ladders — II* (1983), §5.2 p. 123 and §5.6 p. 125, [scanned paper](https://www.orionsarm.com/fm_store/OrbitalRings-II.pdf). Birch identifies sudden skyhook-load changes as a threat and proposes inter-ring riders, but does not provide a modern, verified continuous-skyhook state, rider force law, or transient model. Those must not be inferred from his prose.

## Status vocabulary

- **Source fact:** directly transcribed from a cited source with page/figure/table locator. Preserve the architecture and system boundary to which it belongs.
- **Derived:** computed from explicit source-backed inputs with equation, units, and independent arithmetic check.
- **Engineering assumption:** deliberately chosen value/range, with rationale and sensitivity plan; not relabeled as a measured or established fact.
- **Measurement:** captured from a specified instrument/system with calibration and uncertainty.
- **Independent model:** produced by a separately implemented or independently reviewed calculation, with code/version and input digest.
- **Test fixture:** only for validator development. It must never be presented as an engineering value.

Every populated input requires an input_provenance record with classification, at least one evidence_refs item, rationale, and review_state. A reference proves traceability/origin, not correctness. independently_reviewed records a review disposition; it does not constitute a physics pass. A rejected record blocks the field, and the validator rejects test_fixture provenance for scenario-candidate values. If a value is unknown, leave the value null and state Blocked.

## Field-by-field closure contract

| # | Manifest field | Unit / representation | Evidence needed to close the field |
|---:|---|---|---|
| 1 | inertial_frame | Named frame and realization | Declare frame realization, epoch/time scale, origin, axes, and any required Earth orientation model. A bare “ECI” label is insufficient for frame-critical work. |
| 2 | support_frame_transform | SI position/velocity, unit quaternion, angular velocity/acceleration | Provide the transform's direction and convention, quaternion ordering, epoch, origin state, angular velocity/acceleration, and numerical source. Check orthonormality and quaternion normalization. |
| 3 | ring_radius_m | m, strictly positive | Geometry definition: centerline/reference radius and Earth/body datum. Derive from an explicitly stated altitude/reference radius or a source-backed geometry; never infer exact altitude from a rounded historical table alone. |
| 4 | ring_plane_orientation | Unit normal plus plane convention | Normal expressed in the declared inertial frame and its orientation/reference convention. Demonstrate normalization and consistency with the reference point and support coordinates. |
| 5 | initial_ring_position_m | Finite inertial 3-vector, m | Define this as the selected reference **centerline material point**, not the ring's centre of mass. Its norm must match ring_radius_m; it must lie in the declared ring plane. |
| 6 | initial_ring_velocity_m_s | Finite inertial 3-vector, m/s | Velocity of the same reference material point as field 5, at the same epoch. It must lie in the ring plane and satisfy `abs(r·v)/||r|| ≤ acceptance_limits.max_initial_radial_velocity_m_s` for this scenario’s declared circular-reference path. This is only a kinematic tangency check; it does not establish force balance or dynamic stability. |
| 7 | initial_equilibrium_evidence_ref | Immutable artifact/URI reference | Independent evidence for no-fault equilibrium: exact geometry/input manifest, force balance, residuals/tolerances, solver/reference implementation and reviewer disposition. A reference string alone is not evidence of equilibrium. |
| 8 | skyhook_support_locations_rad | At least two unique angles in [0, 2π) | Angle zero and direction convention tied to field 4, support count/topology, each support's identity, and why the chosen spacing covers the claimed load path. |
| 9 | support_load_before_n | Finite 3-vector, N | Direction, point of application, support identity, sign convention and pre-fault load case. Magnitude and decomposition must follow the frozen control-volume boundary. |
| 10 | support_load_after_n | Finite 3-vector, N | Post-fault load for the same support and convention. For this scenario its magnitude must be strictly less than field 9. State whether the removed load is transferred, externally applied, or no longer present in the chosen boundary. |
| 11 | load_change_profile | step or ramp, duration in s, reference | Exact time profile and source/definition. A step requires zero duration; a ramp requires positive duration. Include profile shape, terminal value, and injection uncertainty in the referenced definition. |
| 12 | load_step_time_s | s from manifest epoch, finite and nonnegative | Declare the time origin and event trigger. Include latency and time-step sensitivity in analysis design. |
| 13 | rider_spacing_m | m along the specified path | Rider/coupler topology and distance definition; distinguish arc spacing, straight-line chord distance, and spacing between ring interfaces. |
| 14 | rider_force_law | Immutable equation/model reference | Constitutive force law, direction, valid displacement/velocity range, coupling topology, hysteresis/nonlinearities and failure/saturation behavior. Historical prose is not a force law. |
| 15 | rider_stiffness_n_per_m | N/m, finite and positive | Identified tangent/secant stiffness and operating range, source or measurement, uncertainty, and relationship to the force law in field 14. |
| 16 | rider_damping_n_s_per_m | N·s/m, finite and nonnegative | Damping model/definition, reference frame and range. Zero is permitted only if explicitly justified; damping cannot be used to hide energy loss outside the boundary. |
| 17 | rider_max_coupling_force_n | N, finite and positive | Hardware/model force limit, duration/thermal constraints, load-sharing assumption and saturation transition. Do not infer this from nominal stiffness alone. |
| 18 | actuation_measurement_delay_s | s, finite and nonnegative | Sensor, computation, communications and actuator latency boundary. Provide nominal value, worst-case/uncertainty interval and whether clocks are synchronized. |
| 19 | force_model_reference | Immutable equation/model reference | Gravity and all other force laws, versions, assumptions, domains of validity and independently checked limiting cases. State explicitly which forces are omitted. |
| 20 | energy_boundary | Frame, included bodies, work/flux terms | Enumerate all included subsystems, external work, heat/electrical transfer, material/energy flux, storage terms and signs. Frame must match inertial_frame; scope_complete must not be inferred from empty lists. term_inventory_ref must point to an explicit term inventory for independent review. |
| 21 | momentum_boundary | Frame, included bodies, force/flux terms | Enumerate included bodies, external forces/torques, exchanged momentum and mass flux. Frame must match inertial_frame; term_inventory_ref must point to an explicit term inventory for independent review. Do not substitute angular momentum or scalar speed for vector momentum. |
| 22 | acceptance_limits | m, N, J, kg·m/s, m/s | Before running, justify each limit from clearance/geometry, hardware or independently reviewed numerical error. Energy and momentum residual limits must use the declared control-volume normalization and uncertainty. `max_initial_radial_velocity_m_s` bounds the radial component implied by the declared circular-reference constraint and must be justified from the initial-state source/model uncertainty; it is not a general orbital-safety tolerance. |
| 23 | independent_reference_case | Immutable case/artifact reference | A separate calculation that shares no unexamined implementation path with the target solver; specify what it checks, expected result, tolerances, and reviewer. A copied golden output is not independence. |

## Cross-field invariants

The validator enforces basic shape and selected consistency rules: ring-point norm versus radius, position/velocity with ring plane, circular-reference radial speed bounded by a declared m/s tolerance, a strict decrease in support-load magnitude, unique support angles, matching energy/momentum/inertial frames, explicit from/to transform frames and epoch, term-inventory references for both conservation boundaries, coupling acceptance limit not exceeding the coupler's declared force capacity, and provenance for every populated value. It rejects test_fixture and rejected evidence provenance. These are **necessary contract checks, not sufficient physics checks**.

A valid physical model will need additional checked invariants: tangent velocity for a declared circular reference path; support-load locations consistent with the selected topology; force law and stiffness/damping using compatible coordinates; baseline equilibrium residuals; conservation residuals during the transient; and convergence with time/space discretization. Do not add a generic numerical tolerance without dimensional reasoning and a reference comparison.

## Current disposition

All 23 values and all 23 provenance entries in docs/engineering/scenarios/F_SKYHOOK_LOAD_STEP.json are null. This is intentionally not an executable case. The contract validator should report passed_contract_checks for the correctly declared blocked manifest, while leaving its expected scenario state Blocked and analysis state NotRun. When information is added, each field closes only with its provenance record; malformed, placeholder or mismatched values keep the gate blocked.

NASA’s Meteoroid Environment Model guide states that the inertial position and velocity vectors are perpendicular for a circular orbit, while noting this does not hold for eccentric or transfer orbits: [NASA MEM User Guide](https://fireballs.ndc.nasa.gov/mem/guide/). The checker applies that relation only because this scenario explicitly declares a circular-reference path; it does not infer a full orbit or dynamical equilibrium from tangency.

## Research methods worth adapting—within scope

- Arena & Lacarbonara, “On the stability of magnetically levitated rotating rings” (2017), [DOI 10.1016/j.ijmecsci.2017.07.007](https://doi.org/10.1016/j.ijmecsci.2017.07.007): relevant methods include geometrically exact ring modelling, modal analysis, gyroscopic effects and magnetic restoring-force sensitivity. It is **not** a model of a planetary orbital ring or evidence that the proposed skyhook system is stable.
- [“A finite element based stability test for equilibria of flexible structures in circular orbits” (2008)](https://doi.org/10.1016/j.ijnonlinmec.2008.03.004): a useful reminder to solve equilibrium and then test stability with the correct conserved quantity (angular momentum for the discussed class of orbiting systems), rather than equating convergence of a solver with stability. It does not supply the missing rider or skyhook force laws.
- Vanderbei & Kolemen, “Linear stability of ring systems” (2007), [DOI 10.1086/510457](https://doi.org/10.1086/510457): its planar point-body orbital stability problem can inform linearization/eigenvalue verification for a simplified orbital-particle model; it is not a structural-ring/support solution.

Transfer only methods (independent equilibrium check, perturbation/eigenvalue analysis, sensitivity, and convergence testing), not conclusions or coefficients from physically different systems.

## Next gate

First close fields 1–8 sufficiently to identify one initial geometry and equilibrium case; close fields 9–19 to define the support and rider physics; close fields 20–23 through independent review. Only then is it meaningful to run a support-load step/ramp sweep. Report failed/divergent runs and uncertainty along with successful runs; never turn validator readiness into a claim of survivability.
