# Orbital-ring failure-mode evidence note — Birch, Part II

**Status:** primary-source requirements extraction; not a validated fault model, quantitative safety case, or present-day design recommendation.  
**Sources reviewed:** scanned pages 123 and 125 of Paul Birch, “Orbital Ring Systems and Jacob's Ladders — II,” *Journal of the British Interplanetary Society*, vol. 36, pp. 115–128 (1983); [PDF scan](https://www.orionsarm.com/fm_store/OrbitalRings-II.pdf). The paper explicitly frames Part II as engineering/logistics/safety analysis built on the Part I theoretical concept.

Strict parsing follows [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259.html): object names should be unique because duplicate-name handling differs between parsers, and `NaN`/`Infinity` are not valid JSON numbers. The validator now rejects duplicate object keys and non-standard numeric tokens instead of letting Python's permissive defaults normalize them silently. These are input-integrity checks, not model or physics checks.

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

## 4. Executable scenario gate (contract only)

A versioned template exists at [`docs/engineering/scenarios/F_SKYHOOK_LOAD_STEP.json`](scenarios/F_SKYHOOK_LOAD_STEP.json). It declares **23 required inputs**, all currently null, so the scenario is explicitly `Blocked`, analysis state `NotRun`, and result artifacts empty. The [field-by-field input/evidence ledger](ORBITAL_RING_LOAD_STEP_INPUT_LEDGER.md) defines the closure evidence required for each field. This is intentional: the source describes the failure mechanism, but does not provide a reviewed modern model of the chosen ring geometry and rider force law.

The standard-library validator is [`scripts/engineering/validate_orbital_ring_failure_manifest.py`](../../scripts/engineering/validate_orbital_ring_failure_manifest.py). Each populated value must have an `input_provenance` record (classification, explicit evidence reference, rationale and review state). It also checks frame-transform from/to frames, epoch, quaternion convention and normalized quaternion; ring-point radius/plane consistency; strict load reduction; unique support angles; matching conservation-boundary frames; explicit energy/momentum term-inventory references; rider-capacity limits; and rejection of test-fixture or rejected evidence provenance. A complete fixture may advance only to `ReadyForIndependentReview`; the validator cannot mark a physics model passed or run the transient.

Recorded local checks:

- `python3 -m py_compile scripts/engineering/validate_orbital_ring_failure_manifest.py` succeeded.
- Last executed self-test (against the exact prior validator blob `2d07b86480dea71f3272fbee2a9fde4152727abb`): **73 passed, 0 failed**. The current validator has twelve new fail-closed regressions and has not yet been executed; do not treat the old report as validation of the current source.
- Additional regressions reject `test_fixture` and rejected provenance, empty evidence references, missing from/to-frame metadata, a transform target that differs from the declared inertial frame, and energy/momentum boundaries without term-inventory references.
- Running the validator on the checked-in template returned `passed_contract_checks` for the *manifest contract*, while correctly keeping the scenario `Blocked` with all 23 inputs unresolved and `analysis_state = NotRun`.
- Repeated test and manifest-validation outputs were byte-identical. The exact outputs are preserved in [the validation report](evidence/f_skyhook_load_step_manifest_validation.json) and [the self-test report](evidence/orbital_ring_failure_manifest_validator_selftest.json).

Current validator Git blob: `734fae84c6d5b1fb7b895b56b021a75892546206`; source SHA-256: `532de2bb5707db127fa9a031e233e25100673491e658d0b2173845bfeebf85f4`. Template Git blob: `a65250ba7284b503e1effb804a3a30668445017e`. Template SHA-256: `60c747bf226f436c28f08ca047e77faae2368e200bf72f600e3b84f7dd5234b1`. Validation report Git blob: `8d0f200eb2db33c4c4ef6b8dc07ef98d21e184d6`; report SHA-256: `0aa07355674fb104d0e9bb38ad3bafad3cd768b3557e4bfe012126b25f8c253c`. Self-test report Git blob: `87612c32dfe45c3c0fb8b8588716b35c0457aa67`; report SHA-256: `dc7718642fc57f35ca813fb3be857194c9ec74d9bc9413db9810d8a1432367c5`. The 73-check self-test report is historical evidence for the prior validator source; it is not evidence for the current source. See [validator verification freshness status](evidence/orbital_ring_validator_verification_status.json), which records the current source fingerprint and the twelve tests still needing execution. The manifest-validation report is structural evidence for the current empty v2 template and does not execute the validator's full self-test suite.

## 5. Explicit limits

The paper's claims are theoretical engineering analysis from 1983. They identify mechanisms and possible mitigations, but the scanned sections do not provide modern component qualification, a validated transient model, calibrated input distributions, correlated-fault probabilities, or experimental evidence that the complete system survives the described event. The use of shields, redundancy, or damped couplers must remain a hypothesis until the chosen implementation is modeled, independently checked, and eventually tested.

**Next gate:** specify a continuous-skyhook support-loss test geometry and the inter-ring rider force/kinematic contract. If those fields are not available, keep F-SKYHOOK-LOAD-STEP blocked rather than letting a nominal orbit propagator imply safety.

The scenario contract is now schema/version v2 and requires the scenario-level `initial_path_constraint=circular_reference_path` and checks `abs(r·v)/||r||` against the provenance-backed `acceptance_limits.max_initial_radial_velocity_m_s`. Four regression cases cover valid tangency, excessive radial speed, a missing radial-speed tolerance, and a missing path constraint, and scenario-version mismatch. The limit is a scenario-specific kinematic consistency bound—not an orbit-stability or safety criterion.

The draft-safe workflow also compares the live validator blob/SHA-256 against the committed freshness record before running the suite. It requires 84 unique passing check IDs and byte-identical repeated output, then archives exact output and fingerprints. The corresponding GitHub Actions run is queued; this design does not treat queueing as evidence of a pass.

The reported expected gate is now `Blocked` whenever either field-level blockers or structural/schema errors are present. This avoids contradictory reports in which a malformed manifest says `validation_state=failed_contract_checks` while separately advertising `expected_gate_state=ReadyForIndependentReview`. Regression cases cover a version mismatch, a subset field list, and unexpected provenance keys.

The manifest-contract gate was moved into its own `.github/workflows/orbital-ring-manifest-contract.yml` workflow so it can be scheduled separately from the broad Rust matrix. It fingerprints the validator before preflight, requires the current 84-test expected count, checks deterministic test output and exact template-report equality, and archives the evidence. Its latest run is queued; source-level hardening remains not runtime-verified until that job executes.

The isolated T0 evidence workflow now checks both the v2 fail-closed manifest validator (85 expected tests) and the source-bounded orbital-reference checker (66 expected checks). The latter adds a 600 km circular-orbit reference anchored to Birch Part I p. 476; the source-level report labels it illustrative and it does not populate the failure scenario. The current checker source fingerprint is Git blob `a7c39906e9f5cf34711630e1813e4b4f67527eef`, SHA-256 `c4eabd9a2e02f246d7d337926b9a8f0e422f1eced879d1ed57c2c26b1ffc98c2`. Schema v3 adds the specific mechanical-energy identity and has 66 expected checks, not yet executed; see [the historical geometry reference](ORBITAL_RING_HISTORICAL_GEOMETRY_REFERENCE.md) and [verification status](evidence/orbital_ring_validator_verification_status.json).

An additional regression makes declared-state inconsistency fail closed too: when a fully populated fixture declares `Blocked` instead of the computed review-ready state, the validation report records the schema/contract failure and the expected gate is forcibly reported as `Blocked`. The expanded 85-check suite remains unexecuted pending the isolated workflow.
