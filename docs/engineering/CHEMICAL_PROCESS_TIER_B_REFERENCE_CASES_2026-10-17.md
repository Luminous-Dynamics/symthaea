# Tier-B Chemical Process Reference Cases: Candidate Inventory and Freeze Protocol

**Status:** Candidate-reference inventory; reference outputs are not yet frozen and no simulator run is claimed.  
**Date:** 2026-10-17  
**Parent work:** [CHEM-BENCH-002 #7869](https://github.com/Luminous-Dynamics/symthaea/issues/7869) and [Tier-A benchmark PR #7794](https://github.com/Luminous-Dynamics/symthaea/pull/7794).  
**Repository baseline inspected for this document:** `main@1fd1cfe8311253dbd98fbe1cf881167245c6c56e`.  
**Scope:** Public documentation and reference-case selection only. No hazardous synthesis recipes, plant-control sequences, or physical operating authority.

## Why Tier B must be different from Tier A

Tier A checks the correctness of the benchmark contract using small synthetic fixtures. Tier B asks whether a declared process model agrees with independently sourced reference data or a separate engineering implementation under a *matched context of use*.

Do not conflate these propositions:

```text
synthetic contract correct
!= numerical implementation correct
!= same-model independent reproduction
!= validated against a physical observation
!= qualified for an exact engineering decision
!= plant or product qualified
```

Tier B should contain actual external-example inputs, solver/data versions, and reference outputs frozen from an executed and inspectable source. Documentation examples may provide reasonable starting inputs, but approximate values copied from tutorial prose must not become exact oracle values.

## Candidate B1 — Water mixer mass/composition closure

**Source:** DWSIM official tutorial, [Mixer Basics](https://dwsim.org/tutorials/en/beginner/02-mixer-basics.html).

**Use:** A simple, benign, steady-state material-balance case. This is a strong first external case because inlet stream inputs and the tutorial's reported outlet are explicit.

### Source-reported configuration

- Compound: water.
- Property package: Steam Tables (IAPWS-IF97).
- Inlet 1: 300 K, 101325 Pa, mass flow 100 kg/s.
- Inlet 2: 348 K, 101325 Pa, mass flow 50 kg/s.
- Tutorial-reported outlet: approximately 316 K, 101325 Pa, 150 kg/s, vapor fraction 0.

### What can be frozen immediately

Only exact arithmetic invariants independent of a simulator build:

- Total outlet mass flow must be 150 kg/s for the stated non-reactive mixer.
- Total mass residual is computed independently as inlet sum minus outlet.
- The case cannot assert plant safety, utility adequacy, water quality, or equipment qualification.

### What must await a pinned simulation

- Exact outlet temperature and vapor fraction under the exact installed DWSIM version, property-package version, pressure conventions and numeric tolerances.
- Any solver-specific convergence status and output digest.

The tutorial labels its approximately 316 K outlet as approximate; do not use 316.000 K as an exact golden number. Save raw outputs and exact versions first, then freeze a justified tolerance. A run against a different property package or version is a different evidence context.

## Candidate B2 — Heater/cooler energy balance and sign conventions

**Source:** DWSIM official tutorial, [Heater and Cooler](https://dwsim.org/tutorials/en/beginner/03-heater-cooler.html).

**Use:** A useful second case because it exposes energy accounting, enthalpy-reference semantics, duty sign conventions and conservation across sequential unit operations.

### Source-reported configuration

- Compound: water.
- Property package: Steam Tables (IAPWS-IF97).
- Feed: 298.15 K, 101325 Pa, mass flow 1 kg/s.
- Heater target outlet: 353.15 K.
- Cooler target outlet: 298.15 K.
- Pressure drop: 0.
- Efficiency: 100%.
- Tutorial-reported heater duty: about +230 kW; cooler duty: about -230 kW using the tutorial's sign convention.

### Required assertions

- Material flow is preserved by both units under the stated configuration.
- Duty is independently checked from the matched inlet/outlet enthalpy states using the declared convention: `Q = m_dot * (h_out - h_in)`.
- Across heater then cooler, net heat exchange returns near zero under identical feed/property assumptions; do not assume duties are precisely +230/-230 kW.
- Enthalpy reference states, property-package identity and pressure conditions must be identical or explicitly reconciled before comparing.

The tutorial's duties are approximate reference hints, not exact constants. Freeze numerical expectations only after actually executing the selected DWSIM version and independently calculating the heat balance.

## Candidate B3 — Water/ethanol flash separation

**Source:** DWSIM official tutorial, [Simple Flash Drum](https://dwsim.org/tutorials/en/beginner/04-simple-flash-drum.html).

**Use:** A multi-outlet separation case that exercises phase split accounting and the distinction between recovery, enrichment, and product purity.

### Source-reported configuration

- Feed compounds: water and ethanol.
- Property package: NRTL.
- Feed: 300 K, 101325 Pa, 1 kg/s, 50/50 mass fractions.
- Heater outlet target: 360 K, zero pressure drop, 100% efficiency.
- Flash drum receives the heated stream.

### Tutorial-reported qualitative/range expectations

- Total vapor flow around 0.4–0.8 kg/s.
- Total liquid flow around 0.2–0.6 kg/s.
- Vapor ethanol mass fraction above 0.50.
- Liquid ethanol mass fraction below 0.50.
- Total mass conserved at 1 kg/s.

These are candidate sanity bounds from the tutorial, not a precise quantitative benchmark. Bounds alone are insufficient to establish NRTL accuracy, product purity at an industrial specification, separation recovery, or process economics.

### Required freeze fields

Exact DWSIM version; NRTL configuration and interaction parameter source; component identity and order; basis of composition (mass vs mole fraction); phase convention; flash/heater model settings; exact input file and transitive input hashes; raw outlet stream composition, temperature, pressure, phase and flow; convergence statuses; independent total/component mass balance; numerical tolerance provenance.

Do not silently substitute a different NRTL parameter set. If parameter source/currentness cannot be established, mark the relevant result unresolved or reference-limited instead of laundering it into a pass.

## Candidate B4 — Dynamic reactor-network reference (Cantera)

**Source:** Cantera official [Reactors and Reactor Networks](https://cantera.org/stable/reference/reactors/index.html) and official [Python example index](https://cantera.org/stable/examples/python/index.html).

**Purpose:** Represent transient integration, steady-state criteria, stiffness and solver configuration, instead of testing only steady-state unit-operation balances.

**State:** Candidate family selected; exact example/input is not frozen by this document.

Before admission to the frozen corpus, select one public example with:
- a source-controlled exact mechanism/data file and license;
- a documented reaction/network configuration and initial/boundary conditions;
- declared independent variable and units (time or distance);
- selected integrator and tolerances;
- a reference output or independent analytical/numerical comparator with known scope;
- expected event/steady-state criteria and convergence diagnostics;
- captured Cantera version, mechanism-file hash and transitive input digests.

Do not compare Cantera reactor dynamics against a DWSIM steady-state unit operation as if they modeled the same equations. If no independent reference value exists, classify the case as an adapter/trajectory-reproducibility test, not a model-accuracy test. A solver returning success is not evidence of physical validation.

## Machine-readable candidate registry

The candidate set is also recorded in [`CHEMICAL_PROCESS_TIER_B_CANDIDATES_V1.json`](fixtures/CHEMICAL_PROCESS_TIER_B_CANDIDATES_V1.json). The registry stores case IDs, source URLs, input conditions, analytical invariants, tutorial-only hints, freeze blockers, claim ceilings and hard-gated acceptance targets.

Registry invariants:
- All five candidates explicitly report `NOT_RUN`; the file does not imply DWSIM or Cantera was executed.
- Only the water mixer’s exact total mass-flow sum is frozen as an analytical invariant (150 kg/s); state variables remain unfrozen.
- Approximate heater/cooler duties and broad flash-drum ranges are tagged as non-gold hints.
- The Cantera case stays design-only until a specific example and independent comparator are selected.
- A paired common-mode challenge makes dependency independence testable rather than a self-declared Boolean.
- Missing outputs, skipped runs, and reference unavailability cannot count as PASS.

This registry has been read back from the branch and structurally checked for parseability, unique IDs, source references, valid result statuses, explicit freeze blockers and non-frozen simulator outputs. Those are registry-integrity checks—not scientific validation or a simulator run.

## Freeze protocol for all Tier-B cases

The owner implementing each case must archive these items before setting gold outputs:

1. **Primary source:** official URL, retrieved date, source version/revision, and exact citation/page or section.
2. **Input package:** exact native project/input files, embedded and transitive data, hashes, license/provenance, canonical species/component order and units.
3. **Model context:** property package, equation-of-state or activity model, interaction parameter set, reference states, phase assumptions, boundary/initial conditions, unit-operation assumptions and limitations.
4. **Execution:** solver/build version, OS/runtime, numerical methods, tolerances, convergence and termination status, deterministic settings where available, stdout/stderr and full raw outputs.
5. **Independent check:** independently derived conservation equations or matched reference comparison; identify shared data/models to detect correlated error.
6. **Acceptance:** tolerances preregistered from source uncertainty and numerical requirements; exact expected result; permitted deviation; failed/unsupported domains.
7. **Receipt:** source commit, case/corpus version, input digest set, solver/data versions, output digests, checker version, result and claim ceiling.

A user-readable tutorial screenshot, a copied approximate number or a successful solver exit is not a complete reference receipt.

## Common-mode independence challenge

A separate benchmark should intentionally vary the *evidence dependency graph* while keeping a prediction comparison superficially the same:

- Case A: solver 1 and solver 2 use distinct implementations and independently sourced reference data.
- Case B: solver 1 and solver 2 have different solvers but consume the same property package/correlation or an identical stale data file.

The checker must report that Case B has a shared evidence dependency; numerical agreement does not establish independent corroboration. Use input hashes and dependency manifests rather than labels such as `independent=true`.

## Initial benchmarks and proposed targets

### Data/integrity target

- 100% of frozen Tier-B cases capture exact transitive input/data hashes and version context.
- 100% of cases state whether reference and candidate share property data, correlation, parameters or code.
- Zero missing, skipped, failed or unsupported reference executions are silently counted as PASS.

### Numerical agreement target

- For each case, use case-specific normalized mass, component, energy or state residuals.
- Choose and freeze numerical acceptance tolerances *before* running the candidate implementation. Tolerances must account for the actual reference precision, solver error and model context; do not impose one universal (10^{-8}) threshold on thermodynamic models.
- At least three matched-condition cases should pass reproducibly before adding more complex flowsheet optimization tests.
- Report absolute and normalized error, convergence state and output uncertainty. No aggregate score can mask one failed hard invariant.

### Scientific validity target

- A DWSIM-to-DWSIM version replay is reproducibility evidence, not independent validation.
- Cross-simulator agreement is only counted when equations, inputs, property methods and quantity conventions genuinely match.
- A comparison whose independent reference is absent remains `REFERENCE_UNAVAILABLE` or `UNRESOLVED`, never a pass.
- Physical validity requires relevant observations and VVUQ under a declared context of use; Tier-B numerical comparisons alone do not qualify a real plant or product.

## Suggested implementation sequence

1. Start with B1. Freeze a DWSIM input project and run version only after the operator actually produces the artifacts and output logs. Until then only exact mass balance is analytically checkable.
2. Add B2 as the first energy balance; establish enthalpy convention and duty sign semantics before freezing expected numeric duties.
3. Add B3 only after confirming the NRTL parameter source and archiving raw outputs.
4. Select B4 from Cantera's public examples after pinning its mechanism and independent comparator.
5. Build the common-mode dependency challenge around immutable hashes.
6. Route results through Tier-A receipt generation and the evaluator, extending the schema explicitly rather than creating a second report format.

## Claim ceiling

This document identifies candidate public reference cases and a freeze protocol. It does not report a DWSIM or Cantera run; it does not freeze approximate tutorial outputs as gold values; it does not establish model accuracy, reaction feasibility, yield, purity, safety, economics, process qualification, regulatory compliance or plant operation authority.

## References

1. DWSIM, [Mixer Basics](https://dwsim.org/tutorials/en/beginner/02-mixer-basics.html).
2. DWSIM, [Heater and Cooler](https://dwsim.org/tutorials/en/beginner/03-heater-cooler.html).
3. DWSIM, [Simple Flash Drum](https://dwsim.org/tutorials/en/beginner/04-simple-flash-drum.html).
4. DWSIM, [Streams, State and Balances](https://dwsim.org/tutorials/en/fundamentals/01-streams-state-and-balances.html) and [Recycles and Convergence](https://dwsim.org/tutorials/en/fundamentals/03-recycles-and-convergence.html).
5. Cantera, [Reactors and Reactor Networks](https://cantera.org/stable/reference/reactors/index.html) and [Examples](https://cantera.org/stable/examples/python/index.html).
6. NIST, [Chemistry WebBook SRD 69](https://webbook.nist.gov/chemistry/).
7. NIST, [On Methods of Uncertainty Quantification in Computational Modeling and Simulation: Harmonizing Modeling and Metrology Practices](https://doi.org/10.6028/NIST.AMS.100-85), October 2026.
8. NIST, [Digital Twins Workshops Summary Report (NISTIR 8620)](https://doi.org/10.6028/NIST.IR.8620), July 2026.
