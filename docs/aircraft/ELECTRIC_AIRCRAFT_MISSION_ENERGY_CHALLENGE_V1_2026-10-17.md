# Electric-Aircraft Mission-Energy Challenge and Qualification Contract

**Date:** 2026-10-17  
**Status:** proposed benchmark contract; synthetic/reference scaffold only  
**Tracking:** [AIRROAD-ELEC-001 #7805](https://github.com/Luminous-Dynamics/symthaea/issues/7805)  
**Depends on:** [AIRROAD-003 #7729](https://github.com/Luminous-Dynamics/symthaea/issues/7729), [mass-coupled sizing coordinator PR #7779](https://github.com/Luminous-Dynamics/symthaea/pull/7779), [energy screening PR #7647](https://github.com/Luminous-Dynamics/symthaea/pull/7647), and [decision-value benchmark #7803 / draft PR #7804](https://github.com/Luminous-Dynamics/symthaea/pull/7804).

## 1. The paradigm shift: mission capability, not a single battery number

An electric-aircraft battery must not be represented only by nominal kWh or cell-level Wh/kg. A candidate's usable capability depends on its actual phase-by-phase bus-power demand, state of charge, voltage/current limits, discharge rate, pack architecture, thermal behavior, installation mass, temperature, degradation, and reserve-phase power. Changing installed mass/configuration must regenerate that candidate's mission demand.

The shared engineering question is:

> Under a declared mission family and a bounded evidence model, which energy-system candidates can meet every required phase and reserve condition, what assumptions dominate that conclusion, and which next measurement could change the decision?

This contract defines a benchmark and evidence schema around existing aircraft-energy work. It does not introduce a new optimizer or claim a validated aircraft performance model.

## 2. Research grounding and implications

The following sources support the problem formulation:

- NASA, *A Parametric Battery Model for the Conceptual Design of Electric Aircraft* (NTRS 20240007599): high-level specific-energy/specific-power parameters can behave as a black box because both depend on state of charge and discharge rate. Its abstract reports that inaccessible charge can depend on late-mission power requirements. https://ntrs.nasa.gov/citations/20240007599
- NASA, *Battery Evaluation Profiles for X-57 and Future Urban Electric Aircraft* (NTRS 20205004497): nominal cell-level performance and fixed discharge-efficiency assumptions can misrepresent aircraft-usable energy; thermal and installed-system mass can offset apparent efficiency gains. https://ntrs.nasa.gov/citations/20205004497
- NASA, *Battery Cell-to-Pack Scaling Laws for Electric Aircraft* (NTRS 20210009584): cell-to-pack specific-energy scaling is not linear and varies with chemistry, architecture, safety, thermal and operating requirements. https://ntrs.nasa.gov/citations/20210009584
- Guibert et al., *Optimizing Electric Aircraft Batteries: Tackling Flight and Material Uncertainties With Robust Designs Using a Multifidelity Approach* (2026 AIAA paper): relevant recent work on robust battery optimization under flight/material uncertainty. https://doi.org/10.2514/6.2026-4780
- *Multi-Mission Multidisciplinary Design Optimization of Hybrid-Electric Aircraft Under Probabilistic Operational Scenarios* (published online 28 August 2026): its results report that mission distributions can reshape battery mass and airframe configuration as electrification increases. This is for a hybrid-electric regional-aircraft study, not direct evidence for a specific four-seat electric design. https://doi.org/10.1016/j.ast.2026.113655

The literature establishes that these couplings matter; it does not prove any Symthaea model or proposed aircraft configuration is feasible. Published research results must not be copied as aircraft-specific coefficients without checking applicability and provenance.

## 3. Layers of the challenge

### E0 — Deterministic synthetic oracle

Start with small, hand-calculable cases and an independent oracle, separate from the production sizing coordinator. This qualifies arithmetic and contracts before using any real-world model.

The fixture schema must identify:

- stable challenge and candidate IDs;
- source/configuration digest and exact model version;
- units and reference conditions;
- mission phases in a fixed, declared order;
- each phase's duration, nominal bus power, lower/upper demand bounds, and peak/continuous classification;
- reserve and terminal-phase requirements;
- pack-level usable energy and power assumptions, current/voltage boundaries and installed mass;
- temperature, degradation and end-of-life semantics where applicable;
- hard constraints and their reference values;
- known-answer outputs, tolerances and independently computed oracle version;
- terminal disposition: feasible within model, infeasible, indeterminate, evaluation failed, or did not converge.

Every synthetic parameter must be labelled synthetic. Do not silently substitute nominal values for absent battery, aerodynamic or propulsion data.

### E1 — Adversarial energy-and-power cases

At minimum, the suite must contain:

1. **Energy-limited:** required usable energy is the active constraint; all power limits have known margin.
2. **Peak-power-limited:** total nominal kWh appears sufficient, but a late-mission power demand violates voltage/current or peak-power limits.
3. **Continuous-power-limited:** a sustained segment violates continuous power or thermal limits despite short-time peak capability.
4. **Peak-duration:** rated peak power exists, but not for the demanded duration.
5. **Reserve-maneuver-limited:** nominal mission completes, yet the declared reserve/terminal maneuver cannot be completed.
6. **Mass-feedback:** adding installed battery mass changes aircraft demand; a deliberately fixed-profile comparator must disagree with the coupled reference case.
7. **Usable-energy vs nominal-energy:** cell-level advertised energy is insufficient as a proxy for pack- and mission-usable energy.
8. **Cold/hot conditions:** declared synthetic derating changes energy, voltage, power, thermal margin or feasible status.
9. **End-of-life:** capacity fade and resistance growth are separate declared variables; do not model both as the same simple percentage loss unless the fixture explicitly defines that abstraction.
10. **Mission family:** several missions share a single installed battery. It is not permissible to resize the battery independently for each mission and then report those candidates as one aircraft.
11. **Common-mode uncertainty:** one uncertain environmental or model parameter affects multiple phases consistently.
12. **No feasible candidate:** no candidate satisfies the hard constraints; the correct result is an empty feasible set, not a winner.
13. **Non-convergence:** a sizing iteration hits its iteration bound or fails its closure criterion; the disposition must not be feasible.
14. **Bad evidence:** missing maps, mismatched digests, invalid units, reversed bounds, out-of-domain inputs, and missing reserve assumptions fail closed.

### E2 — Mission-family robustness

Define an explicit family of missions representative of intended operations: nominal, range/route variants, payload variants, weather/temperature variations, diversion/reserve, and aged/cold/hot battery scenarios. Probabilities may be used only when their source and calibration status are declared. If only a finite set or interval bounds are available, report set-based outcomes rather than fabricated expected values.

Report separately:

- per-mission feasibility and active limiting constraint;
- joint mission-family feasibility for one installed design;
- energy, peak-power and thermal margins;
- maximum take-off mass and mass-closure residual;
- candidate ordering and frontier membership changes across scenarios;
- which assumptions alter the decision, not merely which ones have the largest numerical variance.

Do not collapse mission probability, physical safety, operating cost, passenger preference and environmental impact into an unnamed scalar score.

### E3 — Decision-value benchmark

Connect to #7803 / PR #7804 to compare how policies choose the next evaluation. Use identical initial evidence and budgets, paired seeds, held-out scenarios and an evaluator whose hidden truth is unavailable to the policy until it commits to an action.

Compare at least:
- seeded random/space-filling selection;
- selection by the largest parameter uncertainty;
- energy-only sensitivity ranking;
- fixed low-/high-fidelity allocation;
- decision-aware selection aimed at reducing regret or resolving a specific feasibility ambiguity, when the applicable probability or uncertainty-set semantics are explicit.

An experiment's value should be assessed by its effect on feasibility, regret, frontier relationships or selection stability. High uncertainty in an irrelevant parameter should not automatically receive priority. If no available measurement could change the decision, the benchmark should permit a zero-value/abstention response.

## 4. Independent analytical fixture for mass coupling

A small synthetic fixture may use the following intentionally simplified relationship to provide a hand-checkable oracle. It is not an aircraft physics law.

Let:
- m be installed pack mass in kg;
- E0 be the non-pack portion of mission energy in kWh;
- beta be the declared synthetic increment of mission energy per kg of installed pack, in kWh/kg;
- epsilon be pack-level nominal specific energy in kWh/kg;
- u be a declared usable fraction, strictly between 0 and 1.

Define mission energy E(m) = E0 + beta * m and pack mass relation m = E(m) / (u * epsilon).

When u * epsilon > beta, the unique positive solution is:

    m* = E0 / (u * epsilon - beta)

When u * epsilon <= beta, this simple model has no finite positive fixed point and the expected synthetic disposition is infeasible/non-convergent according to the fixture contract. The production coordinator must not turn a numerical iteration stop into a valid mass closure.

This fixture should be extended with a separate peak-power constraint and, where appropriate, voltage/current limits. Keep the energy equation and peak-power oracle separate so that a result cannot pass by satisfying only one. Use exact rational or high-precision arithmetic in the test oracle where practical, then explicitly declare floating-point tolerances for the Rust result.

## 5. Evidence and model boundaries

Each model/data input must declare:

- artifact locator and immutable revision/digest;
- units or explicit non-applicability;
- reference conditions and applicability domain;
- uncertainty representation and its meaning;
- known limitations and omitted physics;
- independent validation status and source;
- model implementation/configuration identity.

A digest match establishes identity, not truth. A published reference dataset is not the same as an independently validated physics model, which is not the same as physical flight evidence. Never claim airworthiness or physical safety from a synthetic challenge pass.

Candidate-specific aerodynamic/propulsion demand generation remains owned by #7729 and its verified successor. The mass-coupled coordinator remains the orchestration layer in PR #7779. The benchmark should call those public APIs where available; do not copy their production implementation into a second crate. The robust-frontier module reports relations under supplied intervals; it does not create or validate the intervals.

## 6. Proposed acceptance thresholds

These are targets to verify, not measured results:

| Gate | Proposed target |
|---|---|
| Deterministic sizing oracle | 100% agreement on independently known-answer fixtures within declared numerical tolerance |
| False feasibility | Zero false-feasible outputs on preregistered synthetic cases |
| Mass-coupling negative control | 100% detection of the deliberately incorrect fixed-profile result in the mass-feedback fixtures |
| Required constraint accounting | 100% explicit disposition for energy, continuous power, peak power/duration, reserve, mass, thermal/temperature and convergence when applicable |
| Missing/invalid evidence | Zero silent defaults; all preregistered malformed/out-of-domain inputs fail closed |
| Repeatability | Identical report for identical pinned inputs/configuration under the declared deterministic execution contract |
| Mission-family consistency | One installed configuration is evaluated against every required mission; per-mission resizing cannot count as a family pass |
| Decision-aware selection | Beats uncertainty-only and random baselines on preregistered decision-boundary cases under equal evaluation budget, with paired effect sizes and uncertainty reported |
| No-value negative control | No fabricated positive decision value when all available experiments are irrelevant |
| Physical credibility | Separate held-out validation and calibrated uncertainty evidence; never inferred from synthetic tests |

For real reference aircraft, report numerical error by quantity and operating condition. Do not set a universal percentage accuracy gate before identifying the reference dataset, measurement uncertainty, model validity domain and intended use. A single benchmark fit must not be achieved by tuning the model to the one reference aircraft and then treating that agreement as independent validation.

## 7. Execution order

1. Freeze E0 fixture schema and hand-calculated known answers.
2. Add energy-only, power-limited, reserve-limited, coupled-mass and non-convergence cases.
3. Compare the independent oracle to the mass-coupled coordinator at exact branch heads.
4. Add battery mission-profile adapters only for source data with traceable provenance and applicability limits.
5. Run fixed-budget decision-selection baselines and preserve per-case receipts.
6. Validate candidate-specific models on independent references; only later add qualified physical evidence.

## 8. Out of scope

This contract does not establish a production battery electrochemistry model, flight-control law, aircraft performance prediction, battery safety case, design qualification, certification, or airworthiness. All proposed aircraft results remain conceptual until their model, evidence, verification, validation and approval gates are independently satisfied.
