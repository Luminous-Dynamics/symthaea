# FUS-ECON-001 — Independent Fusion Economics Qualification Protocol

**Status:** proposal for review; no production equation or default is certified by this document.  
**Scope:** public Symthaea repository, economics and reactor-architecture exploration.  
**Audit subject observed:** `main` at `b9af191f380d635f0459b24284c1d1fd6a5516aa` on 2026-10-15. File blob IDs observed during the audit: `economics.rs` `3039c7eeb6e50133aa7ef5b34f71f0a488ff1035`; `reactor_architectures.rs` `f8b17a590bad1216162274db70a8bb000085aff1`.

This is a verification protocol and a benchmark proposal, not a production-code correction, engineering qualification, or claim of commercial fusion viability.

## 1. Paradigm shift: qualify claims, not just numbers

A computed number should not be treated as trustworthy merely because the function returned a finite float, the optimizer converged, or a test passed. Keep these states separate:

1. **Implemented:** code computes a specified expression.
2. **Equation-checked:** the expression and units have been independently derived.
3. **Input-qualified:** empirical inputs have traceable sources, dates, units, regions, currency year and applicability.
4. **Verification-tested:** implementation matches independent known-answer and adversarial cases.
5. **Model-validated:** comparison with relevant observations establishes performance for a declared intended use.
6. **Decision-comparable:** system boundaries, financing, lifetime, output basis and evidence quality are aligned across compared designs.

A state may advance only when its stated evidence gate passes. Missing inputs are not zero; convergence is not validation; a cost estimate is not an observed commercial cost; and a Pareto optimum is only optimal within the modeled, declared design space.

### Minimum provenance contract for a scored quantity

Every quantity that can influence a comparison or ranking should bind:

- value, canonical unit and energy/output basis;
- whether it is a physical identity, sourced empirical datum, analyst scenario, placeholder, or unresolved/contradictory assumption;
- source URL, publication/data version and access date when applicable;
- jurisdiction, currency and currency year for economic inputs;
- model identity/version, subject configuration, applicable operating range and uncertainty representation;
- explicit include/exclude/missing disposition for cost and plant subsystems;
- resulting claim ceiling and exact upstream input/artifact hashes.

Ranking must be blocked when a required quantity is unknown, contradictory, outside applicability, or silently defaulted. The program may still display a diagnostic scenario, but that scenario must remain labelled as such.

## 2. Independent reaction-energy and stoichiometry fixtures

Freeze the basis before changing any default. For one MWh of reaction energy:

```text
N_reactions = 3.6e9 J / (Q_MeV × 1e6 eV/MeV × 1.602176634e-19 J/eV)
mass_i = N_reactions × isotope_mass_i × 1.66053906660e-27 kg/u
```

The reference values below use tabulated isotopic masses from NIST and reaction Q-values listed by IAEA. They are ideal *reactant-consumption lower bounds per MWh of fusion reaction energy (thermal basis)*. They are not fuel-cycle throughput, fuel injected into a plant, tritium breeding requirements, or mass per net-electric MWh.

| Reaction | Energy basis | Ideal reactants consumed per MWh(th) | Current helper value | Audit disposition |
|---|---|---:|---:|---|
| D–T, 17.6 MeV | One D and one T per reaction | D ≈ 4.270 mg; T ≈ 6.394 mg | D = 2.1 mg; T = 3.2 mg | Both defaults are about half of the ideal thermal lower bound; basis/derivation unresolved |
| D–³He, 18.35 MeV | One D and one ³He per reaction | D ≈ 4.095 mg; ³He ≈ 6.133 mg | D = 1.9 mg; ³He = 2.9 mg (stored as 0.0029 g) | Both defaults are about half of the ideal thermal lower bound; comment says “2.9 g” while code value equals 2.9 mg |
| D–D, 3.27 MeV branch | Two D nuclei per reaction | D total ≈ 45.963 mg | 11.5 mg | Inconsistent with the stated Q value and reactant count; investigate before use |

D–D has two principal branches, listed by IAEA at approximately 3.27 MeV and 4.03 MeV. The 45.963 mg figure is specifically based on 3.27 MeV; if a 50:50 branch mixture is approximated by its 3.65 MeV mean Q-value, the ideal consumption is ≈41.177 mg D/MWh(th). The helper's comment of **87,000 MWh/kg** also conflicts with the stated 3.27 MeV and two-deuteron basis, which imply approximately **21,756 MWh/kg**. Treat the exact branch model as a separate decision to freeze; do not replace this issue with a guessed single default.

### Energy-basis requirement

The existing LCOE path calculates annual MWh from `power_kw × capacity_factor × 8.76`, while `FuelCosts` documents consumption only as mass per MWh. That does not by itself establish whether the MWh is fusion-thermal, gross-electric, or net-electric.

Require explicit conversion for each pathway:

- **MWh(th):** reaction energy released under the declared reaction/branch assumptions;
- **MWh(gross electric):** after the declared thermal/direct-conversion model;
- **MWh(net electric):** after auxiliary and recirculating loads, and plant availability/downtime when using a lifetime basis.

At a declared thermal-to-gross-electric efficiency \(\eta\), the ideal reactant mass per gross-electric MWh is the thermal-basis amount divided by \(\eta\), before losses, burn fraction, recycling, tritium breeding/inventory, handling and recirculating power. Net-electric accounting needs the complete energy balance, not only this efficiency division.

### Tests for these fixtures

- Use a second implementation that does not call the production conversion/helper.
- Check dimensional algebra and fixed numerical fixtures for D–T and D–³He, and separate branch-specific plus declared branch-mixture D–D fixtures.
- Target relative error ≤1e-8 for deterministic arithmetic against the frozen full-precision fixture; round display values separately.
- Doubling target energy doubles reaction count and mass; doubling Q at fixed energy halves them; changing isotope mass changes only the corresponding mass calculation.
- Any changed reaction Q-value, branch mixture, isotope source/version, energy basis or conversion convention invalidates the affected fixture receipt.
- Test the electric-basis conversion against explicit efficiencies (including an invalid zero/non-finite efficiency). Keep ideal stoichiometry independent from plant fuel-cycle losses.

Sources: [IAEA World Fusion Outlook 2023, Table 1](https://www-pub.iaea.org/MTCD/Publications/PDF/FusionOutlook2023_web.pdf); [NIST atomic weights and isotopic compositions](https://pml.nist.gov/cgi-bin/Compositions/stand_alone.pl); [UKAEA PROCESS fusion reaction overview](https://ukaea.github.io/PROCESS/physics-models/fusion_reactions/plasma_reactions/).

## 3. Economic-equation audit and acceptance fixtures

### LCOE and annual energy

Declare whether power and output are gross or net electric, and use the same output basis for every numerator/denominator term.

- The capital-recovery factor (CRF) must be defined for the zero-discount limit: at (r=0), CRF = (1/n), not the current formula's indeterminate 0/0.
- Enforce finite, physically meaningful values and clear domain errors for capacity, capacity factor, lifetime, discount rate, component costs, overhaul interval and fuel use.
- Test each LCOE component's units as currency/MWh and make the denominator identical across annualized capital, O&M, and fuel components.
- Separate simple payback, discounted payback, NPV and IRR in result semantics and reporting. They answer different questions.
- Label `lifetime_cost_usd` as undiscounted and state every included/excluded category; do not imply it is a discounted lifecycle total.
- Do not represent omitted taxes, insurance, decommissioning, construction financing/interest during construction, contingency, degradation, replacements, downtime or fuel-cycle infrastructure as zero. Use explicit `included`, `excluded_by_scenario`, or `missing_evidence` dispositions.

### IRR

The present implementation's `annual_savings / capital.total() - discount_rate / 2.0` is an approximation, not a general solution of the cash-flow equation. Keep it out of claims labelled IRR until replaced or given a different name.

Reference tests must include:
- a unique-root cash flow with a known IRR and normalized NPV residual ≤1e-8 of initial investment;
- a zero-rate/zero-discount case;
- a no-root case with an explicit `no_root` disposition;
- a non-conventional cash-flow case with multiple roots, which must not return one silently selected value;
- cash-flow timelines with uneven construction, replacement and decommissioning years;
- fractional lifetime policy stated explicitly (rounding/truncation must not happen implicitly).

### Break-even search

Before binary search, validate finite inputs, ordered bounds, a valid target, and opposite-sign/bracketed residuals at endpoints. State when monotonicity is assumed and verify it for the supported parameterization. Use a typed result such as `bracketed_root`, `target_already_met`, `not_bracketed`, `invalid_input`, or `non_finite_evaluation`; a bare float is insufficient. Retain the exact evaluated endpoints and target residual in the result.

### Economic scenario provenance

Every default in `FuelCosts`, `CapitalCosts`, `OmCosts`, `EconomicEngine` and `EnergyComparison` must be classified. A hard-coded value is not a sourced market estimate merely because it has a comment. For cost scenarios require source/date, currency year, location/jurisdiction, system boundary, financing convention, technology maturity and uncertainty range. Keep conflicting deuterium-price assumptions in the reactor-architecture and economics helpers visible as distinct assumptions until evidence reconciles them.

## 4. Benchmark ladder and proposed quantitative gates

These gates are for Symthaea's evidence and software workflow. They are not a claim that an actual fusion plant meets a performance target.

| Gate | Benchmark | Proposed pass condition |
|---|---|---|
| B0 — arithmetic | Independent reaction-mass and unit fixtures | All declared known-answer fixtures pass at ≤1e-8 relative error; independent checker has no production-helper dependency |
| B1 — numeric robustness | NaN, infinities, negative/zero, overflow, singular CRF, zero denominators, invalid intervals and unbracketed roots | 100% yield typed error/status; no invalid value is presented as a qualified number |
| B2 — cash-flow semantics | Known unique-root, no-root, multiple-root and non-conventional cash flows | Correct disposition for every fixture; unique-root normalized NPV residual ≤1e-8; multiple roots are disclosed |
| B3 — provenance completeness | Every quantity consumed by an optimizer and every comparison input | 100% of decision-influencing values carry units, basis, evidence/scenario class and provenance; missing required evidence blocks qualified ranking |
| B4 — failure retention | Invalid, infeasible, unsupported, warning, non-converged and incomplete candidates | 100% receive a durable, explicit disposition and remain in the corpus; none is silently dropped or re-labelled as feasible |
| B5 — receipt integrity | Change source, model, input bytes, units, mapping, scenario or environment identity | 100% of such subject changes make prior receipts stale/rejected for current-subject claims |
| B6 — external baseline | Reproduce one official PROCESS v3.4.1 reference input from the exact pinned commit | Record exact tag+commit, license, inputs, environment, command, exit status, stdout/stderr, raw outputs and independent normalized manifest; set field-specific tolerances before first comparison |
| B7 — lifecycle uncertainty | Seeded sensitivity/Monte Carlo over lifetime, component replacement, repair duration, availability, recirculating power, financing and cost ranges | Publish median and stated interval/quantiles plus dominant sensitivity; retain assumptions and failures. Do not publish a single “best” LCOE without the scenario definition |
| B8 — optimizer value | Compare grid/random and one-factor-at-a-time baselines against Symthaea using the same exact-solver budget, fixed variables/ranges and held-out parameter strata | Report feasible-candidate yield, Pareto quality with frozen scaling/reference point, top-K recovery, solver calls, failures and uncertainty over ≥3 frozen seeds. A superiority claim requires a predeclared metric and a confidence interval that excludes no improvement |
| B9 — efficiency stretch | Same quality/feasibility threshold at reduced expensive-solver cost | Aim for ≥2× fewer exact-solver evaluations than the best simple baseline at comparable held-out candidate quality; treat this as an aspirational research target until measured |

For B8, an optional stretch goal is ≥10% improvement in a predeclared Pareto metric at equal solver budget. Do not choose or change the metric after seeing outcomes. If the baseline already performs as well or better, report that result and keep the simpler method.

### External reference baseline — exact candidate pinned

The official release tag `v3.4.1` currently resolves to PROCESS commit `04a8363725f26b59dda0f5aa1520e67aa7014fde` (root tree `154167c7c914407dc58ed3595f86b66b7822eac6`). Use the immutable commit SHA as the subject identity; keep the human-readable tag as metadata.

Recommended first case: the officially documented fixed-design evaluation input `examples/data/large_tokamak_eval_IN.DAT`, Git blob `464a9a84baf8b9a6dd3987725cd67c60c8f47b53`. The v3.4.1 official example `examples/single_model_evaluation.ex.py` initializes a `SingleRun` from this input and describes it as a no-optimization evaluation point. This is preferable as the first deterministic engineering baseline because it separates a fixed-input model run from optimizer behaviour.

Secondary case: `examples/data/large_tokamak_IN.DAT`, Git blob `b489fca8ab110655ae622d13a8d093dc152e962e`. The official introduction example documents running it with `process -i data/large_tokamak_IN.DAT`; include it as a later end-to-end scenario after the fixed evaluation case is stable. Do not mix the two cases into one baseline.

Source-pinned references:
- [PROCESS v3.4.1 release](https://github.com/ukaea/PROCESS/releases/tag/v3.4.1)
- [Pinned release commit](https://github.com/ukaea/PROCESS/commit/04a8363725f26b59dda0f5aa1520e67aa7014fde)
- [Documented evaluation input](https://github.com/ukaea/PROCESS/blob/v3.4.1/examples/data/large_tokamak_eval_IN.DAT)
- [Official single-model evaluation example](https://github.com/ukaea/PROCESS/blob/v3.4.1/examples/single_model_evaluation.ex.py)
- [Official introduction example](https://github.com/ukaea/PROCESS/blob/v3.4.1/examples/introduction.ex.py)

Before declaring reproduction, still capture the runner/OS, compiler/interpreter and lock/environment identity, install/build log, exact command, exit status, stdout/stderr, raw `MFILE.DAT`/`OUT.DAT` outputs and an independently recomputed normalized manifest. Record missing/unstable fields and define field-specific comparison rules before examining the actual output. The upstream Git blob ID pins source bytes, not the output or runtime environment.

PROCESS is a reference model, not an oracle; an exact reproduction establishes reproducibility of a model run, not physical validation.

Research supports integrated lifecycle modelling: the 2025 STEP concept study describes a trade-off in which a larger radial build raises capital cost but can lower through-life replacement costs by extending component life. A separate 2025 PROCESS-based availability sensitivity study shows strong coupling among component lifetime, maintenance duration, availability and LCOE. Those are exactly the non-local trade-offs the benchmark should expose, but their numerical results remain conditional on the selected concept and assumptions.

Sources: [Chapman (2025), *Developing Integrated Cost Models for Fusion Power Plants*](https://link.springer.com/article/10.1007/s10894-025-00515-1); [Foster, Knight & Muldrew (2025), *Component lifetimes and their impact on the availability of commercial fusion power plants*](https://doi.org/10.1016/j.fusengdes.2025.115231); [DOE Fusion Science and Technology Roadmap, finalized 9 June 2026](https://www.energy.gov/articles/energy-department-releases-finalized-fusion-science-and-technology-roadmap-accelerate).

## 5. System-level research hypotheses worth testing

These are hypotheses for controlled comparison, not facts to bake into a scoring function:

1. **Design for replaceability, not only peak performance.** Search component lifetime, access/maintenance strategy, shielding, and downtime together with initial machine size/cost. Record total installed and through-life replacement cost.
2. **Optimize net exported energy, not fusion gain or gross power in isolation.** Explicitly account for auxiliary heating/current drive, cryogenics, pumping, thermal conversion, startup/standby and availability.
3. **Treat uncertainty as a design variable.** Keep parameter uncertainty separate from scenario variability and numerical error. Show when the identity of the Pareto frontier changes across plausible scenarios.
4. **Make the next measurement an optimization output.** When two designs' ranking changes across unresolved inputs, identify the cheapest credible evidence that could discriminate them (for example a materials lifetime measurement or a maintenance-time estimate). This is value-of-information analysis, not permission to treat a predicted value as evidence.
5. **Compare specialist models only through qualified interfaces.** PROCESS, Bluemira, DESC and SIMSOPT have differing scopes; use each only for outputs and cases it supports. Do not create whole-plant values from missing specialist outputs or rank architectures at unequal fidelity.

A STEP component-lifetime study uses 80% availability as a target in specific modeled scenarios, with assumptions around neutron fluence and maintenance time. This can serve as a **scenario benchmark**, not a universal commercial-fusion pass line. The platform's immediate success criteria are methodological (correctness, completeness, reproducibility and honest claim limits); plant performance targets must remain scenario- and architecture-specific.

## 6. Execution order — do not wait on CI for analytical work

1. Freeze this audit subject, source citations, equations, units and expected fixtures.
2. Independently verify the fixtures with a small calculation external to the production helpers; store the full precision and derivation.
3. Build the finding matrix: issue, exact source location, affected output, consequence, confidence, proposed narrow repair and test.
4. Reproduce PROCESS's official supported example in a pinned local/research environment and capture raw artifacts. Do not wait for repository CI; use local/domain checks when available and label the exact checks executed.
5. Implement the narrowest correction on a feature branch with adversarial and metamorphic tests; inspect downstream consumers before changing public semantics.
6. Generate a machine-readable receipt and independently recompute it from the captured artifacts. A queued CI run is neither pass nor fail evidence; missing CI status stays unknown.
7. Only after B0–B6 pass, compare baseline search strategies under a frozen evaluation budget and test value-of-information/uncertainty-aware search.

## 7. Claim ceiling

Passing deterministic equation tests supports implementation correctness for the tested equations and units. It does **not** establish realistic fuel consumption, tritium self-sufficiency, blanket performance, a viable reactor, validated capital costs, commercially competitive LCOE, safety, licensing, or readiness for plant operation. Those claims require their own relevant evidence and review.

## Audit sources inspected

- Symthaea `crates/core/symthaea-core/src/physics/economics.rs` at the audit subject above.
- Symthaea `crates/core/symthaea-core/src/physics/reactor_architectures.rs` at the audit subject above.
- [FUS-ECON-001 issue #7512](https://github.com/Luminous-Dynamics/symthaea/issues/7512), [FUS-SYS-001 issue #7509](https://github.com/Luminous-Dynamics/symthaea/issues/7509), [FUS-OPS-001 issue #7510](https://github.com/Luminous-Dynamics/symthaea/issues/7510), and [FUS-ARCH-001 issue #7511](https://github.com/Luminous-Dynamics/symthaea/issues/7511).

**Current disposition:** preliminary independent calculations identify clear discrepancies; this document proposes verification gates. No production code was changed here, no formula is marked qualified, and no benchmark is reported as passed.
