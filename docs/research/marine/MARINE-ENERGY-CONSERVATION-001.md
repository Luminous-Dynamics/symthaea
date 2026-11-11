# MARINE-ENERGY-CONSERVATION-001
## First-law component and whole-system energy balance audit

**Status:** research specification v0.1.0  
**Created:** 11 November 2026  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Related:** source-input ledger (#11501), trial evidence (#11430), regulatory route accounting (#11469), gas-resolved evidence (#11532), implementation issue #11595

> This is an engineering accounting contract, not an adopted statutory energy formula, exergy analysis, MRV verifier decision, physical qualification or vessel performance claim.

## 1. Paradigm shift: close energy at every transformation boundary

The existing source-input ledger prevents double-counting by including only primary/substitute energy inputs and excluding generator output, battery discharge, shaft output and conversion losses from the source total. That is necessary but not sufficient: it does not prove the energy flows reconcile across the engine, generator, converter, battery, motor, shaft, propeller, WAPS, heat-recovery and hotel-load boundaries.

The new ledger audits each component over one aligned reporting window:

    Energy in - Energy out - Conversion losses - Increase in stored energy = Residual

A battery discharge is represented as a negative stored-energy change. A battery charge is a positive stored-energy change. The same internal transfer is outgoing from one component and incoming to another, so it cancels from the whole-system balance. Only the external boundary inputs and outputs remain in the system total.

The first implementation uses integer micro-GJ (one micro-GJ is one joule). It does not mix energy quantities with power, fuel mass, emissions, carbon-equivalent values or exergy.

## 2. System boundary and graph

A balance record includes a frozen vessel configuration digest and service fingerprint, one reporting window and timebase, components and their conversion losses/storage changes, directed internal transfers, external input/output flows, evidence-source IDs and per-component uncertainty budgets.

External inputs may include chemical fuel energy, shore electricity, onboard energy sources or wind-derived mechanical work when an applicable engineering method exists. External outputs may include useful propulsion work, hotel service or energy exported ashore. Wind-assistance counterfactual avoided fuel is not itself an energy flow and must not be inserted as measured work.

Component carriers include fuel chemical, shore/onboard electricity, shaft mechanical, propulsive mechanical, ambient wind mechanical and recovered thermal energy. All are expressed in the same first-law energy unit, while retaining carrier identity to avoid confusing energy quality or origin.

## 3. Node and system equations

For component i:

    R_i = sum(E_in,i) - sum(E_out,i) - L_i - DeltaS_i

where L_i is declared conversion loss and DeltaS_i is stored-energy increase (negative for discharge).

For the whole declared system:

    R_system = E_external,in - E_external,out - sum(L_i) - sum(DeltaS_i)

Internal transfers cancel. The auditor calculates both component and system residuals and checks them against supplied expanded-uncertainty budgets.

A residual within a declared budget is only BALANCED_WITHIN_DECLARED_UNCERTAINTY. It is not proof that meters, boundary definitions, calibrations or uncertainty propagation are correct. Engineering allowance is not accepted as metrology uncertainty and cannot create a qualifying pass.

Unknown flows/losses/storage changes make the balance INCOMPLETE, never zero. A residual outside the declared budget is UNBALANCED; malformed or unresolved flow graphs are INVALID.

## 4. Why use a separate first-law audit

The source-input ledger answers how much primary/substitute source energy crossed its declared source boundary without double-counting downstream conversion outputs. The conservation audit answers whether energy transfers, losses and storage changes reconcile at every component and system boundary.

They are complementary: source input totals support energy/carbon intensity and route economics; component balances identify missing or contradictory energy records; conversion losses expose where architecture/power-management changes may have leverage; storage deltas prevent battery charging/discharging from being mistaken for free generation; wind-assist work remains separate from counterfactual avoided fuel and statutory credits.

This is not exergy analysis: equal joules of fuel chemical energy, electricity, low-temperature heat and propulsive work do not have equal usefulness or carbon consequence.

## 5. Industry and standards crosswalk

- IEC 60092-501:2025 covers electric propulsion plant design, installation and testing, including generators/prime movers, batteries and other sources, switchboards, converters, propulsion motors, control/monitoring/safety and podded drives. The balance graph should preserve equipment/configuration boundaries. It does not itself provide this whole-vessel first-law accounting method. https://webstore.iec.ch/en/publication/92837
- US MARAD Energy Efficiency and Alternative Fuels Technical Guide (2024) describes series/parallel hybrid arrangements and battery charge/discharge from generators, shore power and renewable inputs. Useful architecture context, not a vessel-specific validation standard. https://www.maritime.dot.gov/sites/marad.dot.gov/files/2025-03/Energy%20Eff.%20and%20Alt.%20Fuels%20Technical%20Guide%20-%202024.pdf
- ISO 15016:2025 remains a speed/power trial-analysis reference within scope; it is not a complete component energy ledger. https://www.iso.org/standard/82080.html
- ISO/TS 25128:2026 defines a mechanical propulsion energy-consumption test for small craft but explicitly excludes wind, wind-assisted hybrid and wave propulsion. It must not be presented as a cargo WAPS validation standard. https://www.iso.org/standard/89049.html
- The source-input energy ledger remains the upstream source-boundary companion: MARINE-ENERGY-SOURCE-LEDGER-001.

## 6. Falsification suite and proposed benchmarks

Software/accounting targets (not physical performance targets): 100% exact integer agreement on known-answer component/system balances; zero internal-flow double-counting in whole-system totals; zero unknown-to-zero conversion; 100% endpoint/evidence-source resolution; zero false-balanced outcomes in adversarial fixtures; overflow invalid with no wraparound; deterministic replay.

Report for every component: inflow, outflow, loss, storage change, residual, expanded uncertainty, carrier mix and status. Report system external inputs/outputs, loss subtotal, storage delta, residual and combined tolerance. Never hide a bad component behind a system aggregate or vice versa.

A future physical research target is to reduce unexplained residual relative to an independently measured, uncertainty-qualified balance across the declared operating envelope. No universal percentage tolerance is claimed here; it must be established from metrology, test method, equipment and operating regime.

## 7. Claim ceiling

A successful audit means only that supplied first-law quantities close within supplied, referenced uncertainty budgets. It does not verify sensor authenticity, measurement calibration, uncertainty propagation, fuel LHV, physical model, exergy, emissions factors, safety or regulatory compliance. The synthetic fixture is software test data and never measured vessel evidence.