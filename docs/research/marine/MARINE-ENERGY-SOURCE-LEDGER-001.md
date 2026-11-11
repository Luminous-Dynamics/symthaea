# MARINE-ENERGY-SOURCE-LEDGER-001
## Deterministic source-input accounting and conversion-flow anti-double-counting

**Status:** research specification v0.1.0  
**Created:** 10 November 2026  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Related:** #11430 trial evidence; #11469 regulatory routes; #11501 WAPS energy MRV.

> This is a research accounting contract, not an adopted statutory energy formula, MRV verifier decision or legal compliance report. Its input boundary is explicit and versioned; a future regulation may define a different boundary.

## 1. Paradigm shift: account for energy-source inputs and conversion flows as a graph

A ship's energy record is not a flat list of all measured power/energy values. A generator consumes fuel and produces electrical energy; a battery receives charge and later discharges; a motor converts electricity to shaft energy; propulsion produces useful work. Summing every meter reading double-counts energy that crosses multiple measurement boundaries.

This programme therefore models the energy system as a directed provenance graph:
- **Primary inputs:** fuel energy consumed by onboard fuel consumers.
- **Substitute inputs:** shore electricity, renewable energy generated onboard and wind-assistance energy when determined under a declared method.
- **Conversion outputs:** generator output, battery discharge, motor/shaft output.
- **Conversion losses:** heat, conversion and distribution losses.

For the declared source-input boundary, only primary and substitute inputs contribute to the research source-energy total. Conversion outputs and losses remain linked evidence but are not added again. Every conversion output must identify its parent input(s). This prevents counting fuel at a generator inlet and the generator's electrical output as two independent sources.

## 2. Internal numeric representation

The kernel uses integer micro-gigajoules:
- 1 micro-GJ = 10^-6 GJ = 1,000 joules = 1 kJ.
- Energy input values are non-negative integers.
- Missing/unknown input is an explicit null plus reason, never zero.
- Totals use checked integer addition; overflow is invalid.
- No floating-point accumulation occurs in the source ledger.

Conversion from calibrated fuel mass/LHV, electrical meters or a future WAPS method happens upstream in a separately versioned calculation and must preserve the original measurement, conversion factor, units, uncertainty and provenance. The ledger does not decide those methods.

## 3. Flow-role contract

| Source type | Flow role | Included in source-input total |
|---|---|---|
| Fuel consumed by a main/auxiliary/boiler consumer | primary_input | Yes |
| Shore electricity | substitute_input | Yes |
| Onboard renewable electricity | substitute_input | Yes |
| Wind-assistance energy | substitute_input | Yes, only when determined under the declared method |
| Generator electrical output | conversion_output | No |
| Battery discharge | conversion_output | No |
| Propulsion shaft output | conversion_output | No |
| Conversion/distribution loss | loss | No |

The graph also binds source IDs, measurement IDs, consumer boundaries, parent-source IDs, method IDs and evidence-source IDs. Unknown source inputs make the computed total incomplete. They are not replaced with zero.

## 4. WAPS and regulatory boundary

COM(2026)620 is a pending proposal. A wind-assistance entry may remain unknown until its applicable standard/certification method exists. The source ledger is independent of that legal status: it can still record the WAPS device, evidence and unknown method state.

The ledger's research total must not be described as an EU MRV/FuelEU statutory figure. The future law and implementing acts may specify exact boundaries, source treatment, factors and validation. The v0.1 basis is labelled research_source_input_boundary_v1.

Research wind contribution and counterfactual avoided fuel remain separate from the regulatory wind-energy amount under #11501.

## 5. Falsification and known-answer controls

Required controls:
1. Fuel + shore power + known wind input sum exactly in integer micro-GJ.
2. Generator output, battery discharge and shaft output are not added to source-input totals.
3. An unknown primary/substitute source produces an incomplete total with known subtotal and unresolved source ID.
4. Unknown energy is not silently treated as zero.
5. Duplicate source or measurement IDs are invalid.
6. Primary/substitute rows cannot have parents; conversion outputs/losses must have parent source IDs.
7. Missing parent references and dependency cycles are invalid.
8. A source type/flow-role mismatch is invalid.
9. Overflow is invalid and cannot wrap.
10. Replay from the same record is deterministic.

## 6. Benchmarks and claim ceiling

Research targets:
- 100% exact integer agreement on known-answer input totals.
- 0 double-counted conversion outputs or losses.
- 0 unknown-to-zero conversions.
- 100% parent-reference resolution for conversion flows.
- 0 dependency cycles.
- 100% measurement/evidence/method identity for all included input rows.

A source-ledger total is only as trustworthy as its input measurement and conversion method. It does not prove physical energy truth, causal savings, WAPS certification, statutory MRV/FuelEU compliance or vessel approval.
