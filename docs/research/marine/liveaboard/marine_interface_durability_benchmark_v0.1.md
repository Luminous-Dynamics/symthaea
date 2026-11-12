# Marine service-interface durability and post-exposure serviceability — V0.1

**Status:** research profile and proposed laboratory pilot; no specimens have been exposed or physically tested.  
**Parent:** [MARINE-INTERFACE-DURABILITY-001 #11097](https://github.com/Luminous-Dynamics/symthaea/issues/11097)  
**Related:** [MARINE-SERVICEABILITY-001 #11049](https://github.com/Luminous-Dynamics/symthaea/issues/11049), [LOS18-DEMONSTRATOR-001 #11093](https://github.com/Luminous-Dynamics/symthaea/pull/11093)

## 1. Question

Does a marine service interface remain inspectable, safely operable, extractable, replaceable and requalifiable after its declared environmental exposure and repeated maintenance history?

New-condition modularity is not sufficient. Salt, wet/dry cycles, vibration, thermal cycles, UV where relevant, galvanic interaction, contamination, seal ageing and repeated mating/fastening can change removal force, sealing, electrical contact, access and reassembly quality.

This is a specimen- and method-bounded test plan. It does not predict years at sea.

## 2. Reuse owners; do not fork the engineering stack

- Symthaea #11049 owns marine service-task/access/return-to-service semantics.
- Symthaea #11064 owns instrument identity, calibration, time synchronization, raw observations and campaign binding.
- Symthaea #4863 owns generic robotics degradation/reliability/maintainability profiles, which can be reused for degradation-state vocabulary.
- Symtropy #3331 owns component identity, material/inspection lineage and as-built revision.
- Symtropy #2048 owns design-for-manufacture and lifecycle interface considerations.
- Symtropy #4056 owns generic maintenance task phases and return-to-service semantics.
- Symthaea #11093 owns the staged LOS-18 physical demonstrator architecture.

This work owns only marine interface exposure/cycle profiles, matched specimen design, post-exposure serviceability observables and the evidence integration. It does not create a second failure-rate model, maintenance planner, geometry engine, corrosion-protection authority, connector standard, DAQ system or approval ledger.

## 3. Interfaces under test

Start with safe, coupon-scale or inert fixtures:
1. mechanical slide/carrier rails, stops and retention;
2. fastener and locking interfaces, including representative material pairs;
3. hatch/access-panel hinges, latches, seals and drainage features;
4. de-energized electrical connector surrogates for mating mechanics and seal/access testing;
5. non-pressurized fluid-coupling surrogates for access and replaceable-seal mechanics.

Record the complete interface stack: material substrates and coatings, geometry and tolerance, fasteners/locking, seals, drains, connectors/couplings, access, inspection and maintenance instructions. Do not infer universal interchangeability from a common connector shape.

Live batteries, energized high-voltage connectors, propulsion, fuel, oil, refrigerant, pressurized systems, seawater intakes and primary structural load paths are outside the first laboratory pilot. Any later powered/pressurized/load-bearing trial requires separate product-specific engineering and safety approval.

## 4. Environmental and maintenance exposures

Choose each exposure from the actual product/environment specification; do not combine arbitrary cycles into an invented universal marine profile:
- cyclic salt mist / salt-laden atmosphere;
- wet/dry, condensation and humidity;
- temperature/thermal cycling;
- vibration and shock;
- UV exposure for relevant polymer, seal and coating materials;
- ingress/water exposure for the declared installation location;
- galvanic/material-pair exposure and coating damage;
- dust, sand, biological fouling and residues only where relevant;
- mating/separation, opening/closing, fastening and extraction cycles;
- documented cleaning, inspection, lubrication and seal-renewal intervals.

The exact sequence, severity, duration, cycle count, sample size and acceptance rule come from the applicable product specification, environmental classification, manufacturer and competent test plan. Accelerated lab exposure is not a universal conversion to calendar years or sea hours.

## 5. Standards and use boundaries

- IEC 60068-2-52:2017 describes cyclic salt-mist environmental testing for components/equipment designed for salt-laden atmospheres. Choose the method/severity from an applicable product requirement and document the whole sequence: https://webstore.iec.ch/en/publication/30839
- ISO 9227:2022 specifies neutral, acetic-acid and copper-accelerated salt-spray methods. Its official description says these methods are not intended as a universal ranking of different materials or a predictor of long-term corrosion resistance. Salt-spray chamber hours must not be translated into years at sea: https://www.iso.org/standard/81744.html
- ISO 9227:2022/Amd 1:2024 is a published amendment; the applicable edition/amendments must be checked when the test protocol is frozen: https://www.iso.org/standard/87301.html
- IEC 60512-9-3:2011 defines connector mechanical-operation endurance testing (engaging/separating cycles) with an electrical load. Use only where the connector family and declared conditions fit the method's scope: https://webstore.iec.ch/en/publication/2412
- DNV describes maritime approval, inspection and technical support for corrosion-protection systems; it is an industry comparator and technical service, not a substitute for project-specific rules: https://www.dnv.com/services/certification-materials-components/materials-and-corrosion-protection/corrosion-protection-systems/

Salt-mist, salt-spray, ingress, connector endurance, vibration and thermal tests are distinct. One test's acceptance cannot be silently reused as acceptance for another.

## 6. Matched specimen protocol

1. Freeze interface identity, design revision, substrate, finish/coating, seal, fastener, connector/coupling, assembly process and instructions.
2. Measure initial dimensions at declared datums; record baseline extraction/actuation force or torque and applicable electrical/sealing measurements.
3. Record instruments, calibration, method, uncertainty, environmental conditions and raw artifacts.
4. Assign matched specimens to a declared unexposed control and exposure/cycle groups. Record randomization and specimen count.
5. Apply the selected exposure exactly as specified by the approved laboratory protocol.
6. Inspect using declared coverage and method; record corrosion, coating damage, ingress, looseness, distortion, wear, seizure and visible defects.
7. Conduct the same service task, with the same task completion criteria, tools and spare identity across interfaces.
8. Record phase-level time, tool changes, damage, assistance, disassembly depth and incomplete/blocked task outcomes.
9. Reassemble and repeat the applicable functional, electrical, sealing or mechanical checks under approved methods.
10. Preserve failed, damaged, reworked, censored, aborted and inconclusive specimens. Reserve separate specimens for destructive investigation and keep specimen lineage.

## 7. Observables

Report per specimen and exposure stratum:
- exact material/coating/seal/fastener/connector/coupling revisions;
- exposure sequence, chamber/fixture conditions and cycle counts;
- inspection method, observer/qualification where applicable and coverage;
- dimensional changes with datum, instrument and uncertainty;
- extraction/insertion/actuation force or torque;
- fastener seizure, damage, removal tool and replacement burden;
- contact resistance/continuity/insulation only on a suitably guarded specimen and approved electrical procedure;
- leak/ingress/seal performance only for a declared, rated test method and safe test boundary;
- cleaning, lubrication, seal/connector replacement and reassembly time;
- task completion, rework, errors, damaged parts and tool changes;
- post-maintenance test completion and authorized disposition;
- added protection mass/volume/cost and inspection burden;
- specimen attrition, missing data and uncertainty.

Report changes relative to each specimen's own baseline as well as the matched-control comparison. Do not pool different substrates or exposure sequences into one lifetime ranking.

## 8. Negative controls

- Missing material, specimen, interface revision or exposure history -> Indeterminate.
- Missing initial baseline or matched unexposed control -> no causal comparative claim.
- Unknown chamber conditions or unlogged exposure deviation -> no qualified exposure claim.
- Salt-spray hours converted to years of service -> invalid claim.
- Incompatible substrate comparison presented as a universal material ranking -> invalid.
- Seized/damaged connector, fastener or carrier hidden as successful service -> invalid.
- Failed post-service seal/electrical/mechanical check treated as restored -> invalid.
- Changed geometry, finish, seal, torque or material silently pooled with the original subject -> invalid.
- Missing calibration or measurement uncertainty -> affected observable is ineligible.
- Energized, pressurized or load-bearing specimen outside approved scope -> blocked.

## 9. Metrics and initial gates

Keep a metric vector:
- retained serviceability by exposure stratum;
- change in extraction/actuation force or torque;
- post-exposure seal/ingress or electrical performance;
- service phase time, tool changes and disassembly depth;
- seizure/damage/rework and specimen failure rates;
- inspection and protection-system burden;
- added mass/volume/cost;
- uncertainty, sample count and censored/incomplete outcomes.

Initial contract targets (not physical results):
- 100% admitted specimens have exact identity and exposure-history binding.
- 100% measurements carry method, units, instrument/calibration and uncertainty as applicable.
- Zero false pass on frozen seized/damaged/leaking/electrically out-of-limit/wrong-revision controls.
- All failed, aborted, blocked, reworked and inconclusive specimens remain in denominators and reports.
- No lifetime prediction is admitted from salt-spray/cyclic-mist hours alone.
- Numeric limits, sample sizes and exposure severities are fixed only after the actual interface and applicable product specification are selected.

## 10. Deliverables and integration

- Versioned marine interface-exposure profile.
- Specimen identity and pre/post measurement record composed with #11064 and #3331.
- Matched unexposed/exposed and serviced/unserviced corpus, including negative controls.
- Test-method applicability matrix with explicit claim ceilings.
- Laboratory pilot plan for inert/de-energized/non-pressurized coupons.
- Post-exposure serviceability report and Pareto comparison including protection burden.
- Exact findings fed back into #11049 service cases and the #11093 LOS-18 interface-control register.

## 11. Claim ceiling

A result qualifies only the exact tested specimen and declared exposure/test protocol. It does not establish years at sea, fleet reliability, universal corrosion resistance, suitability in every marine climate, structural integrity, electrical safety, pressure integrity, class/flag compliance or permission to service a vessel underway. Those claims require applicable product methods, sufficient samples and field exposure, exact configuration lineage, qualified professional review and competent authority decisions.
