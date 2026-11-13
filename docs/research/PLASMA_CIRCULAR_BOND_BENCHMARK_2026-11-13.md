# PLASMA-CIRCULAR-BOND-001 — Service, controlled release and rebond qualification

**Date:** 2026-11-13  
**Issue:** [#11976](https://github.com/Luminous-Dynamics/symthaea/issues/11976)  
**Status:** S0 synthetic evidence contract; no physical qualification

## 1. Research thesis

The dominant manufacturing objective has traditionally been to create the strongest, most durable permanent bond. For many multi-material products, that is incomplete: the joint must also be repairable, separable or recyclable when the product reaches end of life.

This benchmark defines a **bond-and-release lifecycle window**:
1. the joint performs above its service floor during use;
2. it remains stable through declared aging and load conditions;
3. it releases when the specified trigger is deliberately applied;
4. the release does not damage protected substrates or neighboring joints;
5. recovered components can be inspected, repaired, recycled or rebonded with a new traceable process lineage.

The plasma-specific question is not whether plasma makes an adhesive debondable. It is whether plasma preparation is compatible with an existing debond-on-demand (DoD), reworkable or recyclable adhesive system and measurably improves the complete lifecycle relative to matched non-plasma preparation controls.

No new adhesive chemistry, process controller, generic optimizer, material registry or release authority is introduced.

## 2. Ownership and integration

- [PLASMA-SURFACE-001 #11795](https://github.com/Luminous-Dynamics/symthaea/issues/11795): material/adhesive profile and in-service bond acceptance.
- [PLASMA-LIFE-001 #11971](https://github.com/Luminous-Dynamics/symthaea/issues/11971): durability, aging, censoring and failure-mode evidence.
- [PLASMA-OED-001 #11907](https://github.com/Luminous-Dynamics/symthaea/issues/11907): experiment selection and decision value.
- [PLASMA-VM-001 #11847](https://github.com/Luminous-Dynamics/symthaea/issues/11847): telemetry-to-product outcome prediction.
- [PLASMA-REGION-TRUTH-001 #11952](https://github.com/Luminous-Dynamics/symthaea/issues/11952): true physical support and spatial measurement lineage.
- MAT-013 #4344 and MAT-INTERFACE-001 #6227: canonical specimen and interface lineage.

The adapter owns only the application decision contract: service, trigger/release, collateral and recovery/rebond evidence obligations. The shared owners retain their respective registries, estimators and evidence-receipt schemas.

## 3. Current research comparators

The technical opportunity is real but strongly chemistry- and stimulus-dependent.

### A. Chemical-triggered depolymerizable polyurethane hot-melt films (2026)

Kang et al., *Chemical Engineering Journal* (1 October 2026), DOI [10.1016/j.cej.2026.179322](https://doi.org/10.1016/j.cej.2026.179322), report low-tack PU hot-melt films with stimulus-responsive sacrificial units. Their studied system undergoes fluoride-triggered backbone depolymerization and room-temperature interfacial separation, with substrate recovery for plastics and fibre-reinforced composites.

The chemistry is specific to the reported polymer architecture and trigger. It is not evidence that ordinary polyurethane adhesives can be chemically debonded or that a plasma-prepared interface will preserve the trigger response.

### B. Dynamic EVA/HDPE interfacial vitrimer hot-melt adhesives (2026)

Liu et al., *Chemical Engineering Journal* 535 (1 May 2026), DOI [10.1016/j.cej.2026.175552](https://doi.org/10.1016/j.cej.2026.175552), report an interfacial dynamic epoxy–anhydride network in EVA/HDPE blends. Their paper reports steel lap-shear up to 4.4 MPa, thermal self-healing and thermal debonding/rebonding; the material retained more than 93% of properties after five reprocessing cycles in the reported tests.

This is a strong comparator for service performance plus dynamic repair/reprocessing. It is not a plasma-preparation result and must not be pooled with the chemically gated PU film.

### C. Thermal/resistive debonding of thermoplastic GFRP laminates (2025)

A 2025 *Journal of Advanced Joining Processes* study, DOI [10.1016/j.jajp.2025.100364](https://doi.org/10.1016/j.jajp.2025.100364), compares thermally expandable particles and localized resistive heating in MMA-bonded thermoplastic GFRP laminates. Localized heating was more effective in the studied specimens; TEP-only oven triggering had limited effect.

This is a comparator for spatially localized release, substrate compatibility and the importance of measuring the actual trigger field.

### D. Localized Joule heating for GFRP disassembly and remanufacturing (2026)

Chanda et al., *Composites Part A: Applied Science and Manufacturing* 210 (November 2026), DOI [10.1016/j.compositesa.2026.110073](https://doi.org/10.1016/j.compositesa.2026.110073), report a localized Joule-heating approach using carbon veils and thermally expandable particles in epoxy-bonded GFRP. Their abstract reports a 92.9% reduction in lap-shear strength after triggered debonding, fractography indicating localized adhesive degradation without adherend damage, and successful rebonding of recovered substrates with comparable joint performance.

These are the authors' results for their specific GFRP/epoxy system. The benchmark must compare actual residual substrate properties and rebond quality, not infer them from the debond percentage.

### E. Solvolytically debondable adhesive systems (2025)

Meschut et al., published online 15 December 2025, DOI [10.1080/00218464.2025.2600590](https://doi.org/10.1080/00218464.2025.2600590), discuss solvolytically debondable epoxy systems for repair and recycling and describe exploratory mechanical and thermal characterisation. This is a candidate chemistry family, not a universal solvent recipe or an assumption of industrial readiness.

These comparator papers demonstrate different release mechanisms and endpoints. Their numerical results are not directly rankable because the substrates, adhesives, joint geometries, triggers and service requirements differ.

## 4. Paradigm: qualify a lifecycle, not a single bond strength

### 4.1 Service performance

A DoD joint must first meet the same engineering service requirements as the permanent incumbent appropriate to the application. Include:
- initial and aged static strength distributions;
- fatigue or sustained-load behaviour when relevant;
- adhesive/interface/cohesive failure modes;
- creep and environmental retention;
- premature-release and accidental-trigger susceptibility;
- material/adhesive-lot and production-day transfer.

A weak joint that is easy to remove is not a successful controlled-release product.

### 4.2 Trigger selectivity

A trigger is qualified against a frozen envelope and target region. Measure:
- actual stimulus modality and source configuration;
- delivered energy/time/temperature or chemical exposure, as physically applicable;
- spatial extent and uncertainty of trigger delivery;
- target release success;
- neighboring protected-joint response;
- collateral damage or off-target degradation.

Thermal, Joule/induction, photothermal/UV, chemical/solvolytic and other triggers are different technologies. Compare only those admissible for the actual material and product. S0 does not prescribe trigger settings or authorize their use.

### 4.3 Substrate recovery

After release, inspect:
- adherend integrity and retained mechanical properties;
- dimensional change, cracking, delamination or thermal damage;
- adhesive residue and contamination;
- recoverable mass/area and separation purity;
- compatibility with the next repair, rebond or recycling process.

A visual separation is not proof of substrate recovery.

### 4.4 Rebond and repeat cycles

For every claimed reuse cycle, issue a new preparation, adhesive, cure, specimen and test lineage. Report:
- rebond strength distribution;
- service/aging retention after rebond;
- recovered substrate condition;
- process/adhesive additions or replacement;
- cycle-to-cycle variability and failure cause.

Do not count repeated measurements on one specimen as independent component recoveries.

## 5. Lifecycle state machine

The intended lifecycle is:

    MATERIAL_RECEIVED
        -> PREPARED (plasma or matched comparator)
        -> BONDED_AND_CURED
        -> SERVICE_QUALIFIED
        -> TRIGGER_AUTHORIZED_BY_EXTERNAL_PROCESS
        -> RELEASE_OBSERVED
        -> SUBSTRATE_RECOVERY_VERIFIED
        -> REPAIR / REBOND / RECYCLING_ROUTE
        -> NEXT_CYCLE_QUALIFICATION

A failed or unknown step remains FAILED, BLOCKED or UNKNOWN. It cannot be skipped to claim a complete circular lifecycle.

TRIGGER_AUTHORIZED_BY_EXTERNAL_PROCESS is an evidence state only. This benchmark has no equipment actuation authority.

## 6. S0 evidence contract

Every lifecycle receipt binds:
- canonical material/specimen/joint lineage (MAT-013);
- canonical pre/post interface states (MAT-INTERFACE);
- exact plasma run or conventional preparation and configuration digest;
- adhesive chemistry/product/lot and cure record;
- service test protocol, measured strength and acceptance floor;
- aging/load/environment history and test receipt;
- declared trigger modality and actual trigger modality;
- target and protected neighboring joint identities;
- trigger history, calibration and observation digest;
- release outcome, force/time and failure-mode evidence;
- substrate damage, residue and recovered-material evidence;
- rebond cycle number, new process/adhesive lineage and test result;
- independent specimen/lot/day groups;
- evidence class: synthetic, physical or inferred.

Spatial evidence must retain its true support under #11952. A specimen-level lap-shear result is not a local trigger map, and a single release observation does not establish a population release probability.

## 7. Frozen synthetic challenge benchmark

The fixture has 15 cases:

| ID | Expected | Required behaviour |
|---|---|---|
| PREMATURE_RELEASE | REJECTED | Joint releases before commanded trigger during service |
| SERVICE_STRENGTH_PASS | PASS | Service and aged strength exceed their frozen floors |
| TRIGGER_OUTSIDE_ENVELOPE | BLOCKED | Required stimulus exceeds declared envelope |
| SELECTIVE_RELEASE | PASS | Target releases and protected neighboring joints remain intact |
| COLLATERAL_RELEASE | REJECTED | An untriggered protected joint releases |
| SUBSTRATE_DAMAGE | REJECTED | Recovered substrate exceeds damage allowance |
| RESIDUE_UNQUALIFIED | BLOCKED | Residue/recovery evidence missing |
| REBOND_CYCLE_UNVERIFIED | BLOCKED | Rebond claim lacks fresh process/test lineage |
| REBOND_STRENGTH_DECAY | REJECTED | Rebond falls below the service-strength floor |
| TRIGGER_HISTORY_MISSING | BLOCKED | Trigger history/calibration evidence absent |
| PLASMA_COMPATIBILITY_UNTESTED | BLOCKED | Plasma/trigger chemistry compatibility has no matched evidence |
| WRONG_TRIGGER_MODALITY | REJECTED | Actual trigger differs from frozen adhesive-system trigger |
| MATERIAL_IDENTITY_MISMATCH | REJECTED | Released article identity does not match source specimen |
| MANUAL_FORCE_ONLY | REJECTED | Destructive force is mislabeled as qualified on-demand release |
| COMPLETE_SYNTHETIC | PASS | Internally consistent synthetic lifecycle, still not physical qualification |

Expected counts are 3 PASS, 5 BLOCKED and 7 REJECTED. The Rust evaluator and independent source-side replay must match every row before these are accepted as benchmark results.

## 8. Matched industrial baselines

For one exact material/adhesive/joint profile, compare:
1. permanent incumbent adhesive and best incumbent surface preparation;
2. same DoD adhesive without plasma;
3. same DoD adhesive with incumbent surface preparation (primer/corona/flame where appropriate);
4. same DoD adhesive with plasma preparation;
5. alternative trigger modalities only where admissible;
6. best applicable incumbent repair/rework/disassembly process.

Do not compare permanent and temporary adhesives using one unqualified strength target. Define the product service floor and lifecycle use case before choosing baselines.

## 9. Metrics and prospective targets

Report service, release and recovery separately:
- initial/aged strength distribution and fatigue;
- premature-release probability;
- target release probability and release force/time;
- trigger energy, temperature and chemical demand;
- selectivity and collateral-release rate;
- substrate damage and retained properties;
- residue/contamination and recovered-material purity;
- rebond strength and service retention by cycle;
- component recovery yield;
- full lifecycle cost, energy, waste, recovered value and time-to-repair.

Prospective goals (not standards or achieved results):
- zero premature-release events on known-truth negatives;
- zero collateral release on protected-joint controls;
- 100% lifecycle lineage from source specimen through release/rebond;
- at least 90% release success within the frozen trigger envelope on adequately powered physical trials;
- at least 90% of recovered substrates satisfy the predeclared preservation/reuse floor;
- no rebond-cycle service claim without independent physical tests;
- lifecycle cost compared at equivalent service and safety floors.

No composite scalar score may hide service performance, collateral release or substrate damage.

## 10. Physical promotion protocol

Before a laboratory study:
- freeze one material/adhesive/joint and intended service environment;
- select an existing DoD chemistry and approved trigger pathway with the supplier/facility;
- define service floor, trigger envelope, protected neighboring components and substrate preservation criteria;
- randomize/block specimens and preparation arms;
- measure service/aging before triggering;
- measure target and off-target trigger response and local substrate condition;
- quantify residue/recovery and fracture surfaces;
- carry new process lineage for every rebond cycle;
- reserve unseen specimens/lots/days for blinded confirmation;
- obtain all applicable chemical, electrical, thermal and facility safety reviews outside this software benchmark.

## 11. Claim ceiling

S0 validates only synthetic identity, phase-order, service/release/rebond evidence obligations and refusal semantics. It does not prove adhesive chemistry, safe trigger settings, product service life, selective release in real assemblies, recyclability, substrate preservation, life-cycle savings, standards compliance or physical actuation.
