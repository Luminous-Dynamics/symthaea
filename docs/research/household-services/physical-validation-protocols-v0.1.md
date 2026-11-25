# Verified Household Services — first physical-validation protocol set

Version 0.1 · 4 November 2026

**Status:** preregistration template and campaign-design proposal. Not a test standard, safety procedure, product specification or authorization to build/install equipment.

## Purpose

Move the first three programmes from synthetic software evidence to controlled, independently reviewable physical experiments:

1. **InstantClean:** independently accepted dishware service under matched loads and total-resource accounting.
2. **WaterLoop:** end-use-qualified water actually delivered at named household fixtures through declared supply and power disturbances.
3. **HomeRenew:** correct, safe, requalified restoration of useful function, with recurrence tracked against operating exposure.

The programmes share an evidence format but not physical acceptance criteria. A dishwasher endpoint cannot qualify water quality; a water-delivery test cannot qualify a repair; and none can infer another product's safety.

## 1. Common pre-registration contract

No confirmatory physical result should be collected until the following fields are frozen.

### Identity and scope
- Protocol ID and revision.
- Product/programme and exact functional unit.
- Device model, market variant, serial number, hardware revision, critical component revisions and firmware/software digest.
- Installation/site, lab, operator/technician IDs and dates.
- Named comparator models, exact revisions, programme/settings and manufacturer-approved accessories.
- Intended use, exclusions, duty profile and qualified operating envelope.

### Inputs and conditioning
- Exact load/task manifest and physical item identities.
- Consumables, detergent/filter/repair-part supplier, catalogue ID, batch/lot and revision.
- Water source, hardness, conductivity, temperature, pressure and any relevant source chemistry.
- Ambient temperature/humidity and electrical supply state.
- Preconditioning, storage, cleaning and soil-ageing instructions.
- Randomization seed and run order.

### Metrology

For every observation, record:
- Measurand and unit.
- Value and timestamp/window.
- Sensor/instrument manufacturer, model, serial, range and revision.
- Calibration certificate reference, validity interval and as-found/as-left status.
- Measurement location/boundary and topology drawing revision.
- Resolution, accuracy/specification and uncertainty budget.
- Blank, reference, recovery and drift checks as applicable.
- Missingness, saturation, stale state and any manual intervention.

A calibration flag or hash string does not establish calibration suitability. The referenced certificate, method and uncertainty budget must be independently reviewed.

### Outcomes and disposition
- Preregistered primary endpoint and hard acceptance limits.
- Separately defined secondary endpoints.
- Independent assessor and assay/method revision.
- Uncertainty-aware pass / fail / indeterminate rule.
- All attempts, retries, aborts, rewash/rework, interventions and excluded records.
- Exact reason for each abstention or unresolved result.
- Data archive and code digests, with raw and derived data retained.

Never remove a failed attempt from the service denominator. Exclusions must be preregistered, justified and reported.

## 2. InstantClean — matched physical service campaign

### Standards and incumbent set

Use the licensed corrected IEC 60436:2025 edition, including COR1:2026, for applicable household dishwasher performance methods. It includes a unit-variation method, multi-compartment method and updated low-power/network-standby measurement. IEC 60436 is a performance measurement method, not a safety standard or minimum-performance rule.

Official references:
- https://webstore.iec.ch/en/publication/80459
- https://webstore.iec.ch/en/publication/112799

For microbiological relative reduction, use the separate IEC TS 63331:2024 path; do not describe it as sanitization, disinfection or sterilization:
- https://webstore.iec.ch/en/publication/66647

Any sanitizing claim requires the distinct applicable certification/qualification path, such as NSF/ANSI 184 where relevant. Do not infer sanitization from temperature, camera or cleaning scores:
- https://www.nsf.org/consumer-resources/articles/dishwasher-certification

Comparator candidates should include:
- A premium mainstream household unit with an exact manufacturer Eco programme.
- A South African market model with a frozen local-market variant and documented cycle data.
- A multi-compartment/drawer product if the same functional unit and load/task can be fairly matched.

Current research anchors include Miele G 7655 SCVi XXL AutoDos (14 place settings; 0.54 kWh and 8.4 L on Eco, 6 L on Automatic) and Beko DIN28428 South Africa (14 place settings; 0.93 kWh and 9.5 L published). These are source-ledger starting points, not substitutes for exact model procurement and matched measurement. Programme numbers must never be mixed.

The announced Miele G8 Diamond is a roadmap comparator only at this snapshot; the manufacturer states a Germany/Austria/Switzerland launch in April 2027.

### Functional units and load strata

Report at least two denominators:
1. **Per independently accepted item:** dish, cup, utensil or other predefined object that passes every required endpoint.
2. **Per standardized full/partial load:** frozen load manifest and exact rack/compartment arrangement.

Use:
- Standardized reference items and soil preparation under the selected licensed method.
- A separately defined household-realistic item mix.
- Occupancy strata chosen before testing (for example 1, 4, 8, 10 and rated-capacity loads).
- Difficult geometry/material/soil strata represented separately.

Do not count rated place settings as accepted items. A rated-capacity load counts only if every required item outcome is accepted; partial-load results must not be extrapolated to full-load capacity.

### Independent physical endpoints

At minimum, maintain distinct endpoint families:
- Residual soil removal (soil-specific, surface-specific).
- Final-rinse chemical residue (bulk and item-surface as relevant).
- Cross-transfer to clean witness items.
- Drying/remaining moisture.
- Materials/surface damage or unacceptable alteration.
- Relative microbiological reduction only when separately planned and qualified.

No single camera, turbidity, conductivity, time, temperature or process signal is ground truth. The 2026 fluorescence study for oatmeal/porridge residue on rice bowls is an assay lead with a narrow validity envelope, not a universal endpoint method:
https://doi.org/10.1016/j.foodres.2026.118853

For each assay, establish blanks, positive/negative controls, matrix interference, recovery, detection capability, repeatability, uncertainty and an explicit valid soil/surface range before using it to label training or confirmation runs.

Blind outcome assessors to candidate and comparator identity where practical.

### Whole-service resource accounting

Use a common system boundary from user request to final accepted-item availability. Count:
- Wall electricity, including standby/networked standby and preheat.
- Upstream hot-water energy if an external hot-water inlet is used.
- Fresh water at machine inlet.
- Detergent/consumable mass actually dispensed.
- Cycle duration, queue/wait, unload/reload and operator time.
- Purge, drain losses, cleaning/maintenance, filter changes and all rewashes.
- Thermal-storage losses and shared utilities across concurrent compartments.

Do not allocate shared pump/heater energy by arbitrary per-cycle averages. Meter the common service window and report how allocation to item/task was derived.

Primary service metrics:
- Accepted items per elapsed wall-clock hour.
- Time to first independently accepted item.
- First-pass accepted load fraction.
- Rewash/rework fraction.
- kWh, L and consumable mass per accepted item.
- Final-rinse residue and cross-transfer distributions.
- Noise and operator time.
- Unit-to-unit variation and maintenance burden.

Hard gates: every required cleaning/rinse/cross-transfer/material acceptance constraint must pass. Energy, water or duration cannot compensate for a failed quality/safety endpoint.

### Pilot and confirmation sequence

**A0 — metrology feasibility:** reference assay capability, instrument calibration, boundary audit and repeatability; no superiority claim.

**A1 — within-unit pilot:** same frozen unit, operator, batch and protocol; estimate run-to-run and assay variance. Use for method refinement only.

**A2 — model selection/development:** compare deterministic baseline, physical-model controller and adaptive/AI controller. Ablate each added capability. Do not use the locked confirmation set.

**A3 — locked matched confirmation:** freeze candidate software, hardware, process, assay, thresholds and analysis plan. Randomize run order and blind assessors. Compare to incumbent on same load, programme class, water and detergent references.

**A4 — independent replication and production variation:** new laboratory/operator and production-representative unit sample. Analyse unit, lab, day, batch and operator effects rather than pooling them away.

No fixed universal sample count is prescribed here. Use the A1 variance components to preregister confirmatory power/precision, independent-unit definition, clustering and multiplicity correction. A dish/image/frame is not an independent machine-cycle trial.

## 3. WaterLoop — household water continuity campaign

### Reference installation and comparison

Compare separately specified system configurations under the same frozen supply and demand profile:
1. Conventional municipal-backup tank and pressure-controlled pump.
2. Tank/pump plus independently specified leak monitoring/isolation.
3. WaterLoop candidate with the same required end-use service.

JoJo already publishes an automatic municipal fill, tank, pump and pressure-controller backup arrangement. Moen Flo already markets pressure/flow/temperature monitoring, AI-assisted usage learning, micro-leak checks and automatic shutoff. The benchmark must not treat either underlying feature set as novel.

Reference sources:
- https://www.jojo.co.za/water-storage-solutions/municipal-backup-water-storage-solutions/
- https://flo.moen.com/pages/flo-smart-water-monitor

Freeze the reference installation's tank, pump, float/control, pipework, check/backflow components, isolation valves, power/backup, bypass and outlet configuration. Installation and commissioning must be handled by appropriately qualified personnel under applicable local requirements.

### Functional unit and measurements

The primary unit is **litres of a specifically qualified end-use service delivered at a named household fixture while meeting its declared minimum pressure and flow envelope**.

Measure at relevant points:
- Municipal inlet flow and pressure over time.
- Tank level using at least one independent reference method.
- Usable tank volume at the declared delivery boundary.
- Outlet flow and pressure at each named fixture.
- Pump state/current/energy and grid/backup-power state.
- Losses, overflow, purge, leakage and any recirculation.
- Water-quality samples and method per intended end use.
- Maintenance, filter/media age and time since source/last quality check.

Do not call gross tank volume “available water” without a functioning delivery path. Do not treat tank level as water quality. Do not infer water flow from valve-open feedback or a refill command.

### Disturbance matrix

Include separately specified, repeatable profiles:
- Stable municipal supply.
- Zero municipal flow for 6, 24 and 48 hours.
- Low pressure and intermittent supply.
- Staggered restoration with pressure surges.
- Grid failure with and without independently qualified backup power.
- Pump unavailable.
- Stuck/stale level sensor and conflicting independent reference.
- Small continuous unexplained flow and controlled leak injection on a non-hazardous test rig.
- Stale/missing quality evidence.
- Recovery after outage and source switching.

Use actual time-series flow and pressure where possible. Categorical labels such as “unstable recovery” are not adequate to calculate service duration or stability.

### Primary and secondary metrics

Primary:
- Pressure-qualified litres delivered per fixture.
- Pressure-qualified service hours under an observed essential-demand profile.
- Time to first restored qualified service after source/power return.

Secondary:
- Leak detection latency, missed events and false alarms.
- False shutoff frequency and isolation success on a controlled test rig.
- Pump/standby energy per qualified delivered litre.
- Overflow, water loss, maintenance and filter cost.
- Uncertainty of usable reserve and actual delivery.
- Quality-assay coverage by end-use class.

Do not set a universal leak shutoff threshold in this protocol. A live automatic shutoff is a separate safety/installation qualification and must not be enabled by a software benchmark result.

### Water-quality and reuse boundaries

Define a separate acceptance method for each named use. Potable/cooking, bathing, toilet flushing, irrigation and cleaning are not one interchangeable “water quality” class.

The first WaterLoop pilot should not assume kitchen or dishwasher wastewater is suitable for its initial reuse profile. Treatment performance, microbial hazards, cross-connections, backflow protection and compliance require independent domain qualification. This benchmark does not authorize potable reuse or automatic source switching.

## 4. HomeRenew — repair and restoration campaign

### Initial safe task set

Start with de-energized, depressurized, low-risk service/bench tasks using documented isolation and qualified supervision. Begin with replaceable non-critical filter/cartridge housings, seals on a dedicated test fixture, simple mechanical assemblies and textile repair/maintenance tasks.

Mains electrical, hot-water, pressure-bearing, chemical, hygiene-critical or safety-interlock tasks require the product-specific safe work procedure, qualified personnel and approved test equipment. The synthetic repair corpus is not a work instruction.

### Exact task definition

For each task freeze:
- Fault seed and diagnostic truth (hidden from technician where appropriate).
- Product/device/revision and initial condition.
- Allowed tools, manual/diagnostic access and technician skill profile.
- Exact spare identity, revision, supplier, local stock status, quoted lead time and price.
- Task start/stop events and allowable interventions.
- Affected interfaces and mandatory post-repair functional/safety checks.
- Follow-up duty profile and recurrence definition.

A catalogue listing is not local stock. A mechanically fitting spare is not a qualified replacement. A repair manual is not independent diagnostic access. A successful restart is not restored service.

### Primary metrics

- Correct fault identification rate.
- Time to diagnosis and qualified restoration (median, P90).
- First-time qualified restoration fraction.
- Part fill rate and delivered lead time.
- Total repair cost and service downtime.
- Post-repair functional and product-specific safety qualification.
- Recurrence rate per operating hour or service cycles, with exposure and censoring.
- Number and cost of referrals/holds when safe test capability is unavailable.

Track both calendar time and operating exposure. “No recurrence observed” is valid only over the stated follow-up interval and exposure. A 30-day interval must not be called lifetime durability.

### Repair arms

Compare:
1. Human technician with existing documentation and same instruments.
2. Human + Symthaea diagnostic support using the same instruments and documentation access.
3. Where applicable, independent repairer using vendor tools/access.
4. Candidate local/offline service workflow.

The AI/helper system must not become the sole authority for safety state, fault truth, part compatibility or post-repair qualification. Measure whether each proposed observation or diagnostic action changes the decision and reduces total restoration cost without increasing unsafe errors or false diagnosis.

## 5. Cross-cutting study design and statistics

### Cluster and variance model

The independent unit depends on the claim:
- InstantClean throughput/false acceptance: physical run or load under the frozen protocol, not item image count.
- WaterLoop availability: defined outage/demand episode per independently configured installation.
- HomeRenew recurrence: repaired unit/task exposed to a defined duty profile, with right-censoring.

Record crossed/nested factors: device/unit, production lot, component supplier, site/lab, operator, day/session, consumable/soil/water batch and household. Use a justified mixed-effects/hierarchical model or robust alternative. Report variance components, confidence intervals, exclusions, missingness and worst-performing strata.

A zero-false-acceptance calculation such as the one-sided 95% bound 1 - 0.05^(1/n) applies only to independent Bernoulli trials. It cannot be justified by counting correlated dishes/images/measurements separately. Determine confirmatory sample sizes from pilot variance, the intended false-acceptance risk, stratum structure and multiplicity.

### Locked evidence stages

- E1: within-unit repeatability only.
- E2: independent-laboratory reproducibility.
- E3: production-unit variation.
- E4: real-household validity.

A result is not promoted from E1 to E2 by adding more repeats at the same lab. A successful bench campaign is not household validity. Household pilots do not automatically represent all climates, users or regions.

### Stop / pivot gates

Stop or redesign if:
- Independent assay recovery/uncertainty is inadequate.
- A required quality or safety endpoint fails.
- The result depends on a hidden retry or omitted preparation/maintenance resource.
- A common-mode failure defeats a claimed resilience path.
- An unqualified substitute or cloud-only service access blocks repair.
- Between-unit variation removes the prototype advantage.
- Lifecycle or operator costs erase the service benefit.

Proceed only when the next evidence grade is preregistered and the required test resources, safety authority and independent assessors are available.

## 6. Data package and reproducibility

Each campaign should publish, subject to safety/security/privacy review:
- Preregistration manifest and frozen protocol.
- Comparator configuration/firmware and exact device revision ledger.
- Per-run raw measurements and derived observations.
- Instrument/calibration/uncertainty records.
- Independent assay reports and blind assessor key after lock.
- Randomization and analysis source.
- Every attempt, retry, intervention, failure, repair and censoring event.
- Machine-readable service receipts, dispositions and unresolved claims.
- Exact input, code, analysis and output digests.
- A claims matrix separating synthetic, modeled, bench, inter-lab, production and household evidence.

Personal household data should be minimized, consented and de-identified. Publish enough metadata for independent review without exposing personal information or physical security details.

## 7. Decision rule

The first physical campaign should not ask “did we beat the industry?” in one overall score.

It should ask:
1. Did the candidate satisfy every mandatory quality and safety constraint?
2. Did it deliver the same or better independently accepted service under a matched workload?
3. Did total resource use, time, cost, rework and maintenance improve?
4. Did the improvement survive uncertainty, unit variation and external replication?
5. Is the gain still present in realistic household use?

Only then claim a competitive advantage—and state exactly which product, programme, endpoint, population and evidence grade support it.
