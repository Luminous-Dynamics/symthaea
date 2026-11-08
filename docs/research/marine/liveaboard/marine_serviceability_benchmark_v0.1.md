# Marine at-sea serviceability and return-to-service benchmark — V0.1

**Status:** marine-specific benchmark profile and synthetic fixtures; no actual vessel geometry, service procedure, maintenance duration or safety approval is claimed.  
**Parent:** [MARINE-SERVICEABILITY-001 #11049](https://github.com/Luminous-Dynamics/symthaea/issues/11049)

## 1. Purpose

A module is not truly replaceable because it has a connector or can be removed in a CAD assembly. It is replaceable only when a permitted, qualified task can diagnose it, isolate it, access and extract it, replace it with the correct revision, restore all affected services, complete the required tests and receive authorized release.

The benchmark asks whether an LOS-18-class vessel design restores critical services with less time, fewer disassembly dependencies and lower logistics burden while preserving safety, structural/watertight/fire boundaries and exact configuration lineage.

## 2. Reuse owners; no parallel maintenance stack

- Symtropy [AERO-SERVICE-LOOP-001 #4056](https://github.com/Luminous-Dynamics/symtropy/issues/4056) owns the generic maintenance task, phase, parts, skill, time-to-repair and return-to-service contract. This benchmark supplies the marine context profile; it does not fork the evaluator.
- Symtropy [SPATIAL-DESIGN-001 #1766](https://github.com/Luminous-Dynamics/symtropy/issues/1766) owns generic task/space geometry and occupancy semantics. Consume exact realized geometry only when available and current.
- Symtropy [SHIPBUILD-CO-DESIGN-001 #2048](https://github.com/Luminous-Dynamics/symtropy/issues/2048) owns design-for-manufacture, shipyard build sequence, inspection and repair access at assembly/design time.
- Symtropy [COMPONENT-DIGITAL-THREAD-001 #3331](https://github.com/Luminous-Dynamics/symtropy/issues/3331) owns component identity, material/inspection lineage, as-built survey and repair history.
- Symtropy [SHIPYARD-SPACE-001 #2050](https://github.com/Luminous-Dynamics/symtropy/issues/2050) owns shipyard block staging/transport/access, not service access within an operating vessel.
- Symthaea #10994 owns measured watermaker and equipment-service performance evidence; #10967 owns task/location-specific motion/comfort; #10990 owns exact mission evidence joining.
- Mycelix remains the authorized custody/provenance/inspection receipt owner; signed records do not establish physical truth or professional approval.

## 3. Research grounding and limits

The 2026 shipyard production-planning review by Bordbar et al. finds hybrid discrete-event simulation + optimisation the most operationally credible methodological class under bounded conditions, while emphasizing the limited availability of open industrial datasets and weak longitudinal yard-wide KPI validation. Apply its evidence hierarchy—synthetic simulation, historical-data replay, shop-floor integration, sustained live KPI validation—rather than claiming superiority from a rich digital-twin architecture alone. https://doi.org/10.3390/jmse14040396

ISO 12215-7:2020 was confirmed current in 2026. Its scope covers relevant multihull craft up to the specified 24 m hull/load-line length boundary, including local loads/scantlings and structures supporting windows, hatches and doors. This is a reminder that access openings and service hatches must be integrated with qualified structural design; it is not a serviceability acceptance standard. https://www.iso.org/standard/73457.html

Fincantieri and Generative Bionics announced a four-year shipyard humanoid-welding partnership in February 2026, with initial yard tests planned by end-2026. This indicates industrial investment in robotics but is not evidence of certified autonomous ship production or a quantified productivity improvement. https://www.fincantieri.com/en/newsroom/press-releases/2026/fincantieri-and-generative-bionics-launch-an-industrial-partnership-for-the-development-of-a-humanoid-welding-robot-for-shipyards/

## 4. Marine-specific operating contexts

Keep these contexts distinct:
- **Afloat / underway:** default not permitted for invasive maintenance unless the exact task-specific procedure and operating envelope explicitly authorize it. The scheduler cannot invent permission.
- **Afloat / secured or sheltered:** vessel stationary/secured, with actual motion, heel, weather, access, shore service and lifting availability explicitly declared.
- **Alongside / drydock / service yard:** shore lifting, external access, tooling and power may be available only where documented in the case.

A manufacturer estimate, CAD clearance or synthetic task duration is not a measured maintenance outcome. Synthetic test fixtures must never be presented as actual LOS-18 geometry, real part specifications, authorized high-voltage procedures, or safe work instructions.

## 5. V0 equipment families

Start with six bounded service families:
1. watermaker prefilter/cartridge;
2. watermaker high-pressure pump;
3. seawater strainer / intake valve;
4. refrigeration or HVAC compressor;
5. inverter / DC-DC converter;
6. battery module.

The initial fixtures deliberately include feasible, blocked and indeterminate cases. High-voltage battery/inverter tasks are represented only as prerequisite/authority checks; this benchmark does not generate step-by-step hazardous maintenance procedures.

## 6. Serviceability evidence contract

Every case should bind:
- exact profile, configuration, mission, operating-case and condition digests;
- asset serial, part number, equipment revision, as-built reference, installed location and parent assembly;
- task trigger and diagnostic evidence;
- context (underway, secured afloat, alongside, drydock), permission and qualified environment envelope;
- required skill/authority/crew count and tooling;
- current approved task-procedure reference and safety preconditions supplied by the responsible owner;
- access path, aperture, obstructions, removable panels, hand/tool clearance, posture/crew access and extraction/lift envelope, all in a named coordinate frame;
- mass / centre-of-gravity interval and lifting/handling authority where relevant; this benchmark does not calculate stability;
- isolation / safe-state evidence for electrical, stored-energy, pressure, fluid, thermal, fire and flood hazards;
- fluid drainage, spill containment and residual disposition;
- exact spare identity/revision/effectivity, stock state, custody and substitution approval;
- separate phase time intervals: isolate, diagnose, prepare, access/disassemble, extract, replace/repair, reinstall, reconnect, test, update as-built/configuration and authorized return-to-service;
- inspection and post-maintenance tests, method/revision, instrument/calibration and acceptance authority;
- explicit evidence class, units, source, uncertainty, raw digest, missingness and currentness.

Unknown dimensions, mass, safety boundary, approved procedure or lifting capability remain Unknown/Unsupported—not inferred from geometry or a rendering.

## 7. Independent verification and dispositions

A future verifier must reconstruct task eligibility from independent case facts rather than trust a planner's `feasible=true` flag.

- `synthetic_task_feasible`: only the declared synthetic prerequisites and task geometry are internally consistent.
- `blocked`: a known prerequisite, permission, spare or access condition prevents task execution.
- `indeterminate`: a required critical fact is missing, stale, conflicting or outside the declared evidence boundary.
- `unsupported`: a physical/engineering semantic is outside the implemented verifier's declared scope.

Return-to-service is separately gated. A physically completed swap does not establish return-to-service until required post-maintenance tests, configuration updates and authorized sign-off are present. The only success state in this V0 corpus is `requirements_satisfied_synthetic_only`; it is not an actual release.

## 8. Negative controls

- Extraction only possible after removing an unrelated critical system -> not serviceable under the declared task.
- Access dimensions or as-built geometry stale/missing -> Indeterminate, not pass.
- No approved task procedure or permission for underway work -> Blocked.
- Missing seawater/flood isolation evidence before opening an intake path -> Blocked/Indeterminate, never executable.
- Missing HV isolation, competent authorization, qualified tooling or approved procedure -> Blocked; do not generate operating instructions.
- Wrong serial/revision or incompatible spare -> rejected/blocked.
- Unknown module mass/lift point/handling envelope -> Indeterminate/Unsupported.
- Repair without leak/insulation/functional/calibration test as required -> no return-to-service.
- Missing residual/spill handling path -> no environmental or service-completion claim.
- Estimated task durations never become measured MTTR.
- Changed component revision invalidates dependent as-built and mission evidence until requalified.
- A task may not claim underway/sea-state capability unless an approved task-specific envelope and evidence support it.

## 9. Metrics and fair baselines

Compare fixed-installation, replaceable-module and redundant/degraded-mode alternatives on identical mission, task list, environment and service requirements.

Report by task and context:
- diagnosis time, hands-on time, access/disassembly time, extraction/replacement time, test/recommission time and return-to-service time separately;
- crew size, authorized skill, technician-hours, tools and lifting aids;
- number of unrelated systems removed and disassembly depth;
- access/extraction clearance margin with geometry uncertainty;
- parts commonality, stock availability, substitute approval and replenishment lead time;
- service downtime versus hands-on repair time;
- repeat-failure or rework only from measured exposure data;
- added interface mass/volume, sealing/connector count, inspections and maintenance burden;
- blocked/failed/indeterminate share and worst operating context.

Never compare average task time when one design cannot perform the task in the required context. Never hide blocked cases by calculating MTTR only over successes.

Initial gates (targets, not results):
- zero false synthetic return-to-service dispositions on hard-negative controls;
- 100% required tasks include their safety/permission and post-maintenance evidence references;
- 100% task phases distinguish measured, estimated, synthetic and unknown;
- zero unknown extraction/handling facts converted to feasibility;
- 100% wrong-identity/revision/spare controls blocked;
- deterministic replay of the frozen synthetic corpus.

## 10. Physical evidence ladder

1. Requirements and approved procedure / equipment documentation.
2. Exact realized geometry and interface fit-up survey.
3. Human factors / access mock-up with correct representative envelopes.
4. Non-hazardous, low-voltage and non-pressurized module service trial.
5. Professionally supervised equipment-specific service trial.
6. On-vessel task validation under explicitly permitted conditions, with vessel-specific safety review.
7. Fleet/task-time evidence with sufficient exposure before reliability or MTTR claims.

No physical task should be attempted based on this synthetic benchmark alone. High-voltage, pressurized, propulsion, seawater-intake, fire/flood or structural boundary work requires the applicable approved procedure, qualified personnel and competent oversight.

## 11. Claim ceiling

This benchmark tests declared information, geometry, access and evidence semantics. It does not establish real MTTR, human-factor validity, stability during component removal, high-voltage safety, watertight/fire integrity, seaworthiness, class/flag approval or authority to perform maintenance.
