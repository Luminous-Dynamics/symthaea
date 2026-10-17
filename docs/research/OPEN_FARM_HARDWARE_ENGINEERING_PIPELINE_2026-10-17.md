# Open Farm Hardware: Engineering, Manufacturing, and Evidence Pipeline

**Status:** research/design contract; not a released machine design  
**Date:** 2026-10-17  
**Scope:** open-source farm tools, implements, equipment, and robotics platforms  
**Safety posture:** advisory/read-only until a task- and configuration-specific engineering case permits a supervised physical trial

## Thesis

Symthaea should help people engineer and locally manufacture useful farm tools and robotics platforms—but not by generating attractive CAD and calling it finished.

The goal is a reproducible design-to-field evidence pipeline: start from a farmer-defined service and current-practice baseline; inspect existing open designs; select a bounded design question; explore parameters; calculate material, manufacturing, service and lifecycle costs; prepare editable source, drawings and build instructions; define tests; record physical results; and publish a revision that independent builders can reproduce.

The unit of value is not a CAD model or a robot. It is a successfully completed, economically defensible, repairable farm service with traceable evidence.

## Existing work to compose, not replace

- Symthaea agricultural systems/robotics research matrix: [draft PR #7796](https://github.com/Luminous-Dynamics/symthaea/pull/7796).
- Evidence-bounded platform selection, including manual work, existing equipment and “no robot”: [issue #7797](https://github.com/Luminous-Dynamics/symthaea/issues/7797) and [draft benchmark kernel PR #7817](https://github.com/Luminous-Dynamics/symthaea/pull/7817).
- Task-specific operating domain and safety case: [issue #7818](https://github.com/Luminous-Dynamics/symthaea/issues/7818).
- Agricultural human-robot-interaction data and limits: [issue #7828](https://github.com/Luminous-Dynamics/symthaea/issues/7828).
- Symtropy owns value-of-information planning and bounded field-robot service economics: [#1661](https://github.com/Luminous-Dynamics/symtropy/issues/1661), [#1814](https://github.com/Luminous-Dynamics/symtropy/issues/1814), [#1816](https://github.com/Luminous-Dynamics/symtropy/issues/1816).
- Sol Atlas owns site/spatial evidence; local climate/food resilience work starts with [AGRO-CLIMATE-001 #107](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/107).
- Mycelix owns cooperative producer commitments and fulfillment contracts: [MYC-FOOD-001 #5001](https://github.com/Luminous-Dynamics/mycelix/issues/5001).

This pipeline must not create a duplicate safety assessor, geospatial source of truth, optimizer, inventory ledger, or identity system.

## Paradigm shifts

| Traditional path | Proposed shift |
|---|---|
| Ask AI to invent a robot | Define one farmer-observed task, then compare existing/manual/tool/robot alternatives |
| CAD renders as “done” | Editable native source + exported formats + build procedure + verification evidence |
| Bill of materials as a shopping list | Versioned BOM with equivalent parts, provenance, availability date, lead time, and local substitutions |
| Optimize purchase price | Optimize cost per successfully completed service, including travel, supervision, downtime, consumables, repairs, and seasonal idle time |
| Assume a precision factory | Design for the tools, materials, tolerances, skills, and spares available at specific fabrication hubs |
| One canonical machine | Modular, versioned interfaces and tested modules with explicit compatibility |
| Simulation interpreted as proof | Simulation, bench verification, shadow mode and physical field validation remain distinct evidence classes |
| Failures hidden by aggregate score | Publish exclusions, hard blockers, failures, repair events, uncertainty, and unsupported conditions |

Open-source design is not simply placing files online. OSHWA calls for making the preferred design source available for study, modification, manufacture and redistribution. Choose each design's hardware, software, documentation and dataset licenses deliberately; do not assume one license covers every artifact. See [OSHWA's definition](https://oshwa.org/definition/) and the three variants of [CERN-OHL v2](https://cern-ohl.web.cern.ch/home).

## Symthaea-assisted workflow

### 0. Define the service and baseline

A farmer and engineer specify the job and success criterion; site, crop/livestock system, season and operating window; current manual/mechanized method and measured cost; terrain, geometry, obstacles and operating exclusions; available power, water, transport, tools, fabrication methods, budget, skills, repairs and parts lead times; and the people affected and telemetry they agree to share.

Include doing nothing new. If a better work practice, hand tool, rental, contractor or existing implement is more defensible, return it.

### 1. Search and qualify existing designs first

Build an evidence-linked design library rather than copying files indiscriminately. Start with [FarmHack's tool library](https://farmhack.org/node/786), [Open Source Ecology's plans and Global Village Construction Set](https://wiki.opensourceecology.org/wiki/GVCS), and [FarmBot's open hardware, CAD and BOM](https://farm.bot/pages/open-source), alongside other upstream designs with explicit component-level revision and license metadata.

For every candidate, record source URL and exact revision, artifact hashes, license and attribution, editable native files, drawing units/datums, BOM completeness, required process/tolerance, published tests and failures, manufacturing assumptions, operating limits, price and region/date of each quote, and whether evidence is measured, vendor-claimed, inferred or synthetic. If provenance, license or critical dimensions are unclear, block reuse pending review.

### 2. Define requirements and interfaces

Use versioned request/candidate schemas (proposed, not implemented yet) for the intended service and non-goals; user/site operating envelope and hard no-go conditions; geometry and interfaces with units, tolerances, loads, materials and mating-part revisions; energy, power, thermal, environmental ingress, cleaning/corrosion and duty cycle; fabrication-site requirements; serviceability; and evidence/reviewer/change-trigger requirements.

Never invent universal allowable loads, safety distances or tolerances. Limits need an engineering basis and qualified review.

### 3. Explore the design space with bounded tools

Symthaea may help translate requirements to typed constraints, search prior designs, propose parameter sweeps, find inconsistent requirements, rank measurements that could change the decision, compare trade-offs, and explain the supporting evidence.

Use mature open tools such as FreeCAD, OpenSCAD, KiCad, OpenFOAM or CalculiX when appropriate to the artifact and task. Generated code/parameters require review. Pin tool versions and inputs. Use an independent checker that does not import the production ranking function.

Symtropy handles relevant deterministic scenarios, service-cost bounds and trade-off comparison under its existing ownership. Simulation assumptions and calibration must be explicit. A simulated pass is not field-safety qualification.

### 4. Design for local manufacturing and repair

A release package should include, as applicable: native editable CAD/source and neutral exports; dimensioned drawings, coordinate frames/datums, critical tolerances and inspection methods; a structured BOM with quantity, unit, material/grade, process, finish, allowed equivalencies, criticality, source/license/part number, local quote/date, lead time and substitutes; wiring diagrams, firmware/software, calibration and configuration files; tools/jigs/consumables and assembly checks; maintenance and fault isolation; replaceable modules and spares; end-of-life guidance; and a reproducible manifest with SHA-256 digests.

A substitution must declare potential effects on mass, strength, current/thermal limits, fit, calibration or the qualified operating envelope. “Equivalent” must not silently mean “same safety evidence.”

### 5. Test in stages

1. Schema, units, provenance/license, BOM and hard-constraint validation.
2. Independent hand-calculated and deterministic analysis fixtures.
3. Component/bench tests, inspection and calibration.
4. Fault injection and simulation with declared model boundaries.
5. Effects-disabled shadow operation against independent observations.
6. Supervised, bounded field trial after a reviewed task-specific safety case.
7. Replication at a second fabrication site and field validation across relevant sites/seasons before broadening claims.

Use [ISO 18497-1:2024](https://www.iso.org/standard/82684.html) as a reference for partially automated/semi-autonomous/autonomous agricultural machinery design principles, and [ISO 18497-4:2024](https://www.iso.org/standard/82688.html) for verification and validation methods. Referencing a standard does not demonstrate compliance; applicable machine-specific and local requirements need qualified review.

### 6. Release evidence alongside each design revision

Bind requirements and operating envelope, source files and revision, BOM and quote context, toolchain/configuration and artifact hashes, test method/results, failure modes and exclusions, safety-review status, instructions/maintenance, and compatibility/change history. Mycelix may record consented contribution, inspection receipts, custody and cooperative commitments. Neither reputation nor a credential proves machine safety.

## First design target: choose after field discovery

Do not start by committing to an autonomous tractor. Compare with farmers and local fabricators:
1. a manual or tractor-mounted implement improvement for an expensive repetitive job;
2. a repairable low-voltage sensor/power/tool interface or environmental logger;
3. a low-speed, supervised, non-contact field-scouting platform assembled from locally available components.

Choose only after identifying a low-risk, observable task, current-practice baseline, available parts, a competent reviewer, repair path and test site. Exclude spraying, autonomous high-energy tools, public-road operation and unvalidated animal interaction from the first prototype.

## Proposed benchmark suite

These are qualification gates to validate, not claims about current system performance.

### Requirements and evidence
- 100% exact expected decisions on a frozen, hand-reviewed synthetic corpus covering missing dimensions, unit mismatch, incompatible interfaces, stale prices, incompatible parts and explicit hard constraints.
- Zero false accepts of declared hard safety/physical constraints inside that corpus; one false accept blocks the relevant claim.
- Each source, license, revision, quote and measurement has a provenance state, version/date and digest when obtainable.
- Identical versioned inputs and toolchain produce deterministic reports and manifests.
- Independent checker uses separate expected outcomes and never calls the production ranking function.

### Manufacturing and replication
- 100% of safety- or fit-critical interfaces have units, datums/tolerances, inspection methods and compatibility revision specified before release.
- Complete BOM line coverage for every required assembly item; unknown quantities/costs block complete-cost claims.
- Two independent fabricators reproduce one nominated configuration from the released package; record missing instructions, rework, assembly hours, deviations and failures.
- Every required part has a traceable source/lead time or reviewed local substitution path.
- Compare actual build and repair costs with current practice in local currency; do not claim universal cost reduction.

### Service economics
- Report cost per successfully completed service, not just acquisition cost or theoretical throughput.
- Include capital/financing if supplied, consumables, travel, setup/calibration, supervision, energy, cleaning, downtime, repair lead time, seasonal idle time and defined output denominator.
- Use bounded scenarios where probabilities are unsupported. Never fabricate distributions or confidence intervals.
- Include manual/no-robot baseline and permit “evidence insufficient” or “do not automate.”

### Robotics and safety
- Zero false accepts of frozen hard constraints. Dataset perception scores, simulation and bench tests remain separate from machine-specific physical validation.
- Report intervention rate, stop/fallback behaviour, recovery time, task success, crop/soil damage, near misses and excluded conditions.
- Changes to tool, firmware, model, sensor, calibration, speed, site envelope or safety function trigger explicit re-assessment.
- No direct actuation, spraying, watering, procurement or physical work order is authorized by this research pipeline.

### Accessibility and stewardship
- First release includes repair guide, parts list, license notice, failure-report path and telemetry data/export plan if relevant.
- Assess energy, water, materials, soil disturbance/compaction, chemical exposure where applicable, worker burden, local skills and end-of-life recovery.
- Publish independently replicated outcomes and negative results.

## Cross-project boundaries

| Project | Owns | Does not own |
|---|---|---|
| Symthaea | Evidence synthesis, requirement reasoning, bounded candidate review, uncertainty/abstention and reports | Authoritative mechanics, safety certification or autonomous authority |
| Symtropy | Deterministic scenarios, service-cost envelopes, value-of-information and canonical trade-off comparison | Site ground truth or safety approval |
| Sol Atlas | Geospatial/site sources and their declared spatial/temporal scale | Mechanical qualification |
| Mycelix | Consented coordination, member permissions, contribution and fulfillment receipts | Measurement truth or physical safety proof |

## Field validation and governance

Recruit producers and repair/fabrication people before selecting a design. Observe the task and current failure/cost; ask what they can source and repair; agree on risk and outcomes. Make consent and compensation explicit. Keep farm coordinates, operational telemetry and know-how private by default.

A pilot report includes participants/non-participants, exact hardware revisions, environment, task denominator, source/data gaps, adverse outcomes, operator interventions, maintenance burden, per-site results and uncertainty. A single demonstration does not prove general farm readiness.

## References

- [Open Source Hardware Association definition](https://oshwa.org/definition/).
- [CERN Open Hardware Licence v2 variants](https://cern-ohl.web.cern.ch/home).
- [FarmHack tool library](https://farmhack.org/node/786).
- [Open Source Ecology Global Village Construction Set](https://wiki.opensourceecology.org/wiki/GVCS) and [modular development](https://www.opensourceecology.org/development/).
- [FarmBot open-source CAD/BOM](https://farm.bot/pages/open-source).
- [ISO 18497-1:2024](https://www.iso.org/standard/82684.html) and [ISO 18497-4:2024](https://www.iso.org/standard/82688.html).
- [AgGateway ADAPT Standard v2](https://adaptstandard.org/) for agricultural production-data exchange, not mechanical CAD.
- Symthaea/Symtropy/Mycelix/Sol Atlas tasks linked under “Existing work to compose.”

## Acceptance state

- [x] Research and canonical workstream boundaries linked.
- [x] Design-to-field evidence lifecycle and initial candidate classes documented.
- [x] Synthetic targets and staged safety boundary defined.
- [ ] Typed schemas implemented and validated.
- [ ] Independent corpus and checker implemented.
- [ ] One existing design qualified for a named service/site.
- [ ] Physical bench tests run.
- [ ] Independent local build replication completed.
- [ ] Supervised field trial and service-economic outcomes independently reviewed.
