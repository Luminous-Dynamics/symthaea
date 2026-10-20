# AtomFab: PCB-to-product productive-closure roadmap

**Status:** research architecture and benchmark contract  
**Scope:** Symthaea + Symtropy + existing fabrication, electronics, and Mycelix owners  
**Non-claim:** this document does not establish a qualified physical factory, qualify any process, or authorize machine actuation.

## 1. Thesis

AtomFab should be a **manufacturing-capability compiler and coordinated family of process cells**, not one purportedly universal machine.

The long-term goal is broad productive capability: design, simulate, manufacture, assemble, inspect, repair, and reproduce useful physical products, including PCBs and electronics. Capability must be represented by process, material, geometry, equipment, metrology, environment, and qualification profiles. Different processes may live on different machines or sites.

The governing distinctions are:

~~~text
design exists              != design is manufacturable
route is representable     != route is feasible for available equipment
machine advertises ability != machine is qualified for this exact envelope
program was executed       != artifact conforms
bare PCB was fabricated    != PCB is assembled and functional
board assembly is local    != all components / semiconductor dies are local
~~~

“Make everything” is the direction of travel, not a near-term capability claim. The product objective is **productive closure**: for a stated product and acceptance profile, show which capabilities are local, outsourced, imported, missing, or qualified; then improve that boundary without hiding remaining dependencies.

## 2. Reuse existing owners; do not fork their responsibilities

- **Symthaea process semantics:** [MFG-PROC-000 #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686) and process identity/extensions [#5688](https://github.com/Luminous-Dynamics/symthaea/issues/5688) own process identities, families, capabilities, recipes, plans, and qualification boundaries.
- **Symthaea geometry/fabrication:** the existing fabrication kernel owns supported geometry, slicing/toolpath-adjacent work, validation and the governed fabrication gateway.
- **Electronics/compute:** [COMPUTE-MFG-001 #5796](https://github.com/Luminous-Dynamics/symthaea/issues/5796), [HMI-MFG-001B #5799](https://github.com/Luminous-Dynamics/symthaea/issues/5799), [SEMI-ROADMAP-001 #5867](https://github.com/Luminous-Dynamics/symthaea/issues/5867), and the existing circuits, silicon, device, EMC, materials, metrology and as-built owners.
- **Symtropy:** workcell geometry, kinematics, reachability, collision checks, material handling, resource conflicts, scheduling scenarios, and route comparison. A simulation is a model, not inspection evidence.
- **Mycelix:** work orders, external services, material and lot lineage, supplier/site records, provenance, and production history.
- **Canonical engineering owners:** reuse existing quantity/unit/frame, component/BOM identity, material-state, observation, calibration and evidence-independence contracts.

AtomFab coordinates these owners. It must not become a second BOM authority, materials database, machine-control runtime, inspection ledger or capability-closure engine.

## 3. PCB is a complete process chain, not just milling

A PCB product should carry separate versioned subjects for requirements and schematic; native editable board layout; library/footprint dependencies; ERC/DRC reports; BOM and approved substitutions; manufacturing exports; bare-board process plan and as-fabricated board; assembly plan and placement data; as-assembled board; required electrical, functional and calibration evidence; and released configuration plus repair history.

Use existing tools and formats where suitable. KiCad can export Gerber layer files and Excellon/Gerber drill files via its command-line tooling. These files are a machine-readable handoff, not proof of manufacturability or conformity. See [KiCad CLI documentation](https://docs.kicad.org/9.0/en/cli/cli.html).

### 3.1 Distinct bare-board routes

| Route | Scope | Explicit limits |
|---|---|---|
| **P0 — outsourced bare board, local assembly** | Qualified supplier fabricates the declared stack-up; local assembly, bring-up, inspection and repair | Fabrication is an external service, not local closure |
| **P1 — desktop isolation milling / drilling** | Copper-clad prototype board, simple low-voltage designs, local fixturing and alignment | Normally unplated holes; no assumption of plated vias, industrial solder mask/finish, dense routing, controlled impedance or multilayer registration |
| **P2 — controlled image/etch prototype route** | Research route for selected single/double-sided boards | Requires engineered containment, ventilation, waste handling, process monitoring and metrology; not automatically a production line |
| **P3 — conventional multilayer rigid PCB** | Lamination, inner-layer processing, drilling, desmear/plating, outer-layer processing, solder mask, legend, finish, routing and electrical testing | Long-term staged equipment R&D; every capability and process window needs its own qualification |
| **P4 — specialized boards** | Flex/rigid-flex, HDI/microvia, RF/controlled-impedance, heavy-copper, embedded-component or high-reliability boards | Separate material, geometry, process, metrology and reliability profiles; no inherited qualification |

A milled board is a good first local demonstrator, but P1 must reject a design that requires plated vias or features outside its proven envelope. Visual inspection and apparent net connectivity do not establish hidden-layer quality or long-term reliability.

### 3.2 Assembly and test are separate operations

A complete electronics route may include stencil or controlled paste dispensing, component placement, reflow, through-hole soldering, inspection, rework, in-circuit testing, and functional testing. AOI does not reveal every hidden joint defect; X-ray or other methods may be required by the design and risk profile.

Track bare-board acceptance and assembly acceptance separately. Relevant references include [IPC-2221C, board design standards](https://www.ipc.org/ipc-board-design-standards), [IPC-6012F, rigid-board qualification](https://www.ipc.org/news-release/ipc-releases-ipc-6012f-qualification-and-performance-specification-rigid-printed), [IPC-A-600, printed-board acceptability](https://www.ipc.org/meet-your-standards), and [IPC-J-STD-001J / IPC-A-610J, assembly process and acceptability](https://www.ipc.org/news-release/ipc-releases-j-revisions-two-leading-standards-electronics-assembly). Select the relevant revision and product class for the actual contract. Listing a standard is not the same as qualification or compliance.

## 4. First physical article: repairable low-voltage controller

Start with a simple 2-layer low-voltage controller / sensor board, such as a workshop sensor interface or wired macropad. Keep the first board free of mains voltage, high-energy power conversion and safety-critical control. Select the final article after explicit design-for-manufacture review.

Compare two honest routes for the same frozen requirements:

- **P0:** outsource bare-board fabrication; perform local assembly, bring-up, inspection and repair.
- **P1:** locally mill/drill a constrained prototype board; assemble and test locally, reporting every feature this route cannot provide.

Prefer documented schematic/layout source, a BOM with compatibility-classified alternates, test points, standard connectors, replaceable modules, versioned firmware, and an accessible programming/debug path.

Do not move connector coordinates, netlist, test points, calibration references or acceptance limits after seeing candidate outcomes and count that as process improvement. Such changes create a new design / benchmark generation.

## 5. First device design: separate process cells

### Cell A — PCB prototype mill

Design inputs: work area; required trace/space and drill features; spindle/tool envelope; fixturing; Z-reference method; probing; two-sided registration if used; tool wear; dust extraction; and inspection method.

Core modules: rigid frame; motion and spindle subsystems; appropriate workholding; tool setter and height probing; board-registration camera or equivalent datum system; enclosure and extraction; controller interface; operator controls; metrology and verification artifacts.

Do not treat commanded step size or advertised positioning resolution as achieved copper-feature accuracy. Measure the actual geometry under the declared substrate, tool, speed and setup envelope.

### Cell B — board assembly

Subsystems: ESD-controlled work surface, repeatable locating, solder-paste application or stencil, feeder/placement system where justified, calibrated reflow, inspection and rework tools, board identity capture, and electrical/functional test fixtures.

[OpenPnP](https://openpnp.org/) is a relevant open-source pick-and-place software and hardware reference. Reuse established tooling where it meets the target rather than prematurely rebuilding motion, vision, feeders and placement logic.

### Cell C — advanced bare-board research line

Treat lamination, chemical/electrochemical processing, via metallization, solder-mask application, surface finish, and associated treatment/waste systems as a distinct R&D facility. Integrate process safety, environmental controls, chemical handling, monitoring and qualified metrology into facility design from inception. This roadmap intentionally does not provide hazardous chemical recipes or process setpoints.

The early decision should usually be **outsource advanced bare boards, localize assembly and test, then research a specific missing process**—not build an industrial chemical line without evidence of its strategic or economic value.

### Cell D — inspection and bring-up

Use staged measurement:
- Geometry and registration: calibrated dimensional checks and visual inspection.
- Bare board: opens/shorts against the frozen netlist, with test-coverage limits stated.
- Assembly: polarity, placement and solder inspection; additional inspection for hidden joints where required.
- Function: controlled power-on, current limits, interface tests, firmware identity and defined operating scenarios.
- Reliability: only the environmental, thermal-cycle, vibration or endurance tests required by the intended application.

Every measurement binds instrument identity, calibration state, profile, article revision and raw/result evidence reference.

## 6. Symthaea / Symtropy contracts

### Symthaea should produce

- Typed process/family/capability requirements, not a closed enum of every process.
- Deterministic candidate process plans with intermediate states and inspection hold points.
- Design-for-manufacture findings bound to exact design revision and route profile.
- Multi-objective route comparisons with cost, lead time, yield, quality, energy, waste, repairability and import dependence kept separate.
- Explicit unknowns, blockers, unsupported transformations and required human decisions.
- Experiment proposals that name the hypothesis, observation, acceptance test and held-out evidence.

HDC/CfC predictions do not qualify a fabrication process by themselves. Prefer deterministic constraints and conventional engineering solvers where adequate; introduce learned models only with comparative, held-out evidence.

### Symtropy should model

- Machine envelopes and calibrated coordinate frames.
- Fixture, tool, board, feeder, robot and operator exclusion geometry.
- Reachability, clearance, setup changes and handoff costs.
- Shared resources, alternative routes, queues and unavailable-machine scenarios.
- Model residuals and uncertainty alongside nominal predictions.

The planner must be able to return “no feasible plan under this capability profile.” An objective score cannot override a hard geometry, electrical, process, quality or safety constraint.

### Authority boundary

~~~text
represent -> simulate -> plan -> prepare -> authorize -> execute -> inspect -> qualify
~~~

Every transition needs its own contract and evidence. AI recommendations must not bypass machine-level guards, emergency stops, process interlocks, operator permissions or independent safety controls.

## 7. Benchmark and comparison framework

Use matched requirements and disclose the denominator. Compare: (1) manual bench workflow; (2) desktop in-house fabrication/assembly; (3) established commercial board fabrication/assembly; and later (4) a more automated cell. Do not compare a simple milled board against a high-density multilayer product or a first article against mature factory yield.

| Metric | Measurement contract |
|---|---|
| Route feasibility | Positive and negative known-answer cases; exact rejection reason |
| Design handoff | Reproducible export bundle, profile/design digests, DRC/DFM disposition |
| Bare-board geometry | Measured trace/space, hole and registration deviations against the declared profile |
| Electrical integrity | Open/short results and coverage against frozen netlist |
| Assembly yield | First-pass accepted boards / boards started, sample count and confidence interval |
| Functional acceptance | Required tests passed, with firmware and test-profile identity |
| Rework / repair | Time, replacement parts, cause, retest and residual defects |
| Total cost | Cost per accepted board including labour, setup, consumables, depreciation, utilities, inspection, scrap and rework |
| Lead time | Design freeze to accepted article, with supplier queue and shipping reported separately |
| Resource footprint | Material, energy, water and waste inside an explicit boundary |
| Adaptability | Effort and failure rate when design, material or machine profile changes |
| Traceability | Required design, BOM/lot, process, equipment/tool, calibration, inspection and as-built lineage |
| Productive closure | Local, outsourced, imported, unavailable and qualified capabilities by subsystem |

### Hard gates for the first synthetic suite

- 100% agreement with independently derived expected dispositions for a frozen known-answer corpus.
- 100% rejection of registered invalid routes, such as asking a P1 mill route to provide plated-through vias.
- Missing design files, unknown units, stale calibration, unsupported features, mismatched revisions or incomplete required inspection yield **incomplete** / **reject**, never silent success.
- Reordering semantically unordered inputs does not change canonical identity; changing material, netlist, profile, process revision or protected datum changes the relevant subject identity.
- Missing or invalid negative controls fail the run.
- Each receipt binds exact Git HEAD, schema/profile/corpus digests, tool versions and outcome.
- Synthetic fixtures prove the contract only. They cannot qualify a physical mill, PCB process, production yield or electrical reliability.

### Physical promotion gates

Freeze board, BOM, route, process parameters, instruments and acceptance criteria before confirmatory trials. Use calibrated inspection and reference artifacts where applicable. Keep exploratory tuning separate from held-out confirmation. Report failures and attempted-board counts, not just successful photographs.

Choose numerical targets for trace/space, drilling, yield, cycle time, cost and reliability from the actual board and intended use; measure the baseline first. Do not invent one accuracy target for every board family.

## 8. Research progression

**Phase 0 — software-first (start immediately)**
- Extend existing MFG-PROC and productive-closure owners; do not create a competing ontology.
- The initial 16-case route corpus is `docs/research/fixtures/ATOMFAB_PCB_ROUTE_CORPUS_V0_1.json`; implement and independently validate its oracle before adding machine adapters.
- Bind native design, fabrication exports, BOM, process plan, as-built identity and test receipts.
- Add offline deterministic validation and an independent oracle before machine adapters.

**Phase 1 — first useful product**
- Design and release a low-voltage controller board.
- Compare outsourced bare-board + local assembly against locally milled prototype + local assembly.
- Establish repair, rework and repeated-build data.

**Phase 2 — reliable local assembly**
- Add repeatable paste/stencil, placement, reflow, optical inspection and test fixtures as economics justify.
- Use OpenPnP or comparable tooling where useful; build adapters rather than unnecessary forks.

**Phase 3 — repeatable workshop production**
- Multiple board families, batch scheduling, controlled revisions, material/lot lineage, yield and cost accounting.
- Add robotic handling and cell scheduling in Symtropy after physical work envelopes and handoffs are measured.

**Phase 4 — selected advanced bare-board capability**
- Pick one valuable gap (for example drilling/registration, vias or finish); evaluate technical risk, EHS burden, throughput, facility cost and external alternatives.
- Qualify that process as a bounded program. Qualification does not transfer automatically across material stack-ups.

**Phase 5 — microelectronics and semiconductor**
- Compose [SEMI-ROADMAP-001 #5867](https://github.com/Luminous-Dynamics/symthaea/issues/5867) and its design/process/packaging/test owners.
- Keep chip design, external foundry fabrication, packaging/test and working system integration distinct.
- Atomic-layer deposition/etching and atomically precise methods remain specialized research tracks, not evidence of a universal atom-by-atom factory.

## 9. Paradigm shifts worth pursuing

1. **Machine-centric -> capability-centric:** search across heterogeneous equipment instead of forcing every product through one machine.
2. **File handoff -> verified digital thread:** connect native design, recipe, as-built artifact, metrology and repair history through explicit identity/lineage.
3. **Nominal simulation -> residual-aware twins:** predictions carry limits and are compared with calibrated observations.
4. **Yield-only -> accepted-product economics:** optimize cost and time per conforming, maintainable product without relaxing hard quality constraints.
5. **Automation-first -> repairability-first:** test points, modular components, documented source and compatible substitutes reduce lifecycle dependence.
6. **Universal autonomy -> bounded, composable closure:** prove one new capability at a time and identify tooling, metrology, feedstocks and imports that remain blockers.
7. **AI as authority -> AI as experiment planner:** learned systems propose routes and experiments; interlocks, human authorization and qualification remain independent.

## 10. Research references

- NIST, [Digital Twins for Advanced Manufacturing](https://www.nist.gov/programs-projects/digital-twins-advanced-manufacturing): interoperability, validation and quantified uncertainty.
- NIST, [Digital Twins Workshops Summary Report, 21 July 2026](https://www.nist.gov/publications/digital-twins-workshops-summary-report): current priorities include interoperability, VVUQ, cybersecurity and workforce readiness.
- KiCad, [command-line manufacturing outputs](https://docs.kicad.org/9.0/en/cli/cli.html): scripted Gerber/drill output.
- OpenPnP, [open-source SMT pick-and-place](https://openpnp.org/): software and buildable/modifiable hardware reference.
- IPC, [IPC-6012F rigid-board qualification](https://www.ipc.org/news-release/ipc-releases-ipc-6012f-qualification-and-performance-specification-rigid-printed), [board design standards](https://www.ipc.org/ipc-board-design-standards), and [assembly standards overview](https://www.ipc.org/news-release/ipc-releases-j-revisions-two-leading-standards-electronics-assembly).
- OPC Foundation, [OPC UA for Machine Tools](https://reference.opcfoundation.org/specs/OPC-40501-1/1): common machine monitoring and job-management model.
- ISO/ASTM, [52920:2023 additive-manufacturing qualification principles](https://committee.iso.org/standard/76911.html): precedent for per-process/site qualification rather than blanket machine claims.

## Initial corpus check

The committed synthetic route corpus contains 16 unique cases: 2 `route_admissible`, 5 `reject`, and 9 `incomplete`. A separate rule-based pass over the fetched, committed JSON independently recomputed all 16 expected dispositions/reason codes. This is an in-session corpus sanity check, not a committed Rust evaluator, CI pass, or physical-process validation.

The execution work is tracked by [COMPUTE-MFG-001C #8223](https://github.com/Luminous-Dynamics/symthaea/issues/8223) and the Symtropy workcell benchmark [PCB-CELL-001 #2385](https://github.com/Luminous-Dynamics/symtropy/issues/2385).

## Exit criterion

For a bounded electronics product, AtomFab can select and explain feasible routes; emit a reproducible manufacturing package; distinguish outsourced from local capability; assemble and test the board; preserve required lineage; and produce an accepted artifact against a frozen profile. It can reject infeasible candidates and reproduce the evidence chain on a second run.

Passing proves only that bounded end-to-end capability. It does not prove universal manufacturing, industrial multilayer PCB production, semiconductor self-sufficiency, or atomically precise fabrication.
