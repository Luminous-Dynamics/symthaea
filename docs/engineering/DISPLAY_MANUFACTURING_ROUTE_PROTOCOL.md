# Display Manufacturing Route Protocol

**Status:** proposed route contract; all route entries remain UNQUALIFIED  
**Tracking:** [VIS-MFG-001 #8012](https://github.com/Luminous-Dynamics/symthaea/issues/8012)  
**Display yield co-design:** [MICRODISP-YIELD-001 #7868](https://github.com/Luminous-Dynamics/symthaea/issues/7868)  
**Parent program:** [VIS-TECH-001 #7820](https://github.com/Luminous-Dynamics/symthaea/issues/7820)

## 1. Paradigm shift: optimize the calibrated-good module

The unit of manufacturing success is not a transferred emitter, a deposited film, a bonded die, a good wafer site, or a nominally bright panel. It is a **final calibrated-good module** that meets a declared optical/electrical/reliability profile, with traceable process history and uncertainty.

The benchmark should compare designs and routes on cost, material/energy use, cycle time and final-good output while keeping hard optical/reliability constraints explicit. The route with the highest local stage yield or smallest pixel pitch may lose after optical integration, repair, calibration, thermal drift or final test.

No single scalar should replace this vector. Cross-architecture comparisons are valid only under matched target module requirements or a declared application-level equivalence contract. A large-area LCD, OLEDoS microdisplay, Micro-LED module and waveguide combiner do not share one universal “best route” objective.

## 2. Research basis and what can be transferred

- Park et al., *Laser & Photonics Reviews* (first published 29 July 2026), review Micro-LED/CMOS integration routes and show that bottlenecks change with scaling: yield/placement, interconnect/alignment, and then process integration/thermal budget. [DOI 10.1002/lpor.71634](https://doi.org/10.1002/lpor.71634)
- Hong et al. review six Micro-LED transfer/integration families and the high cost of very small defect rates across large pixel populations. [DOI 10.1088/2631-7990/ae25a8](https://doi.org/10.1088/2631-7990/ae25a8). The 99.99999% figure is a literature frontier, not our result or a universal acceptance threshold.
- Qiao et al. (May 2026) study laser repair bonding in a specific panel-level Micro-LED stack, using coupled thermo-mechanical modeling and experiments. [DOI 10.1016/j.mssp.2026.110452](https://doi.org/10.1016/j.mssp.2026.110452). Their setup-specific parameter window must not be copied as a universal recipe or imported process qualification.
- The OLEDoS review describes the CMOS backplane/frontplane split, OLED stack formation, encapsulation, inspection, dicing and mounting. [DOI 10.1016/j.nxnano.2025.100132](https://doi.org/10.1016/j.nxnano.2025.100132)
- Hu et al. report a full-color nanoimprint-fabricated waveguide with high uniformity in one prototype. [DOI 10.1002/sdtp.18177](https://doi.org/10.1002/sdtp.18177). This supports including replication/grating uniformity as a measurement axis; it does not establish general production yield.
- A survey of LCD mura detection treats defects across array, cell and module manufacturing stages, supporting stage-specific metrology rather than final-only inspection. [Crystals, 2021](https://www.mdpi.com/2073-4352/11/12/1444)

The machine-readable profiles reference these papers only as route context. Literature describes candidate operations and known challenges; it is not a local recipe, equipment-capability profile, or proof that a transformation occurred.

## 3. Initial route-profile corpus

Manifest: [`display-manufacturing-route-profiles-v0.1.json`](display-manufacturing-route-profiles-v0.1.json)  
Schema: [`display-manufacturing-route-profiles.schema.json`](display-manufacturing-route-profiles.schema.json)

The current data-only corpus has four route families:

1. **Transfer-based Micro-LED on active backplane** — incoming emitter lot/bin map, backplane acceptance, transfer-map preflight, placement, bonding/interconnect, post-transfer inspection, conditional repair/retest, route-specific color integration, protection/package, calibration, final reliability/disposition.
2. **OLED-on-silicon (OLEDoS)** — CMOS backplane acceptance, surface/anode preparation, OLED frontplane stack, electrode/encapsulation, in-process inspection, dicing, package/optics integration, calibration and final reliability/disposition.
3. **TFT-LCD panel** — TFT-array process/inspection, color-filter substrate/inspection, cell preparation and assembly, cell inspection/repair, polarizer/backlight/driver module assembly, power/image-quality test and final disposition.
4. **AR light engine + waveguide** — light-engine acceptance, waveguide blank, route-specific grating fabrication, grating metrology, optional coatings/layers, optomechanical alignment, angle-resolved optical metrology, calibration and final reliability/disposition.

This is a representative first slice, not an exhaustive list of display technologies. LCOS, OLED projection, other micro-OLED stacks, DLP/LCOS projectors, direct-view OLED and alternative waveguide/combiner routes can become separate profiles when specific process ownership and evidence need justify them. Conditional steps must stay conditional; the corpus does not imply all routes share the same stack.

## 4. Process-step evidence contract

Every step records:

- a route/profile revision, stage ID and sequence;
- a technology-specific step description and reference to the existing generic process semantics owner;
- exact input subject/state and *intended* output state;
- conditionality, including route/architecture conditions;
- inspection hold-point and required evidence;
- failure modes expected to be detected or preserved;
- evidence status, which remains UNQUALIFIED in this v0.1 manifest.

The manifest does not create a new process ontology. Generic process-flow semantics remain with [MFG-PROC-000 #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686) and semiconductor-specific state transitions with [SEMI-PROC-001 #5888](https://github.com/Luminous-Dynamics/symthaea/issues/5888). Qualification composition remains with [SEMI-QUAL-001 #5892](https://github.com/Luminous-Dynamics/symthaea/issues/5892). Route entries are specializations/adapters for those contracts, not standalone execution models.

Use these boundaries strictly:

`step specified ≠ step executable ≠ step executed ≠ output observed ≠ output accepted`

`component passed ≠ final display passed`

`one successful unit ≠ stable process capability`

`repaired output ≠ erased first-pass failure`

No recipe values, chemical/gas conditions, laser operating points, or actuation authority are embedded in this corpus.

## 5. Core benchmark definitions

### Yield denominators

For each stage, capture attempted/inspected/accepted/rejected/reworked/scrapped/unknown counts and their lineage. Define, where appropriate:

- **First-pass yield:** unique input units accepted at the first specified test divided by unique units tested at that stage.
- **Post-repair yield:** unique units passing the defined retest after allowed repair, reported separately from first-pass yield and repair attempts.
- **Final calibrated-good yield:** unique incoming candidate units that meet all mandatory final optical, electrical, calibration and declared reliability requirements divided by unique incoming candidate units.
- **Defect escape rate:** observed defective units found at a downstream gate divided by the relevant inspected population, with clear detection opportunity and denominator.
- **Inspection precision/recall:** class-specific, using a preregistered ground-truth review method and explicit uninspected coverage.

Do not silently remove repairs, scrap, failed tests, rework loops, aborted lots or unknown outcomes. Do not multiply stage yields under an unverified independence assumption. If only component-stage data exist, report the stage statistics or a clearly labeled scenario/bound; do not call it observed whole-line yield.

### Economic and resource outputs

Report the following separately:

- **Cost per calibrated-good module:** explicitly defined cost numerator (materials, consumables, energy, inspection/test, rework, scrap, equipment/facility-time allocation and amortization assumptions) divided by accepted final-good modules. If the denominator is zero, report undefined/not evaluated rather than smoothing it.
- **Final-good throughput:** accepted calibrated-good modules per declared elapsed production time, recording holds, rework and retest. Report machine throughput separately from end-to-end throughput.
- **Cycle time and WIP:** include waits for inspection, repair, requalification and downstream capacity.
- **Resource intensity:** energy, material consumed/scrapped, solvents/consumables where measured, and water/utility use where relevant per calibrated-good module.
- **Inspection effectiveness:** precision/recall by defect class, coverage, and the stage at which escaped defects are detected.
- **Quality/reliability vector:** optical/electrical performance, uniformity, calibration stability, power/thermal conditions, and measured aging/reliability results where actually executed.

Report uncertainty and between-lot/wafer variation for key quantities. These dimensions support a Pareto frontier; they should not be hidden in a single “manufacturability score.”

## 6. Robust process window and measurement placement

Optimize a robust feasible region, not a single nominal setpoint. The process-window claim must state what parameters and disturbances were varied, which measurements support the bounds, and what output constraints are protected. When correlations or clustered defects are possible, preserve joint/lot-level evidence or conservative bounds rather than defaulting to an independent Bernoulli model.

Metrology is an optimization decision too. Use the shared decision-aware experiment-selection work ([VIS-EXP-001 #7871](https://github.com/Luminous-Dynamics/symthaea/issues/7871)) to decide which inspection or calibration measurement is most likely to change route selection, avoid expensive downstream scrap or discriminate competing explanations. More sensors are not automatically better: measure the reduction in decision error/scrap risk against instrument cost, line time and false-call cost.

Inspection should generally be evaluated early enough to prevent adding value to unrecoverable bad units, but a proposed checkpoint is only useful if its defect detection performance, false-positive cost and process relationship are measured.

## 7. Proposed acceptance targets (not achieved results)

### Integrity / stage transition gates

- 100% of route stages use exact input/output subject references, route revision and explicit process-semantics owner.
- 100% of accepted transitions bind required post-step measurement, calibration/context, denominator and raw/derived evidence identity.
- Zero output-state promotion when mandatory metrology is absent or contradicts the intended transformation.
- Zero lost first-pass failures or silent denominator changes during repair/retest.
- 100% of conditional/architecture-dependent stages state their condition; no path implies a conditional process ran when it was absent.
- 100% of route source IDs resolve; every route and stage remains UNQUALIFIED until specific process and module claims have real evidence.

### Performance targets after a baseline exists

- **Cost:** initial research target is ≥10% lower cost per calibrated-good module versus a reproducible matched baseline, subject to identical output/quality and declared accounting boundary. Treat as a hypothesis; preregister the costing method before measuring.
- **Yield:** establish a baseline and uncertainty first; then target improvements in final calibrated-good yield without masking stage losses. Do not adopt 99.99999% as a universal threshold.
- **Inspection:** set per-defect-class precision/recall targets from the cost/risk of false negatives versus false positives; critical defect classes cannot be offset by better average metrics elsewhere.
- **Repair:** repair is only beneficial if cost/time and recovered calibrated-good units outweigh repair damage, repeat failure and reliability fallout. Report first-pass and post-repair performance separately.
- **Reliability:** no numerical gate is claimed before a specific device/process, test profile and duration are pinned. A repaired unit must be included in reliability tracking, not assumed equivalent to unrepaired units.

Percent improvement gates only become qualification results after the exact baseline, device/process profile, sample size, uncertainties and raw evidence are available.

## 8. Adversarial controls

At minimum, verify rejection or downgrade when:

1. a process step is marked complete but post-step inspection is missing;
2. equipment or calibration revision changes without a new profile identity;
3. input lot or component identity changes but output lineage stays the same;
4. model prediction disagrees with observed map and the residual is overwritten;
5. a literature profile is presented as locally qualified;
6. a repair overwrites the original failed disposition or denominator;
7. a downstream step consumes an intended-but-unobserved output state;
8. an inspection map omits unmeasured regions but claims full coverage;
9. a failed or censored lot is silently excluded from cost/yield summary;
10. the final optical-calibration requirement is absent but the module is marked good;
11. correlated/clustered defects are multiplied as independent stage probabilities without justification;
12. cross-route comparisons use incompatible target modules or hidden cost boundaries.

Unknowns stay unknown and all unqualified profiles stay unqualified. Failure and discrepancy are useful evidence, not clutter to remove.

## 9. First executable sequence

1. Freeze and independently validate the route manifest/schema and its reference closure.
2. Bind profile stages to canonical MFG-PROC/SEMI-PROC stage/state contracts as those owners qualify. Do not manufacture a parallel process state machine.
3. Add adversarial known-answer fixtures for the evidence-state/lineage transitions.
4. Reuse the existing microdisplay yield arithmetic for expected counts, Wilson intervals and repair arithmetic; avoid duplicate yield math unless a concrete API gap is documented.
5. Run a synthetic route comparison with deliberately correlated defect scenarios and an explicit no-information case, keeping all outputs synthetic.
6. Select one physical component/inspection process only when instruments, a named profile and appropriate qualified process authority are available. First compare model to held-out measured data.
7. Preserve failures, actual scrap/rework, calibration and uncertainty; do not report a physical yield or cost improvement from this planning corpus.

## 10. References and limitations

This protocol uses published literature for route discovery and bottleneck identification, not recipe extraction. It is not a production engineering instruction, process qualification, equipment certification, fab execution plan, or claim of Luminous manufacturing capability. A human-reviewed, device-specific route with qualified inputs and raw measurements is required before any production decision.
