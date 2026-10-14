# Living Concrete and Better Rail Systems: Research Program

**Status:** Research plan and hypothesis registry. This document does not establish a concrete formulation, biological safety, rail-system fitness, code qualification, or authority to construct, inspect, maintain, or operate infrastructure.

**Date:** 2026-10-14

## Executive direction

Yes: Symthaea should help search for *living or bioactive concrete*, but it should not begin by optimizing bacterial growth or inventing a biological recipe. It should compare damage-prevention and self-healing concepts against matched conventional controls, then select candidates only after independent tests show improved performance under the intended exposure conditions.

In parallel, the highest-value near-term railway work is likely **condition-based maintenance and whole-network reliability**, not a speculative new rail alloy or a general-purpose AI dispatch controller. Federal Railroad Administration (FRA) reports RR 26-01 and RR 26-02 describe development and implementation of track-geometry forecasts, including statistical point/segment approaches and the value of encoding maintenance events as explanatory variables. These methods provide a strong baseline Symthaea should attempt to match before claiming additional predictive value.

## Program ownership

- **Symthaea:** research question generation, candidate representation, calibrated property-specific prediction, candidate ranking, uncertainty, experiment manifests and claim boundaries.
- **Symtropy:** deterministic counterfactual scenarios where its documented physical capabilities are sufficient. Its existing rigid-body engine is not presumed to be a validated cementitious fracture, geotechnical, finite-element, wheel–rail contact, or rail-track degradation solver.
- **Sol-Atlas:** read-only geospatial context: climate, terrain, catchment, flood exposure, materials sources, communities, corridor dependencies and observed asset-condition projections with source/time/resolution metadata.
- **Mycelix:** canonical provenance/source rights, review lineage, materials/maintenance history, procurement and commons governance. A vote is not an engineering test.
- **Laboratories, operators, licensed engineers and authorities:** choose and interpret methods, qualify materials, approve designs, and authorize field trials or operational changes.

Compose existing work rather than creating duplicate roots:
- Symthaea [TRANSPORT-ENG-001 #6244](https://github.com/Luminous-Dynamics/symthaea/issues/6244) and [owner coverage PR #6246](https://github.com/Luminous-Dynamics/symthaea/pull/6246).
- Mycelix [Living-City concept contracts PR #5018](https://github.com/Luminous-Dynamics/mycelix/pull/5018) and [source/evidence contracts PR #5029](https://github.com/Luminous-Dynamics/mycelix/pull/5029).
- Sol-Atlas [evidence-bounded transport projection PR #111](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/111).
- Symtropy [engineering digital-twin boundary #1467](https://github.com/Luminous-Dynamics/symtropy/issues/1467).

## A. Living concrete / bioactive cementitious materials

### A1. Why it is plausible

Cementitious materials can undergo limited autogenous healing: unhydrated cement and carbonate precipitation may close some small cracks when water and suitable conditions are present. Additional concepts attempt to improve this behavior using fibers that limit crack opening, crystalline or mineral admixtures, healing-agent capsules, and microbial biomineralization.

Microbial approaches often aim to precipitate calcium carbonate at a crack. Dormant spores or other protected biological systems may be carried in a porous aggregate, capsule, hydrogel or other encapsulant and reactivate when the crack environment provides suitable moisture and substrates. The goal is to seal a transport path and improve durability; it is not safe to assume that visible crack closure equals restored load capacity.

Reviews published in 2024–2025 identify promising laboratory results but also describe major comparability and scale-up gaps: strain survival, alkaline/desiccating conditions, nutrient access, crack width and depth, encapsulation damage during mixing, cost, environmental conditions, inconsistent measurements, and limited long-term field evidence. Different studies often test different formulations and endpoints, so reported percentages must not be combined into a universal performance claim.

### A2. Candidate tracks: establish comparators before optimizing microbes

**Track 0 — Conventional baseline and autogenous-healing control**
- A well-characterized concrete/mortar reference using the project's currently accepted cement, aggregates, reinforcement/fibers and curing regime.
- A matched control to quantify ordinary autogenous crack closure.
- Internal curing as a damage-prevention comparator. FHWA reports that internal curing can reduce early-age shrinkage cracking and may extend bridge-deck service life; its effects and any life-extension claim remain application/context specific.
- A fiber-reinforced, controlled-crack-width comparator when appropriate.

**Track 1 — Non-living autonomous-healing control**
- Crystalline/mineral admixture or encapsulated non-biological healing agent.
- Measure whether the healing system changes workability, setting, strength, shrinkage, durability and repeatability before crediting any crack-sealing benefit.
- This is an essential comparator because it may deliver some intended repair benefits without viable microbes or environmental persistence.

**Track 2 — Encapsulated microbial mineralization**
- Evaluate published, appropriately characterized research systems using non-pathogenic, non-engineered strains selected by qualified microbiology partners and documented risk assessment.
- Focus on compatibility with cementitious alkalinity, moisture limitation, temperature cycling and the actual crack microenvironment.
- Treat carrier/encapsulant and nutrient/precursor choices as separate factors. Encapsulation can improve survival but can also introduce voids, alter strength or fail during mixing.
- Evaluate the biomineralization pathway and its by-products. Ureolytic pathways, for example, can create ammonia-related environmental concerns; a high carbonate yield alone is not an adequate environmental score.
- Do not proceed to deliberate environmental release or engineered-strain deployment on the basis of a software ranking.

**Track 3 — Biological/fiber/mineral hybrid**
- Investigate whether crack-width control, biological or mineral healing, and reduced early-age cracking are complementary.
- Use a predeclared factorial or other justified design to test interactions instead of multiplying the best result from separate studies.
- Treat fungal systems and novel engineered living materials as research hypotheses until material-level and biosafety evidence justify further investment.

### A3. Candidate data contract

Each candidate record must contain:

- Stable candidate/formulation ID and version.
- Material scope: mortar, cast concrete, precast, repair mortar, road slab, bridge deck, sleeper, culvert or another explicitly defined use.
- Full composition/source/batch provenance and component reuse rights.
- Healing class: autogenous, internal curing, non-living encapsulated, microbial, fiber/mineral hybrid, or other named mechanism.
- Whether a viable organism is present; taxonomic/strain identity and qualified biosafety assessment references where relevant; whether it is engineered; contained-use/release status.
- Encapsulant/carrier, nutrient/precursor class, expected activation conditions and declared by-product risks.
- Fresh-state constraints: workability, setting time, air content/voids, mixing/survival damage and construction compatibility.
- Target response with unit, method, conditioning, specimen and replicate IDs.
- Model version, measured-versus-predicted status, calibration domain and uncertainty.
- Standards edition/project acceptance criteria supplied by the responsible engineer; no unversioned thresholds.

Do not store a generic scalar called “healing efficiency” as the only endpoint. Distinguish crack-width closure, reduction in water permeability, reduced ingress of relevant agents, retained/recovered mechanical response, durability through cycles, and the ability or inability to heal a reopened crack.

### A4. Experimental design

Use a staged, preregistered comparison:

1. **Question:** define whether the target is crack sealing, durability preservation, corrosion-risk reduction, or another measurable endpoint.
2. **Matched baseline:** use the same matrix, specimen geometry, curing history, loading and environmental cycle across controls and candidates wherever feasible.
3. **Positive and negative controls:** ordinary autogenous healing; uncracked reference; deliberately damaged but unhealed control; a candidate with the healing mechanism disabled or absent where feasible; image/measurement negative controls.
4. **Replicates and blind analysis:** plan replicate counts with a statistician/materials engineer; blind specimen labels during crack-image and outcome assessment where practical.
5. **Environment:** test the intended service envelope for wet/dry cycles, relevant temperatures, load cycling, water chemistry, exposure chemistry and crack widths. Don't choose conditions after seeing which candidate wins.
6. **Measure more than appearance:** crack width and closure distribution, water permeability/penetration, healing depth, mechanical response and stiffness recovery, cyclic/reopening behavior, durability, reinforcement corrosion indicators where relevant, leachate/by-products, cost and life-cycle impacts.
7. **Microbiology:** where viable organisms are involved, have the qualified lab assess viability and activity under the cementitious conditions, identity/purity, persistence, and relevant worker/environmental exposure. Microbial presence or a positive staining image does not prove structural benefit.
8. **Replication and uncertainty:** report per-specimen results, variability, failures, outliers, missing observations and uncertainty intervals; retain unsuccessful formulations.
9. **Independent verification:** an independent checker validates input manifests, units, calculations and held-out assignments, not merely rerunning the same candidate generator.
10. **Promotion:** lab promising → independently replicated lab result → application-specific engineering review → bounded, owner-authorized monitored pilot → extended field follow-up. No direct jump from model to infrastructure deployment.

### A5. Predeclared performance gates

Every candidate is subject first to mandatory engineering and environmental requirements; a Pareto ranking must not trade away a safety constraint.

Relevant measurements, selected for the application, can include:
- compressive/flexural/tensile response as appropriate, stiffness, shrinkage and creep;
- crack-closure rate/distribution and permeability reduction under declared exposure;
- retained response after repeated wet/dry, thermal, load or chemical cycles;
- durability, bond/compatibility with substrate and reinforcement corrosion indicators;
- fresh-state performance and placement/curing robustness;
- biological viability/containment, leaching/by-products, worker exposure and end-of-life behavior;
- material and maintenance cost, greenhouse-gas/lifecycle effects, water/material demand and repairability.

Use minimum practically important improvement margins defined before running the blinded evaluation. Require confidence intervals and predeclared non-inferiority bounds for essential baseline properties, along with a cap on tolerable failure or variance. Do not specify numerical thresholds until the target application, test method, sample design and acceptance authority are known.

### A6. Environmental and worker safety

Potentially living construction materials span formulation, mixing, curing, service life, demolition and end-of-life. Before any open field deployment, consider strain identity and provenance, unintended persistence, genetic stability where relevant, horizontal gene transfer where relevant, interaction with soil/water microbiomes, runoff/leachate, by-products, demolition dust, aerosols, worker protection and waste treatment. Apply qualified institutional and jurisdictional biosafety review. Favor contained laboratory evaluation and non-engineered, appropriately screened systems for an initial research path.

This is preventive risk assessment, not a claim that all proposed strains are harmful. Conversely, ordinary use of a familiar organism is not evidence that a viable strain in a long-lived construction product is automatically safe.

### A7. Evidence model and ML policy

Symthaea should begin with a transparent tabular baseline (for example a simple regularized regression or a tree-based comparator appropriate to sample size) before testing HDC/CfC-based representations. HDC similarity is useful for candidate retrieval, not as a substitute for physical property prediction.

- Split data by study/site/formulation/batch, not random rows alone, to reduce leakage.
- Publish performance by study and exposure regime, not only aggregate error.
- Distinguish interpolation from transfer to a new cement, aggregate source, climate or crack regime.
- Calibrate prediction intervals and measure performance on an untouched holdout.
- If evidence coverage, calibration or domain checks fail, return **insufficient_evidence** and preserve the reason.
- Use active learning to propose the next most informative *approved, bounded* experiment; never let the model autonomously order cultures, release organisms or change a field material specification.

## B. Better railway systems

### B1. Highest-value first move: predict deterioration, not just geometry

FRA research reports offer immediate baselines:
- [RR 26-01 — Track Geometry Predictions Using Point and Segment Processing](https://railroads.fra.dot.gov/elibrary/track-geometry-predictions-using-point-and-segment-processing). The agency reports that point-based SARIMAX and segment-based Prophet approaches forecast track-geometry behavior, and that adding unexpected maintenance events as explanatory data materially improved predictions. The report described case studies as future work at publication.
- [RR 26-02 — Implementation of Advanced Track Geometry Forecasting Models](https://railroads.fra.dot.gov/elibrary/implementation-advanced-track-geometry-forecasting-models), last updated by FRA on July 30, 2026. It describes operational implementation work improving track-geometry filtering and forecasting methods.
- [RR 21-18 — Feasibility of Applying Model-Assisted Probability of Detection to Track Inspection](https://railroads.fra.dot.gov/elibrary/feasibility-applying-model-assisted-probability-detection-track-inspection). This is a useful reference for evaluating inspection effectiveness, especially ultrasonic rail-flaw detection.

These are baselines to reproduce and learn from, not evidence that Symthaea already outperforms them. Public reports do not guarantee unrestricted access to the underlying raw, geolocated measurements. Every data source still needs access/licensing and field-of-use review.

### B2. Railway workstreams

**Railway track condition and preventive maintenance**
- Forecast geometry by location and time; retain spatially correlated defects rather than treating every reading as independent.
- Incorporate verified maintenance history, tamping/grinding/renewal events, weather, load/axle passages and inspection frequency.
- Compare forecasts with the owner's current model and simple persistence/threshold baselines.
- Optimize inspection and maintenance windows under access/possession constraints, using verified work duration and resource availability.
- Measure calibration, missed deterioration, false alarms, lead time and avoided service disruption; don't optimize only forecast RMSE.

**Inspection quality and detection confidence**
- Preserve sensor/instrument identity, calibration, data-quality status, coverage gaps, inspection modality and operator/method version.
- Study probability of detection, false-negative risk and uncertainty by defect type, track/speed/condition and inspection setup.
- Corroborate sensors using independent physical inspections and appropriate non-destructive testing.
- Missing or low-quality inspection evidence must stay unresolved, not become a “no defect” observation.

**Ballast, subgrade, drainage and transition zones**
- Correlate recurring track geometry deterioration with drainage, ballast condition/fouling, settlement, earthworks, bridge approaches and culvert/flood exposure.
- Combine terrain/catchment/climate context in Sol-Atlas with track-condition and maintenance evidence.
- Prefer root-cause and whole-life comparisons over repeatedly treating the same symptom with tamping alone.
- Keep hydrogeological/geotechnical claims bounded until site-specific observations and a suitable validated model exist.

**Rail/wheel interface and high-wear assets**
- Track rail-head wear, rolling-contact fatigue indications, wheel condition, rail grinding history, curves, loads and temperature.
- Compare maintenance timing and inspection strategies against measured defect recurrence, wear and service impacts.
- Prioritize turnouts, crossings, joints/welds and other complex assets as distinct asset classes with their own inspection/maintenance profiles. A single generic “rail health” model is inadequate.

**Timetable, capacity and passenger/freight reliability**
- Only after route/capacity data are trustworthy, model dwell-time variability, conflicts, junction and terminal bottlenecks, missed handoffs, temporary speed restrictions and maintenance possessions.
- Improve schedule robustness and freight transfer reliability rather than maximizing nominal trains/hour at the cost of recovery margin.
- Separate published/planned capacity, observed use, temporary restrictions and hypothetical capacity improvements.
- For energy, evaluate traction efficiency, regenerative braking where the rolling stock and power network support it, coasting/speed profiles, terminal dwell, and electrification only against real service, power-system and lifecycle constraints.

**Resilience and maintainability**
- Assess heat, flood, wildfire/vegetation, embankment movement, power outage, equipment failure, supplier delays and corridor disruption with explicit scenarios.
- Keep alternative routes, repair stock, crew availability, inspection coverage and recovery time visible.
- Measure service continuity, delay minutes, cancellations, time-to-detect, time-to-repair, repeat defect rate, lifecycle cost and access/equity impacts.
- Apply human operational authority and fail-safe procedures; no simulation is dispatch authority.

### B3. A concrete first pilot

**Pilot:** a read-only track-geometry deterioration benchmark with maintenance-history covariates.

Inputs:
- exact segment/chainage or stable pseudonymous location ID;
- timestamps and survey method, instrument/source identity, units, quality flags and coverage;
- measured geometry values and owner-authoritative limits only if a current source references them;
- traffic/load/axle exposure if available and rights-cleared;
- maintenance events (date, location, intervention class, recorded magnitude and uncertainty);
- local weather/environmental covariates where source and resolution permit;
- split/manifest and source rights.

Models:
1. simple persistence and standard alert-threshold baseline;
2. implementation-faithful SARIMAX/segment Prophet baseline where assumptions fit;
3. Symthaea models and HDC/CfC feature experiments;
4. a held-out, independent evaluator that consumes frozen predictions and labels.

Evaluate by *future* time periods and held-out segments/routes; report site-level variation and maintenance-induced resets. Prevent near-duplicate segment leakage. Track MAE/RMSE where useful, but prioritize threshold-crossing calibration, under-prediction of deteriorating segments, lead time, false-alert burden and decision usefulness. Preserve manual review before maintenance scheduling.

The first prototype may use openly permitted synthetic data to validate the pipeline, but it must label those results synthetic and must not claim real-rail predictive performance from them.

### B4. Cross-project contracts

- **Symthaea:** canonical asset/observation identity; separate as-designed/as-built/as-operated/observed/inferred/synthetic states; predictor identity/calibration/uncertainty; condition trajectories and decisions.
- **Symtropy:** exact read-only projections, scenario/configuration identity, simulation capability declaration, numerical integrity, replay and counterfactual outputs. It must not write simulator output back as an observed track state.
- **Sol-Atlas:** separate physical rail/road graph, communication graph, evidence/reconciliation graph and infrastructure-dependency graph. Every layer carries source, timestamp, resolution/CRS and confidence; unresolved contradictions remain visible.
- **Mycelix:** source rights, evidence lineage, operators/reviewer identities, maintenance/procurement/work orders and community/service outcomes.
- **Independent evaluator:** independently recompute data integrity, splits, metrics and report totals; code path or organization independence must be stated accurately.
- **Operator and authority:** own allowable limits, inspection policy, maintenance actions, route safety and operational authorization.

## C. Joint opportunity: living materials for railway infrastructure

Rail offers an interesting application for self-healing concrete, but the specific asset matters:

1. **Concrete sleepers / precast elements:** potentially compare bioactive/self-healing repair formulations for small cracks, but include fatigue, fastening/interface behavior, moisture ingress and dimensional stability. Qualification requirements for rail components are not interchangeable with ordinary concrete cubes.
2. **Slab-track and track-support concrete:** evaluate crack-width control, internal curing and non-living or microbial sealing systems against stiffness, fatigue, settlement compatibility, drainage and repairability.
3. **Culverts, drainage channels, retaining structures and bridge elements:** these can be more appropriate early monitored cases for crack-sealing studies than primary load-critical track elements, subject to the asset owner and engineer's approval.
4. **Trackside sensor mounting and enclosures:** durability, ingress and maintainability may be a useful non-structural pilot with a lower consequence of failure, if the owner agrees.

Do not start a biological concrete pilot in a critical sleeper, turnout, bridge or tunnel merely because it is an exciting application. Select the first pilot by consequence of failure, available baseline evidence, inspectability, reversibility and owner permission.

## D. Prioritization

| Priority | Work package | Why first | Gate |
|---|---|---|---|
| P0 | Upgrade Symthaea material and infrastructure evidence contracts; distinguish measured, lab, literature and simulated claims | Prevents fabricated confidence from entering later models | Focused Rust tests and independent malformed-input controls on exact source head |
| P1 | Track-geometry benchmark and conventional forecast baseline | Existing external methods give falsifiable comparisons and a direct maintenance use case | Rights-cleared dataset, frozen manifest, future/route-held-out evaluation |
| P1 | Concrete healing literature/evidence registry and conventional control matrix | Avoids optimizing biology without physical baseline | All candidate claims linked to methods/source; comparability and risk fields complete |
| P2 | Calibrated property-specific predictor | Enables honest ranking and transfer checks | Predeclared holdout/calibration criteria and independent evaluator |
| P2 | Laboratory self-healing comparisons | Tests whether crack sealing improves material behavior rather than image appearance alone | Qualified lab, matched controls, replicates, biosafety review where relevant |
| P3 | Whole-life track maintenance optimizer and materials-aware digital twin | Uses qualified predictors and constrained model outputs | Site-specific verification/validation, owner review and safe read-only operation |
| P4 | Monitored pilot / real infrastructure recommendation | Highest consequence; must follow prior gates | Qualified professional approval, owner/regulator permission, field QA and withdrawal/rollback plan |

## Research reference set

### Living/self-healing concrete
1. Ranjith & Das (2025), “Self-healing bacterial concrete: A bibliometric overview and state-of-the-art review on fundamentals, techniques and future perspectives,” *Journal of Building Engineering*, 104, 112324. https://doi.org/10.1016/j.jobe.2025.112324
2. Wong et al. (2024), “Advances in microbial self-healing concrete: A critical review of mechanisms, developments, and future directions,” *Science of the Total Environment*, 947, 174553. https://doi.org/10.1016/j.scitotenv.2024.174553
3. “State-of-the-art review on self-healing in mortar, concrete, and composites” (2024). https://doi.org/10.1016/j.cscm.2024.e03298
4. FHWA, “Internally Curing Concrete Produces EPIC² Results” (2023). https://www.fhwa.dot.gov/innovation/innovator/issue98/page_01.html
5. FHWA, “Enhance Concrete Performance with Internal Curing” (2025). https://www.fhwa.dot.gov/innovation/innovator/issue103/page_06.html

### Rail condition, detection and maintenance
6. FRA, RR 26-01, “Track Geometry Predictions Using Point and Segment Processing.” https://railroads.fra.dot.gov/elibrary/track-geometry-predictions-using-point-and-segment-processing
7. FRA, RR 26-02, “Implementation of Advanced Track Geometry Forecasting Models” (page updated 30 July 2026). https://railroads.fra.dot.gov/elibrary/implementation-advanced-track-geometry-forecasting-models
8. FRA, RR 21-18, “Feasibility of Applying Model-Assisted Probability of Detection to Track Inspection.” https://railroads.fra.dot.gov/elibrary/feasibility-applying-model-assisted-probability-detection-track-inspection
9. Network Rail, CP7 Route Services Delivery Plan Year 2 update (2025), describing predictive asset-maintenance and earthworks-insight programmes. https://www.networkrail.co.uk/wp-content/uploads/2025/07/Route-Services-CP7-Delivery-Plan-Year-2-update.pdf

### South African context
10. South African Department of Transport, COTO Standard Specifications for Road and Bridge Works portal. Its page lists October 2020 chapter versions and states the earlier August versions were superseded. Always verify the current applicable edition and project-specific contractual requirements before using numerical limits. https://www.transport.gov.za/?page_id=1155

## Claim ceiling

This plan establishes a research direction, a candidate/measurement vocabulary and falsifiable next steps only. It does not establish that any proposed bioactive concrete is safe, structurally superior, economical or field-ready; that Symthaea outperforms track-geometry forecasting baselines; or that Symtropy's existing physics engine can simulate fracture or railway dynamics adequately. Those claims require measured evidence and an independently reviewed, application-specific validation argument.
