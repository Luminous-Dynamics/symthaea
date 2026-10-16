# Living City Engineering Automation and Verification Contract v0.1

**Status:** proposed implementation contract for the Living City Systems Lab, to be developed on the active Symthaea branch. This is a software and process specification, not a building-code interpretation, engineering design, permit, or construction authorization.

**Goal:** automate the repeatable work of generating, checking, comparing, documenting, and updating urban/building designs while keeping safety-critical analysis independently verified and professional/legal approvals explicit.

## 1. Operating principle

Symthaea should be an **engineering compiler and experiment coordinator**. It translates a versioned brief into candidate designs, invokes discipline tools, checks their outputs, searches the feasible design space, and assembles evidence-backed deliverables. It must not act as the sole authority on whether a design is safe or lawful.

The loop is:

1. Compile requirements and constraints.
2. Assemble and validate a versioned site/evidence package.
3. Generate bounded alternatives using explicit parameters.
4. Run domain solvers and inexpensive screening checks.
5. Independently verify critical outputs and reject failed candidates.
6. Optimize only within locally approved essential-needs and safety constraints.
7. Export interoperable models, quantities, schedules, and review packages.
8. Capture construction changes, tests, and as-built state.
9. Compare predicted and observed outcomes; version the model and lessons learned.

A generated drawing, a valid IFC file, a passing unit test, or a plausible simulation is not proof that a project is buildable. Use different statuses for software correctness, physical-model credibility, legal compliance, professional acceptance, and field commissioning.

## 2. Responsibilities across the active projects

### Symthaea — design compiler and coordinator

- Turn a human-approved project brief into typed requirements, a dependency graph, parameter ranges, and explicit exclusions.
- Generate alternative block, building, service, and infrastructure arrangements.
- Select informative experiments, allocate compute, perform sensitivity analysis, and maintain Pareto alternatives.
- Invoke solver adapters, aggregate evidence, explain constraint failures, and abstain where evidence is incomplete or outside a validated domain.
- Generate design-review checklists, decision records, and draft submission packages.

Symthaea may propose, prioritize, and automate routine checking. It does not sign engineering documents, interpret ambiguous law autonomously, or make unreviewable choices about rights, land, displacement, or public spending.

### Symtropy — physical simulation where the model is appropriate

- Host or connect domain-specific physical models for energy flows, heat transfer, fluid movement, material handling, and other mechanics only when state variables, units, boundary conditions, numerical methods, and validation evidence are explicit.
- Provide fast reduced-order models for exploration only after comparison with a trusted reference solver over a documented validity range.
- Never treat a civic indicator such as affordability, consent, or flourishing as a physical force merely to fit it into a simulation.

### Mycelix — consent, provenance, proposals and accountability

- Record authorization/consent references, proposal versions, accountable reviewers, attestations, appeals, and governance decisions as supported by the maturity of the relevant component.
- Keep sensitive household, health, identity, and property data off broadly replicated/public ledgers; store access-controlled references and minimum necessary attestations instead.
- Do not describe pre-alpha workflows as production-ready or treat ledger consensus as proof that the underlying engineering fact is true.

### Sol-Atlas — spatial review and field-to-model comparison

- Display observations, assumptions, simulated predictions, uncertainty, and missing data as visibly different layers.
- Show impacts per neighbourhood and affected cohort, not only city-wide aggregates.
- Link each displayed result to the exact scenario receipt, source dataset, model version, and open engineering issue.
- Support GIS/BIM issue references without being the authority that decides whether an engineering check passed.

## 3. Required project inputs

Every project must declare the site boundary and coordinate reference system, jurisdiction and authority having jurisdiction, applicable code editions/amendments, intended use and occupancy, design life, climate and hazard basis, service levels, budget and price date, construction method assumptions, lifecycle boundary, and the people/cohorts whose outcomes must be assessed.

Evidence packages should contain, where relevant:

- Survey/control points, cadastral boundaries, easements, right-of-way and land-use constraints.
- Geotechnical report, groundwater, contamination, slope, subsidence and foundation assumptions.
- Terrain, drainage, flood, heat, wind, seismic, wildfire and other locally relevant hazards.
- Weather observations/design weather files and their period, provenance, and applicability.
- Road, path, transit, freight, emergency access and utility networks.
- Water basin constraints, water-quality requirements, sewer/stormwater capacity, and discharge conditions.
- Electrical source, grid constraints, load profiles, protection requirements, islanding and critical-load definitions.
- Building stock, occupancy, demographic and accessibility data at an appropriately privacy-preserving resolution.
- Material specifications, environmental product declarations where available, local procurement limits, cost sources, lead times, and maintenance assumptions.
- Community-ratified service/equity floors, consultation records, alternatives considered, and appeal/correction process.

Every input is classified as **measured, derived, estimated, simulated, assumed, or missing**. Record who supplied it, when, license/legal basis, units, spatial and temporal scope, transformation history, quality limitations, and uncertainty. Missing is not zero. An assumed value is not a measurement. Conflicting sources must be surfaced, not silently averaged.

## 4. What to automate

| Workstream | Automation target | Required boundary/check |
|---|---|---|
| Brief compilation | Convert objectives into typed, traceable constraints and questions | Human approval for interpretation, rights and priorities |
| GIS/evidence ingestion | Validate CRS, units, coverage, timestamps, topology, licensing and missingness | Source provenance and spatial-resolution limits |
| Site feasibility | Screen slope, hazard exposure, access, protected land, utilities and exclusions | Local authority and licensed survey/geotechnical review |
| Urban morphology | Generate block/street/building typologies, solar access, shade, walkability and service catchments | Real topology; equity checks by cohort; no synthetic geography presented as real |
| Building configuration | Parametric massing, orientation, envelope options, space program, accessibility and service zones | Code edition/profile explicit; architect review |
| Structural design | Prepare model inputs and load cases; compare candidate grids/material systems | A qualified structural engineer owns model selection, interpretations and sign-off |
| Geotechnical/foundations | Track ground models and test foundation alternatives where data supports it | No automated substitution for site investigation or geotechnical design |
| Energy/building physics | Batch energy, comfort, passive design, peak load and storage scenarios | Calibrated weather, material properties, schedules, and reference cases |
| Water/wastewater | Demand, network pressure/flow, storage, drought, quality and failure scenarios | Potable/reuse separation and applicable water-quality limits |
| Mobility/logistics | Network accessibility, capacity, emergency access and traffic alternatives | Validate against observed counts and local movement patterns |
| Heat/wind/flood/ecology | Run defined domain models and compound-hazard scenarios | Numerical verification, validation data and uncertainty; ecology not reduced to tree counts |
| Coordination | Check required BIM properties, geometry/topology and inter-discipline clashes | Issue IDs and owners; tolerances declared; human disposition for consequential issues |
| Cost/procurement | Derive quantities from versioned objects, prices, waste, labor, logistics and replacement cycles | Price date, currency, region, source and uncertainty on each estimate |
| Construction sequencing | Generate draft work breakdown, dependencies, access/lift/logistics constraints and alternatives | Contractor means/methods, site safety and actual supply constraints reviewed by humans |
| Documentation | Generate schedules, quantities, design narratives, assumptions, test reports and permit-package indexes | No invented citations, approvals, signatures, or as-built facts |
| Construction monitoring | Compare field tests, scans, delivered product data and RFIs with the approved model | Survey/measurement uncertainty and deviation disposition recorded |
| Operations | Generate asset IDs, maintenance intervals, replacement assumptions and commissioning checks | Operator acceptance and post-occupancy measured outcomes |

Automation should concentrate first on **repeatability and early rejection**: schema validation, unit checking, infeasibility screening, batch scenario runs, quantities, traceability, change-impact analysis, report generation, and regression testing. These can reduce engineering effort without pretending that a generative model is a licensed professional.

## 5. Interoperability contract

Prefer open, version-pinned exchange formats over a bespoke city-model silo.

- **IFC 4.3 / ISO 16739-1:2024:** exchangeable building and infrastructure asset objects, properties, quantities, spatial structure, and relationships.
- **IDS 1.0:** machine-readable information requirements for which IFC objects, classifications, properties, units and values must be delivered.
- **BCF:** issue and coordination exchange linked to model elements, including location/view and resolution state.
- **OGC CityGML 3.0:** semantic 3D city-scale objects and spatial context; maintain declared mappings between city objects and BIM assets.
- **GeoPackage or another declared GIS exchange:** for geospatial layers, accompanied by CRS, licensing, extent, resolution, and provenance.

Do not assume an IFC schema check proves that the geometry is structurally sound, that the model meets the local code, or that every receiving application interpreted the exchange correctly. Add round-trip tests with at least two independent readers for supported exchanges and check critical quantities/coordinates after translation. Pin schema and model-view requirements per project; do not chase a draft version in a construction workflow without an explicit migration review.

## 6. Domain-solver integration strategy

Use adapters with a common execution contract rather than making Symthaea reimplement each mature engineering discipline. Every adapter must declare its version, inputs and outputs, units, boundary conditions, supported model range, resource needs, known limitations, benchmark suite, error/exit behavior, and provenance.

Initial candidates for feasibility testing—not integrations already completed:

| Domain | Candidate tool or standard | Integration objective |
|---|---|---|
| Whole-building energy and water use | EnergyPlus | Compare envelope, schedules, passive measures, loads and energy/water outcomes |
| Drinking-water distribution | EPA EPANET | Test flow, pressure, storage, pump energy, water age/quality, and outages |
| Urban/building airflow and heat transfer | OpenFOAM | Targeted CFD studies where simplified models are insufficient; not every candidate needs CFD |
| Traffic and network operations | Eclipse SUMO | Analyze mobility and traffic scenarios against a validated network and demand basis |
| Structural and geotechnical analysis | Qualified structural/FEM and geotechnical solvers selected per project | Compare load cases, design assumptions, mesh convergence, and code-specific checks |

Solver selection is an engineering decision. A tool being free/open-source does not guarantee suitability, correct setup, validation for a particular case, regulatory acceptability, or compatibility with the applicable code. Evaluate licensing, reproducible execution, input/output APIs, numerical robustness, benchmark credibility, and the availability of independent reviewers before integrating it into an approval gate.

Each solver adapter returns structured results plus native logs/artifacts. A nonzero exit, missing output, unrecognized warning, invalid unit, or failed convergence must never be converted into a successful scenario. Warning classifications are versioned; safety-significant warnings need explicit disposition.

## 7. Verification and validation: independent layers

Use the risk-based Verification, Validation and Uncertainty Quantification (VVUQ) approach: more consequential predictions require stronger evidence. The generator and verifier should not share all the same assumptions or calculation code, because a shared bug can make both agree incorrectly.

### Layer A — data and schema integrity

- Validate source manifests, coordinate systems, units, schema versions, geometry validity, required properties and source licenses.
- Reject invalid, stale, conflicting or incomplete decision-critical inputs unless an approved fallback rule exists.
- Keep measured, derived, assumed and simulated values separate throughout transformations.
- Detect accidental use of a dataset outside its geographic, temporal, licensing or accuracy scope.

### Layer B — mathematics and software correctness

- Unit/dimensional analysis and valid ranges for every physical field.
- Conservation checks where applicable: mass/water balance, energy balance, electrical power balance, material quantities and carbon accounting.
- Property-based, boundary, mutation/negative-control and regression tests.
- Numerical convergence and mesh/time-step sensitivity where the solver requires them.
- Deterministic replay for deterministic components. For stochastic or parallel solvers, use recorded seeds, tolerance contracts and declared hardware/nondeterminism limits.
- Differential checks against an independently implemented reference, analytical solution, trusted benchmark or second solver where appropriate.

### Layer C — physical model credibility

- Verify that equations are implemented correctly; validate separately that the chosen equations and parameters represent the intended real-world process.
- Benchmark against analytical cases, laboratory/field measurements, published reference problems or historical operating records.
- Calibrate only on a declared calibration set; reserve genuinely held-out data for evaluation.
- Report error and calibration by relevant location, season, operating range and affected cohort. State extrapolation/OOD status and abstain where unsupported.
- Propagate input and parameter uncertainty to result distributions; publish sensitivity and failure cases, not only a point estimate.

### Layer D — independent engineering checks

- Use a separate checker, alternate calculation path, or hand/reference calculation for critical load, safety and conservation quantities.
- Run jurisdiction-specific rule checks against the approved code edition, amendments, occupancy, construction type, climate zone, and project attributes.
- Run BIM information checks, spatial/geometry checks, accessible-route and egress checks, clash checks, and quantities consistency checks.
- Require accountable discipline review for unresolved high-severity issues. A machine-readable rule may flag a possible violation but must not silently adjudicate ambiguous code text.

### Layer E — project and field validation

- Compare the model with site measurements and known operating data before it supports consequential decisions.
- During construction, compare approved models with RFIs, substitutions, shop drawings, test results and as-built capture.
- Commission installed systems under declared operating and fault conditions; record acceptance criteria and witnessed test evidence.
- After occupancy, compare measured service, comfort, cost, water, energy, mobility and equity outcomes with pre-registered predictions and a comparison baseline.

A correct implementation is not necessarily a valid physical model; a valid model is not necessarily code-compliant; a code check is not a permit; and a permit is not proof of correct construction.

## 8. Evidence manifest and run receipt

Each publishable result should carry a machine-verifiable receipt. The minimum receipt fields are:

- Receipt schema/version and unique run ID.
- Repository, exact commit SHA, build/profile, relevant feature flags, dependency lockfile digest, and toolchain versions.
- Scenario/requirements ID and revision; site boundary and coordinate-reference-system identity.
- Input evidence-manifest digest and every source item's URI/identifier, license, retrieval time, units, transforms and quality label.
- Ratified constraints/floors and the decision record that authorized them.
- Exact solver/container/tool versions, parameter sets, initial/boundary conditions, seeds, environment constraints and execution status.
- Output artifact digests, unit/metric definitions, uncertainty summary, warning list, failed constraints and all validation result IDs.
- Independent verifier identity/version, verifier result, review references, and explicit statuses for every required gate.
- Reproduction command and retention locations for input/output artifacts and logs.

Use SHA-256 or an agreed cryptographic digest over canonical serialized manifests and artifacts; define canonicalization and schema evolution before signing anything. The receipt must be independently recomputable. A text field saying “verified” is not verification: verification means a checker ran against a named artifact digest with a declared version, returned an explicit result, and preserved its evidence.

Allowed gate states are **PASS, FAIL, BLOCKED, NOT RUN, and NOT APPLICABLE**. Missing evidence is BLOCKED or NOT RUN, never PASS. NOT APPLICABLE requires a reason and approval. The overall state is fail-closed: any required failed/blocked/not-run gate prevents promotion, while still allowing safe exploratory runs to continue.

## 9. Promotion gates

| Gate | Name | Minimum exit conditions |
|---|---|---|
| G0 | Brief and evidence | Requirements are approved; site/jurisdiction defined; decision-critical sources are traceable; unknowns/exclusions are explicit |
| G1 | Concept feasibility | Geometry and resource budgets valid; hazard and essential-access screens run; all hard floors evaluated |
| G2 | Model credibility | Domain solvers pass their own benchmark protocols; calibration/validation domain and uncertainty are declared |
| G3 | Engineering coordination | Discipline checks, load cases, units, conservation tests and independent critical checks pass; remaining issues are dispositioned |
| G4 | Coordinated BIM | Declared IFC/IDS requirements pass; supported round-trips pass; critical clashes are resolved; quantities reconcile within approved tolerances |
| G5 | Construction package review | Site-specific drawings/specifications/calculations, quantities, construction assumptions, safety reviews and local approval package are complete; required licensed professionals approve/sign |
| G6 | Construction and commissioning | Field deviations are dispositioned; installed systems are tested to acceptance criteria; as-built assets and maintenance data are accepted |
| G7 | Outcome validation | Measured outcomes are compared with the baseline and predictions; error, uncertainty, distributional effects and failures are published with privacy protections |

Passing an earlier gate authorizes only the next stage. A conceptual candidate must never be labeled construction-ready. Approval gates and professional responsibilities are selected for the specific jurisdiction and project; this document does not replace them.

## 10. Benchmark and acceptance suite

Start with deterministic synthetic fixtures to test the pipeline, then progress to public, license-compatible reference cases, and only later to a real pilot.

### Software and evidence invariants

- 100% of decision-bearing metrics link to evidence, a declared assumption, or a named model output.
- No missing value becomes zero without a domain-specific, visible and approved imputation rule.
- Every required safety-critical case is either executed or blocks promotion.
- Every result includes exact revisions, units, configurations and validation results.
- Invalid inputs, missing outputs, failed solver convergence, unknown warnings and digest mismatches fail closed.
- Deterministic components replay under the same pinned inputs and versions; numerical solvers use defined tolerance contracts.
- Negative controls prove that the checker catches intentionally introduced defects.
- Independent check paths catch deliberately corrupted outputs and stale artifacts.
- Pareto comparison occurs only among candidates satisfying required hard floors; infeasible cases and unresolved trade-offs remain visible.

### Physical and planning benchmark families

- Simple structural reference problems with analytical/reference results and mesh-convergence checks.
- Energy and thermal cases with known geometry, weather, schedules, and reference measurements.
- Water-network cases with reference pressures/flows and demand/outage scenarios.
- GIS accessibility fixtures with known travel-time/reachability answers and boundary conditions.
- BIM exchange fixtures testing geometry, units, GUID stability, required properties, quantities and round-trip change detection.
- Heat + outage, drought + energy shortfall, flood + communications failure, material/utility interruption and other locally relevant compound hazards.
- Out-of-sample historical or field comparisons, with pre-registered error metrics and explicit limits on generalization.

Do not invent one universal acceptable error number for every domain. For each metric and domain, the accountable engineer defines an acceptance tolerance from the decision consequence, measurement error, reference quality and applicable standard before seeing test outcomes. Preserve failed cases and negative controls.

## 11. Construction deliverables to automate

As the model matures, generate a traceable package rather than a rendered master plan alone:

- Site constraints plan and survey/data provenance register.
- Parametric massing and block layout, street/public-space plan, accessible-route network, landscape/ecological plan.
- Building and infrastructure BIM exchange models; schedules of spaces, equipment, materials and asset identifiers.
- Discipline calculation inputs/outputs, assumptions, load cases, model cards, uncertainty summaries and check reports.
- Clash/coordination issue register with owners, severity, evidence, due dates and dispositions.
- Quantity take-offs, bill of materials, lifecycle-cost model, price-date and supplier assumptions, lead-time risks, embodied-carbon methodology.
- Construction sequencing and logistics model, risk register and inspection/test plan.
- Draft drawings/specifications and permit-package index tied to model/revision IDs; required professional signatures must be added by the authorized people.
- Commissioning scripts/checklists, acceptance criteria, sensor/asset registry, as-built model and operational maintenance plan.

Outputs must clearly indicate whether they are concept, feasibility, coordinated design, submitted-for-approval, approved, or as-built. Automated quantities and cost estimates must show their extraction method, exclusions, waste factors, price source/date, and uncertainty.

## 12. Implementation order for Symthaea

**P0 — in the current living-city PR**
1. Specify versioned evidence manifests, run receipts and strict gate states.
2. Add deterministic synthetic scenario fixtures for normal, heat, water shortage, outage, and compound stress; preserve known-bad negative controls.
3. Add manifest/receipt validation that rejects missing provenance and inconsistent digest/schema/status fields.
4. Add a documented verifier interface whose result is independent of the scenario generator's success status.
5. Update the PR verification statement only from evidence produced against its exact current head.

**P1 — before using real datasets for design decisions**
1. Create a single licensed, bounded, public reference study area with baseline and ground-truth/benchmark sources.
2. Replace mocked geography in the town prototype; make CRS, grid frequency, unit systems, time horizon, climate and resource parameters explicit configuration.
3. Separate physical production accounting from Tend issuance; Φ must not be an input to monetary issuance or basic service entitlements.
4. Add adapters as separate modules/processes with pinned versions, native logs, structured outputs and benchmark suites.
5. Establish the first energy + water + heat/outage integration slice; add mobility and structural/architectural tooling only with defined verification cases.

**P2 — before producing construction packages**
1. Implement IFC 4.3 exchange, IDS-based information checks, BCF issue exchange, and CityGML-to-BIM spatial mapping for a declared use case.
2. Add jurisdiction profiles sourced from the authority having jurisdiction, with versioned rules and domain-specific review.
3. Add independent checks for critical structural, fire/egress, electrical, water, accessibility and environmental requirements relevant to that project.
4. Add quantity/cost/schedule reconciliation and change-impact analysis.
5. Establish independent professional review and a formal sign-off/approval workflow.

**P3 — construction and operations**
1. Ingest inspections, test certificates, product substitutions, RFIs, field measurements and as-built scans with provenance.
2. Compare as-built versus approved model and block unresolved safety-critical deviations from being hidden.
3. Commission systems against explicit tests; synchronize only validated operational observations.
4. Use post-occupancy evidence to assess predictive errors, maintenance burden, equity effects and real outcomes.

## 13. First executable pilot

Use a small, explicitly bounded neighbourhood and evaluate six alternatives against the same baseline: business as usual; critical-facility energy resilience; heat-safe routes/cooling; water efficiency and reuse; accessible mixed-use housing with anti-displacement measures; and an integrated portfolio.

Minimum stress set: baseline operations, design heat event, drought case, 72-hour outage for named critical assets, plus at least one locally justified compound hazard. For each design, publish feasibility failures, Pareto trade-offs, distributional outcomes, sensitivity, uncertainty, solver status and complete receipt.

**Exit criterion for the pilot:** a third party can reconstruct the input bundle, run the pinned workflow, independently check the defined invariants, understand the limits of every model, and reproduce the stated result. This does not alone authorize construction; jurisdictional review, professional approvals, procurement, site investigation, and field commissioning remain necessary.

## References

- buildingSMART, IFC 4.3 / ISO 16739-1:2024: https://standards.buildingsmart.org/IFC/RELEASE/IFC4_3/
- buildingSMART, Information Delivery Specification (IDS): https://technical.buildingsmart.org/Projects/Information-Delivery-Specification-Ids/
- buildingSMART, BIM Collaboration Format (BCF): https://technical.buildingsmart.org/standards/bcf/
- OGC, CityGML 3.0: https://www.ogc.org/standards/citygml/
- US EPA, EPANET: https://www.epa.gov/water-research/epanet
- EnergyPlus documentation: https://energyplus.readthedocs.io/en/latest/
- OpenFOAM Foundation: https://openfoam.org/
- Eclipse SUMO documentation: https://eclipse.dev/sumo/docs/
- NIST, On Methods of Uncertainty Quantification in Computational Modeling and Simulation (NIST AMS 100-85, 2026): https://doi.org/10.6028/NIST.AMS.100-85
- NIST, A Summary of Industrial Verification, Validation and Uncertainty Quantification Procedures in Computational Fluid Dynamics (NISTIR 8298): https://doi.org/10.6028/NIST.IR.8298
- NIST, Digital Twins for Advanced Manufacturing / VVUQ: https://www.nist.gov/programs-projects/digital-twins-advanced-manufacturing

## Decision rule

Automate all work that can be made explicit, repeatable and independently checked. Use AI to expand the design space and learn from feedback. Keep safety-critical claims tied to domain models, measurements, applicable codes, accountable professional judgement and field evidence.

## 14. Paradigm shift and first executable benchmark suite

The next shift is from testing isolated predictions to testing whether a proposed settlement remains safe and useful when interacting systems fail together. This is **robust, evidence-constrained design**, not just bigger simulations or a single optimizer score.

The active Rust registry is crates/domains/symthaea-scenarios/src/living_city_benchmarks.rs. It defines a versioned initial suite with one baseline, three individual hazards, and three compound hazards:

| Case ID | Stress | Required model domains |
|---|---|---|
| baseline-operating-envelope | Baseline / business-as-usual reference | Thermal, energy, water, mobility, critical services, housing/access, ecology/materials |
| heat-design-event | Locally defined extreme heat | Thermal, cooling access, energy, critical services |
| drought-design-case | Locally defined drought | Water, energy, ecology/materials |
| critical-services-72h-grid-outage | 72-hour grid outage for named critical assets | Energy, water, communications, critical services, emergency access |
| heat-plus-72h-grid-outage | Heat + 72-hour grid outage | Thermal, cooling access, energy, communications, critical services |
| drought-plus-energy-shortfall | Drought + constrained energy availability | Water, energy, critical services, ecology/materials |
| flood-plus-communications-failure | Flood + loss of communications | Stormwater, communications, emergency access, critical services, mobility |

These IDs are test definitions, **not pre-filled hazard severities or claims about any real site**. The profile for each case must be selected and versioned for the study area; its evidence bundle must contain the site/climate/boundary conditions and relevant local assumptions. The 72-hour outage is an initial test condition, not a universal design requirement or a claim that any system will survive that duration.

### Initial software acceptance criteria

- Registry contains exactly one baseline, correctly formed single-hazard and compound-hazard cases, unique IDs and at least one declared model domain per case.
- Every required case has exactly one local profile binding with an evidence digest.
- Unknown/duplicate execution records are rejected.
- A PASS record requires independent verifier identity/version and a digest of the exact result artifact.
- A failed mandatory case yields FAILED; missing, blocked or not-run mandatory cases yield BLOCKED; only a fully recorded suite is eligible for the next workflow stage.
- Negative controls show that stale/missing records and deliberately failed compound-hazard cases are detected.
- Passing this software contract means only that coverage/result metadata meets the workflow contract. It does not validate the underlying physical equations, prove calibrated predictions, or authorize construction.

### Paradigm shifts to prioritize

1. **From nominal optimization to robust feasibility.** First satisfy locally authorized rights and safety constraints; then test worst-case/compound conditions and uncertainty. Do not let average performance hide a critical failure in one neighbourhood or cohort.
2. **From one giant digital twin to federated evidence-backed models.** Retain domain-specific models and connect them through explicit units, boundary conditions and provenance. Each model should own its validity limits; the orchestrator should expose disagreement rather than blend it away.
3. **From generative confidence to independent falsification.** Give the verifier a separate code path, preserve negative controls, and deliberately corrupt artifacts/configurations to prove the verifier catches them.
4. **From data accumulation to decision-relevant evidence.** Use sensitivity analysis and value-of-information experiments to identify which new survey, sensor, simulation, or field test is most likely to change a design decision. Do not collect data merely to make the twin look comprehensive.
5. **From design completion to a digital thread.** Keep requirement IDs, model inputs, outputs, BIM objects, quantities, construction changes, commissioning tests and post-occupancy observations linked across versions.
6. **From sustainability proxies to lifecycle accounting.** Count embodied and operational impacts, upstream/downstream effects, maintenance and end-of-life. Keep ecological indicators and social outcomes distinct rather than claiming one score represents flourishing.

### Research anchors

- NIST, On Methods of Uncertainty Quantification in Computational Modeling and Simulation: Harmonizing Modeling and Metrology Practices, NIST AMS 100-85 (published 5 October 2026): https://doi.org/10.6028/NIST.AMS.100-85. Use its risk-based credibility framing: verification and uncertainty effort should scale with model influence and decision consequence.
- NASA-STD-7009B, Standard for Models and Simulations (active version listed by NASA): https://standards.nasa.gov/standard/nasa/nasa-std-7009. Useful reference for defining project-specific acceptance criteria and model credibility evidence.
- buildingSMART official standards server: https://standards.buildingsmart.org/. Pin the released IFC 4.3.2.0 / ISO 16739-1:2024 and IDS 1.0 releases for the initial information-exchange workflow rather than silently tracking in-development schemas.

## 15. Current verifier implementation boundary

The Rust evidence module now implements SHA-256 over canonical manifest/receipt bytes and verification of retained source/output bytes against declared digests. It also checks that source/output byte sets match the declared IDs and that a run receipt references the exact canonical evidence manifest and scenario identity. Synthetic tests include NIST SHA-256 vectors and altered-byte/metadata negative controls.

Relevant APIs are sha256_hex, evidence_manifest_sha256, verify_evidence_manifest_digest, scenario_run_receipt_sha256, verify_scenario_run_receipt_digest, verify_manifest_source_bytes, verify_receipt_output_bytes, and verify_receipt_manifest_link in crates/domains/symthaea-scenarios/src/living_city_evidence.rs.

An offline verifier command is now present at crates/domains/symthaea-scenarios/src/bin/living_city_verify.rs. Bundle schema v2 reads and validates the versioned hazard profile and profile binding plus the serialized service graph, verifies retained source/output bytes, computes a canonical configuration digest over the profile/binding/graph, and reports the deterministic dependency cascade. It is not yet wired into the scenario orchestrator/CI promotion gates and does not authenticate a trusted signer. Access-controlled artifact retention, a signing/attestation policy where authorship matters, and exact-head execution evidence remain required before this can be called a fully verified engineering pipeline. Tests have been added but were not run in the available environment because Rust/Cargo are absent; skipped CI is not a pass.

## 16. Offline run-package verifier

The verifier executable is at crates/domains/symthaea-scenarios/src/bin/living_city_verify.rs. Bundle schema v3 loads a typed evidence manifest, scenario-run receipt, versioned hazard profile, profile binding, service dependency graph, and evidence-linked city-assessment package. It uses strict JSON parsing, rejects unknown fields, constrains file paths to the canonical package root (including symlink resolution), streams SHA-256 across large source/output files, verifies exact source/output ID sets, binds the receipt to the manifest, evaluates the hazard profile against the declared graph, independently recomputes the city metrics/floor assessment, and verifies the configuration digest binds the profile, binding, canonical graph, city assessment, and built-in grid-reference suite identity.

### Package layout

A run directory contains bundle.json, manifest.json, receipt.json, receipt.sha256, hazard-profile.json, profile-binding.json, service-graph.json, city-assessment.json, and all retained source/output files. A bundle schema v3 example:

~~~json
{
  "schema_version": 3,
  "manifest_path": "manifest.json",
  "receipt_path": "receipt.json",
  "receipt_sha256_path": "receipt.sha256",
  "hazard_profile_path": "hazard-profile.json",
  "profile_binding_path": "profile-binding.json",
  "service_graph_path": "service-graph.json",
  "city_assessment_path": "city-assessment.json",
  "source_files": [
    {"id": "heat-basis", "path": "sources/heat-basis.json"},
    {"id": "outage-basis", "path": "sources/outage-basis.json"},
    {"id": "city-model-output", "path": "sources/city-model-output.json"},
    {"id": "floor-ratification", "path": "sources/floor-ratification.json"},
    {"id": "city-metrics-payload", "path": "sources/city-metrics.json"},
    {"id": "city-assessment-package", "path": "city-assessment.json"}
  ],
  "output_files": [
    {"id": "metrics", "path": "outputs/metrics.json"}
  ]
}
~~~

The service graph has schema version 1 and typed nodes/dependencies, for example:

~~~json
{
  "schema_version": 1,
  "nodes": [
    {"id": "grid", "domain": "electricity"},
    {"id": "cooling-centre", "domain": "cooling"}
  ],
  "dependencies": [
    {
      "dependent_id": "cooling-centre",
      "prerequisite_id": "grid",
      "mode": "required"
    }
  ]
}
~~~

The manifest and receipt use the versioned Rust types, with enum values in snake_case; unknown fields are rejected. Every non-missing manifest item needs exactly one source-file binding with the same ID and matching file digest. Every receipt output needs exactly one output-file binding. Explicitly missing sources have no source-file binding. A missing source can be structurally recorded, but whether it blocks progression depends on the project's mandatory evidence gates and decision-critical requirements.

Every hazard parameter and initiating service failure must reference an item in the exact manifest for the same benchmark scenario, and the item must not be marked missing. The profile binding must identify the profile revision and match its recomputed canonical digest. Each benchmark-declared hazard needs at least one finite parameter with an explicit unit and evidence ID.

The city-assessment package contains the ten decision-bearing city metrics, six locally configured non-compensatory floors, a per-metric evidence-lineage map, and an explicit floor-ratification reference. Every metric must cite at least one non-missing evidence item; the exact canonical metrics JSON payload is retained as city-metrics-payload and its digest must match the embedded numeric values. The exact compact city-assessment JSON bytes are retained as city-assessment-package and must match its canonical digest. Floors must reference an evidence item classified governance_decision, and the ratification reference must match that item's source URI. This establishes source/digest linkage—not that the derivation formula is valid, the evidence is truthful, or a genuine public mandate exists.

The receipt's configuration_sha256 binds five canonical components: the hazard profile's canonical bytes, the serialized profile binding, the canonical service graph, the canonical city-assessment package, and the built-in grid reference suite version, suite ID, and ordered declared case IDs. The graph canonicalizer sorts nodes by ID and dependency edges by their derived deterministic order; each component is preceded by its length as an unsigned 64-bit little-endian integer before the combined bytes are hashed. The receipt separately binds to the evidence-manifest digest. The receipt.sha256 sidecar contains the lowercase SHA-256 digest of the receipt's canonical representation, optionally followed by a newline.

All paths must be relative and resolve to regular files inside the run directory. The verifier streams file contents instead of buffering potentially large GIS/BIM/simulation artifacts.

Run from a repository checkout using:

~~~sh
cargo run -p symthaea-scenarios --bin living_city_verify -- path/to/run-package
~~~

On success, stdout contains a JSON report with package_integrity set to verified, the receipt's separate workflow_promotion_state, run/scenario/commit identifiers, recomputed digests, verified source/output counts, a recomputed city_design_feasibility assessment, the hazard-profile identity, the dependency pre-screen's failed/degraded service IDs and propagation rounds, and a four-case LinDistFlow analytical reference suite. A city scenario that violates any configured hard floor is reported as infeasible and cannot be promoted even if all artifacts are correctly hashed and the declared service graph is internally consistent. It checks (1) a single branch with 7.2 kV base, 1 ohm resistance and 100 kW real load; (2) a reactive-only 100 kVAR load; (3) a two-segment radial feeder with downstream 50 kW load aggregation; and (4) -100 kW net distributed generation causing voltage rise. Each bus voltage and expected branch real power is compared against the closed-form linearized DistFlow equation using fixed regression tolerances. A failed reference case prevents workflow promotion. This checks implementation against the simplified model's analytical equations; it does not validate nonlinear AC power flow, utility-grid adequacy or actual outage behavior. Package integrity and workflow eligibility are separate: an internally consistent run can still have a failed or blocked mandatory engineering gate. The dependency cascade is only the consequence of declared initiating failures over a declared graph. It is not a prediction of actual hydraulic pressure, electrical load flow, thermal response, capacity, failure probability or restoration time. Verification errors exit with status 2 and a diagnostic on stderr.

Tests now include end-to-end synthetic package verification plus infeasible-city floor rejection, city-metric payload corruption, source/output corruption, changed profile, changed graph/configuration digest, receipt-sidecar mismatch, path traversal, unknown-field, bundle-schema and graph-ordering negative controls. The cross-domain crate also contains a fixed analytical LinDistFlow reference check with its own tolerance assertions. These tests have been added but not executed in the current environment.

**Security and evidence scope:** the executable checks content integrity, profile/manifest linkage and declared dependency propagation. It does not authenticate the author's identity or signer's authority, prove that a declared solver ran correctly, validate physical equations, establish jurisdictional compliance, or authorize construction. A package can be internally consistent and still contain a false claim if all files are maliciously replaced together. Trusted signing/attestation and custody policy remain separate work. The CLI has not been compiled or run in this environment because Cargo/rustc are unavailable; treat it as unverified until tested on the exact head.
## 17. Infrastructure dependency propagation pre-screen

The active Rust module crates/domains/symthaea-scenarios/src/living_city_dependencies.rs adds an explicit cross-sector dependency graph. It accepts declared services, required/degraded dependency edges, and a set of initiating failures. Required dependencies propagate outages to a fixed point; degraded dependencies report reduced service without automatically declaring it unavailable. The result lists each service's status and the failed prerequisites that explain it.

The initial negative-control fixture models a grid outage propagating through a water pump, cooling centre and telecom tower to emergency dispatch and a hospital. Other tests cover degraded-only dependencies, cyclic dependency graphs, empty initiating failures, duplicate nodes/edges/initiators, self-dependencies and unknown endpoints.

This is a **deterministic topology/logic pre-screen**, not a physical simulator. It cannot calculate electrical flows, hydraulic pressure, cooling demand, capacity thresholds, failure probability, restoration time or common-cause failures. Initial failures must come from a versioned hazard profile, field observations or a validated domain solver. Use the graph to expose which subsystems become unavailable under declared assumptions before launching expensive multiphysics analysis.

### Acceptance benchmarks for the dependency layer

- Deterministic result ordering for a fixed graph and initiating-failure set.
- 100% detection of undeclared endpoints, duplicate IDs/edges, self-dependencies and unknown initiating failures in the negative-control suite.
- Monotonic failure propagation: adding initiating failures cannot restore a previously failed node in the same static scenario.
- All reported failed/degraded services include explicit direct failed-prerequisite references where applicable.
- Cycles terminate, and the number of propagation rounds remains bounded by the finite number of graph nodes.
- Permuting input node/edge order does not change canonical outcome statuses or causal prerequisite sets.
- A static graph pass never gets interpreted as a reliability, safety or outage-duration prediction.

The next integration is to map stress-case profiles to initiating-failure sets and compare the graph's expected cascades with the results from at least one reference-domain solver. Keep capacity and timing effects in the relevant physical model; do not add arbitrary edge weights just to make this graph imitate physics.

## 18. Evidence-linked hazard profiles and dependency graph bridge

The active module crates/domains/symthaea-scenarios/src/living_city_hazard_profiles.rs connects the benchmark registry, evidence manifest and dependency propagation layer.

A profile identifies its schema version, profile/revision ID, benchmark ID, study-area ID, jurisdiction and time horizon. For each declared hazard, it requires at least one named parameter with an explicit numeric value, unit and evidence-manifest item ID. Each initiating failure names a service node, causal hazard, evidence item and rationale. Profile digests use canonicalized JSON after normalizing unordered parameter and failure collections.

Evaluation rejects:
- An unknown benchmark, invalid benchmark registry, profile/binding revision or stale profile digest.
- A mismatch between the manifest scenario ID and benchmark ID.
- A parameter for a hazard not declared by the benchmark, missing hazard coverage, duplicate parameters, NaN/infinite values, or missing units.
- A parameter or initiating failure whose evidence ID is absent or marked missing.
- Initiating failures for unknown service IDs, undeclared hazard causes, or duplicate cause/evidence tuples.
- A baseline profile that injects hazard stressors or hazard-caused initial failures.
- Malformed service dependency graphs.

Only after these checks does the module pass the declared initiating-failure set to the deterministic dependency propagation evaluator. The resulting causal cascade is therefore tied to a versioned profile and evidence package rather than to an unlabelled list of failed nodes.

The profile bridge is a screening and provenance contract, not a physics engine. An assumed input is still an assumption even though it has an evidence ID; a profile must not label an assumed temperature, outage, flood depth or service failure as measured. Bundle schema v2 now loads the profile and service graph in the offline CLI, verifies the evidence bytes, recomputes the configuration digest and reports the cascade. Wiring the CLI result into scenario orchestration/promotion, followed by one reference-domain solver, remains the next milestone.

### Standards research and interoperability implications

- The official buildingSMART standards server identifies IFC 4.3.2.0 / IFC4X3_ADD2 as the final official release aligned with ISO 16739-1:2024. IFC 4.4 and IFC 5 listings are in development; do not track them implicitly for the first construction exchange profile. Source: https://standards.buildingsmart.org/ and https://technical.buildingsmart.org/standards/ifc/ifc-schema-specifications/
- OGC CityGML 3.0 defines a semantic conceptual model for city-scale objects that can be implemented through different encodings. Keep the urban information model separate from the chosen file/API encoding, and test semantic round-trips rather than assuming a GML-only representation. Source: https://www.ogc.org/standards/citygml/ and https://docs.ogc.org/is/20-010/20-010.html
- OGC's urban digital twin discussion emphasizes integrating infrastructure, the natural environment and people, with dynamic observations/sensor feeds and interoperable geospatial services. This supports keeping service outcomes, ecological context, and distributions of human impact linked to spatial assets without collapsing them into one score. Source: https://docs.ogc.org/dp/24-025.html

### Additional acceptance criteria for profile-to-cascade integration

1. Every numeric hazard parameter is bound to a non-missing evidence-manifest item; the retained bytes for that item must pass the offline verifier.
2. Every initiating failure links to a declared hazard and a service ID in the declared graph; no implicit nodes, defaults, or unknown IDs.
3. The binding's profile digest is recomputed from canonical profile bytes; stale profile values cannot reuse an old binding.
4. The same profile and graph produce the same ordered service outcomes regardless of source collection ordering.
5. A declared assumption stays visibly classified as an assumption; evidence presence does not promote it to a measurement.
6. The topology screen is compared with a documented reference-domain model before its cascades are used in real design decisions.

The built-in reference runner is crates/domains/symthaea-scenarios/src/living_city_grid_reference.rs. Suite version 2 contains four deterministic cases: real-power voltage drop, reactive-power voltage drop, a two-segment feeder that checks downstream load aggregation, and DER-driven voltage rise. Every result declares validation_kind as analytical_equation_regression and carries a machine-readable solver_envelope noting that the model is balanced, radial and linearized, drops the quadratic loss term, and does not cover unbalanced phases, meshed networks, transients or nonlinear AC flow. Its report exposes all case results, the maximum voltage/power error per case, and the number of buses checked. The suite identity, solver envelope and ordered case IDs are included in the verifier's report; the suite identity and ordered case IDs are included in the verifier configuration digest. Changing the declared suite requires a version/identity update and regenerating the run receipt.
