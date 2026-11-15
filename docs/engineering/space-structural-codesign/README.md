# Space Structural Co-Design v0.1

**Programme owner:** Symthaea design intelligence.  
**Physics execution:** Symtropy and independently qualified external backends.  
**Tracking:** [SPACE-STRUCT-CODESIGN-001 #12319](https://github.com/Luminous-Dynamics/symthaea/issues/12319).  
**Initial application:** cislunar orbital structures, rotating artificial-gravity habitats and orbital shipyards.  
**Status:** research contract and synthetic conformance corpus; no design or physical result is qualified.

## 1. Decision

Symthaea should act as an **adaptive engineering scientist** for space structures. It should convert mission/settlement requirements into candidate design questions, select the next admissible analysis or experiment, and compare results under uncertainty and finite engineering budgets.

Symthaea should not become a general-purpose structural solver, duplicate the FEM/mesh/eigen-solver stack, or self-authorize engineering decisions.

Symtropy remains the deterministic execution, replay, and independent-verification side. Qualified external tools remain the physics backends. Physical tests and as-built observations remain distinct evidence.

The core loop is:

`requirements → candidate architecture → analysis subject → admissibility check → qualified solver/experiment → independent receipt → discrepancy/UQ update → robust frontier → next decision`

A generated shape is not a solved structure. A solver run is not physical truth. A Pareto candidate is not approved for construction, flight or occupancy.

## 2. Existing-code baseline

Symthaea's `symthaea-structural` currently provides a bounded reference layer:
- rectangular, circular and hollow-circular section properties;
- four canonical statically determinate Euler–Bernoulli beam cases;
- axial stress, strain and elongation;
- Euler buckling reference load;
- 2D statically determinate pin-jointed truss method-of-joints solution;
- convenience material presets for A36 steel and 6061-T6 aluminium.

Its README explicitly excludes frames, indeterminate/stiffness-method structures and dynamics. Those are not implied capabilities.

Relevant Symthaea owners:
- [ENG-STRUCT-002 #5329](https://github.com/Luminous-Dynamics/symthaea/issues/5329): 3D structural solver admission.
- [ENG-MESH-001 #5332](https://github.com/Luminous-Dynamics/symthaea/issues/5332): FE mesh identity, Gmsh translation and convergence provenance.
- [STRUCT-REF-001 #6062](https://github.com/Luminous-Dynamics/symthaea/issues/6062): existing reference statics qualification.
- [STRUCT-CONSOLIDATE-001 #6078](https://github.com/Luminous-Dynamics/symthaea/issues/6078): reconcile duplicated structural equations.
- [ENG-COUPLING-001 #6173](https://github.com/Luminous-Dynamics/symthaea/issues/6173): cross-domain transfer semantics.
- [ENG-OPT-001 #5635](https://github.com/Luminous-Dynamics/symthaea/issues/5635), [robust Pareto #7787](https://github.com/Luminous-Dynamics/symthaea/issues/7787), and [decision-impact benchmark #7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803): reuse existing design/decision methods.
- [adaptive engineering loop #10902](https://github.com/Luminous-Dynamics/symthaea/issues/10902): reuse the dry-run study/proposal/receipt pattern.
- [settlement systems #7687](https://github.com/Luminous-Dynamics/symthaea/issues/7687) and [artificial-gravity human research #12222](https://github.com/Luminous-Dynamics/symthaea/issues/12222): mission and human-factor consumers.

Relevant Symtropy owners:
- [rotating prestress reference #5353 / PR #5358](https://github.com/Luminous-Dynamics/symtropy/pull/5358);
- [FEM qualification #2303 and #2321](https://github.com/Luminous-Dynamics/symtropy/issues/2303);
- [sparse modal path #2616](https://github.com/Luminous-Dynamics/symtropy/issues/2616);
- [orbital truss acceptance #4263](https://github.com/Luminous-Dynamics/symtropy/issues/4263);
- [rotating habitat structure trade study #5280](https://github.com/Luminous-Dynamics/symtropy/issues/5280);
- [cislunar industrialization gate #5341](https://github.com/Luminous-Dynamics/symtropy/issues/5341).



### Cross-stack integration precedent

Symtropy [ENERGY-SYSTEMS-STUDIO-001 #2721](https://github.com/Luminous-Dynamics/symtropy/issues/2721) already establishes the general collaboration pattern in another domain: Symthaea proposes candidate architectures and policies; Symtropy executes deterministic twins and fault replay; Sol-Atlas supplies spatial/resource evidence; Mycelix preserves provenance and approvals. This space-structure programme specializes that pattern for structural model applicability, mesh/material/load identity and solver qualification. It does not replace or fork the generic integration pattern.


### Structural baseline gates that remain draft-only

As of 15 November 2026, two closely related Symthaea changes are still open draft PRs, not merged baseline capabilities:

- [#6067 — fail-closed checked structural reference facade](https://github.com/Luminous-Dynamics/symthaea/pull/6067). Its checked-domain API must be reviewed and merged before claim-bearing consumers depend on it.
- [#6081 — reaction-force dimensionality repair and proxy-semantics limits](https://github.com/Luminous-Dynamics/symthaea/pull/6081). This addresses a dimensional mismatch in the fabrication analytical path where a force-valued API had exposed stored energy. Until the correction is merged and qualified, do not use that legacy API as a force oracle.

The space proposal emitter must consume only checked/qualified structural outputs, or label legacy/reference outputs as restricted. The existing built-in section/material presets are convenience defaults, not batch/process-conditioned material evidence. These are explicit dependency gates, not new replacement implementations.

## 3. Paradigm shifts

### 3.1 Candidate and fidelity are chosen together

The engineering question is not just "which geometry is best?" It is also "which model can validly answer the question, at what cost, and what evidence would be needed next?"

Symthaea proposes a candidate and an admissible evaluation. Symtropy checks the exact analysis subject and executes only a qualified model/backend. When no model covers the question, the result is an explicit abstention.

### 3.2 Compile engineering intent into a solver-neutral subject

A candidate must bind exact geometry/configuration generation, material state, process/batch lineage, coordinate frames, load/support cases, model assumptions, requested quantities, mesh and backend requirements.

The following substitutions are invalid:
- same friendly geometry name for a changed geometry;
- a surface/fabrication mesh used as a volume FE mesh;
- a material label used instead of actual alloy/process-state evidence;
- a field named "pressure" or "force" used without quantity/frame/support/time semantics;
- a solver's process exit code used as convergence or correctness.

### 3.3 Design for manufacturability, inspection and repair

Manufacturing and serviceability are design variables, not post-processing checklists. The candidate contract should preserve:
- segmentation and replaceable interfaces;
- joint access and tool envelopes;
- inspection datums, fiducials and sensor/access provisions;
- assembly sequence and tolerance stack;
- material/process route and batch-specific property evidence;
- local versus imported replacement parts;
- recycling and repair constraints.

### 3.4 Robust feasibility before scalar optimization

Hard constraints are not compensable by a weighted score. Keep separate objective axes for structural mass, stiffness/dynamic response, life, repair burden, manufacturing feasibility, Earth-launched mass, process energy, availability and lifecycle cost.

Use existing interval/set-valued robust-Pareto semantics. An interval is not a probability distribution. Do not infer likelihoods from interval midpoints or widths.

### 3.5 Decision value over information gain

Choose the next analysis because it may change a consequential design decision, resolve a hard constraint, distinguish surviving robust alternatives or falsify a model assumption—not merely because it has high predicted variance or entropy.

Count the full cost of the engineering loop: failed runs, model escalation, data preparation, inspection, physical tests, and required later validation.

### 3.6 Manufacturing material evidence is part of the constitutive model

For off-Earth products, a generic material name is insufficient. Material inputs must bind batch/process-conditioned composition, porosity, defects, anisotropy, thermal history, surface state, test method, temperature range, uncertainty and provenance.

Lunar regolith occurrence does not establish a qualified structural material. Bulk shielding and aggregate acceptance do not establish pressure-boundary, fatigue, joint or precision-component suitability.

### 3.7 Open-world abstention and discrepancy learning

Unsupported shell, joint, contact, gyroscopic, fatigue, thermal or nonlinear requests remain out of domain until an applicable model is admitted. A Symthaea surrogate cannot silently replace a solver. Failed predictions remain in the corpus and update discrepancy/uncertainty; they are not deleted or tuned away.

## 4. Cross-project contract

The machine-readable companion corpus uses these semantic objects:

### EngineeringDesignCandidateV1
Binds the design generation, requirements, geometry/configuration, variable vector, exact material/process references, load/environment set, manufacturing/repair/inspection profile, hard constraints, objectives, uncertainty semantics, evidence gaps and claim ceiling.

### StructuralAnalysisRequestV1
Binds the exact candidate and geometry generation, material and load profiles, frames/units/signs, model family, included/excluded physics, mesh reference if needed, quantities of interest, allowlisted backend/profile, convergence policy, oracle references, resource budget and claim ceiling.

### StructuralAnalysisReceiptRefV1
References exact request/model/mesh/backend identity, raw artifacts and digests, execution/termination/convergence states, per-observable results, residuals, uncertainty, applicability, independent-oracle disposition, evidence class and claim ceiling.

### DesignEvidenceUpdateV1
References an exact receipt and candidate/model generation. It keeps numerical residual, model discrepancy and physical observation error separate; any model update creates a new generation and requalification obligation.

These are semantic handoff contracts. They must reuse the existing repositories' canonical IDs, digests, evidence registries, unit and frame systems rather than creating competing ontologies or registries.

### 4.1 Dry-run proposal emitter (initial executable slice)

The `symthaea-engineering` facade exports `space_structural_codesign`, a pure proposal emitter. It does not optimize, solve, execute, contact Symtropy, or authorize a design.

V0.1 can emit only:
- `ThinRotatingRingV0_1` using the exact Symtropy ring model-profile reference;
- `UniformRadialSpokeV0_1` using the exact Symtropy spoke model-profile reference.

The emitted proposal binds caller-supplied request, mission, candidate, geometry, material, process-batch (when present), load-case, frame, model-profile and backend references. It pins the reviewed draft Symtropy source commit `540fe6d2fc2fd99daa5277cbc2fe768db722cda9`, declares draft maturity/claim ceiling, and sets solver execution, execution authority, design-release authority and mesh use to false.

The emitter abstains on unsupported profiles/physics/observables, malformed identities, profile/backend substitution, non-synthetic use of the draft profile, generic/unresolved material evidence, and off-Earth batch properties not admitted by a qualified material profile. It never invents artifact digests: the canonical artifact owner must supply them and the Symtropy receiver must independently resolve/check them.

Implementation and wire artefacts:
- `crates/domains/symthaea-engineering/src/space_structural_codesign.rs`
- `proposal-v0.1.schema.json`
- `proposal-example-ring-v0.1.json`

The module has 13 focused unit tests. The synthetic example is a wire-shape fixture, not a habitat design or material recommendation. Compilation and test status must be read from exact PR-head CI; this authoring check does not claim a Rust pass.

## 5. Initial candidate families

The first synthetic campaign includes:
1. idealized thin rotating ring, admissible only for the narrow Symtropy analytical ring reference;
2. idealized radial spoke, admissible only for the narrow analytical spoke reference;
3. ring + spoke load sharing, out of domain until a coupled assembly model is qualified;
4. segmented ring with joint compliance, blocked until joint properties and inspection evidence exist;
5. counter-rotating assemblies, requiring complete rotor, bearing/interface and dynamic models;
6. orbital pin-jointed truss, only within the existing 2D determinate reference envelope;
7. lunar bulk shielding with an imported pressure boundary, a partial substitution case only;
8. locally manufactured non-pressure structural product, blocked until batch-specific material/process evidence is admitted;
9. robust comparison under declared material/joint/mass-growth uncertainty;
10. inspectability-first comparison where added mass/time is traded against qualified inspection and repair capability.

Every fixture is synthetic and intentionally avoids asserting real habitat dimensions, material properties or stress allowables.

## 6. SOTA and comparison strategy

AI-assisted topology optimization is an active research area. Recent reviews identify surrogate modeling, generative topology methods, physics-informed learning, reinforcement learning, manufacturing constraints, uncertainty quantification and multi-physics optimization as active techniques. They also identify domain shift, physical consistency, manufacturability and validation as unresolved barriers. A 2026 review frames the integration as topology-optimization informatics, while a September 2026 review specifically stresses the distinction between numerical feasibility, fabrication feasibility, experimental validation and industrial deployment.

Our competitive proposition should therefore be **evidence-qualified design decisions**, not a claim that a generative model replaces FEM or classical topology optimization.

Compare under identical candidate sets, physics, reference truth, hardware and budget:
- fixed-fidelity analysis;
- sequential discipline-by-discipline workflow;
- equal-budget grid/sweep or space-filling sampling;
- conventional MDO/DOE;
- evolutionary Pareto search;
- Bayesian optimization only where probabilistic assumptions are defensible;
- uncertainty-only / information-gain evaluation selection;
- decision-value / regret-reduction selection;
- robust-feasibility and minimax-regret selection.

Report:
- cost/time to first robust-feasible design;
- false-feasible and missed-feasible rates with denominators and intervals;
- robust-frontier recovery and regret against held-out reference results;
- evaluation/solver/experiment cost, including failures and later fidelity escalation;
- out-of-domain abstention precision/recall;
- uncertainty calibration/coverage on held-out geometries, loads, materials and regimes;
- operating-case coverage;
- objective/constraint/evidence coverage gaps;
- total compute, memory and engineering hours per independently supported design decision.

Stretch target: at least 20% lower total evaluation cost to reach a matched robust-feasible frontier on held-out campaigns without increasing false-feasible decisions. This is an aspirational research target, not an achieved result or industry standard.

## 7. Evidence gates and negative controls

No candidate may become decision-eligible when any of these are present:
- stale geometry/configuration generation;
- absent or friendly-name-only material evidence;
- missing load/support or frame/sign mismatch;
- unsupported pressure/contact/fatigue/gyroscopic physics requested from a simpler backend;
- mesh identity/profile mismatch or hidden remeshing;
- candidate mutation after analysis request;
- solver termination success with missing convergence or requested outputs;
- surrogate output represented as solver/measurement evidence;
- two backends agree after receiving the same wrong load translation;
- interval bounds converted into invented probabilities;
- high-local-content architecture depends on an unverified critical imported part;
- a scalar score compensates for hard structural/service infeasibility.

The result must preserve rejected, failed, out-of-domain, unresolved and missing-evidence cases in the accounting.

## 8. Implementation sequence

1. Freeze this design-to-analysis contract and the hostile synthetic corpus.
2. Reuse the existing Symthaea aircraft design-loop dry-run pattern for a space-structure proposal emitter; do not create a second generic engineering orchestrator.
3. Add a Symtropy conformance adapter that checks exact subject identity and applicability before backend dispatch.
4. Use the Symtropy rotating ring/spoke reference only for its narrow supported equation family; its PR remains draft until merged/qualified.
5. Compose sparse modal and external FEM backends only after their own exact case qualifications.
6. Add adaptive fidelity and decision-impact selection through existing Symthaea #7787/#7803/#10902.
7. Connect local-material candidates to Symtropy #5341 and Planetary Industrial Ecology #1604 only after product/batch evidence is admitted.

## 9. Claim ceiling

This package does not add or qualify 3D structural FEA, rotating flexible modes, pressure boundaries, shell/contact/nonlinear/fatigue behavior, CFD, habitat safety, human tolerance, flight certification or construction authority. A synthetic benchmark pass proves only the frozen software contract on that corpus. It does not validate physical models or hardware.

## 10. References

- [NASA 2026 ISAM State of Play](https://www.nasa.gov/isam/)
- [Topology Optimization Informatics review (2026)](https://www.techscience.com/CMES/v147n1/67149/html)
- [AI-aided design optimization and topology optimization review (2026)](https://journals.sagepub.com/doi/10.1177/16878132261489711)
- [Artificial intelligence for design of spatial structures and joints (2026)](https://www.sciencedirect.com/science/article/pii/S2352710226000872)
- [Symtropy rotating prestress reference PR #5358](https://github.com/Luminous-Dynamics/symtropy/pull/5358)
- [Symtropy cislunar industrialization gate #5341](https://github.com/Luminous-Dynamics/symtropy/issues/5341)

---

**First deliverable status:** contract and synthetic conformance corpus. No new solver, optimizer, mesh generator, physics ledger or evidence registry is introduced.