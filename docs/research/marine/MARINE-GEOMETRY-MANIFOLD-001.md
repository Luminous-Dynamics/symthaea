# MARINE-GEOMETRY-MANIFOLD-001
## Feasibility-constrained generative hull geometry

**Status:** Research specification v0.1  
**Created:** 2026-11-15  
**Owner:** Symthaea (candidate generation and geometry admission)  
**Parent:** [HYDRO-CO-DESIGN-001 #8082](https://github.com/Luminous-Dynamics/symthaea/issues/8082)  
**Tracking issue:** [#12302](https://github.com/Luminous-Dynamics/symthaea/issues/12302)

## 1. Purpose

Develop a geometry pipeline that can propose new marine hull surfaces and reject geometrically, hydrostatically, integrationally, or manufacturing-infeasible candidates before expensive hydrodynamic evaluation.

The design candidate is a realizable, revisioned engineering artifact—not a latent vector, image, optimizer score, or parameter row.

This work owns:
- geometry representation and candidate identity;
- constrained parametric/free-form candidate generation interfaces;
- geometry admission and explicit geometry-quality diagnostics;
- manufacturing-profile-aware preliminary checks;
- benchmark design for geometry feasibility, diversity, and search utility.

It does not own a Navier–Stokes, RANS, URANS, LES, SPH, cavitation, or free-surface solver; a second active-learning or multi-fidelity optimizer; physical-trial validation, vessel approval, or navigation authority; or a duplicate CAD platform, GIS, evidence graph, or shipyard planner.

## 2. Why now: current research frontier

Zheng, Wang and Chen, “The generative marine hull form optimization based on offline data-driven evolutionary algorithm,” *Ocean Engineering* 358 (15 June 2026), 125935, DOI [10.1016/j.oceaneng.2026.125935](https://doi.org/10.1016/j.oceaneng.2026.125935), studies a learned latent manifold of historical hull geometries combined with expensive data-driven evolutionary optimization. It identifies parent-hull deformation and surrogate bias as limitations of conventional simulation-based design.

This is a comparator and research trigger, not evidence that a learned generator is universally better. We must compare against parametric/free-form deformation, space-filling designs, evolutionary baselines, and the validation-aware multi-fidelity work in [#11513](https://github.com/Luminous-Dynamics/symthaea/issues/11513).

A separate 2026 ITTC campaign compares KVLCC2 added resistance in regular and irregular waves across ten facilities. It is a valuable seakeeping reference family, but it does not make all geometry, wave, scale, or observable comparisons interchangeable. See [Gerhardt et al. (2026)](https://doi.org/10.1016/j.oceaneng.2026.127723).

## 3. System composition and owner boundaries

| Owner | Responsibility |
|---|---|
| Symthaea #8082 | Mission-bounded hull/propulsor research and candidate questions |
| This work / Symthaea | Geometry representations, candidate provenance, geometry admission |
| Symthaea #11513 | Multi-fidelity selection, evaluation cost and validated Pareto search |
| Symthaea #11442 | External specialist solver interoperation and per-observable V&V |
| Symthaea #8146 | Rights-aware data admission and leakage-resistant splits |
| Symthaea #11765 | Matched-service, interval-aware Pareto relations and hard gates |
| Symtropy #2048 | Shipyard design-for-manufacture and assembly constraints |
| Sol-Atlas #809 | Source-qualified spatial/temporal test-site context |
| Mycelix #5968 | Owner-native evidence references, authority and cross-stack reconstruction |

The hand-off is by exact, immutable artifact references and digests. No system may promote another system's result into a claim outside that owner's authority.

## 4. Paradigm

Treat the geometry pipeline as:

mission + design envelope + hull family + process profile -> candidate geometry + admission report

The generator is not the evaluator. The geometry checker is not a CFD solver. The optimizer may only consume observations from the owning, qualified evaluation path.

Investigate several representations rather than force one universal shape model:

1. **Section-based / parametric families:** interpretable principal dimensions and controlled studies.
2. **Free-form surface families:** B-spline, NURBS, or equivalent smooth surface-control representations for local shape changes.
3. **Learned latent proposals:** optional, only when source geometry rights, corpus coverage, training split, model revision, calibration and out-of-distribution behavior are documented.
4. **Assemblies:** hull, appendages, shafts/pods, propellers, rudders, intakes and foils as separately revisioned components with explicit reference frames and interfaces.

Generator output must preserve diversity. A candidate set that collapses onto one unsupported design-space region is not a successful search.

## 5. Candidate manifest

Every candidate must bind at least:

- stable candidate ID and schema version;
- mission, vessel class, hull-family and operating-envelope profile;
- source kind: baseline, parametric, free-form, or learned generator;
- exact parent geometry or generator/model revision and immutable configuration;
- geometry artifact digest, native format, neutral-exchange format and tessellation digest if distinct;
- coordinate reference frame, axis convention, origin and units;
- principal dimensions and loading/displacement target, including source or calculation status;
- hull surface and component identities/revisions;
- material and manufacturing process profile, or explicit unknown;
- generation seed and deterministic execution identity where applicable;
- geometry-admission profile/version and complete per-check report;
- supersession links; never mutate an existing candidate in place.

A hash identifies bytes. It does not authenticate the publisher, establish data rights, prove that the geometry is correct, or prove manufacturing feasibility.

## 6. Geometry admission

A candidate must receive one typed disposition:
- **ADMISSIBLE:** all mandatory checks in the selected geometry profile are supported and pass;
- **INCOMPLETE:** required inputs or checks are missing;
- **INCOMPARABLE:** inputs use incompatible units, frames, profiles, or semantics;
- **INVALID:** malformed geometry or violated structural invariants;
- **BLOCKED:** an explicit non-compensable profile constraint fails.

### 6.1 Structural and surface checks

Where required by the selected hull family:
- finite coordinates and explicit units/frame;
- closed/watertight displacement surface;
- manifoldness, orientability, consistent outward normals and positive enclosed volume;
- self-intersections, open seams, duplicate faces, zero-area or degenerate elements;
- tessellation quality and declared tessellation settings;
- configurable continuity/fairness diagnostics at section joins, chines, appendage roots and declared transitions;
- no silent scale, axis, mirror, sign or frame conversion.

Native B-rep/CAD validity and tessellated mesh validity are separate checks. A valid mesh is not proof of a valid native solid or a buildable surface.

### 6.2 Naval-architecture and interface checks

- declared length, beam, draft, waterline, displacement/volume and loading target;
- explicit water density, gravity, trim and draft semantics for any hydrostatic calculation;
- versioned shaft/pod, propulsor, rudder, intake, foil, appendage and mounting interfaces;
- collision, clearance, access and assembly-space checks against exact component revisions.

Geometry admission must not claim that stability, seakeeping, resistance, cavitation, structural adequacy, maneuverability or class requirements have passed. Those remain separate, qualified analysis owners.

### 6.3 Manufacturing-aware checks

Manufacturing constraints are process-specific, not universal:
- plate/panel developability and curvature limits;
- minimum radius/thickness and panel segmentation;
- weld/joint access, assembly sequence and inspection access where applicable;
- distinct profiles for steel, aluminium, composites and specialist/additive processes;
- material, tooling and process identity.

An unknown process limit is INCOMPLETE, not an assumed pass. The checker must not silently deform a candidate into compliance.

## 7. Search and evaluation protocol

1. Freeze mission, useful service, design envelope, hull family, baseline geometry and process profile.
2. Freeze geometry corpus rights, family splits and exact artifact digests.
3. Generate candidates with bounded, deterministic settings.
4. Run geometry admission before mesh generation or expensive physics evaluation.
5. Send admitted candidates and their immutable manifests to #11513 for fidelity selection.
6. Execute specialist hydrodynamics through #11442; retain mesh, solver, convergence, uncertainty and raw-output identities.
7. Compare candidates only under matched service and conditions; use #11765's interval-aware and non-compensable decision semantics.
8. Evaluate final selections on blind hull-family and operating-condition holdouts.
9. Preserve invalid, inconclusive and dominated candidates; do not publish best-run-only results.

Do not mix KCS, KVLCC2, Series 60, ONRT or Delft 372 datasets into one generic score. Their hull families, observables, scales and physical regimes differ.

## 8. Benchmark baselines and metrics

### Baselines

At equal declared evaluation and validation budget, compare:
- conventional parent-hull parametric/free-form deformation;
- Sobol or Latin-hypercube space filling;
- a fixed evolutionary baseline (e.g. NSGA-II) where objectives are compatible;
- learned latent-space geometry plus offline evolutionary search, only where a rights-cleared comparable corpus/implementation exists;
- the existing validation-aware multi-fidelity search from #11513.

### Metrics

**Geometry integrity**
- false-accept rate on adversarial invalid geometry;
- feasible-candidate yield;
- deterministic replay and artifact identity;
- principal-dimension / hydrostatic-target mismatch where independent values exist.

**Design-space search**
- diversity and coverage of declared geometry space;
- valid candidates per unit compute;
- meshing success and cell-quality distribution;
- high-fidelity query count and total cost, including mesh generation and mandatory final validation.

**Validated engineering outcome**
- Pareto hypervolume/gap, rank inversion and validation regret;
- mission-energy / useful-service improvement with uncertainty;
- held-out generalization by entire hull family and operating-condition group;
- buildability, material/process compatibility and fabrication complexity, reported separately.

Do not use geometry ratios, latent scores, or unvalidated surrogate predictions as hydrodynamic performance.

## 9. Negative controls

The frozen fixture suite must include:
- inverted normals and negative signed volume;
- open seams, self-intersection and non-manifold edges;
- non-finite coordinates, duplicate faces and degenerate elements;
- unit-scale or coordinate-frame mismatch;
- duplicate candidate IDs and changed artifact under unchanged digest;
- hull/appendage/propulsor collision and impossible declared clearance;
- principal dimensions/displacement outside the frozen design envelope;
- geometry-family leakage between training/calibration and blind holdout;
- an attractive proxy score attached to a candidate that fails a geometry or manufacturing hard gate.

## 10. Proposed targets — aspirations, not results

- 100% rejection of the frozen invalid-geometry negative-control suite.
- 100% byte-identical candidate manifests under identical inputs, generator revision and seed.
- Zero solver calls for candidates that fail mandatory geometry admission.
- Zero physical-performance or safety claims derived from latent scores or geometry ratios alone.
- At equal high-fidelity budget, target at least 10% improvement in independently validated Pareto hypervolume over the strongest frozen baseline on two compatible benchmark families.
- Stretch: 20% lower total cost to reach the same validated frontier without worse held-out constraint violations, rank inversion or validation regret.

These are research aspirations. Freeze normalization, objective ranges, baseline implementations, uncertainty protocol and budget before scoring. Publish a null or negative result if targets are not met.

## 11. Initial implementation increments

**G0 — Contract and fixtures**
- versioned candidate and admission-report schemas;
- neutral geometry fixture corpus with valid/adversarial cases;
- explicit claim ceiling and process-profile semantics.

**G1 — Deterministic admission**
- Rust-first, dependency-light CLI;
- strict finite/unit/frame and digest checks;
- structural/topological checks supported by the selected geometry adapter;
- typed dispositions and deterministic report.

**G2 — Candidate proposal interface**
- baseline and bounded parametric/free-form generators;
- candidate diversity and geometry-family coverage diagnostics;
- optional learned latent proposal adapter, separately qualified.

**G3 — Evidence-coupled search**
- exact hand-off to #11513 and #11442;
- blind hull-family holdouts;
- equal-budget baseline comparison and independent validation.

**G4 — Manufacturing and lifecycle composition**
- process-specific DFM checks with Symtropy #2048;
- source-qualified site context when required from Sol-Atlas #809;
- owner-native artifact references for Mycelix #5968.

## 12. Exit / claim boundary

A passing geometry audit establishes only that the exact candidate satisfies its declared geometry-admission profile. It does not establish correct CFD, resistance reduction, propulsive efficiency, seakeeping, structural adequacy, ecological acceptability, class approval, seaworthiness or real-world manufacturing success.

No SOTA or superiority claim is permitted until the generator is compared against frozen baselines at equal evaluation cost and selected candidates survive independent held-out validation.
