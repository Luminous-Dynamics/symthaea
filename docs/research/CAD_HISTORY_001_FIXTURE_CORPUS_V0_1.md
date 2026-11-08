# CAD-HISTORY-001 — Synthetic edit-preserving feature-history corpus v0.1

**Status:** DESIGN_ONLY / synthetic fixture specification  
**Parent:** [CAD-HISTORY-001 #10510](https://github.com/Luminous-Dynamics/symthaea/issues/10510)  
**Geometry authority:** compose [FAB-GEOM-001 #4927](https://github.com/Luminous-Dynamics/symthaea/issues/4927); this corpus does not define a competing geometry identity or CAD kernel.  
**Execution:** no CAD kernel, FreeCAD, STEP importer, benchmark runner, or external task has been executed for this document.

## 1. Evaluation thesis

The benchmark must evaluate a recovered design's **functional edit contract**, not guess the one historical feature sequence that created a final solid. More than one timeline may be valid for the same boundary representation.

A candidate may be accepted when it:
1. rebuilds in a pinned native CAD kernel;
2. is geometrically equivalent to the permitted source within case-specific tolerances;
3. exposes stable, semantically meaningful parameters;
4. survives the required interventions while preserving stated invariants; and
5. represents non-identifiability honestly.

The original source authoring history is not ground truth unless independent provenance establishes it. Face and edge indices are not stable semantic identifiers across kernel operations unless the kernel contract explicitly guarantees that property.

## 2. Corpus structure

The proposed v0.1 corpus has six positive functional fixtures and four adversarial controls. All values below are synthetic, in millimetres unless noted. They are design specifications—not prebuilt CAD files and not verified kernel outputs.

Each fixture should ultimately have three separate records:
- **candidate-visible task:** source solid, permitted reference dimensions, design requirements and allowed edits;
- **evaluator-only contract:** invariants, edit sequence, comparison tolerances and acceptable dispositions;
- **provenance record:** fixture generator revision, kernel/export revision, source artifact digest, evaluator revision and exact unit/frame policy.

The evaluator must not expose hidden edit sequences or expected dispositions to the candidate during a scored run. Public examples may be used for development, but the held-out evaluator corpus must be separately versioned and access-controlled.

## 3. Positive functional fixtures

### CADH-P01 — patterned mounting plate

**Source geometry:** plate envelope 120 × 80 × 8. Four Ø8 through-holes are centred at (15,20), (105,20), (15,60), and (105,60). Datum origin is the lower-left corner of the plate's bottom face; +Z is thickness.

**Permitted candidate histories:** one sketch + pad + two-direction rectangular hole pattern; pad + four independently dimensioned hole features; any other native history that rebuilds and satisfies the edit contract.

**Observable design intent:** the two hole columns remain 15 mm from the short-end datums; two rows remain 20 mm from the long-side datums; paired holes remain symmetric about the plate midplanes.

**Intervention:** increase plate length from 120 to 140 mm. Preserve the 15 mm end margins, equal column spacing, 20 mm row offsets, hole diameter and thickness.

**Acceptance:** successful rebuild; initial geometry within declared case tolerance; post-edit centres at x = 15 and 125 mm and y = 20 and 60 mm; no lost/duplicated holes; datum and envelope invariants satisfied.

**Why it matters:** several histories reproduce the source solid exactly, but the intended relational constraints are observable only under intervention.

### CADH-P02 — stepped shaft

**Source geometry:** axis-aligned shaft along +Z, total length 100 mm, with three contiguous cylindrical regions: Ø20 × 20 mm, Ø30 × 60 mm, Ø20 × 20 mm. Shoulder planes are at z = 20 and z = 80 mm.

**Permitted candidate histories:** one revolved axial profile; sequential coaxial pads; a constrained base cylinder with additive/subtractive features; any other editable history satisfying the interventions.

**Observable design intent:** both end journals remain Ø20 × 20 mm; the central journal remains Ø30 mm; the central journal length is the named variable; the two shoulders remain square and coaxial.

**Intervention:** change central journal length from 60 to 70 mm. Preserve both end-journal lengths and diameters; total shaft length becomes 110 mm; the second shoulder moves by 10 mm.

**Acceptance:** successful rebuild; cylinder coaxiality; exact dimensional constraints within the kernel's declared modelling tolerance; both end interfaces unchanged; central length changed by the requested amount.

**Why it matters:** the feature tree may vary widely. Only measurable design behavior should determine acceptance.

### CADH-P03 — thin-wall enclosure

**Source geometry:** open-top rectangular enclosure with outer envelope 100 × 60 × 40 mm, nominal wall thickness 2 mm, nominal floor thickness 2 mm, and a rectangular inner cavity. The outer bottom face is the primary datum.

**Permitted candidate histories:** block + shell + face removal; outer solid minus a parameterized inner cavity; sketch-and-pad wall construction; any editable history satisfying the contract.

**Observable design intent:** outer envelope and mounting datum are fixed; nominal wall and floor thicknesses are independently named parameters; the enclosure remains open at the top.

**Intervention:** increase wall thickness from 2 to 2.5 mm while preserving the outer envelope and floor thickness. The cavity shrinks symmetrically about the outer side-wall datums; the floor remains 2 mm.

**Acceptance:** successful rebuild; outer envelope unchanged; wall thickness measured at all four side walls is 2.5 mm within tolerance; floor remains 2 mm; cavity remains open and connected; no self-intersections or invalid topology.

**Why it matters:** static shape equivalence is insufficient when wall thickness and outer dimensions have different functional roles.

### CADH-P04 — rounded rectangular plate

**Source geometry:** plate envelope 80 × 40 × 6 mm with four vertical corner radii of 5 mm. Top and bottom faces are planar; vertical side walls are tangent to the corner arcs.

**Permitted candidate histories:** rounded-rectangle sketch + pad; rectangular pad + four vertical-edge fillets; equivalent native feature construction.

**Observable design intent:** overall 80 × 40 × 6 mm envelope is retained; all four corner radii are equal and parameterized as one named value.

**Intervention:** change corner radius from 5 to 8 mm.

**Acceptance:** successful rebuild; outer envelope unchanged; all four corners measure radius 8 mm; side-wall tangent continuity is preserved; no accidental top/bottom edge fillets are introduced.

**Why it matters:** a fillet feature and a rounded sketch can produce the same source solid. The correct choice is determined by the named edit contract, not by historical imitation.

### CADH-P05 — equally spaced slot array

**Source geometry:** plate envelope 90 × 40 × 8 mm with three identical through-slots, each 6 mm wide × 20 mm long, centred along the x-axis at x = 15, 45, and 75 mm; slot long axes are parallel to Y.

**Permitted candidate histories:** sketch + pad + linear feature pattern; three independent slot cuts; a constrained multi-slot sketch; other editable histories that preserve the contract.

**Observable design intent:** the slot count and end margins are named; slot width/length are shared parameters; slots are equally spaced.

**Intervention:** change slot count from 3 to 4 while retaining 15 mm end margins, 6 mm width, 20 mm length and equal spacing. Required centres: x = 15, 35, 55, 75 mm.

**Acceptance:** exactly four through-slots; equal centre spacing; both end margins 15 mm; slot dimensions unchanged; no failed rebuild or unintended cut.

**Why it matters:** this tests whether a reconstruction captures array relations rather than only four/fewer local hole-like cuts.

### CADH-P06 — right-angle mounting bracket

**Source geometry:** L-bracket with 80 × 50 mm orthogonal flange envelopes, 6 mm plate thickness, and two Ø6 through-holes on each flange. Hole centre locations and the two mounting datum planes are supplied by a dimensioned drawing in the task variant.

**Permitted candidate histories:** two orthogonal pads joined by a constrained sketch; one multi-profile pad; equivalent additive construction; a mirrored/linked flange only if the resulting history remains editable and rebuildable.

**Observable design intent:** flange interface planes stay perpendicular; 6 mm thickness is shared; mounting-hole datums and edge margins are preserved; changing flange width must not move the mounting datum plane.

**Intervention:** change the horizontal flange width from 80 to 90 mm while preserving its mounting datum plane, thickness, hole diameter, hole edge margin and perpendicularity. The vertical flange and its holes remain unchanged.

**Acceptance:** successful rebuild; only the horizontal flange envelope changes; both datum planes and hole constraints remain satisfied; no unintended gap, overlap or disconnected solid is introduced.

**Why it matters:** it evaluates interface preservation and cross-feature dependencies rather than isolated primitive reconstruction.

## 4. Adversarial and selective-prediction controls

### CADH-N01 — unit and scale ambiguity

**Input:** imported unitless solid plus a drawing whose nominal dimension reads 100 without a declared unit or scale note. Two scale interpretations are geometrically plausible.

**Required disposition:** request clarification or return INDETERMINATE. Do not silently pick mm, cm or inches. A numerical value without a declared unit is not an engineering dimension.

**False-pass condition:** candidate chooses a unit, rebuilds a plausible object and reports high confidence without identifying the ambiguity.

### CADH-N02 — invalid boundary representation

**Input:** open shell, non-manifold edge, self-intersection, or a body with an invalid face orientation. The fixture generator must pin exactly which defect is present and preserve the raw input bytes.

**Required disposition:** reject the input or return an explicit, separately identified repair proposal. A repaired output must not be described as a faithful reconstruction unless the repair is explicitly accepted and the source/repair difference is recorded.

**False-pass condition:** kernel healing silently changes geometry and the candidate reports exact reconstruction.

### CADH-N03 — conflicting references

**Input:** STEP geometry implies 100 mm between two datums; accompanying drawing explicitly specifies 105 mm, with no revision authority or change notice.

**Required disposition:** identify the conflict and ask which source controls. Preserve both observations and their provenance. Do not average dimensions or silently privilege the model.

**False-pass condition:** candidate silently picks either source and reports compliance with both.

### CADH-N04 — unsupported or impossible edit

**Input:** valid source geometry, but requested intervention cannot be satisfied by the supported feature/kernel profile, or constraints are mutually inconsistent (for example, a fixed outer envelope and a requested cavity expansion that violates a declared minimum wall thickness).

**Required disposition:** explain the conflicting constraints, return NOT_SUPPORTED / INFEASIBLE / clarification as appropriate, and do not emit an accepted design.

**False-pass condition:** candidate relaxes a hard constraint without authorization or returns an unchanged artifact as though the edit succeeded.

## 5. Cross-fixture metamorphic controls

The evaluator should include paired transformations that preserve known semantics and transformations that must change them:

1. Reordering source feature candidates must not alter the set of acceptable functional outcomes.
2. Renaming arbitrary face/edge indices must not change semantic feature identity or edit outcomes.
3. Re-serializing the same canonical source geometry under a pinned lossless profile must preserve geometry identity; changing units or coordinate-frame metadata must change the interpretation and require explicit conversion.
4. A uniformly translated part and its datums must preserve relative dimensions and edit outcomes.
5. Scaling every length and every length-valued tolerance by the same positive factor must preserve dimensionless geometric and edit metrics.
6. A duplicated or omitted required feature must fail the relevant topology/edit gate even when global volume difference remains small.
7. Two distinct feature histories that produce the same source solid must both be accepted when both satisfy the stated interventions; do not reward guessing the private authoring history.
8. An edit that changes one named parameter must preserve all unrelated declared invariants.
9. Candidate portfolio order must not affect the selected acceptable set or evaluator result.
10. An ambiguous-input perturbation must change the disposition from a confident single answer to clarification/abstention when the evidence ceases to identify the required unit, datum or design intent.

## 6. Evaluation protocol

Do not collapse the following into one weighted score:

| Measure | Unit / reporting | Hard gate |
|---|---|---|
| Native rebuild validity | per case and per edit sequence | every accepted artifact rebuilds |
| Geometric equivalence | symmetric surface distance, Hausdorff distance, volume/area deviation, datum alignment; per-case tolerance | case-specific geometry tolerances met |
| Parameter semantics | required parameters exposed and correctly bound | all required parameters exist |
| Edit success | successful required interventions / attempts | every required edit succeeds or is explicitly non-supported |
| Intent preservation | invariant pass / total invariants; report each invariant | no critical interface or dimensional invariant violated |
| Topological health | kernel validity, manifoldness and connected-body expectations | no silent invalid topology |
| Ambiguity handling | candidate coverage, unjustified-uniqueness rate, clarification/abstention quality | no false certainty on N01/N03 or ambiguous variants |
| Reliability | failures, timeouts, retries, missing outputs and complete denominator | all cases accounted for |
| Efficiency | wall time, CPU/RAM if measured, tokens, tool calls, model and kernel cost | cost per accepted editable artifact reported |

Use tolerances defined for each kernel, exchange format, part scale and task. A single global millimetre tolerance is not sufficient. Record the comparison method and whether it uses native BREP, tessellated mesh, or both. A mesh comparison is auxiliary evidence; it cannot alone establish native parametric editability.

## 7.1 Relation to Parametric CAD Bench V3

The November 2, 2026 V3 paper describes a pinned FreeCAD environment and a separate programmatic verifier that checks native CAD artifacts for geometry and task-specific specification consistency. It explicitly notes that these checks are proxies for requirements and do not certify arbitrary future editability or manufacturing readiness. V3 also does not require exact reproduction of a private reference feature history; alternative constructions can satisfy the same engineering requirements.

The September 24, 2026 public snapshot is particularly useful for locating the gap: Claude Opus 5.5 / Claude Code reports 84.66% on image-to-CAD but 39.69% on create-and-edit; GPT-6 Astra / Codex reports 69.70% on image-to-CAD and 44.37% on create-and-edit. Overall means (61.03% and 56.87%) should not obscure this workflow split. Our additional intervention corpus tests whether inferred relations survive *new, explicitly specified downstream edits*, beyond matching the initial drawing/text task. This is an extension, not a claim that V3 omitted all edit evaluation.

The v3 paper identifies complete model–agent systems, not isolated model capabilities. Matched comparisons must therefore pin and disclose both the model and the agent/tool stack. Keep V3's mean continuous reward distinct from our proposed per-invariant and per-edit hard gates.

Reference: https://arxiv.org/abs/2611.02659 and https://cadbench.ai/leaderboard

## 8. Matched competitive experiment

After the corpus is materialized and its evaluator independently reviewed, compare these arms on the exact same task set and budgets:

- pinned CAD kernel/API baseline with hand-authored or scripted features;
- strongest available model without CAD tools;
- same model with the standard CAD API and rebuild feedback;
- same model with bounded candidate-history generation and semantic edit search;
- same model with Symthaea orchestration, with retrieval, PhysicsIR/typed dimensions, candidate portfolio and verification components ablated separately;
- qualified human novice and experienced parametric CAD designer cohorts, reported separately.

Keep model revision, effort, prompt, CAD kernel/build, task inputs, permitted tool calls, retries, time/token budgets and evaluator fixed. Use paired per-task comparisons and report uncertainty, cost and failed tasks. The public Parametric CAD Bench V3 is a related but separate comparator: it has 100 tasks (30 text-to-CAD, 30 create-and-edit, 40 engineering-drawing reconstruction) and scores mean continuous task reward, with failed/unscored trials counted as zero. Its September 24, 2026 publisher snapshot and its later arXiv paper must be cited as their own versioned results; do not convert that score into geometric-equivalence pass rate. See https://cadbench.ai/leaderboard and https://arxiv.org/abs/2611.02659.

## 9. Prospective gates

These are hypotheses to preregister after a pilot, not present Luminous results or industry standards:

- 100% of accepted artifacts rebuild in the exact pinned kernel.
- Zero false PASS across N01–N04 and the corresponding evaluator mutations.
- 100% of required interventions either meet their invariants or receive an explicit supported non-success disposition; silent relaxation is zero-tolerance.
- At least 95% of accepted artifacts meet their case-specific geometry equivalence tolerances.
- At least +10 percentage points on held-out semantic edit success over the strongest matched model+CAD baseline, with no degradation in rebuild validity or critical invariants.
- At least 20% lower cost per accepted editable artifact at non-inferior geometry and edit quality.
- Zero unjustified unique-history claims on deliberately ambiguous source cases.
- 100% task-accounting coverage: accepted, rejected, abstained, unsupported, timed out and ungraded cases remain in the denominator.

Do not freeze these as success criteria until pilot variance, task difficulty, independent evaluator review and the actual kernel's numeric tolerances have been characterized.

## 10. Runner-independent next work

1. Generate the six positive source solids from a declared synthetic construction script, but keep alternate valid histories for evaluator testing.
2. Produce N01–N04 as exact negative fixtures; verify that the intended defect/conflict is present in the source artifact itself.
3. Independently review case definitions and edit invariants before committing binary CAD files.
4. Pin FreeCAD/kernel version, STEP export settings, coordinate conventions, tessellation settings and geometry comparison implementation.
5. Keep candidate-visible inputs separate from evaluator-only edit sequences and labels.
6. Map feature/parameter IDs to FAB-GEOM-001's existing validated geometry and feature-identity contracts. Do not add a second geometry subject, BREP identity or evidence receipt.
7. Only then create a runner adapter and compare against Parametric CAD Bench V3 under its published rules.

## 11. Claim ceiling

This corpus is a synthetic design specification only. It establishes no CAD-kernel implementation, geometry equivalence result, parametric reconstruction accuracy, manufacturability, structural adequacy, GD&T compliance, material suitability, product safety, regulatory compliance or physical validation.

A candidate feature history can be a valid editable representation without being the original author's history. A geometry-equivalent solid is not automatically functionally equivalent; a functionally equivalent parametric model is not automatically manufacturable or safe.


## 7.2 Scan-to-CAD extension: metrology-aware reconstruction

Add a seventh task family when traceable scan inputs can be materialized. This is not the same task as reconstructing a history-less STEP/BREP solid: the input is a set of **measured observations**, with calibration, uncertainty, registration, surface-interaction and occlusion limitations.

### CADH-P07 — scanned stepped shaft

**Nominal synthetic design:** a stepped shaft matching CADH-P02 (Ø20 × 20 mm, Ø30 × 60 mm, Ø20 × 20 mm; nominal total length 100 mm).

**Measured-input specification:** multi-view scan points covering the accessible outer surface, with a declared instrument profile, calibration state, pointwise or regional uncertainty, exact unit/frame and multi-view registration covariance. The scan must include a masked underside/end-face region in one case variant and a deliberately worn band in another.

**Two separate task targets:**
- **As-built replica:** reproduce the measured article geometry and preserve the observed wear/deviation.
- **Nominal design inference:** infer the likely nominal stepped-shaft design from the scan plus explicit design requirements, but report wear/unknown regions and measurement uncertainty separately.

**Required output:** editable feature-based shaft model, semantic parameters, per-feature uncertainty and datum/frame bindings; point-cloud residual distribution and coverage; occluded regions marked UNKNOWN; and the declared target type.

**Intervention:** change the central journal length from 60 to 70 mm while preserving the end-journal dimensions and shared axis. The reconstructed feature history must survive the requested edit. A pure mesh that matches the scan but cannot be edited fails the task.

**Negative controls:** scanner scale bias, multi-view registration drift, surface dropout, reflective/coated patches, hidden bore/underside, point-cloud outliers, synthetic hole filling, and wear mistaken for nominal design.

**Acceptance:** traceable scan profile and coverage; uncertainty-aware feature dimensions; no synthetic/occluded region represented as measured input; successful native-kernel rebuild; semantic edit success; separate as-built vs nominal dispositions. No universal dimensional tolerance is imposed: tolerances must derive from the task's instrument/calibration/reference-artefact/part-scale profile.

### Scan-to-CAD metrics

Report separately:
- dimensional error by feature/scale and calibrated reference;
- surface residual distribution, percentile tails, coverage and outlier handling;
- calibration and registration uncertainty propagation;
- uncertainty coverage against traceable artefacts;
- feature/parameter recovery and correct abstention under occlusion;
- native parametric editability and rebuild success;
- cost/time per independently accepted editable model and human correction burden.

Hexagon Geomagic Design X and Creaform Scan-to-CAD are industry comparators for point-cloud/mesh processing, registration, feature extraction, editable feature history and CAD transfer. NIST's work on VDI/VDE 2634-2 / ISO 10360-13 underscores calibration/systematic-error effects in structured-light metrology. A 2026 review summarizes uncertainty sources including surface/sensor interaction, calibration, occlusion, registration drift, mesh/surfacing, XCT segmentation and task-specific uncertainty interpretation.

References:
- OpenAI / Chip Ganassi Racing scan-to-CAD workflow (2026-11-03): https://openai.com/index/accelerating-engineering-for-the-physical-world/
- Hexagon Geomagic Design X: https://hexagon.com/products/geomagic-design-x
- Creaform Scan-to-CAD: https://www.creaform3d.com/en/products/software/creaform-metrology-suite/scan-to-cad-software-module
- NIST (2024): https://www.nist.gov/publications/vdivde-2634-2-and-iso-10360-13-performance-evaluation-tests-and-systematic-errors
- 2026 scan-based reverse-engineering uncertainty review: https://www.mdpi.com/2076-3417/16/3/1229

Compose ENG-TOL-001 #4883 for dimensional/metrology requirements and ROB-REALIZE-001 #4859 for as-built article lineage; do not create another measurement system or as-built authority.


## 7.3 Parametric CAD Bench V3 exact public run contract (retrieved 2026-11-05)

Use V3 as a distinct external baseline. The current official run page specifies:
- Harbor 0.23.0;
- FreeCAD 1.1.0;
- gnucleus-freecad-validator 0.6.0 with v2 scorer;
- Harbor task schema 1.3;
- 100 tasks: 30 text-to-CAD creation, 30 create-and-edit, 40 image-to-CAD engineering drawings;
- one attempt per task, four-way concurrency, zero retries, 9,000-second per-task time limit;
- network disabled except selected model provider, separate evaluator/reference geometry, agent state limited to task workspace;
- failures and unscored cases count as zero reward.

The current public leaderboard is not a single homogeneous capability score. On the official page retrieved 2026-11-05, Opus 5.5 / Claude Code has 61.03 overall but 39.69 create-and-edit and 84.66 image-to-CAD; GPT-6 Astra / Codex has 56.87 overall, 44.37 create-and-edit and 69.70 image-to-CAD. Current whole-run costs ($573.94 and $317.81 respectively) are totals for the 100-task jobs, not per-case costs.

The benchmark's continuous reward mixes geometry similarity and specification consistency; it does not establish future arbitrary edit preservation, metrology traceability, manufacturing feasibility or physical safety. The CAD-HISTORY extension therefore focuses on downstream interventions and protected semantic invariants that are separately measured.

References:
- https://cadbench.ai/
- https://cadbench.ai/benchmark
- https://cadbench.ai/news/cad-bench-v3-paper
- https://arxiv.org/abs/2611.02659



## 9. Runner-independent source geometry materialization (2026-11-05)

A separate development fixture package is tracked in [ENGINEERING-FIXTURE-MATERIALIZATION-001 #10622](https://github.com/Luminous-Dynamics/symthaea/issues/10622).

All six positive fixture geometries now have a source STEP B-rep and a corresponding semantic-edit STEP variant (12 artifacts total). The pinned local generator used CadQuery 2.8.0 with OCP / Open CASCADE 7.9.3.1 and Python 3.13.5. STEP output is AP214 `AUTOMOTIVE_DESIGN`, not AP242. The Open CASCADE wall-clock header timestamp was normalized before hashing; two clean generator runs reproduced all outputs byte-for-byte.

Every source/edit STEP artifact was re-imported and passed the kernel's B-rep validity check. All 12 volumes matched independently derived analytic volume references within 1e-5 mm³. The machine-readable materialization manifest binds dimensions, bounds, expected volumes and exact artifact SHA-256 values.

Important claim boundary: these are history-less input solids for a future reconstruction evaluation. The generator source is parametric, but no FreeCAD .FCStd feature-history answer, native FreeCAD rebuild, CADBench V3 run, agent score, edit-preservation score, manufacturing validation, or physical test has been produced. The actual future candidate must still return an editable native CAD artifact and pass its required edit sequence.
