# Luminous Atelier — paradigm shifts and competitive benchmark plan

**Date:** 2026-11-14  
**Status:** research synthesis and proposed benchmark programme; no SOTA claim is made.

## 1. Strategic thesis

Do not compete to generate the most photorealistic fashion images. Build a human-controlled, evidence-native system that carries a person's intent through editable design, fit specification, pattern semantics, material/process feasibility, manufacturing, inspection, repair and lifecycle records.

The complete product has two equally important starting points:

1. **Create:** design a garment from a blank brief, preset, sketch, description, existing pattern or reference asset.
2. **Use what exists:** plan combinations, repair, alter, adapt, lend, rent or reuse existing garments before any new-item recommendation.

No-purchase and no-manufacture outcomes must remain valid. Low-frequency ceremonial, archival, performance, cultural or sentimental garments must not be assigned lower value simply because they are worn less often.

## 2. Paradigm shifts

### 2.1 From prompt-to-image to a typed garment design language

Treat design intent as structured, versioned data: silhouette, ease field, component topology, materials, joining methods, closures, fit intent, accessibility requirements, cost ceilings, machine constraints, rights and privacy. Images are references or previews, not canonical product truth.

### 2.2 From size labels to user-controlled spatial fit intent

Represent ease as local and direction-aware quantities where the geometric model supports it. Let users preserve or intentionally modify fit intent across poses and body shapes. Keep measured dimensions, inferred geometry, virtual fit, wearer feedback and physical fit trials as separate evidence.

### 2.3 From a single best design to a feasible Pareto set

Hard constraints are non-compensatory. Unknown hard gates do not pass. Soft objectives describe a person's trade-offs and should generate several non-dominated feasible alternatives. A high appearance score cannot offset an unacceptable closure, budget or accessibility constraint.

### 2.4 From AI replacing designers to human-authorized co-design

Every system suggestion should be inspectable, attributable to a brief revision, selectively adoptable, reversible and recorded as accepted/rejected/edited/deferred by a human actor. Measure control and agency, not only output quality.

### 2.5 From virtual appearance to calibrated physical evidence

A rendered garment is not a material test. A geometric fit pass is not wearer comfort. A fabric label is not measured composition. A supplier declaration is not physical verification. Physical calibration and independent validation remain mandatory for physical claims.

### 2.6 From a garment to a wardrobe and lifecycle system

Optimize the user's actual clothing needs over their wardrobe, climate, care constraints, repair options and preferences. Treat repair, alteration, modular replacement and second-hand sourcing as first-class options. New manufacture should solve a user-declared gap.

### 2.7 From static fabric catalogues to evidence-aware inverse material design

Search for fibre/yarn/fabric/process combinations that could meet a declared property envelope. Reject candidates outside the model's calibrated domain, expose uncertainty and select physical coupons by value of information. Never promote surrogate predictions to measured material properties.

### 2.8 From disconnected tools to a semantic garment compiler

Use a canonical, versioned intermediate representation with adapters to DXF-AAMA, SVG, OpenSloper JSON, CLO/Browzwear interchange where available, and machine-oriented formats such as Knitout for relevant knitting workflows. Preserve units, grain direction, seam allowances, notches, piece identity, joining topology, grading rules and source revisions. Report unsupported or lossy fields explicitly.

### 2.9 From a circularity label to measured recovery outcomes

Design for sorting, disassembly, repair and recovery at design time, but qualify downstream acceptance physically. Fibre identification, garment-category classification, sort-stream purity, recovery yield and environmental benefit are separate claims.

## 3. Competitive landscape (verified 2026-11-14)

### Professional CAD and virtual sampling

- CLO 2026.1 (official release, 13 August 2026) reports 29 new features including pattern property/revision history, grading review, new knit stitch types, seam ripping/tearing and GPU simulation across platforms. CLO 2026.0 added sketch-on-avatar to pattern, automatic POM/grading and fabric-aware strain maps. This is a mature baseline; Atelier should integrate rather than imitate every CAD function.
- Browzwear's VStitcher product page markets 95% first-time-right, 2x faster time-to-market and 50% fewer physical samples. Treat these as vendor claims that require a matched independent study, not universal benchmark truth.
- An August 2026 comparative study evaluated CLO and VStitcher using eight virtual dresses, four corresponding physical garments and 15 experienced reviewers. It identified workflow-specific strengths and noted virtual fabric smoothness and avatar proportion limitations. This supports matched physical validation rather than screenshot-only comparisons.

### Human-AI co-design and pattern generation

- EasyFashion (arXiv:2609.18483, 16 September 2026; revised 9 October) combines reference images/text, reconstructed personal avatars, iterative style/size refinement, virtual try-on and sewing-pattern generation. It shows that an end-to-end user-facing pipeline is already an active frontier. Atelier should differentiate on local-first privacy, explicit objective negotiation, accessible design, editable semantic geometry, evidence provenance and wardrobe-first optimization.
- EASE (August 2026) treats ease as an explicit spatially varying, anisotropic design variable and optimizes 2D sewing patterns with geometric/seam constraints; it reports a manufactured pant validation. Atelier should benchmark local ease transfer, pose adaptation and pattern semantics rather than claim the concept is novel.
- A 2026 multimodal sketch/text-to-pattern paper reports 96.2% pattern accuracy, but its comparison table says its method lacks pattern-level ground truth and uses pseudo-patterns derived from fashion imagery. Do not quote this number as proof of pattern correctness without reproducing the metric and validation data.
- A September 2026 human-AI sustainable fashion study describes 4,200 design records, 350 expert-rated designs, 864 consumers and 426 production patterns. Reproduce its sampling and rubric before comparing; pattern records do not establish physical manufacture.

### Cloth simulation and robotics

- RGBench (AAAI 2026) reports over 6,000 garment meshes and claims roughly 20% lower simulation error and 3x faster runtime for its GarmentDynamics simulator against evaluated baselines. The currently public Hugging Face evaluation subset is smaller: nine captured garments, grasp/fold/fling actions and roughly 100 samples; two garments have non-manifold meshes excluded from published baselines. Reproduce the currently downloadable subset first and do not claim access to the full asset library/simulator until released.
- Cloth manipulation benchmarks should distinguish grasp, flatten, fold, fling, alignment, seam following, feeding and sewing. Success in unfolding is not autonomous garment production.

### Textile sensing and circularity

- TexSortML (2026) contains 751 post-consumer garments, 11 visual categories and three presentation stages. Its ViT result drops 9.3 percentage points from easiest to hardest presentation versus 16.6–16.9 points for its CNN comparators. It labels visual garment category/presentation, not fibre composition; reported model sizes differ substantially.
- NIST NIR-SORT covers a separate spectral fibre/composition task. Visual classification must never infer composition, and a spectral composition estimate must never imply recyclability or recovery acceptance.
- European Commission textile DPP page lists Q4 2027 as the planned adoption of the textile ESPR delegated act; timing is indicative and technical requirements are still forthcoming. Keep projections versioned and adaptable; do not advertise compliance prematurely.
- UNEP InTex Phase II in South Africa (August 2025–April 2028) works with CARES and CSIR. CSIR's policy work includes a national textile LCI database and EPR feasibility work. The 2026 national textile LCI CFP closed 27 September 2026. Do not claim selection/partnership or clone the forthcoming official database.
- South African GoN 8145 draft EPR regulations (Gazette 55679, 5 November 2026) invite comments until 5 December 2026. This is a draft consultation, not proof that textiles are currently subject to a new textile EPR fee or product notice. Track exact notice scope and status.

## 4. AtelierBench lanes

### AB-1: Intent and designer agency

Matched arms: expert without AI; conventional CAD; image-only generation; parametric pattern generation; complete Luminous workflow.

Metrics: requirement adherence; style-family-specific blinded design ratings; time to first acceptable concept; time to accepted production package; suggestion acceptance/rejection/edit/reversal; designer control; authorship perception; inter-rater agreement; provenance completeness.

### AB-2: Pattern and semantic interchange

Metrics: pattern-piece identity preservation; unit/grain/notch/seam-allowance/topology round-trip; local seam-length error; grading-rule preservation; 2D/3D correspondence; explicit loss reports; import/export determinism.

Hard target: zero silent semantic loss for supported fields. Unsupported fields must fail or be clearly marked.

### AB-3: Fit and movement

Metrics: local ease error by body region and direction; fit landmark registration; geometric clearance; pose-dependent strain; pattern-to-article dimensional deviations; task acceptance; wearer-reported comfort/fit kept distinct.

Stratify by garment family, body profile, posture, mobility/accessibility requirements and fabric family. Do not publish a single average that hides subgroup failures.

### AB-4: Material and cloth physics

Metrics: measured-vs-simulated deformation, drape, bending, stretch/recovery, shear, friction, settling and task outcome. Report calibration and held-out specimen error separately, plus runtime/memory Pareto fronts.

### AB-5: Manufacturing

Metrics: pattern manufacturability, seam operation compatibility, nesting utilization, material waste, cycle time, first-pass yield, rework, rejected output and machine/tool capability coverage.

### AB-6: Wardrobe-first planning

Metrics: user-declared need coverage; false need-gap recommendations; outfit compatibility; repair/alteration success; new-item count; total user cost/time; style satisfaction; protected-garment respect; privacy.

### AB-7: Circularity and passport evidence

Metrics: measured composition coverage, lot/revision identity, source-class provenance, public/authorized disclosure leakage, stale/revoked evidence detection, repair/reuse event receipts, physical recovery-grade acceptance, mass-balance closure and environmental assessment completeness.

## 5. Matched baseline design

- Use the same briefs, garments, patterns, measurements/manikins, fabric parameters, machine constraints, time budget and evaluation rubric.
- Record exact software version, hardware, operator expertise, model/version, parameter budget, manual corrections and license constraints.
- Use garment-, pattern-family-, fabric-lot- and body-profile-grouped splits. Prevent image/video frames or textile replicates from crossing splits.
- Keep synthetic, simulated, supplier-declared, measured and independently adjudicated evidence separate.
- Preregister primary outcomes; use blinded review and confidence intervals; report failures, exclusions, missing data and lower-tail performance.
- For small exact problems, compare heuristics against exhaustive or independently implemented oracles. A Pareto score is not a substitute for feasibility.

## 6. Proposed targets (hypotheses, not achieved)

- 100% of accepted design suggestions bind to the exact brief revision/digest and human decision.
- Zero hard-constraint violations admitted into an engineering handoff.
- Zero missing/unknown hard gates treated as passing.
- Zero passing gate without an evidence reference.
- 100% preservation of supported pattern semantics across tested interchange paths.
- 30% lower median time to independently accepted production package vs matched CAD-only baseline, with no reduction in quality or agency.
- 25% fewer physical sample iterations to same independently accepted fit/quality; validate against human/CAD control.
- 20% reduction in matched cutting waste without reducing accepted output or changing functional unit.
- 30% fewer physical experiments to reach a material target vs matched space-filling baseline, with held-out physical validation.
- Zero unnecessary purchase recommendation when user has declared a no-new-item constraint.
- Zero unsupported physical, environmental, composition, durability, comfort or regulatory claims.

Freeze quantitative thresholds only after exact baseline reproduction, measurement repeatability, task scope and power/sample-size analysis.

## 7. First engineering vertical

Use the non-PPE woven overshirt already selected in APPAREL-ENG-TWIN-001 as the first controlled vertical. Start with five styles: Everyday Core, Heat-Adaptive, Adaptive Ease, Repair-First and Creative/Statement. Each is a starting configuration, not a quality or sustainability claim.

The first Rust slice (`atelier-brief-core`) validates brief identity, hard-constraint consistency, relative objective definitions and suggestion provenance; it provides a fail-closed gate for downstream handoff. It intentionally does not reimplement Symtropy's Pareto engine, CAD, physics or pattern generation.

## 8. Evidence and claim ceiling

A passing schema or Rust semantic test validates only contract behavior. It does not establish pattern correctness, fit, comfort, fabric physics, durability, safety, manufacturing yield, circularity, environmental superiority or regulatory compliance. These require owner-specific measurement, physical trials, human studies and external review.

## 9. Primary sources

- CLO 2026.1 official release: https://www.clo3d.com/en/resources/notices/1442
- CLO 2026.0 official release: https://www.clo3d.com/en/resources/notices/1374
- Browzwear VStitcher official product page: https://browzwear.com/products/v-stitcher
- EasyFashion: https://arxiv.org/abs/2609.18483
- EASE: https://doi.org/10.1016/j.cag.2026.104644
- Human-AI sustainable fashion co-design: https://doi.org/10.1177/24723444261488123
- RGBench website: https://rgbench.github.io/
- RGBench current evaluation dataset: https://huggingface.co/datasets/RGBench/RGBench-Cloth-Sim2Real-v1
- TexSortML: https://doi.org/10.3390/s26196236
- GarmentCodeData: https://doi.org/10.3929/ethz-b-000673889
- European Commission textile DPP: https://single-market-economy.ec.europa.eu/single-market/digital-product-passport/textile-apparel_en
- UNEP InTex South Africa: https://www.unep.org/en/intex
- South African Government public notice register: https://www.gov.za/documents/public-comment
