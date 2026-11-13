# TEXTILE-SORT-FUSION-001 — visual + spectral textile evidence v0.1

**Status:** executable Rust evidence-admission contract and 15-case synthetic corpus. No real dataset has been downloaded, no fibre classifier has been trained, and no physical sorting line has been evaluated.

**Owner:** Symthaea `symthaea-materials::textile_sensing`

**Research issue:** [TEXTILE-SORT-FUSION-001 #11743](https://github.com/Luminous-Dynamics/symthaea/issues/11743)

**Parent programme:** [APPAREL-ENG-TWIN-001 #11731](https://github.com/Luminous-Dynamics/symthaea/issues/11731)

---

## 1. Problem and research thesis

Textile sorting is not one classification task. A real garment may require separate answers to:

1. What visual garment family and presentation state are visible?
2. What fibre composition is supported by a calibrated material sensor?
3. Which portions of the material remain unknown, blended, contaminated or out-of-domain?
4. What evidence is required before a material stream may be routed to a particular physical recovery process?
5. Did the physical sort actually produce the declared output lot and grade?

The system must keep those questions independent. A shirt/jacket/dress label does not establish cotton/polyester/wool composition. NIR spectra do not establish garment category, contamination, wearability, recyclability or output grade. A classifier prediction does not prove a physical sort occurred.

**Paradigm shift:** move from forced top-1 labels to an evidence-bearing sensing cascade with independent modality results, calibrated uncertainty, explicit open-set abstention and downstream physical outcome measurement.

## 2. Existing ownership

- Symthaea [TEXTILE-HYG-MFG-000 #5984](https://github.com/Luminous-Dynamics/symthaea/issues/5984) owns textile product-function evidence and its qualification ceiling.
- Symthaea [symthaea-materials](https://github.com/Luminous-Dynamics/symthaea/tree/main/crates/domains/symthaea-materials) owns shared material representation, HDC-based property search and related material reasoning. This v0.1 slice is added as `textile_sensing`; it does not create a second material database.
- Symtropy textile robotics [#4865](https://github.com/Luminous-Dynamics/symtropy/issues/4865) owns cloth handling/manipulation trials and physical action evidence.
- Symtropy RGBench reproduction [#4870](https://github.com/Luminous-Dynamics/symtropy/issues/4870) owns cloth-model Sim2Real benchmark admission.
- Mycelix textile recovery [PR #5057](https://github.com/Luminous-Dynamics/mycelix/pull/5057) owns recovery event/secondary-fibre evidence conformance; its combined D6S integration candidate is [PR #6095](https://github.com/Luminous-Dynamics/mycelix/pull/6095).
- Mycelix textile product-passport projection [#6092](https://github.com/Luminous-Dynamics/mycelix/issues/6092) owns a controlled projection over admitted product/lifecycle records.

The sensing module is not a recovery eligibility engine, product passport, inventory ledger, sensor calibration service or robot controller.

## 3. Latest public benchmark sources

### 3.1 TexSortML — visual garment category and presentation state

Paper: https://doi.org/10.3390/s26196236
Dataset DOI: https://doi.org/10.5281/zenodo.22143896

The 30 September 2026 paper describes:
- 751 post-consumer garments;
- 11 visual categories;
- 2,253 annotated frames;
- 3 presentation stages for each garment;
- garment-level five-fold validation, keeping all frames for one garment in the same fold.

Reported ViT accuracy is 95.23%, 91.90% and 85.94% across the three stages (9.3 percentage-point first-to-third drop). The reported ResNet-18 and ResNet-50 drops are 16.9 and 16.6 points. These numbers are paper-reported, not independently reproduced by Luminous.

Limitations:
- small and class-imbalanced sample;
- garments were sourced from collection containers and had already been pre-sorted at one facility;
- its labels describe visual garment family/presentation, not fibre composition;
- a very large ViT (about 305.6M parameters in the reported setup) is compared with substantially smaller ResNets, so parameter/compute-matched comparisons are required before claiming architecture superiority.

### 3.2 NIST NIR-SORT — fibre/composition evidence

Catalog: https://catalog.data.gov/dataset/near-infrared-spectra-of-origin-defined-and-real-world-textiles-nir-sort-a-spectroscopic-a
Paper: https://doi.org/10.1038/s41597-026-07434-6
Record DOI: https://doi.org/10.18126/61v0-rf98

The NIST catalog describes 113 fabric specimens:
- 70 known-provenance fabrics with known fibre content and varied dyes/finishes;
- 20 known-provenance undyed blends;
- 11 post-consumer textiles;
- 12 pre-consumer textiles.

The accompanying Scientific Data paper describes NIR-SORT versions 1.0 and 2.0, a sample library of 113 fabrics and a separate fibre-specimen library. NIST's NIR-SORT 2.0 release notes report 61 new custom-blend specimens, a new handheld profile (Handheld_F), and a standoff benchtop experiment (Standoff_Q). Keep fabric specimens, fibre references, and replicate spectra as distinct entities in the intake manifest; never count replicates as independent specimens.

Nominally seven NIR replicates per fabric specimen are described, using:
- a benchtop instrument covering 833–2500 nm;
- two handheld instrument ranges covering 1550–1950 nm.

The catalog lists approximately 193 MB spectroscopy data and 10.6 GB microscopy data. Start with spectroscopy; microscopy is a later opt-in lane. Verify repository and file-level rights/attribution before copying, downloading in CI or redistributing any data.

### 3.3 Existing industrial sorting and trim-removal baseline

Do not frame the research gap as “industry has no trim removal.” Valvan publicly describes a commercial textile sorting sequence with:
- **Fibersort:** NIR-based fibre/blend prediction plus RGB colour sorting;
- **Trimclean:** a downstream system described as extracting trims including zippers, buttons and labels from fibre-sorted material.

The supplier positions Trimclean as a way to improve feedstock purity for textile-to-textile recycling. This makes Fibersort + Trimclean a direct incumbent baseline for any proposed sensing-to-disassembly line. Supplier descriptions establish advertised capabilities, not independently reproduced yield, damage, purity, uptime or operating-cost results.

A 2025 Refashion technical-monitoring comparison also lists NIR/hyperspectral systems including Pellenc ST Mistral+ CONNECT and TOMRA AUTOSORT, alongside Fibersort. Its capacity figures are context-specific and should not be treated as directly comparable without common feedstock, presentation, purity and uptime definitions.

The September 2026 AI literature review's finding that only one study in its 173-paper corpus addressed attachment removal/separability is a **research-publication gap**, not proof that commercial trim-extraction machinery does not exist. Our challenge is to benchmark the full process—including unknown constructions, partial removal, textile damage, mass balance and downstream acceptance—against the installed industrial workflow, not just against a vision classifier.

Primary industry sources:
- Valvan Fibersort and Trimclean: https://www.valvan.com/en/solutions/textile-sorting-recycling
- Refashion technical monitoring (sorting and unravelling technologies): https://media-pro.refashion.fr/2025/11/240428_synthese_veille-technos-tri-delissage_vf-en.pdf

### 3.4 Task separation

| Benchmark | Valid question | It does not establish |
|---|---|---|
| TexSortML | Which visual garment category/presentation state is visible? | Fibre composition or recovery eligibility |
| NIR-SORT | What fibre/composition evidence is supported by this instrument/specimen/method? | Garment category, contamination or recyclability |
| Mycelix recovery evidence | Is a declared physical recovery event and output-lot claim sufficiently evidenced? | Sorting model accuracy or sensor calibration |

Fusion must not collapse these three evidence planes.

## 4. Executable Rust v0.1

The new module is:

`crates/domains/symthaea-materials/src/textile_sensing.rs`

It exports through `symthaea-materials::textile_sensing`.

### 4.1 NIR admission

An NIR composition estimate is admitted only when:
- observation, specimen, subject, instrument, replicate-group and evidence identities are present;
- instrument identity matches the declared instrument profile;
- observation, calibration and profile share the exact calibration-profile identity;
- calibration was not revoked and was valid at acquisition time;
- the observed wavelength interval is within the declared qualified profile;
- signal quality is valid and the composition estimate is structurally/arithmetically coherent;
- fibre fractions close to exactly 1,000,000 ppm, duplicate fibre identities are absent, and each uncertainty interval contains its point estimate.

Expired-at-acquisition, saturated, low-signal, out-of-domain or missing-composition observations abstain as `Unknown`; explicit calibration revocation or malformed/contradictory identity/units is rejected. A declared unknown fraction is retained and yields an unknown disposition rather than a guessed fibre identity.

This is an admission validator for externally produced estimates. It does not implement a NIR classifier, absorbance preprocessing or instrument calibration routine.

### 4.2 Visual garment observation

Visual observations have a separate type containing:
- garment category;
- presentation state;
- model identity;
- confidence;
- garment and subject identity;
- source image/annotation evidence reference.

The type has no fibre-composition field. Confidence floors are caller-declared policy inputs; they are not treated as universal thresholds.

### 4.3 Paired multimodal observations

When both modalities are supplied, the pair is bound only if both refer to the same physical subject. The assessment preserves the visual and spectral results separately.

No visual-to-fibre label transfer exists in this contract. If visual-only data arrive, the spectral result remains absent; if modalities identify different items, pair binding is rejected, even though each source's independent assessment is retained.

### 4.4 Leakage-safe split validation

`validate_grouped_splits` enforces:
- TexSortML-style frames for one garment identity remain in one train/validation/test/fold partition.
- NIR-SORT replicate spectra for one physical specimen remain in one partition.
- row identities are unique within each grouping kind.

This prevents multiple views/replicates of one physical item from leaking across the train/test boundary and inflating performance.

### 4.5 Evidence-qualified recovery routes

The new module `symthaea-materials::textile_recovery` evaluates candidate routes separately from sensor admission:

- **Reuse as-is:** requires physical inventory, human condition inspection, safety/hygiene pass and independent receiving acceptance.
- **Repair:** requires a repairable condition, physical repair trial and independent post-repair acceptance.
- **Component recovery:** requires a physical component-recovery trial, mass balance and independent component receiver acceptance.
- **Mechanical fibre recycling:** requires admitted calibrated composition evidence, physical opening trial, closed mass balance, measured fibre-length distribution and independent downstream acceptance.
- **Chemical fibre recycling:** requires admitted composition, a physical chemical-compatibility trial, a physical process trial, mass balance and independent downstream acceptance.
- **Downcycling:** requires a physical route trial, mass balance and independent acceptance against a declared receiving specification.

Evidence roles use exact evidence tiers, not a generic “higher score” hierarchy. Downstream acceptance records include a versioned specification and an explicit Accepted/Rejected/Conditional/Unknown outcome. The issuer and verifier identifiers must differ for independent acceptance. These fields are structurally checked; the library does not cryptographically authenticate them.

Every destructive route binds the trial input mass to the physical lot inventory and checks mass closure against a policy ceiling for measurement uncertainty. Fibre-length bins must be ordered, non-overlapping and close to 1,000,000 ppm. The minimum fibre length and required mass fraction are explicit receiving-policy parameters; no single universal 12.7 mm threshold is asserted.

Unknown condition, chemistry, trial or receiving status remains insufficient evidence. An actual failed trial, rejected output, invalid digest/tier or mass mismatch is ineligible under the declared policy. A route assessment is not an environmental certification, safety certification, legal determination or proof that its evidence sources are honest.

### 4.6 Design-for-recovery component/joint topology

The module `symthaea-materials::textile_design` adds a static design dossier:

- product ID, exact design revision and product family;
- component inventory with role, material reference, mass estimate and replacement intent;
- joint inventory with connected component IDs, joining method, reversibility, access zone, removal operation, tools and predecessor dependencies;
- design intents for repair, disassembly/component recovery and material traceability.

The static audit detects duplicate/dangling component and joint IDs, invalid joint cardinality, unknown materials, replaceable modules without instructions, reversible joints without access/tool/operation, unknown or permanent joints that conflict with the stated recovery scope, and cyclic removal dependencies.

**Recovery-relevant joint scope matters:** a garment can include permanent structural seams outside the intended recovery path while a cuff, pocket, collar, fastener or trim module is intentionally replaceable. Each joint therefore declares whether it is in the target disassembly/recovery path. A disassembly intent with no target joints is incomplete; a permanent joint inside that target path blocks the claim. This avoids the naive rule that every stitch in a garment must be reversible.

The 13-case synthetic design corpus is schema-checked and consumed by Rust tests:
- `docs/research/fixtures/textile_sort_fusion_v0_1/design_cases.schema.json`
- `docs/research/fixtures/textile_sort_fusion_v0_1/design_cases.example.json`

A `Complete` static dossier proves only that its declared component/joint graph is internally coherent. It does not establish seam strength, fatigue life, wash durability, actual access in a physical garment, safe repair, disassembly time, panel retention, or downstream recovery. Those require physical testing and independent acceptance.

## 5. Current synthetic tests

Before corpus integration, the module contained 16 Rust unit tests. The latest change adds an unknown-category/presentation negative control and an executable corpus-binding test (18 test functions total). The 15-case synthetic corpus is now loaded by Rust tests; each case invokes its mapped regression and checks the declared disposition/reason. The Rust unit tests cover the 16 original contract cases plus two new regressions: unknown visual category/presentation abstention and the executable 15-case corpus-to-test mapping (18 test functions total). The original cases cover:
1. valid current-calibration composition admission;
2. expired calibration abstention;
3. revoked calibration rejection;
4. instrument/profile mismatch rejection;
5. wavelength-domain abstention;
6. saturated-signal abstention;
7. missing composition abstention;
8. invalid fraction-total rejection;
9. preserving an unknown blend fraction without forcing a label;
10. visual-only input not producing fibre-composition evidence;
11. visual/spectral subject mismatch;
12. visual confidence below a declared threshold;
13. garment-identity split leakage;
14. physical-specimen replicate leakage;
15. valid grouped replicate/fold assignment;
16. unknown visual category or presentation state abstention.

A separate synthetic fixture corpus is provided at:
- `docs/research/fixtures/textile_sort_fusion_v0_1/sensor_cases.schema.json`
- `docs/research/fixtures/textile_sort_fusion_v0_1/sensor_cases.example.json`

The JSON corpus is consumed by the Rust test suite. Each synthetic case ID maps to an executable regression case; the test checks unique IDs, corpus/schema identity, expected disposition and expected reason. This proves only the fixture-to-contract regression mapping, not performance on the real NIST or TexSortML datasets.

The recovery-route contract has a separate 27-case synthetic corpus:
- `docs/research/fixtures/textile_sort_fusion_v0_1/recovery_cases.schema.json`
- `docs/research/fixtures/textile_sort_fusion_v0_1/recovery_cases.example.json`

The recovery corpus is also consumed by Rust tests. It covers six route families and adversarial states including model-only composition, unresolved fibre fraction, missing/short fibre-length output, chemical compatibility fail/unknown, downstream reject/conditional, self-issued acceptance, malformed digest, duplicate evidence ID and mass-input mismatch. Its expected outcomes are bound to the route tests. No real specimen or processor acceptance is asserted.

The design-for-recovery module adds a separate 13-case synthetic topology corpus, also consumed by Rust tests. It covers complete modular topology, duplicate/dangling identities, missing tools/instructions, permanent target joints, unknown construction, circular dependencies, and the distinction between recovery-target joints and permanent seams outside the declared recovery path.

## 6. Benchmark programme to exceed SOTA

### B0 — reproduce both public baselines independently

TexSortML:
- reproduce the published garment-level grouped five-fold split;
- report per-class macro-F1, balanced accuracy, precision/recall, confusion and confidence calibration at each presentation stage;
- compare matched model parameter counts, FLOPs, resolution, training data and inference budget.

The September 2026 paper's directly fine-tuned ViT baseline reports **85.94% Stage-3 accuracy** and **83.04% Stage-3 macro-F1** across the most disordered garment presentation. It records 95.23% Stage-1 accuracy, but Stage 1 is not the deployment-defining condition. The weak Zipper/Sweat Jacket class and the single-facility sample source are explicit limitations.

| TexSortML benchmark gate | Published anchor | Proposed stretch objective (not achieved) |
|---|---:|---:|
| Same grouped five-fold reproduction | ViT Stage 3: 85.94% accuracy / 83.04% macro-F1 | Reproduce both within ±1.0 pp before claiming a model improvement |
| Most-disordered Stage 3 | 85.94% accuracy / 83.04% macro-F1 | ≥88.0% accuracy and ≥85.5% macro-F1 on the identical split, taxonomy and compute budget |
| Weakest-class performance | Zipper/Sweat Jacket is the weakest class; 17 garments | At least +5 pp worst-class recall without reducing overall macro-F1; report garment-level confidence intervals |
| Domain transfer | Single facility / one camera setup; external-facility transfer untested | A separate multi-facility held-out test, with no facility overlap in train/validation; freeze targets after first independent facility dataset |
| Open-set operation | Not established by the published top-1 task | ≤1% false acceptance of unknown categories/material states at a separately reported ≥90% accepted coverage on a preregistered unknown set |

These targets are deliberately above the published benchmark but modest enough to make a clean reproduction the first gate. An improvement is not accepted if it comes from more training data, larger compute, leakage, taxonomy changes or easier evaluation examples unless those changes are reported as separate experimental arms.

NIR-SORT:
- group all replicates by physical specimen;
- stratify by instrument (benchtop vs each handheld instrument);
- compare raw/normalized spectra and simple classical baselines (PCA + SVM, PLS-DA where methodologically appropriate) with a version-pinned learned baseline;
- preserve preprocessing, method, reference composition measurement and calibration identity;
- reproduce instrument-specific baselines before cross-instrument fusion;
- report blend-fraction MAE in percentage points, per-fibre precision/recall, prediction-interval coverage and specimen-grouped confidence intervals.

The NIR-SORT 2.0 custom-blend specimens and Handheld_F / Standoff_Q profiles create a ratio-generalization and instrument-transfer challenge. The objective is not just to classify broad fibre labels: estimate composition with bounded uncertainty, preserve unknown fractions and evaluate entire held-out specimens/instruments. A proposed stretch gate is to reduce the composition-error increase on held-out instruments by at least 50% relative to a simple instrument-naive baseline, while maintaining nominal 90% prediction-interval coverage within a predeclared tolerance. Freeze exact tolerances after establishing source reference uncertainty; no performance value is claimed yet.

Do not pool the visual and spectral scores into one leaderboard.

### B1 — open-set and domain shift

Adversarial domains:
- unknown blends or fibre families;
- new dyes, finishes, coatings and laminates;
- wet, dirty, worn and aged samples;
- partial garment views, occlusion, overlap and extreme folds;
- handheld-to-benchtop instrument shift;
- low signal, saturation, wavelength truncation and calibration expiry;
- conflicting replicate measurements and changed sensor firmware.

Metrics:
- macro-F1 and worst-class recall;
- per-class precision/recall and blend error;
- unknown/OOD false-accept rate;
- coverage-risk curves and selective prediction;
- expected calibration error/Brier score;
- confidence/interval coverage;
- instrument-transfer degradation;
- time to abstain/request remeasurement.

The most important metric may be **false acceptance of an unknown material** rather than the highest top-1 accuracy.

### B2 — sensor fusion

Compare matched models/policies:
1. visual-only;
2. NIR-only;
3. late fusion with separate calibrated modality outputs;
4. learned fusion with explicit missing-modality handling;
5. conservative evidence cascade that requests another measurement on conflict/OOD.

All arms receive the same physical specimen IDs, training splits, calibration boundaries and compute budget. The cascade must never invent fibre composition from a garment label or treat a supplier declaration as a spectral measurement.

### B4 — design-for-repair and design-for-disassembly

Compare conventional and modular/reversible garment construction under matched function, materials, wearer comfort and durability:
- static dossier completeness and number of unknown joins/material references;
- percentage of replaceable components with versioned repair instructions and available spare components;
- disassembly success, time per target joint, tool changes and operator skill requirements;
- panel retention, textile damage, broken yarns, residual attachment mass and component salvage;
- seam strength, fatigue, abrasion and wash-cycle performance before and after repair/disassembly;
- repair success, repair time/cost, post-repair function and additional useful-life cycles;
- downstream fibre-length/yarn quality after opening each garment construction;
- added manufacturing cost, production time, material consumption and component inventory burden.

Run at least three arms: (1) conventional sewn construction, (2) selectively reversible modules with permanent non-target seams, and (3) highly modular construction. This supports a Pareto analysis rather than forcing all joints to become reversible. A design that is easy to disassemble but fails seam fatigue or wearer function is not an improvement.

A 2025 Refashion design guide identifies fasteners, labels, linings, elastics, rivets, padding, decorations and electronics as recycling disruptors; ISO 5157:2023 defines vocabulary for repairability, disassembly and recyclability; Cradle to Cradle guidance recognises strategic placement/accessibility of trims as design-for-disassembly features. These are design references, not substitutes for physical trials.

### B3 — physical sorting, route selection and recovery

Only after a guarded instrumented cell exists, measure:
- item-level correct sort / reject / abstain and route-selection coverage-risk;
- mass per physical output stream, with scale uncertainty and mass-balance residual;
- garment reuse/repair acceptance and useful life where a longitudinal receiving/use test is possible;
- component reuse acceptance, not merely successful extraction;
- residual trim by attachment family and process stage;
- for mechanical opening: fibre-length distribution, short-fibre/fines share, opening efficiency, accepted spinnable mass, yarn imperfections/strength and receiving-spinner acceptance;
- for chemical routes: polymer purity, molecular/functional properties, contaminants, solvent/chemical inputs, process safety and receiving-polymer acceptance;
- downstream product-grade acceptance, not model-predicted eligibility;
- throughput, recirculation, labour, intervention, downtime, utilities and rejects;
- actual output lot identity, custody and versioned receiving specification.

A 7 November 2026 controlled cotton study found that fabric interlacing density was a major predictor of opening efficiency (woven structures: reported correlation r = -0.94) and reported up to 70% of recovered fibre mass above 12.7 mm for low-interlacement weaves under its tested conditions. It demonstrates why fabric construction and measured fibre-length distributions matter; its numbers are not universal acceptance criteria:
https://doi.org/10.1016/j.wasman.2026.115980

A 5 May 2026 industrial-scale cotton study reports that combing and compact ring-spinning improve the handling and performance of mechanically recycled cotton at increasing recycled ratios, emphasizing downstream yarn quality instead of treating opening yield as final success:
https://doi.org/10.1016/j.wasman.2026.115502

Classification accuracy alone cannot prove recyclability, recovery yield or environmental benefit. Each recovery route needs a receiving-process specification and an independent outcome.

## 7. Proposed challenge targets (not achieved)

Freeze after baseline reproduction and measurement repeatability:
- zero admitted spectra from missing, revoked or out-of-validity calibration;
- zero fibre-composition claims derived from visual garment labels;
- 100% garment/specimen group integrity across all train/test splits;
- zero silent imputation of missing/saturated/out-of-domain spectra;
- report selective coverage and risk; choose an operating point with a frozen false-accept ceiling;
- improve macro-F1 and worst-class recall against compute-matched baselines without increasing unknown-material false acceptance;
- demonstrate reduced instrument-transfer degradation on held-out specimens, with a preregistered instrument-held-out split;
- compare physical line arms against Fibersort + Trimclean on matched lots, including residual trim, damage, retained textile length/fines, throughput, labour, utilities and downstream accepted grade;
- 100% of actual physical output lots linked to sort, route-trial, mass-balance and independent receiving-acceptance receipts before material recovery claims;
- route-specific accepted output mass and value per input kg improve against the matched incumbent without increasing unsafe false acceptance, textile damage or downstream rejection;
- mechanical-route improvements are measured through fibre-length distribution and receiving-yarn quality, not shredder throughput alone.

These targets are challenge goals, not predictions or results. Freeze thresholds only after the data protocol, class taxonomy, held-out split and source measurement quality are established.

## 8. Claim ceiling

A passing Rust contract test validates only the declared input/admission and route-record semantics. It does not establish model accuracy, calibrated fibre identification, garment condition, industrial throughput, physical repair/disassembly performance, fibre length, secondary-fibre grade, recycling yield, product safety, lifecycle impact or regulatory conformity. Those require licensed data, independently checked references, calibrated instruments, physical process trials, measured outputs and actual receiving-process acceptance.

The EU Commission currently lists Q4 2027 as the planned adoption window for the textile ESPR delegated act; technical specifications and implementation measures follow adoption, and dates may change. Textile Exchange's Materials Matter Standard V1.0 becomes effective 31 December 2026 and mandatory for relevant Tier 4 audits on 31 December 2027. These are versioned external frameworks, not permission to invent final DPP fields or certification claims:
- https://single-market-economy.ec.europa.eu/single-market/digital-product-passport/textile-apparel_en
- https://textileexchange.org/app/uploads/2025/12/TE-MM-STN-101-V1.0-Materials-Matter-Standard-1.pdf
