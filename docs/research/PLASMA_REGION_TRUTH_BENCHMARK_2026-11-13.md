# PLASMA-REGION-TRUTH-001 — Measurement-support-aware exposure/outcome joins

**Status:** S0 benchmark contract and synthetic evaluator  
**Date:** 2026-11-13  
**Issue:** [#11952](https://github.com/Luminous-Dynamics/symthaea/issues/11952)  
**Claim class:** synthetic contract only; no physical validation

## 1. Research thesis

Spatial process maps are not spatial ground truth. A process image may contain thousands of pixels, while the product measurement may be integrated over a bonded area or an entire specimen. Treating the latter as one label per pixel creates pseudoreplication, leakage, false precision and invalid local-failure claims.

The benchmark introduces a strict rule:

> A claim may not have finer spatial or temporal support than the physical observation that supports it, unless an independently validated measurement operator justifies the reconstruction.

The target is not to produce the densest map. It is to make every spatial claim traceable to the actual support, coordinate transform, time interval, uncertainty and independence structure of its source observation.

This complements the existing plasma programme:
- [PLASMA-EXPOSURE-MAP-001 #11925](https://github.com/Luminous-Dynamics/symthaea/issues/11925) owns delivered exposure descriptors, spatial coverage and geometry transfer.
- [PLASMA-VM-001 #11847](https://github.com/Luminous-Dynamics/symthaea/issues/11847) owns process telemetry to downstream outcome prediction.
- [PLASMA-SURFACE-001 #11795](https://github.com/Luminous-Dynamics/symthaea/issues/11795) owns material/adhesive profiles and product acceptance endpoints.
- [PLASMA-OED-001 #11907](https://github.com/Luminous-Dynamics/symthaea/issues/11907) owns decision-focused measurement selection.
- [MAT-013 #4344](https://github.com/Luminous-Dynamics/symthaea/issues/4344) and [MAT-INTERFACE-001 #6227](https://github.com/Luminous-Dynamics/symthaea/issues/6227) own canonical sample lineage and interface-state identity.
- Symtropy [PLASMA-TWIN-001 #1685](https://github.com/Luminous-Dynamics/symtropy/issues/1685) owns scenario/geometry replay.

This benchmark does not create a second coordinate system, calibration registry, sample registry, optimizer, exposure generator, virtual-metrology model, digital twin or product-release authority.

## 2. Why the problem is physical, not cosmetic

Polášková et al. studied an RF atmospheric-pressure plasma slit jet on moving polypropylene samples of 5 mm and 30 mm thickness. Although the slit-to-top-surface distance was held at 10 mm, treatment efficiency changed with sample thickness under selected gas conditions because the distance to ground altered the electric field and filament behaviour. Edge and centre treatment also differed in selected settings. The authors varied gas feed, power and conveyor speed and measured spatial wettability/chemistry rather than assuming one recipe represented a uniform delivered treatment.

- Polášková et al. (2024, published online 10 October; *Plasma Chemistry and Plasma Processing* 45, 49–68), DOI: [10.1007/s11090-024-10517-0](https://doi.org/10.1007/s11090-024-10517-0).

A 2026 argon cold-atmospheric-plasma study on LDPE/HDPE reports partial hydrophobic recovery after treatment and changes in water-vapour and oxygen barrier properties. The recovery measurements are contact-angle measurements; they are not a substitute for delayed bond-strength outcomes. The barrier changes are a separate product-function endpoint.

- Spanou et al. (first published 7 February 2026), *Argon Cold Plasma Modification of Polyethylene Films to Tailor Surface and Barrier Properties*, DOI: [10.1002/app.70533](https://doi.org/10.1002/app.70533).

A 2026 five-polymer comparison similarly finds that the lowest contact angle does not always correspond to the highest adhesive bond strength, reinforcing that a proxy measurement is not itself a product outcome.

- Péczi-Kovács & Kókai (20 October 2026), *Effect of argon and nitrogen plasma surface treatment on wettability and adhesive bond strength of PP, HIPS, PVC, PC and PET*, DOI: [10.1007/s44245-026-00396-4](https://doi.org/10.1007/s44245-026-00396-4).

The measurement-support question follows directly: if local exposure, surface chemistry and integrated mechanical outcomes are joined incorrectly, the system can learn a spatial relationship that the experiment never measured.

## 3. Paradigm shift: observations are operators over physical support

Do not model a measurement as a scalar attached to a pixel. Model it as a source observation plus a support operator.

For a spatially resolved observation, a generic representation is:

    y_i = h_i(z, S_i, phi_i, t_i) + epsilon_i

where:
- z is the latent physical response field, if a field model is justified;
- S_i is the physical support (point footprint, area, bonded joint or specimen);
- phi_i is the registration/coordinate transform and its uncertainty;
- t_i is temporal support;
- h_i is the measurement operator appropriate to the method;
- epsilon_i is measurement error, including any declared correlated components.

For a validated area-averaging instrument, a linearized form may be suitable:

    y_i = integral_{S_i} w_i(x; phi_i) z(x) dx + epsilon_i,
    integral_{S_i} w_i(x; phi_i) dx = 1.

This is not a universal law for destructive tests. A lap-shear test has a joint-level response operator affected by adherend geometry, adhesive, stress distribution, test method and failure mode. It must not be reinterpreted as a point measurement or a spatial failure-origin measurement without separate evidence.

When registration is uncertain, the operator must retain that uncertainty. Conceptually:

    p(y_i | z) = integral p(y_i | z, phi_i) p(phi_i) dphi_i.

A deterministic transform may be used only where calibration and error bounds justify it. An arbitrary coordinate matrix does not establish registration accuracy.

## 4. Support classes and claim ceiling

Every source observation declares exactly one support class:

| Support class | Physical meaning | Permitted use |
|---|---|---|
| POINT | Local coordinate with a finite, calibrated measurement footprint | Point/footprint-level claim, then aggregation with a declared rule |
| AREA | Polygon, mask or finite instrument footprint with weighting semantics | Area-level claim or explicitly justified aggregation |
| SPECIMEN | Integrated article, coupon or bonded-joint outcome | Specimen/joint-level claim only |
| FIELD | Independently measured spatial field with sampling/interpolation provenance | Field claim only within validated coverage/resolution |
| INFERRED | Model output, not a direct physical observation | Prediction only; never physical ground truth |

The support class does not automatically imply a resolution. A POINT observation still has a finite point-spread/footprint; a FIELD may contain gaps; and an AREA measurement may aggregate nonuniform stress or chemistry.

### Claim-resolution ordering

- A SPECIMEN observation cannot directly support an AREA or POINT label.
- An AREA observation cannot be copied into many POINT labels.
- A POINT observation may support a larger claim only through a declared aggregation rule and a sufficient sampling design.
- A FIELD observation supports only the sampled/interpolated domain and stated resolution; unobserved regions remain unknown.
- An INFERRED observation cannot satisfy a physical-measurement obligation.

The contract is conservative by design. It permits a valid specimen-level outcome to be joined to the matching specimen and bonded-joint geometry. It refuses to pretend that the same outcome localizes the exact failure coordinate.

## 5. Registration and support uncertainty

A valid join binds:
- source and destination coordinate-frame IDs;
- article/mesh/geometry revision;
- transform identity and calibration;
- registration uncertainty or a conservative bound with units;
- support footprint or specimen/joint ID;
- temporal support and synchronization bound;
- source map / observation digests;
- instrument, method, measurand and units;
- physical/synthetic/inferred evidence class;
- common-cause and independence-group identities.

A transform uncertainty bound must be evaluated against the physical decision margin, feature scale and support overlap. A fixed universal "one pixel" threshold is not sufficient: pixel pitch is a sampling choice, not a physical tolerance.

The S0 fixture uses deliberately simple millimetre bounds to test logic:
- registration_bound_mm must not exceed the declared admissible bound;
- the bound must remain strictly inside the decision margin;
- uncertain overlap across conforming and nonconforming regions must produce BLOCKED/ABSTAIN.

These are synthetic contract values, not instrument specifications.

## 6. Aggregated outcomes and spatial inference

Three cases must remain distinct:

1. **Direct local truth.** A spatially resolved physical method measures a local support. Local outcome claims are limited to that method's support and uncertainty.
2. **Integrated truth.** A destructive test returns an outcome for the declared specimen/joint. Preserve it at that level. Do not fan it out across pixels.
3. **Operator-based inference.** An explicit measurement operator maps a latent spatial field to the integrated observation. Local reconstruction requires validated operator assumptions, independent spatial ground truth, uncertainty propagation and held-out validation. A software flag saying "operator valid" is not itself physical validation.

A lap-shear result may be joined to an exposure map at the declared bonded-joint/specimen level. It does not locate the failure origin by itself. Failure-location or local adhesion claims require a suitable independently validated spatial observation.

A whole-coupon average may hide a critical edge region. Conversely, a local measurement cannot establish whole-coupon reliability without a justified sampling/aggregation plan. The benchmark therefore keeps integrated acceptance and local spatial risk as separate outputs.

## 7. Independence, leakage and pseudoreplication

Spatial patches from one coupon share material lot, preparation, treatment run, geometry, operator, calibration and environmental history. They are not independent specimens.

At minimum, splits are grouped by:
- physical specimen / article;
- material lot and conditioning batch;
- plasma run and production day;
- device/source configuration;
- operator and common preparation group where relevant.

No patch-level random split may place the same physical specimen or a shared causal group in both training and holdout. The fixture contains a direct leakage negative control.

Repeated rows, interpolated cells and duplicated support IDs cannot increase the independent observation count.

## 8. Frozen known-answer challenge suite

The S0 fixture contains 18 synthetic cases:

| Case | Expected audit | Reason |
|---|---|---|
| PIXEL_LEAKAGE | REJECTED | Specimen identity overlaps train/test |
| SPECIMEN_TO_PIXEL_SMEAR | REJECTED | Integrated specimen outcome claimed at point resolution |
| AREA_SUPPORT | PASS | Finite footprint, weighting, transform and source identity are explicit |
| REGISTRATION_BOUNDARY | BLOCKED | Transform bound can cross the decision margin |
| EDGE_HOTSPOT | REJECTED | Critical local failure is hidden by a passing whole-area mean |
| UNKNOWN_REGION | BLOCKED | Critical unmeasured region cannot be called conforming |
| GEOMETRY_REVISION_MISMATCH | REJECTED | Measurement and exposure geometry revisions differ |
| TIME_SUPPORT_MISMATCH | BLOCKED | Observation and exposure intervals do not overlap |
| CORRELATED_SPATIAL_SUPPORT | REJECTED | Overlapping/common-cause records are falsely counted as independent |
| INFERRED_FIELD_AS_TRUTH | REJECTED | Inferred output is offered as physical truth |
| LOCALIZATION_OVERCLAIM | REJECTED | Specimen-level destructive strength used to claim local failure location |
| AGGREGATE_OPERATOR | BLOCKED | Local reconstruction requested without a validated measurement operator |
| DUPLICATE_SUPPORT | REJECTED | Repeated source/support identities inflate independent sample count |
| COMPLETE_SYNTHETIC | PASS | Structurally sound synthetic join; still not physical qualification |
| POINT_SUPPORT | PASS | A point-level claim is bounded by its finite calibrated footprint |
| FIELD_SUPPORT | PASS | Field claim binds declared scope, sampling plan, resolution and zero unmapped regions |
| SPECIMEN_SUPPORT | PASS | Integrated outcome remains at specimen/joint level |
| AREA_TO_SPECIMEN_NO_AGGREGATION | BLOCKED | Area evidence cannot be promoted to specimen support without a validated full-support aggregation operator |

PASS means only that the declared join contract is structurally admissible. It never means product acceptance, physical qualification or release.

## 9. Matched baselines and SOTA comparison

Compare all methods with the same source observations, geometry, measurement budget and group splits:

1. Whole-coupon / recipe-only summary.
2. Naïve pixel-label replication (adversarial baseline).
3. Nearest-neighbour registration with no uncertainty propagation (adversarial baseline).
4. Footprint-aware deterministic support join.
5. Support-aware join with transform uncertainty and abstention.
6. Explicit aggregate measurement-operator inference.
7. Spatial model with specimen/lot/day grouped holdout and common-cause covariance.
8. Expert/standard inspection policy using the same physical measurements.

The field is not beaten by producing more pixels. The meaningful comparison is whether the method improves calibrated spatial prediction and product decisions on held-out articles without false local claims or increased false acceptance.

### Benchmark metrics

Report separately:
- invalid support-claim detection;
- support-overlap accuracy where independent spatial truth exists;
- registration-bound coverage and error in physical units;
- local localization error only for methods with local ground truth;
- prediction-interval coverage by support-size and geometry stratum;
- false ACCEPT / false REJECT / abstention;
- worst-region and connected-unknown-region risk;
- split leakage and effective independent specimen count;
- sensitivity to transform-bound, support-kernel and covariance perturbations;
- acquisition cost, destructive-test count, time-to-decision and compute cost.

No single weighted scalar should obscure false acceptance, unknown-region coverage or pseudoreplication.

## 10. Prospective research targets

These are programme targets, not standards or achieved results:

| Metric | Proposed target |
|---|---:|
| Admitted observations with explicit support class and source identity | 100% |
| Specimen outcomes promoted directly to point/pixel truth | 0 |
| Inferred observations counted as physical measurements | 0 |
| Admitted joins retaining geometry/frame/transform lineage and uncertainty | 100% |
| Unresolved registration/support ambiguities silently accepted | 0 |
| False accepts on known-truth synthetic spatial negatives | 0 |
| Grouped split separation at declared physical independence unit | 100% |
| Critical unknown/occluded regions represented explicitly | 100% |
| Regret gap to exact oracle on eligible small synthetic support-assignment cases | ≤10% on ≥90% of cases |

Physical targets must be frozen only after actual instrument resolution, footprint, calibration error, article geometry and laboratory measurement repeatability are known.

## 11. S0 implementation

The implementation tranche is:
1. versioned support-aware benchmark and 18-case known-answer fixture;
2. Rust-native offline audit with PASS/BLOCKED/REJECTED dispositions;
3. mutation tests for support downgrade, geometry mismatch, time mismatch, leakage, duplicate support and inferred-as-physical claims;
4. deterministic replay of all fixture expectations;
5. a narrow adapter boundary back to #11925, #11847, #11795 and #11907.

No automated physical actuation, product release or certification is supported.

## 12. Physical validation protocol (future)

Before any laboratory run:
- freeze one substrate grade, manufacturer, material lot, conditioning and adhesive/primer lot;
- define article geometry, bonded-joint footprint and local coordinate frame;
- record transform calibration and independent registration check artifacts;
- declare each physical measurement's support and resolution before observing outcomes;
- randomize/block specimen locations and treatment order where feasible;
- record raw instrument outputs and destructive failure mode;
- reserve entire specimens/lots/days for blinded validation;
- do not derive local labels from integrated lap-shear results;
- use independent local measurements if local failure localization is a target;
- preregister missing-data, occlusion, outlier and abstention rules.

Standards references remain method-family pointers only. For rigid plastic lap shear, ASTM D3163-01(2023) is listed as active; for FRP lap shear, ASTM D5868-01(2023) is listed as active. Exact licensed editions and procedures must be selected by a qualified laboratory for the actual materials/joint. The current D3163 page cautions against treating single-lap results as design allowables for different structural joints without further analysis. No compliance claim is made.

References:
- [ASTM D3163-01(2023)](https://store.astm.org/standards/d3163)
- [ASTM D5868-01(2023)](https://store.astm.org/standards/d5868)
- [ASTM adhesive standards index](https://store.astm.org/products-services/standards-and-publications/standards/adhesive-standards.html)

## 13. Claim ceiling

A synthetic audit PASS validates only support, source identity, geometry/time lineage, group-separation and refusal semantics for the fixture. It does not prove physical registration accuracy, local adhesion, local failure location, virtual-metrology accuracy, bond reliability, manufacturing yield, cost savings, standards compliance, safety or actuation authority.

Physical promotion requires calibrated instruments, exact specimen and geometry lineage, validated support/resolution, preregistered matched experiments, blinded independent specimens, appropriate destructive ground truth and laboratory authorization.


## 14. Industry baseline and differentiation

The published commercial baseline already includes substantial process monitoring and inline integration. Luminous should not claim novelty for those capabilities:

| Vendor / published system | Publicly documented capability | Competitive interpretation |
|---|---|---|
| Plasmatreat Openair-Plasma / PCU | Inline integration; process-control unit and generator; process-parameter recording, deviation warnings and preventive-maintenance signals; voltage measurement at generator, jet or across jets; optional three-fold nozzle monitoring | Process telemetry, alarms, multi-nozzle voltage monitoring and inline integration are incumbent capabilities |
| Surfx Atomflo AFC-600 | RF power supply and auto-tuning; mass-flow control for plasma/process gases; real-time RF power, gas-flow and source-temperature control; recipe storage; out-of-spec alarms; Ethernet remote control and data logging | Recipe control, gas/power/temperature monitoring and digital trace logging are incumbent capabilities |
| Plasmatreat surface testing | Test inks for quick surface-energy screening; contact-angle measurement / measurement-system analysis for more precise, documentable wettability checks | Wetting and surface-energy screening are existing QA tools, not a sufficient Luminous differentiator |

Sources:
- [Plasmatreat Openair-Plasma systems and PCU](https://www.plasmatreat.ch/en/products-and-services): inline integration and process-control claims are described in the vendor's product overview.
- [Plasmatreat surface testing](https://www.plasmatreat.com/en/products-and-services/test-and-validation): test inks are described as quick screening; contact angle / MSA as more precise surface analysis.
- [Surfx Atomflo AFC-600 vendor datasheet](https://www.surfxtechnologies.com/wp-content/uploads/2023/08/AFC-600-Datasheet-1.4.pdf): controller capabilities and interfaces.

This is a review of publicly documented product capabilities, not a teardown or independent performance audit of proprietary competitor systems. We must not infer that a vendor lacks an undisclosed capability.

### What Luminous should compete on instead

| Dimension | Incumbent comparison protocol | Luminous differentiator to prove |
|---|---|---|
| Process-control conformance | Compare vendor alarms / recorded parameters at matched recipe and line speed | Source-bound evidence and explicit uncertainty, not another alarm dashboard |
| Delivered spatial exposure | Compare recipe-only and vendor monitoring against independently calibrated local witness maps | Geometry-, thickness-, edge- and motion-aware coverage with registration bounds |
| Product outcome | Same substrate lot, adhesive, geometry, cure and aging; include the best incumbent primer/preparation route | Predict the actual product endpoint, not only wettability or plasma health |
| Spatial claims | Check the actual support of each instrument and destructive test | No specimen-to-pixel leakage; explicit POINT / AREA / SPECIMEN / FIELD / INFERRED semantics |
| Transfer | Hold out whole lots, production days, device configurations and geometries | Calibrated risk and abstention on unseen material/geometry regimes |
| Economics | Same useful service unit, quality floor and full resource boundary | Cost per accepted durable bond, including gas, power, cooling, extraction, motion, inspection, scrap, rework and downtime |
| Evidence | Audit what the incumbent can export and what remains proprietary | Reproducible observation lineage, registration and decision receipts |

The commercial comparison must use the vendor's best applicable process plus its normal quality-control policy as the incumbent, not an untreated control alone. Match throughput, product acceptance threshold, inspection budget, material and adhesive profile, and lifecycle boundary. A brochure comparison cannot establish superiority.

### Product-level scorecard

At minimum, publish:
- first-pass yield and escaped-defect risk;
- bond-strength distribution and failure-mode distribution;
- retention after the predeclared storage / environmental aging schedule;
- worst-lot, worst-day and worst-geometry performance;
- spatial coverage and critical unknown-region area;
- false acceptance, false rejection and abstention;
- destructive inspection count and cost/time to decision;
- full-system energy and resource consumption per accepted durable bond;
- availability, downtime, maintenance burden and line throughput.

Process alarms and surface-wetting tests are useful baseline inputs. They do not substitute for the physical product outcomes.

## 15. Recommended physical comparison campaign

A first external-laboratory campaign should be intentionally small in scope but strong in design:

1. Freeze one substrate grade and lot family, one exact adhesive/joint geometry, and one realistic application acceptance floor.
2. Obtain and run the vendor-recommended treatment and its normal inspection policy as the primary incumbent.
3. Include matched untreated, plasma-only, best-applicable-primer-only and plasma-plus-promoter arms where the chemistry and application permit them. Do not assume a combined treatment is always admissible.
4. Measure local surface observables only at their true instrument footprint. Keep destructive lap-shear results at the exact bonded-joint/specimen support.
5. Record geometry, substrate thickness, edge distance, grounding, nozzle pose, speed/acceleration, gas, electrical waveform and temperature with synchronized calibration lineage.
6. Measure initial and delayed/aged mechanical outcomes, failure mode, and any substrate-function preservation endpoint that matters to the application (for example, barrier performance for packaging).
7. Randomize/block run order and specimen positions. Keep entire specimens out of training/holdout splits; reserve at least one material lot and one production day for blinded confirmation.
8. Conduct measurement-system analysis for contact angle, registration and any local imaging/metrology before fitting a spatial predictor.
9. Determine replicate counts from pilot variance, the minimum practically important effect, acceptable false-accept risk and power; do not select sample size by convenience.
10. Compare cost per accepted durable bond and risk-coverage at the same line speed and inspection budget.

The planned 0 h, 1 h, 4 h, 24 h, 72 h, 6 d and 18 d surface-aging observations are a hypothesis-driven schedule, not a universal aging law. Bond outcomes must be measured separately at the timepoints relevant to the actual production workflow. Extend to 30/60 d only where pretreated-part storage makes those horizons commercially relevant.

### Promotion gates

- **S0 software:** synthetic fixture, schema and deterministic audit only.
- **S1 metrology feasibility:** calibration, repeatability, registration, support footprints and data lineage pass an independent measurement-system review.
- **S2 physical mechanism study:** preregistered, randomized matched experiments with destructive outcomes and declared aging.
- **S3 held-out qualification study:** whole unseen specimens/lots/days/geometries, calibrated uncertainty, no increased escaped-defect risk.
- **S4 production pilot:** same product specification and throughput as incumbent, independent quality oversight, full lifecycle economics and rollback conditions.

Progression between stages is not automatic. A failure or unmeasured condition returns the relevant claim to BLOCKED or REJECTED.
