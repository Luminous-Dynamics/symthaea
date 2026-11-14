# PLASMA-CLAIM-TYPE-001 — method-scoped engineering claims

**Date:** 2026-11-14  
**Status:** S0 structural audit and synthetic known-answer fixture  
**Issue:** [#12026](https://github.com/Luminous-Dynamics/symthaea/issues/12026)

## Executive thesis

A measurement does not support every claim that can be written beside it.

The same numerical strength value can mean different things under different adherends, geometries, loading modes, environments and test methods. A contact angle is not a bond strength. An apparent lap-shear value is not automatically a structural design allowable. A qualitative wedge durability observation is not a quantitative fracture toughness. A short-term coupon result is not a service-life prediction.

The paradigm shift is to treat engineering claims as **typed, scope-bounded objects**. A claim must identify its exact observation type, method edition, material pair, geometry, protocol and underlying specimen/data references. Stronger claims require explicit additional evidence bridges; they cannot be inferred from a method name, a scalar metric or an unqualified model score.

This audit is intended as a claim-scope layer in the Bond Qualification Passport. It complements the spatial-support, durability and qualification-transfer workstreams without replacing them.

## Why this matters now

### ASTM D1002-26 — current metal-to-metal comparative scope

ASTM lists D1002-26 as active and last updated **2026-11-11**. Its scope is apparent shear strength for adhesively bonded metal-to-metal single-lap specimens under specified preparation and test conditions. The published significance/use text calls the method primarily comparative and warns that misuse of its values as structural design-allowable stress can lead to product failure, property damage and human injury. It also cautions that values from small single-lap coupons can differ from dissimilar adherends or a different bonding process.

Source: https://store.astm.org/d1002-26.html

### ISO 4587:2003 — lap-shear measurement, not design information

ISO describes ISO 4587:2003 as a tensile lap-shear method for rigid-to-rigid bonded assemblies using a standard specimen and specified preparation/testing conditions. Its public abstract explicitly says the procedure does not provide design information. ISO lists the 2003 edition as current after its 2021 review.

Source: https://www.iso.org/standard/34852.html

### ASTM D3762-24 — qualitative wedge durability, not fracture strength

ASTM describes D3762-24 as a primarily qualitative wedge test that discriminates adherend surface-preparation and environmental-durability differences and reports crack growth and failure mode. Its default application is aluminium-to-aluminium; other metals/plastics require consideration of adherend thickness and rigidity. The standard explicitly says it is not a quantitative fracture-strength method and points to D3433 for fracture strength.

Source: https://store.astm.org/standards/d3762

### Plasma-specific consequence

Péczi-Kovács and Kókai (published 2026-10-20) compared atmospheric argon and nitrogen plasma treatment on PP, HIPS, PVC, PC and PET/PET-G adhesive systems. The paper reports that nitrogen generally generated lower water contact angles, while argon delivered comparable or higher bond strength for several polymer–adhesive combinations. The study's result is material/adhesive-specific; it is not a universal gas ranking. Its lap-shear data were collected using three specimens per treatment category, a useful source-reconstruction anchor but not sufficient by itself for a broad manufacturing reliability claim.

DOI: https://doi.org/10.1007/s44245-026-00396-4

## What the audit does

The S0 registry currently maps four observation families:

| Method ID | Pinned edition | Material pair | Geometry | Endpoint | Narrow supported claim |
|---|---|---|---|---|---|
| ISO_4587 | 2003 | rigid-to-rigid | single lap | tensile lap-shear strength | method observation |
| ASTM_D1002 | 2026 | metal-to-metal | single lap | apparent shear strength | method observation or cautious comparative screening |
| ASTM_D3762 | 2024 | aluminium-to-aluminium | wedge | qualitative wedge durability | method observation or qualitative surface-durability screen |
| WATER_CONTACT_ANGLE_PROTOCOL | protocol-defined-v1 | material surface | sessile drop | water contact angle | wettability observation |

The method IDs, edition pins, endpoint mapping, geometry and claim ceilings are explicit in `xtask/src/plasma_claim_scope.rs`. The fixture contains only synthetic examples; it is not a source of normative standards requirements. The source registry links to official standard records and does not reproduce licensed standards text.

### Dispositions

- **SUPPORTED** means that the requested *measurement claim* fits the encoded method/edition/material/geometry/endpoint map and required structural provenance fields are present. It does not mean the experiment has been independently witnessed, the standard was correctly executed, the product is qualified, or the result is suitable for design.
- **BLOCKED** means the method/edition is unknown, source/protocol references are missing or malformed, specimen identity is incomplete, or material suitability needs an external controlled review that this audit cannot authenticate.
- **REJECTED** means a known method/material/geometry/endpoint mismatch or a claim escalation exceeds the encoded method scope.

The digest validator checks lowercase 64-character hexadecimal syntax only. It does not fetch, authenticate or recompute the referenced raw artifact. Artifact existence, authenticity, signer authority, instrument calibration and laboratory competence remain independent obligations. This limitation is intentional and explicit.

An extra boolean such as `standard_compliant=true`, `design_qualified=true`, `equivalent=true` or `physically_validated=true` cannot authorize a result. Unknown fields are rejected by the fixture deserializer.

## Frozen challenge fixture

Run:

```bash
cargo run -p xtask -- plasma-claim-scope-audit
```

The committed v0.1 fixture contains **19 synthetic known-answer cases**:

| Expected disposition | Cases | Challenge families |
|---|---:|---|
| SUPPORTED | 4 | exact in-scope lap-shear, cautious metal-to-metal comparison, aluminium wedge durability, wettability-only |
| BLOCKED | 7 | non-aluminium D3762 suitability, unrecognized edition, missing protocol/data digest, duplicate specimen IDs, unknown method, malformed digest |
| REJECTED | 8 | design allowable, service life, polymer use of D1002, fracture-strength promotion from D3762, contact-angle-to-bond-strength promotion, endpoint and geometry mismatch |

The audit output prints a narrow claim ceiling for each SUPPORTED case. All counts are expected fixture dispositions, not observed physical-performance results.

## Industry comparison: what is already strong

Plasmatreat already documents optical monitoring of the plasma stream, travel/rotation and effective-distance monitoring, media-flow control, and PCU monitoring of plasma generation, gas flow, power, pressure and plasma stability. Its own materials also acknowledge that surface activation is often not directly visible inline and that downstream gluing, painting, bonding or coating reveals the actual result.

Sources:
- https://www.plasmatreat.com/en/products-and-services/openair-plasmar-process-monitoring
- https://www.plasmatreat.com/en/products-and-services/process-control

Surfx describes Atomflo-INT as an integration-ready atmospheric plasma system with real-time monitoring, process control, Ethernet/IP integration, automated logging and traceability.

Source:
- https://www.surfxtechnologies.com/products/atomflo-int/

These vendor capabilities are real competitive baselines. A generic dashboard, a second process logger, or a claim that Luminous invented plasma telemetry would be weak differentiation.

The opportunity is a vendor-neutral **downstream claim and qualification layer**: it should be able to say what the test method actually measured, what claim is admitted, what evidence is missing, and what stronger test or independent bridge is required. It should consume vendor telemetry and lab records rather than replace the generator/PCU.

## Paradigm shift: a typed evidence-to-claim graph

The long-term Passport should model evidence and claims as a directed, typed graph:

1. **Raw evidence:** calibrated instrument readings, specimen identity, process telemetry, images, source artifacts and laboratory records.
2. **Method-defined observation:** a result under a named edition, geometry, material pair, protocol and test conditions.
3. **Product endpoint:** a predeclared acceptance property, aged retention, failure mode, fatigue measure or functional outcome.
4. **Engineering inference:** equivalence, reliability, lifetime, mechanism transfer or design claim, each with its own statistical/physical validation requirements.
5. **Decision:** qualified, bridge required, blocked, rejected, suspended or revoked under the governing quality authority.

No upward edge is implicit. Each edge needs its own stated assumptions, evidence, uncertainty, independent holdout and claim ceiling.

Examples of prohibited implicit promotions:
- water contact angle → adhesive bond strength;
- one single-lap coupon value → structural allowable;
- short-term initial lap-shear → service-life estimate;
- a model score or a non-significant p-value → physical equivalence;
- a stable plasma process signal → accepted downstream product;
- an aluminium wedge result → a polymer/composite result without method-suitability review.

## Competitive benchmark programme

The following are prospective engineering targets, not current results or industry standards:

| Metric | Target |
|---|---:|
| Detection of planted unsupported claim promotions | 100% |
| False SUPPORTED outcomes on known hard negatives | 0 |
| Acceptance of an unrecognized edition as current | 0 |
| Detection of missing/duplicate specimen identities and missing artifact references | 100% |
| SUPPORTED outcomes with explicit narrow claim ceiling | 100% |
| Unsupported design-allowable, service-life or certification promotion | 0 |

Compare at equal fixture coverage and data:
1. untyped spreadsheet/manual review;
2. method-name-only rules;
3. exact-edition rules;
4. the proposed edition + material + geometry + endpoint + typed-claim audit.

Report false-support rate, BLOCKED rate, reason coverage, review time and unsupported-promotion escape rate separately. A high BLOCKED rate on unknown methods can be appropriate; do not turn abstention into a failed prediction metric.

To claim superiority over industrial quality systems, a future blinded evaluation must use real test records, independent reviewers, controlled method editions and intentionally planted overclaims. Synthetic fixtures demonstrate contract behaviour only.

## Next physical and standards campaign

1. Obtain the controlled copies of the standards applicable to the chosen customer/product; the public metadata is not a substitute for the normative text.
2. Select one application-specific test ladder: for example, ISO 4587 comparative screening plus a separately justified durability/aging method and a geometry-relevant fatigue/fracture method where required.
3. Preregister endpoints, method editions, adherend geometry, adhesive thickness/cure, conditioning, specimen counts, invalid-test rules and failure-mode taxonomy.
4. Preserve every specimen, including invalids and substrate failures; do not report only peak or passing coupons.
5. Use whole-lot and whole-day holdouts for generalization. Random coupon splits from one batch are not enough.
6. Require an independent materials/test authority to review any bridge from coupon results to product acceptance, design or service claims.

## Validation and claim ceiling

- Rust audit and unit tests are authored but have not been compiled or executed.
- Fixture replay is synthetic and does not execute ISO/ASTM test procedures.
- Hash fields are checked for syntax; bytes and signatures are not authenticated.
- No instrument calibration, laboratory accreditation, standard compliance, material equivalence, bond strength, fatigue life, service lifetime, product qualification or design allowable is claimed.
- No plasma control, equipment actuation or product-release authority is introduced.

The intended result is a more trustworthy Bond Qualification Passport: every claim should carry the exact evidence that supports it and a precise statement of what it does **not** establish.


## New SOTA update — November 2026: evaluation must respect physical clusters

A methodological review published **3 November 2026** directly addresses the small, hierarchical datasets typical of manufacturing experiments:

Smakova et al., *Machine Learning on Small Experimental Datasets in Manufacturing: A Review of Data Leakage, Grouped Validation, and Uncertainty Quantification*, Information 17(11), 1095. DOI: https://doi.org/10.3390/info17111095

This is not a plasma-joining benchmark, but it is directly relevant methodology. Its reproducible simulation shows why random row-wise splits are invalid when measurements share physical specimens. At ICC 0.74, row-wise CV estimated RMSE 0.458 against true new-specimen error 0.727; at ICC 0.92, it estimated 0.558 against true error 1.258. Grouped CV remained close to the true error in those simulations. The paper also warns that tuning and selecting models on the same folds produces optimistic estimates.

The review further makes an important small-sample point about conformal uncertainty:
- for 90% split-conformal intervals, at least 9 independent calibration clusters are needed for a finite distribution-free interval;
- for 95%, at least 19 independent calibration clusters are needed;
- these counts refer to independent calibration clusters (such as specimens or lots), not rows or repeated measurements;
- simultaneous coverage of every measurement within a new cluster can require wider intervals than marginal row-level coverage.

These are mathematical sample-size limits for the stated split-conformal construction, not universal sample-size recommendations. With few clusters, use model-based intervals only with explicit assumptions or consider cluster-level CV+/jackknife+ alternatives; do not report nominal coverage that the design cannot support.

Source: https://www.mdpi.com/2078-2489/17/11/1095

### Consequence for the Luminous benchmark

The current prospective 85–95% empirical coverage target must be evaluated using:
1. a declared independent unit (specimen, material lot, adhesive batch, plasma head, production day, line or site);
2. grouped/nested grouped validation for every data-driven model-selection path;
3. completely untouched whole-group test sets for the final transfer claim;
4. cluster-level calibration and cluster-level resampling;
5. interval width and decision-risk reporting alongside coverage;
6. a reported number of independent groups and an explicit statement when the dataset is too small for the requested nominal coverage.

A random coupon split is not an acceptable SOTA comparison for a transfer-to-new-lot or transfer-to-new-line use case.

The review's recommended baseline families also matter: regularized/polynomial response-surface regression, mixed-effects models, Gaussian processes, tree ensembles/boosting, and physics-informed/hybrid models. Tabular foundation models such as TabPFN are worth testing as emerging comparators, but the review notes limited evidence for grouped experimental designs. Deep/tabular models must earn their place on untouched groups, not on row-wise accuracy.

### Aged CFRP bonds: do not assume monotonic degradation

Vigón et al., *Hygrothermal Degradation Effects on Mode I Fracture in Adhesively Bonded Carbon–Epoxy Composites*, Applied Sciences 16(21), 10810, published **31 October 2026**, DOI: https://doi.org/10.3390/app162110810.

This study exposes why a single monotonic aging curve is unsafe. Carbon/epoxy adhesive joints were aged at 60 °C / 70% RH for 1, 2, 4 and 12 weeks. Mode I fracture toughness decreased by about 12% at 2 weeks, then reached about 19% above the unaged value at 12 weeks; the 12-week difference was statistically significant, while the 2-week reduction was not under the paper's analysis. Fatigue-initiation resistance was lowest after one week and improved at longer exposure, while aged crack-growth rates remained higher than unaged values.

The fatigue program used 15 specimens per ageing condition and included run-outs as right-censored observations in Weibull regression. The authors also caution that simultaneous estimation of five Weibull parameters with 15 specimens per condition is sensitive to individual results. The paper is a useful methodological comparator—not evidence about Luminous plasma-treated joints or a general law of aging.

Source: https://www.mdpi.com/2076-3417/16/21/10810

### Reproduction caution from the five-polymer plasma study

The October 2026 PP/HIPS/PVC/PC/PET-G argon/nitrogen paper is a valuable anchor because it measures both wettability and joint performance. However, its design also leaves precise replication questions:
- three lap-shear specimens were tested per treatment category;
- the exact bondline thickness was not measured;
- substrate temperature was not measured directly;
- contact angle and adhesive bonding were performed immediately after treatment to limit recovery;
- visually degraded specimens were excluded from subsequent bonding tests.

These choices are reasonable for a comparative laboratory study, but they mean its reported means should not be treated as a production reliability distribution. A Luminous reproduction should explicitly measure or bound bondline thickness and substrate temperature, preserve excluded/damaged specimens in the dataset with their true disposition, and include treatment-to-bond delay cohorts rather than only immediate bonding.

Source: https://link.springer.com/article/10.1007/s44245-026-00396-4

## Refined benchmark scorecard — no leaderboard metric without a valid split

Add the following to the previously defined claim-scope and transfer metrics:

| Benchmark | Prospective target / reporting rule |
|---|---|
| Group leakage | 0 specimen/lot/head/line overlap between train, calibration and final test partitions for the declared transfer task |
| Nested selection | 100% of preprocessing, feature selection, hyperparameter/model-family choice occurs inside grouped training folds |
| Independent units | Report both row count and independent cluster count for every endpoint |
| Interval validity | Report empirical coverage, interval width and risk–coverage by whole-cluster holdout; report nominal level only when calibration count supports it |
| Cluster calibration | At least 9 independent calibration clusters for 90% split-conformal; at least 19 for 95%; otherwise declare the interval method/coverage limitation |
| Model comparison | Paired out-of-fold per-cluster losses, multiplicity correction, uncertainty on ranking; publish inconclusive results |
| Aging mechanism | Separate static fracture, fatigue initiation, crack growth, sustained-load failure and service reliability; no forced monotonic trend |
| Run-outs | Treat as right-censored with a method appropriate to the endpoint; never encode as failure at the test stop time |
| Physical replication | Record bondline thickness, actual substrate temperature, treatment-to-bond delay and all degraded/invalid specimens |

We should not promise “85–95% coverage” as a universal target from 10 coupons. The target is meaningful only when the independent-group count, calibration method, holdout design and interval width support it.
