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
