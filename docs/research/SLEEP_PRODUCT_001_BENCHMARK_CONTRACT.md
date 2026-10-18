# SLEEP-PRODUCT-001 — Open mattress-topper benchmark contract

**Status:** research contract / synthetic specification only  
**Tracking issue:** [#8039](https://github.com/Luminous-Dynamics/symthaea/issues/8039)  
**Version:** 0.1.0  
**Claim ceiling:** no physical product has been tested by this document.

## 1. Paradigm shift

The objective is not to generate a plausible-looking mattress or optimize a single firmness number. It is to build a traceable loop from human requirements to a manufactured product, evidence, repair, and the next design iteration.

The core shift is from:

- SKU selection by a vague "soft/medium/firm" label to individualized, testable trade-offs;
- a monolithic product to replaceable layers and covers;
- simulation-as-answer to simulation as a hypothesis generator calibrated against physical evidence;
- cheapest purchase price to delivered and lifecycle cost;
- average-only comfort claims to preregistered user evaluation with subgroup reporting;
- nominal recyclability to a demonstrated, geographically plausible recovery route;
- style as an afterthought to aesthetic preference as an explicit, separate design objective;
- proprietary, vendor-locked product records to reproducible open designs and verifiable revisions.

A candidate that is attractive or inexpensive cannot compensate for failing an essential safety, load, durability, or material-identification constraint. Style, comfort, cost, performance, and circularity are separate dimensions on a Pareto frontier; do not collapse them into an opaque weighted score.

## 2. First physical product: modular topper

Use a mattress topper / replaceable comfort-layer assembly as the initial low-tooling product hypothesis. It is smaller and easier to iterate than a full bed, while exercising textile, material, thermal, mechanical, assembly, and human-factors paths.

Explore only a small and explicitly recorded design space at first:

1. one reference construction selected before testing;
2. one simple replaceable-layer construction;
3. one variant that changes a single declared construction variable;
4. optional aesthetic variants that do not silently change the qualified functional bill of materials.

The candidate material set must use exact supplier/product/grade identity where available. Generic labels such as "latex", "recycled foam", "natural fiber", or "breathable fabric" are not sufficient evidence of composition or performance. Do not fabricate supplier data to fill gaps.

No sensor, application, network, cloud account, camera, or learned controller is required. A passive product is the default until a measurable need for electronics is demonstrated.

## 3. Ownership and system boundary

Compose existing Symthaea owners; do not add another materials engine, simulation engine, inventory ledger, lifecycle engine, or trust/evidence graph.

- [ENG-DESIGN-001 #6005](https://github.com/Luminous-Dynamics/symthaea/issues/6005): requirements-to-design digital thread and V&V.
- [ENG-DESIGN-003 #6091](https://github.com/Luminous-Dynamics/symthaea/issues/6091): aesthetics, sensory experience and symbolic design intent.
- [MFG-LIFE-000 #5705](https://github.com/Luminous-Dynamics/symthaea/issues/5705): lifecycle/circularity.
- [TEXTILE-HYG-MFG-000 #5984](https://github.com/Luminous-Dynamics/symthaea/issues/5984): fabrics, covers, seams, washing/reuse and textile product-function evidence.
- Use existing geometry, material, structural/load, thermal, manufacturing and Mycelix provenance owners through qualified interfaces.

This contract owns the consumer challenge and its benchmark protocol only. Any furniture and apparel follow-ons must reuse the same top-level evidence discipline while keeping their product-specific qualification with the relevant owner.

## 4. Evidence model

Every candidate and result must carry a stable ID and version. Separate at least these planes:

| Plane | Minimum content | Cannot establish by itself |
|---|---|---|
| Requirements | user group, use case, posture, climate, budget, constraints | physical feasibility |
| Product identity | design revision, dimensions, layer order, bill of materials, seams/joints, assembly instructions | material quality |
| Source claims | exact supplier, grade, document/date, units, declared uncertainty | independent verification |
| Simulation | model/version, boundary conditions, parameters, outputs, calibration domain | real-world performance |
| Instrumented test | specimen ID, instrument/calibration, method/version, conditioning, raw results, uncertainty, deviations | subjective comfort |
| Human evaluation | protocol, comparison order/randomization, recruitment/sampling, scale, subgroup data, consent/privacy | clinical or medical benefit |
| Lifecycle | repair steps/time, replacement parts, wash/aging/durability, mass by material, recovery route | actual recovery without evidence |
| Economics | quantity, currency/date, local labor, consumables, packaging, shipping, tax/duty where applicable, replacement assumptions | long-term affordability if lifetime remains unknown |

Missing required evidence yields `not_evaluable`, not a pass. A synthetic fixture is useful for testing evaluation logic but can never be relabelled as a measured product result. Preserve raw or safely shareable data and hash the exact source, protocol, and analysis artifacts.

## 5. Proposed benchmarks (targets, not results)

These are hypotheses for the initial engineering campaign. Freeze the final test protocol and the comparator before inspecting candidate results.

### Delivered cost and useful service

- Target at least 20% lower delivered cost than a named, functionally comparable reference where feasible.
- Include material, cutting/sewing, assembly labor, consumables, packaging, shipping, taxes/duties where relevant, and expected replacements.
- Report sensitivity to material price, yield/scrap, labor rate, production volume and freight.
- Report cost per year of useful service only when the service-life assumptions have evidence. Otherwise label it a scenario.

### Comfort and support

- Run a randomized/crossover human evaluation against named reference product(s); pre-register endpoint(s), scoring scale, sample, exclusion rules, and analysis before collecting results.
- Report the distribution and uncertainty, not only the mean, and inspect relevant body types and sleeping postures separately.
- Pair subjective ratings with instrumented support/pressure measurements, but never use pressure maps as a substitute for comfort.
- Predefine an adverse subgroup/posture stop rule. A good aggregate score cannot erase a meaningful harm or failure in a target subgroup.
- Make no diagnosis, therapeutic, or medical-outcome claims.

### Thermal and moisture behavior

- Measure surface temperature and humidity/moisture under a declared conditioning and load protocol.
- Where relevant, characterize textile or multilayer thermal and water-vapour resistance with a suitable apparatus and standard method.
- Distinguish coupon/material results, bench assembly results, and in-use human results. One cannot be silently substituted for another.

### Durability and maintainability

- Measure initial and post-cycle height/deformation, firmness/indentation response, cover/seam damage, and effects of cleaning/reassembly.
- Select the applicable mattress or textile test method with a qualified lab; record every adaptation or deviation. Do not imply conformity merely because a similar apparatus was used.
- Demonstrate replacement of a common wear component with common tools. Initial target: no more than 10 minutes for a trained operator following the published instructions.
- Publish the cost and availability of replacement parts and whether the product remains functional when a replaceable layer is unavailable.

### Circularity

- Report mass fractions by material family, joined-material interfaces, disassembly time/tool count, replaceable fraction, and region-specific recovery destinations.
- Credit material recovery only when a real route accepts the identified material at the relevant grade and contamination level. "Technically recyclable" is not the same as "locally recovered".
- Report repair, refurbishment, component reuse, material recycling and disposal as separate outcomes.

### Aesthetics and personalization

- Offer controlled visual/texture/cover variants to a defined evaluation group.
- Measure preference and willingness-to-live-with separately from support, durability and safety.
- Preserve a clear configuration identity: any style variant that changes material grade, layer orientation, seam/joinery, or other function-bearing detail is a new candidate requiring relevant requalification.

## 6. Standards and their boundaries

These references inform test selection; they do not certify a design.

- [ASTM F1566-14(2021)](https://store.astm.org/f1566-14r21.html) includes comparative mattress firmness, firmness retention, durability, impact and height-change methods. The standard notes that the methods provide comparative data rather than general pass/fail criteria.
- [EN 1957:2012](https://www.nen.nl/en/nen-en-1957-2012-en-177099) covers mattress/bed durability and hardness methods. Its firmness rating is not proof of comfort or quality; applicability to a topper must be checked for the actual assembly.
- [ISO 11092:2026](https://www.iso.org/standard/85998.html) measures thermal and water-vapour resistance under steady-state conditions for relevant textiles and other materials. The conditions do not themselves specify real-world physiological comfort.
- [ISO 12947-2:2016](https://www.iso.org/standard/61058.html) gives a Martindale method for textile specimen breakdown.
- [ISO 13934-1:2013](https://www.iso.org/standard/60676.html) gives a strip method for woven-fabric tensile force and elongation.
- [ISO 20887:2020](https://www.iso.org/standard/69370.html) provides design-for-disassembly/adaptability principles for construction works; use it as an inspiration for reversible interfaces, not as a claim of mattress conformity.

Confirm the current applicable edition, intended use and jurisdictional rules before making a standards-conformity or safety claim.

## 7. Adversarial negative controls

The benchmark must include cases that try to make a weak design look successful:

1. High simulated support score with absent or failing physical validation → no verified physical-performance pass.
2. Strong aggregate comfort score but a materially worse result for a declared subgroup/posture → flag / fail the preregistered subgroup gate.
3. Low ex-factory cost with freight, labor or replacement costs missing → cost outcome is not evaluable.
4. Material described only by a generic family name or an undated supplier claim → material-specific performance claim is blocked.
5. Nominally recyclable mixed layers without an identified recovery route → do not pass the demonstrated-circularity gate.
6. A cover/style change silently modifies construction or component identity → create a new candidate and rerun affected tests.
7. Missing specimen, calibration, uncertainty or test-deviation record → no verified instrumented-test pass.
8. A synthetic fixture is relabelled as measured → evidence-integrity failure.
9. A firmness label is used as evidence of greater comfort → reject the inference.
10. An attractive render is used as evidence of structural stability or manufacturability → reject the inference.

## 8. Execution sequence

1. Freeze the candidate manifest, named comparator(s), requirements, protocol and proposed endpoints.
2. Complete costed CAD/pattern and bill-of-material variants using traceable inputs.
3. Use simulation only to rank hypotheses and declare where the model is valid.
4. Build the smallest useful prototypes and collect instrumented coupon/assembly measurements.
5. Run durability, cleaning and repair tests; update uncertainty and cost records.
6. Conduct the preregistered, consented human comfort/preference study only after safety review and a defensible physical prototype are available.
7. Publish limitations, negative results, and an exact, hashed evidence package. Independent replication is a separate evidence stage.

### Follow-on product families

After the topper benchmark has a usable, audited protocol, adapt the same structure to:

- **Furniture:** an affordable flat-pack desk/storage unit with structural stability, repeated assembly, repair time, material waste, visual preference, and delivered-cost benchmarks. [ISO 7170:2021](https://www.iso.org/standard/76864.html) is a relevant storage-unit test-method reference; it does not itself define universal acceptance criteria.
- **Everyday clothing/workwear:** a repairable garment with fit/mobility, seam and tensile performance, abrasion, laundering, thermal/moisture behavior, material consumption, repair cost, and wearer preference.

Keep the first release small. The paradigm shift is a reproducible process that can improve many products—not a sprawling product catalog with unverified claims.
