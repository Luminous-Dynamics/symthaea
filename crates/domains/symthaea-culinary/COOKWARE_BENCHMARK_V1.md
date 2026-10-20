# Cookware Benchmark V1 — Matched Performance, Evidence, and Craft

**Status:** proposed benchmark contract  
**Schema:** `CookwareBenchmarkManifestV1`  
**Implementation:** `src/cookware.rs`  
**Pilot:** one sauté pan / skillet, compared against established references  
**Important:** passing manifest validation is not product qualification, a food-contact certification, or evidence of superiority.

## 1. Design objective

Evaluate whether a cookware candidate improves a declared cooking task without hiding trade-offs in a single score. This benchmark extends conventional performance comparison with matched test blocks, traceable observations, independent cookware-unit identities, and a separate measure of the practitioner's experience.

The product principle is **augment craft, do not deskill it**. The chef remains the author of the dish and retains meaningful control. Instrumentation and optional assistance must not become compulsory telemetry or a prerequisite for using the tool.

## 2. Industry baseline

Consumer Reports' publicly described frying-pan evaluation includes cooking evenness, food release, nonstick durability, heating speed, simmer performance, handle temperature/sturdiness, cleaning, and sauté performance. Its March 2026 durability description includes accelerated abrasion followed by food-release evaluation. See:

- https://www.consumerreports.org/home-garden/cookware/best-frying-pans-of-the-year-a7222946160/
- https://www.consumerreports.org/home-garden/cookware/least-durable-nonstick-frying-pans-what-to-buy-instead-a4466494459/

ASTM F1786-97(2021) evaluates energy use and cooking performance for commercial gas/electric braising pans, with optional surface-temperature-uniformity measurements. This is a useful methodological analogue for energy and surface measurements, **not** a directly applicable skillet compliance claim:

- https://store.astm.org/f1786-97r21.html

Food-contact claims are separate from thermal performance. The FDA describes food-contact authorization in terms of the specific substance/article, manufacturer, intended conditions of use, and applicable evidence; a broad material label or a thermal test does not establish suitability for all uses:

- https://www.fda.gov/food/food-ingredients-packaging/food-chemical-safety
- https://www.fda.gov/food/packaging-food-contact-substances-fcs/determining-regulatory-status-components-food-contact-material

## 3. Test profiles and matched comparisons

Freeze a versioned protocol before collecting results. Candidate runs in a comparison must share:

- task and pan geometry profiles;
- heat-source/power profile;
- load type, mass, and initial condition;
- ambient/environment profile;
- protocol version and jurisdiction where relevant;
- required metrics and calibration/method references.

Use repeated, randomized blocks so each candidate appears once per block. Record run order and the declared randomization method. A missing run or mismatched context invalidates the comparison rather than being silently filled in.

The protocol's minimum block and independent-unit counts are project choices to be preregistered for each study. V1's default fixture uses two blocks and two physical units per candidate only to exercise the contract; this is **not** asserted to be statistically sufficient for a real product claim. Determine final sample size from the experimental design and expected variability.

## 4. Measurement vector

Do not aggregate these into an opaque "best pan" score before presenting the measurements, uncertainty, and trade-offs separately.

| Metric | Unit in V1 | Notes |
|---|---|---|
| Surface temperature spread | millidegrees Celsius | Define sensor grid, stabilization window, and heat-source conditions |
| Heat-up energy | millijoules | Define start/end states and metering method |
| Recovery time after defined load | milliseconds | Define initial surface state and recovery threshold |
| Maximum handle temperature | millidegrees Celsius | Define measurement location, timing, and ambient conditions |
| Peak food-release force | millinewtons | Only use with a repeatable food/load, procedure, and force measurement |
| Cleanup time | milliseconds | Define residue, wash method, operator, and completion rule |
| Food mass loss | milligrams | Define food mass, cooking endpoint, and collection procedure |
| Pan warp | micrometres | Define reference geometry, temperature state, and measurement fixture |
| Energy per edible serving | millijoules | Define edible yield, serving basis, and included energy boundary |

These metrics are candidate measurements, not a claim that every skillet needs every metric. The intended task profile should select the required subset. Add metrics only with a defined method, unit, outcome meaning, and evidence requirements.

## 5. Evidence contract

Each required observation has a stable evidence reference, measurement-method reference, evidence class, and (for instrument observations) a calibration reference. Distinguish:

- observed instrumental measurement;
- observed practitioner judgment;
- simulation output;
- manufacturer-declared specification;
- synthetic fixture.

For empirical comparisons, V1 requires primary metrics to be instrument-observed and rejects simulation/manufacturer/synthetic values as substitutes. A populated reference identifies a record but does not authenticate the source; evidence review and exact artifact verification remain necessary.

The validator checks internal consistency only. It does not decide whether a measurement instrument is accurate or whether a referenced laboratory dossier is valid.

## 6. Food-contact claim ceiling

If a report claims food-contact suitability, every candidate must reference evidence qualified for the exact declared use and jurisdiction. An unresolved dossier is a blocker to that claim—not a low score that can be compensated by better heat-up time.

V1 validates that the reference and scope are declared. It does not replace competent laboratory testing, toxicological assessment, applicable regulatory review, or the manufacturer's legal obligations.

## 7. Craft experience is a separate outcome

Collect practitioner feedback only under an explicit consent protocol, with a declared survey/instrument version and opaque pseudonymous identifiers. Avoid names, contact details, audio/video, and unrelated personal information in this benchmark schema.

Keep dimensions separate:

- perceived control;
- tactile feedback;
- creative agency;
- learning support;
- enjoyment;
- physical strain.

V1 encodes individual 1–7 responses but does not combine them into a universal craft score. Studies should preregister the target practitioner population, sampling approach, order/randomization, primary outcomes, analysis and non-inferiority margins. Compare manual practice with optional assistance where relevant. Faster preparation is not an overall win if meaningful agency, safety, learning or creative freedom materially deteriorates.

## 8. Initial comparison design

Use at least three reference classes suited to the intended task, where relevant: bonded conductive-core stainless, carbon steel/cast iron, and nonstick. Do not assume one construction wins all cooking tasks, and do not compare unlike pan sizes or loads without a justified normalization strategy.

Randomize run order; repeat across independent physical units; report per-run and aggregate results with uncertainty. Keep laboratory measurements, manufacturer claims, simulation, and practitioner judgments visibly distinct. Any proposed composite ranking must publish its objectives, directions, weights, missing-data handling, and sensitivity to plausible changes.

## 9. Proposed acceptance targets

These are internal engineering targets, not external standards or achieved results:

1. Zero known manifest-integrity violations in the frozen adversarial corpus.
2. Every empirical primary metric has a method, evidence reference, and calibration reference.
3. Every comparison block is complete and context-matched; no physical unit is assigned to two different design candidates.
4. No food-contact suitability claim can pass with unresolved or out-of-scope qualification evidence.
5. Publish the raw metric vector, unit, uncertainty, sample counts, protocol identity, failures, and limitations.
6. Demonstrate improvement on at least one preregistered thermal/usability outcome without violating other declared hard requirements.
7. Establish a practitioner-defined non-inferiority margin for perceived control/creative agency before the craft study; do not claim non-inferiority without enough observed data.
8. Do not claim SOTA from synthetic fixtures, a single pan, one kitchen, or a single narrow task.

## 10. Next steps

- Run this schema's tests and formatting checks on an exact, identified Git head.
- Review the protocol with chefs, cookware engineers, metrology specialists, ergonomics experts, and food-contact compliance professionals.
- Freeze a small, repeatable measurement fixture and compare candidate pans against established references.
- Carry measured results into Symthaea's robust multi-objective evaluation; use Symtropy's existing manufacturing/repairability and planning systems where appropriate.
- Publish failures and cases where a conventional, simpler pan remains the better choice.

The success criterion is better tools and better cooking while preserving human authorship—not replacing chefs or turning craft into compulsory automation.
