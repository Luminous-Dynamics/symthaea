# Independent outcome evidence and stop-rule benchmark

Status: exploratory research contract, 3 November 2026. This document does not establish appliance safety, cleaning efficacy, regulatory compliance, sanitation or a product claim.

## The paradigm shift

Move from "the cycle finished" to **the process has satisfied the required, separately defined endpoints within a qualified scope**.

The controller must distinguish:
1. soil removal from tableware;
2. final-rinse residue / chemical carryover;
3. cross-contamination between items;
4. residual moisture / drying;
5. relative microbiological reduction;
6. sanitization / disinfection claims under their own standards and certification.

Turbidity, conductivity, temperature, camera confidence, pump current and commanded detergent mass are process observations. None is automatically a ground-truth measurement of soil removal. A validated process proxy can support routine assurance only inside a separately qualified and monitored envelope. It must not be re-labelled as an independent outcome assay.

A stop rule may avoid unnecessary extra washing or rinsing, but only when a validated model establishes that stopping does not violate the declared cleaning, rinse, material and (where claimed) microbiological endpoint. Otherwise continue the conservative qualified process or reject the clean claim.

## 2026 sensing research and its limits

### Fluorescence imaging for soil residue

Ma et al., Food Research International 232 (2026), Article 118853, developed 365-nm LED-induced fluorescence imaging for oat-porridge residue on rice bowls. The study used excitation-emission analysis, chromatic adaptation, a YOLOv11-based model and surface-area correction. It reports 86.7% cleaning-score classification accuracy after correction. The per-class accuracies reported for scores 5, 4, 3, 2, 1 and 0 were 100%, 100%, 85%, 65%, 70% and 100%, respectively.

This is promising evidence for more objective cleaning assessment than unaided visual inspection, but it is not a generally validated "clean dish" detector:
- It targets one porridge/oatmeal soil and bowl substrate.
- Intermediate score classification is materially weaker than the extremes.
- It does not establish transfer to oils, proteins, beverages, mixed residues, glass, stainless steel or polymers.
- It evaluates a research imaging station; it does not prove that an in-appliance camera can observe all hidden surfaces or recesses.
- Its score classifier must not be converted into a health or sanitization claim.

Source: https://doi.org/10.1016/j.foodres.2026.118853
PubMed: https://pubmed.ncbi.nlm.nih.gov/41895937/

### Process proxies and rinse chemistry

Conductivity can provide useful evidence for removal of ionic constituents, but does not automatically quantify non-ionic surfactants, enzymes, proteins or all detergent residues. Turbidity of recirculated water can indicate suspended soil, but low turbidity does not prove that soil has left every dish surface; soil can remain attached, settle, adhere to the filter or be hidden in geometry.

Use paired references:
- surface soil residue assay / standardized witness coupon for soil removal;
- final-rinse conductivity plus a formulation-appropriate assay (e.g. TOC or specific analytical marker) for carryover;
- clean witness surfaces placed with soiled items for cross-transfer;
- gravimetric or material-specific residual-moisture measurement for drying.

The chosen method and acceptance threshold must be fixed before the test series and independently reviewed. A SHA-256 field records identity only; it does not authenticate the assay or justify the threshold.

### Microbiology and sanitization are not synonyms

**IEC TS 63331:2024** specifies reproducible methods for measuring relative reduction of microbial contamination on household dishwasher items. The IEC expressly says results support relative statements only and that the document does not address sanitization, disinfection or sterilization.
- Official IEC publication: https://webstore.iec.ch/en/publication/66647

**NSF/ANSI 184-2024** is a separate residential-dishwasher standard. NSF describes certification as demonstrating at least a 5-log (99.999%) bacterial reduction on the sanitizing cycle, with a final-rinse temperature of 150 °F (65.6 °C) as part of its stated requirements. A temperature observation alone is not an NSF certificate; any claim requires the applicable complete standard, prescribed microbiological testing and certification scope.
- NSF dishwasher certification explanation: https://www.nsf.org/consumer-resources/articles/dishwasher-certification
- 2024 standard listing: https://webstore.ansi.org/standards/nsf/NSFANSI1842024

Do not map relative microbiological evidence to sanitization. The quality module therefore names its optional microbiology endpoint RelativeMicrobialReduction.

## Rust implementation

The design crate exports quality::qualify_pilot_evidence.

A QualificationScope fixes:
- exact machine model and programme;
- soil class and surface;
- item-set ID and every item ID whose result is required;
- detergent batch and water batch / hardness;
- inlet and ambient conditions;
- protocol identity and raw archive digest.

A caller supplies EndpointCriterion records for the required endpoints. The criteria should come from the frozen test protocol or applicable licensed standard, not be selected after observing results. The software checks criterion identity and digest shape; it does not authenticate the source or decide whether the numerical limit is legally/technically appropriate.

Every OutcomeObservation is one item × endpoint × physical run. It includes:
- run and item IDs;
- endpoint and evidence class;
- exact assay method / protocol;
- measured value, expanded uncertainty and unit;
- immutable evidence reference and calibration-bundle digest.

The computed guard-band rule is:
- AtMost: pass only if measured value + expanded uncertainty <= limit;
- AtLeast: pass only if measured value - expanded uncertainty >= limit;
- fail when the full uncertainty interval is beyond the unacceptable side;
- indeterminate when the interval overlaps the limit.

An indeterminate result cannot qualify. Unit, protocol or method changes cannot silently mix in one qualification. Process proxies are rejected as substitutes for independent outcome assays.

The default required endpoints are soil residual, final-rinse residue and cross-contamination. Drying and relative microbiological reduction must be explicitly requested, with a separate acceptance criterion. A five-run minimum is a pilot floor only. The module does not calculate statistical power, confidence intervals for a population, or regulatory certification. Criterion IDs, metric IDs and hashes bind records to their declared fields, but the software does not authenticate the criterion issuer or decide whether a limit is scientifically/regulatorily appropriate. Acceptance limits must be preregistered, reviewed and sourced from the applicable test protocol or independent method.

### Scope of the output

A PilotEvidenceEnvelope is:
- limited to the exact scope, criterion IDs, methods, batches and protocol tested;
- not evidence of performance on a different soil, SKU, detergent batch, water hardness, material, programme or geometry;
- not a certificate or generalization to all household use;
- not proof that evidence references or hashes are authentic;
- not proof of sanitation.

It may be used as input to the next controlled scheduler/process-envelope stage. It must not be conflated with a production release permit.

### Scheduler adapter boundary

The scheduler exposes an evidence-bound research adapter: `EvidenceBoundMode::from_pilot_evidence` consumes the pilot envelopes and derives the supported soil mask. For each advertised soil it requires independent outcome evidence at every occupancy from 1 through the mode's declared capacity, all from the same machine, programme, material, chemistry batch, water batch/hardness and operating conditions. `simulate_with_pilot_evidence` uses these derived mode profiles. It cannot authorize sanitization claims and does not derive energy/water coefficients; use measured occupancy curves separately.

The older `simulate` entry point remains for synthetic/offline policy comparison. Its cleaning/rinse boolean is a caller assumption only, not evidence. Neither simulation API certifies a product or authorizes production release.

## Data templates

- outcome-evidence-template.csv: one item × endpoint × run measurement.
- endpoint-criteria-template.csv: preregistered endpoint, method/metric, unit, comparison direction and threshold.
- measurement-template.csv: cycle-level energy, water and process record.
- dose-events-template.csv: commanded and actual chemical release event record.

Keep all raw failures. Do not remove a failed run to create a zero-failure pilot envelope. Preserve the full denominator, rewash, failed cycles and resource use.

## Benchmark protocol

### Reference instruments / methods

- Cleaning: standardized food-soil coupons; residual mass or calibrated residue-area imaging; an independent reference assay to calibrate fluorescence classification.
- Rinse: final-rinse conductivity and pH as process observations, plus formulation-appropriate chemical residue assay (for example TOC or targeted marker assay) as ground truth.
- Cross-transfer: clean witness coupons/item surfaces with a traceable marker, quantified after the cycle.
- Drying: item-specific mass before/after controlled drying or other validated residual-moisture method.
- Microbiology: when required, protocol-defined microbial methods under IEC TS 63331 for relative efficacy or separate NSF/ANSI 184 certification work for a sanitization claim.

### Required challenge matrix

- Soils: porridge/starch, protein, fat/oil, tea/coffee/tannin, and mixed soil after individual soil models are qualified.
- Surfaces: ceramic, glass, stainless steel and representative polymers.
- Geometry: flat plate, cup/mug interior, rim, underside, bottle interior and narrow recess.
- Water hardness: 20, 50, 75, 100, 150 and 200 mg/L as CaCO3 plus actual local inlet-water samples.
- Challenges: blocked nozzle, partially clogged filter, occluded surface, dirty/settled sump, low detergent delivery, excessive dose, failed sensor, disconnected probe, drifted turbidity/conductivity, phase-temperature deviation, incomplete final rinse and cross-transfer.

### Metrics

- Per-endpoint false-pass and false-fail rate against independent reference assay.
- Indeterminate rate and abstention quality.
- Soil-specific sensitivity and specificity; macro-averaged metrics must not hide a weak soil/material class.
- Calibration error and uncertainty coverage, particularly near acceptance boundaries.
- Item-level and load-level pass rates.
- Final-rinse residue and cross-transfer false-pass rates.
- Resource use per verified successful clean including rewash.
- Time saved by safe early stop and extra resource burden from conservative fallback.
- Sensor degradation / nozzle blockage detection and recovery.

### Aspirational research gates

These are targets, not achieved results:
1. Zero false-clean and zero false-rinse passes in the held-out pilot challenge set; report exact denominator and one-sided confidence bound, then power the confirmatory trial.
2. At least 95% sensitivity and 95% specificity for each soil/material stratum before a sensor proxy is eligible for routine operational assurance. Aggregate accuracy alone is insufficient.
3. At least 90% of truly acceptable runs can be released without unnecessary extra rinse, while all uncertain/out-of-scope states abstain or use validated fallback.
4. Uncertainty intervals show their preregistered empirical coverage on held-out days, soil batches, item materials and machine units.
5. Safe-stop control demonstrates a measurable reduction in water/energy/time per successful clean versus fixed-duration baseline without increasing any endpoint failure.
6. A proxy is never used outside its calibration envelope; an invalid/contradictory sensor never authorizes an early release.

The 86.7% oatmeal-only fluorescence classification result is a replication baseline, not a sufficient operational gate. Our target is not to beat its aggregate accuracy by training on more images; it is to demonstrate independent generalization, uncertainty calibration and zero false-clean releases within the validated use envelope.

## Immediate next steps

1. Acquire/reference a controlled fluorescence imaging station and reference assay for the target soil types.
2. Freeze the endpoint criteria and operator protocol before collecting comparative data.
3. Build the per-item evidence archive and negative-control dataset.
4. Compare independent assay, process proxy and fused observer on held-out physical runs.
5. Only then derive a process stop rule and a runtime assurance envelope.
6. Keep cleaning, rinse, cross-transfer, drying and microbiology as independent claims.

Related work: https://github.com/Luminous-Dynamics/symthaea/issues/10225.
