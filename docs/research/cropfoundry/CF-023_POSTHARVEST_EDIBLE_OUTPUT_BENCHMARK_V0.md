# CF-023: marketable edible output and post-harvest quality retention

**Status:** benchmark proposal; not executed. The purpose is to create a defensible evaluation contract for reducing post-harvest losses, not to claim that CropFoundry has improved shelf life or food safety.

## Paradigm shift

Do not optimize harvest yield in isolation. Optimize **safe, marketable edible output delivered**, after handling, storage, processing and transport, subject to cost, energy, time, quality and environmental constraints.

FAO's SDG 12.3.1a Food Loss Index estimates food losses between production and retail. Its portal reports the 2023 fruit-and-vegetable aggregate at 25.4%, but this is a global commodity-basket estimate and must not be interpreted as a Gauteng baseline or a tomato-specific rate ([FAO Food Loss Index](https://www.fao.org/sustainable-development-goals-data-portal/data/indicators/1231-global-food-losses/en)). FAO's 2026 technical platform lists dedicated measurement guidance for harvest and post-harvest losses in fruits and vegetables, emphasizing consistent measurement methods and stage-specific accounting ([FAO food-loss measurement platform](https://www.fao.org/platform-food-loss-waste/en/)).

Evidence quality is a major part of the problem. A 2025 review of food-loss data gaps in fruit-and-vegetable value chains reports that published evidence is uneven by commodity, region and supply-chain stage ([review](https://doi.org/10.3389/fhort.2025.1529040)). This argues against a single universal loss percentage and in favor of traceable batch-by-stage accounting.

## Candidate dataset for schema feasibility, not training

[DS-014: Dataset on tackling postharvest tomato losses in tropical climates using a passive cooling blanket](https://zenodo.org/records/15856413) (Zenodo v3, published 2025-07-10) lists eight XLSX workbooks of roughly 2.3 MB total. The associated 2025 *Scientia Horticulturae* article ([DOI 10.1016/j.scienta.2025.114313](https://doi.org/10.1016/j.scienta.2025.114313); [Zenodo publication record](https://zenodo.org/records/16910974)) describes a Uganda tomato supply-chain pilot and reports measurements such as produce weight, firmness, color, soluble solids content and acidity. The visible Zenodo Rights/License field is not populated; **DS-014 remains quarantined** until the exact artifact rights and terms have been reviewed.

No workbook has been downloaded, parsed, hashed or scored by this workstream. Before use, verify per-file license/access terms, workbook schema, units, batches, replicate counts, intervention assignment, measurement timing, mass balance and exclusion/missingness rules. A paper summary or a visible landing page is not a substitute for the underlying row-level observations or proof of causal identification. Findings in the Uganda study are not evidence of efficacy in South Africa.

## Local mass-balance audit prototype

A local audit command is now being added for CSVs that have already been legitimately acquired and whose source/terms are known. It is intentionally separate from model training:

```bash
cargo run -q -p xtask -- cropfoundry-postharvest-audit \
  --input /path/to/postharvest-lot-stages.csv \
  --source-id DS-014 \
  --source-uri https://zenodo.org/records/15856413 \
  --delimiter ';' \
  --out reports/cropfoundry/postharvest-mass-balance.json
```

The v0 profile expects one row per lot × stage × replicate × measurement time, a non-empty measurement method, and mass values explicitly labelled in kilograms. Each row must account for input mass across non-overlapping marketable output, edible discard, inedible removed fraction, moisture loss and handling loss. A caller can set absolute and relative closure tolerances; defaults are 0.02 kg and 1% of input mass. The command emits per-row residuals, marketable-mass fraction, source/file SHA-256, duplicate-key and schema findings, and a JSON receipt even when checks fail. It exits nonzero for structural or mass-balance failures.

This profile deliberately does not model added wash water, packaging, coatings, gas exchange, or other inputs. Data with such flows need a different balance schema rather than forcing it into this profile. Mass-balance closure is a data-accounting check, **not** a validation of freshness, food safety, biological quality, nutritional retention, or causal effect. The receipt always records training authorization and safety validation as false.

## Evaluation contract

1. **Define the accounting boundary.** For each produce lot, record mass at harvest, after handling, after storage, on arrival, saleable/marketable mass, edible discards, inedible fractions, and the reason for each loss. Avoid double-counting dehydration, sorting rejection and spoilage. Every quantity needs a unit, scale, timepoint and measurement method.
2. **Keep endpoints distinct.** Mass retention, marketability, sensory/quality measures, measured nutrient retention and food safety are different outcomes. A change in color or texture does not prove microbial safety or nutrient content. Safety constraints need validated independent tests and an explicit stop rule.
3. **Start with baselines.** Compare the current documented operation with crop-/condition-appropriate temperature-time thresholds, simple storage-life models and any justified established post-harvest model. Avoid a high-capacity image model until small, transparent baselines are reproducible.
4. **Prevent leakage.** Keep all images, repeated measurements and sub-samples from the same produce lot together. For forecasting, hold out complete future harvest lots and sites/seasons. Randomly splitting images of the same tomato or batch can make a model memorize the specimen or storage trajectory.
5. **Attribute interventions carefully.** A policy comparison needs the assignment mechanism, independent batches, replication, matched harvest maturity, baseline handling, ambient conditions, packing, transport duration, and predeclared outcomes. Without credible assignment/control, report association rather than causal effect.
6. **Measure full system cost.** Include cooling energy or water, packaging, equipment amortization, labor, travel/transport, rejected produce, cleaning, spoilage disposal and emissions where measurable. A storage intervention is not better if it reduces loss but requires unaffordable energy or shifts losses elsewhere.
7. **Report uncertainty and generalization.** Report loss ratios and confidence intervals at each supply-chain stage, each crop and each site. A small single-site experiment supports only a narrow claim. Require external lots and season/site holdouts before transfer claims.

## Proposed targets, to be locked before outcome inspection

- **Stage 0 — data contract:** reconcile mass balance and loss definitions for every batch; unresolved mass differences and unsupported unit conversions must be surfaced, not imputed.
- **Stage 1 — retrospective benchmark:** reproduce a declared baseline on a pinned snapshot and lot-disjoint/chronological split. Report calibration, error, coverage, and performance at each storage horizon; keep this predictive score separate from intervention efficacy.
- **Stage 2 — prospective pilot:** provisional research target of at least a 10% relative reduction in measured marketable-mass loss, with non-inferior predeclared quality, zero additional safety-gate failures, and no increase in fully loaded cost/energy/emissions per marketable kg. Revisit the target using baseline variability, power analysis, partner priorities and a pre-registered protocol before collecting outcomes.
- **Stage 3 — external evidence:** replicate across independent lots and at least two materially distinct storage/transport settings before generalizing beyond the tested conditions.

These are proposed study targets, not achieved results or universal standards. The benchmark cannot pass on appearance prediction alone, and CropFoundry must not emit a food-safety conclusion unless supported by an appropriate validated test and authoritative review.

## Run receipts and provenance

Every scored run must include:
- source artifact URI, file-level license/terms evidence, retrieval timestamp and content digest;
- crop, cultivar when known, batch/lot ID, harvest maturity, source location granularity permitted by consent, and all storage/transport stages;
- units, instrument calibration, weighing procedures, quality/safety methods and missingness;
- split/holdout IDs, code/model commit, baseline, metric definitions, exclusions and complete per-batch predictions;
- intervention allocation/protocol and ethics/partner approvals where relevant.

A synthetically simulated shelf-life curve can test code and accounting but cannot demonstrate freshness, food safety, avoided loss, nutritional retention or farm economics.
