# Ore Assay Semantics and Mass-Accounting Benchmark v0.1

**Owner:** Symthaea  
**Evidence:** synthetic fixtures only  
**Purpose:** establish input semantics before ore-recovery prediction or process optimization

## Research grounding

- The [USGS USMIN Mineral Deposit Database](https://www.usgs.gov/centers/gggsc/science/usmin-mineral-deposit-database) provides deposit/geology/resource context for the United States. Such spatial or reported context is not a sample assay or proof of recoverable supply.
- The [USGS Mineral Commodity Summaries data release](https://doi.org/10.5066/P13XCP3R) provides commodity-level production and resource context, not site-specific assays or process behavior.
- The South African Council for Geoscience [mineral-resource map service](https://maps.geoscience.org.za/hosting/rest/services/RSA_MINERAL_RESOURCE_V2_MIL1/MapServer) exposes regional geological and mineral-resource layers. Map scale, source date and feature semantics must remain attached; a mapped occurrence is not a verified reserve, live operating capacity, assay or extraction permission.
- Mwanga, Rosenkranz and Lamberg, Testing of Ore Comminution Behavior in the Geometallurgical Context (2015), describes why ore variability and comminution response matter to process performance: https://doi.org/10.3390/min5020276

These sources motivate distinctions and later data adapters. They are not evidence that this synthetic validator predicts the behavior or grade of a real deposit.

## Paradigm shift

Don't model ore as an unconditioned commodity name or a map point. Preserve the chain:

deposit / regional context
 -> exact sample and lot
 -> assay, method, detection limit and QC
 -> mineral phases / texture / liberation evidence
 -> process route under a declared capability envelope
 -> measured product grade and recovery
 -> tailings / middlings / water / reagents / energy / residues

Every transition has its own evidence. An element's presence doesn't imply mineral liberation, a bulk assay doesn't fully describe mineral phase distribution, a process simulation doesn't prove measured recovery, and a custody receipt doesn't certify grade.

## v0.1 schema

- All assays use integer parts per million by mass (ppm_mass_fraction).
- Each assay records analyte ID, quantity kind, reporting expression, sample basis and result state.
- Quantity kinds remain distinct: elemental_mass_fraction, oxide_equivalent_mass_fraction, and mineral_phase_mass_fraction.
- Assay basis is explicit: dry_mass or wet_mass.
- Result state is quantified, below_detection, or not_tested.
- A below-detection result carries a positive detection limit and no numeric result; zero is not a valid encoding of below-detection evidence.
- A quantified result carries a value. Where a detection limit is recorded, a quantified value must not be below it.
- For a frozen synthetic sample, mineral phase quantities are dry-mass fractions. Their quantified sum cannot exceed 1,000,000 ppm. Do not sum elemental assays or oxide equivalents together with mineral phases because these are overlapping descriptions of the same material.
- Source record, sample, lot, method, and analytical QC are explicit. A blank source or sample identity blocks the claim.
- Process-flow mass uses dry milligrams for the input and each output stream. Product, middlings, tailings and explicitly modeled loss streams must reconcile within the declared parts-per-million tolerance.
- A flow must refer to the exact sample and lot under test.
- A grade comparison may compare like analyte, quantity kind and basis directly. If kind or basis differs, an explicit conversion-evidence reference and normalized ppm value are required. The presence of a reference is only a software gate; the converter's chemistry and applicability are not verified by v0.1.
- Unknown fields are rejected. IDs are stable within each scenario and duplicate analyte/stream IDs are rejected.

## Frozen known-answer corpus

1. Valid synthetic assay, valid phase fractions, compatible elemental grade comparison, and exact dry-mass closure.
2. Below-detection assay incorrectly supplies numeric zero.
3. Elemental-grade requirement compared to an oxide-equivalent result without a conversion record.
4. Process-flow input points to a different sample/lot.
5. Output stream mass exceeds dry feed mass.
6. Analytical quality-control evidence is unresolved.
7. Quantified mineral-phase mass fractions sum above the physical sample fraction limit.
8. Required source-record provenance is missing.

The evaluator emits a receipt binding exact corpus bytes, validator version and source commit. It returns success only when the frozen expected dispositions and reason codes match exactly.

## Benchmark metrics

Report independently:
- schema/source/lineage validity;
- basis and unit mismatch counts;
- below-detection semantic violations;
- mineral-phase fraction violations;
- dry-mass closure error and rejected flows;
- quality-control/provenance completeness;
- expected negative-control match rate;
- deterministic replay and runtime.

Do not report ore recovery, grade prediction accuracy, reserve confidence or economic value from these fixtures. Those require independent reference measurements and a separate benchmark.

## Next research ladder

1. v0.1 assay contract: exact semantics and negative controls only.
2. v0.2 public-source registry: metadata-only adapters for USGS and South African CGS with scale, dates, source license, CRS, coverage and documented claim scope.
3. v0.3 mineralogy/comminution contract: distinct sample-level mineral-phase, particle-size, liberation and comminution-test evidence; no liberation inference from assay grade alone.
4. v0.4 mass-balanced process scenarios: simple synthetic beneficiation flows with explicit product, middlings, tailings, water, energy and reagent streams.
5. Measured-data pilot: only after a licensed dataset, method comparability, sampling, laboratory QA/QC and site/deposit holdout are established.
6. Decision value-of-information: test whether selecting a new assay/mineralogy/comminution experiment can change the feasible route or Pareto set.

## Cross-stack ownership

- Symthaea: evidence-aware reasoning, hypotheses and experiment selection.
- Symtropy: deterministic geometry/flow simulation where a suitable physical/process model exists.
- Mycelix: exact sample/lot custody, event history, correction and provenance records.
- Sol-Atlas: spatial context and source metadata, not reserve or assay truth.
- Independent validator: units/basis, identities, arithmetic, controls and receipts.

Reuse canonical element, material, measurement, manufacturing and upstream-resource owners. Don't create a parallel periodic table, material ontology, resource registry, generic process engine or provenance ledger.

## Claim ceiling

A passing receipt establishes only software semantics for the declared synthetic fixtures. It does not establish geological truth, mineral identity, assay accuracy, reserves, mine safety, process recovery, product quality, environmental acceptability, economic feasibility or extraction permission.
