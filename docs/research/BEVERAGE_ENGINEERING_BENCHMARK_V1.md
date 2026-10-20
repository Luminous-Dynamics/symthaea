# Beverage Engineering Benchmark V1

Status: research contract and synthetic qualification corpus. No beverage model, real-world performance, food-safety claim, or production improvement is qualified by this document alone.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8232  
Related general culinary benchmark: https://github.com/Luminous-Dynamics/symthaea/issues/7859  
Cross-project process twin: https://github.com/Luminous-Dynamics/symtropy/issues/2387  
Cross-project provenance: https://github.com/Luminous-Dynamics/mycelix/issues/5330  
Cross-project regional evidence: https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/226

## 1. Why this benchmark should exist

Research has already shown that machine learning can predict aspects of beer flavor, and industrial-data digital twins can forecast selected beer-quality attributes. A 2024 Nature Communications study combined chemical and trained sensory analyses across 250 beers with 180,000 consumer reviews; its findings also show why public ratings are not clean ground truth (price and style preferences are confounders). A January 2026 industrial beer digital-twin study reported relative RMSE below 10% for its specific selected quality attributes. A 2026 comparative food-fermentation assessment reported that knowledge-based models can outperform machine learning on some dynamic-prediction tasks and that surrogate models can approximate mechanistic dynamics. These numbers come from different datasets, endpoints and evaluation designs: they must not be ranked as though they were one common benchmark.

The deployment gap is especially actionable. A January 2026 review of automated alcoholic-fermentation monitoring identifies missing public standardized sensor datasets, poor comparability between studies, and limited industrial validation. For wine, every vintage and must can differ; for spirits, heterogeneous and viscous mash, sensor placement and long maturation horizons complicate generalization.

Our opportunity is not to claim that a new AI is better. It is to build a stringent, reproducible test that can tell when a model is actually better, where it fails, and whether acting on it creates value.

## 2. Scope and non-goals

Products: beer, wine, distilled spirits; non-alcoholic fermented beverages may be added as a separate product family rather than mixed into alcohol-containing beverage targets.

Initial tasks:
1. Forecast fermentation trajectories using only information available at a declared prediction cutoff.
2. Estimate fermentation completion time and selected physicochemical quality endpoints.
3. Test chemical-profile prediction separately from blinded expert sensory outcomes and consumer preferences.
4. Detect sensor drift, abnormal trajectories and out-of-distribution batches with calibrated uncertainty.
5. Compare experiment-selection policies under a fixed experimental budget.
6. Later, connect validated batch/process measurements to a process digital twin, provenance records and regional resource context.

Out of scope: automatic actuator control; alcohol-consumption recommendations; proof of food safety, legal compliance, origin authenticity or shelf life; claims that a chemical score equals human preference; and field-scale benefits inferred from synthetic fixtures.

## Initial public-data triage

The first source inventory is intentionally conservative; it distinguishes a source that exists from a source that is licensed, downloaded, parsed, or fit for a particular scientific claim.

| Candidate | Good initial use | Limitation / gate |
|---|---|---|
| Schreurs et al. beer flavor data, Zenodo record 10653704 | Chemistry-to-panel sensory and published-model replication candidate; associated article reports chemical features and trained-panel descriptors for 250 beers | Audit the exact record/file license and checksum before ingestion. RateBeer review scores are explicitly restricted in the paper and are excluded unless separate permission is obtained. The reported random, style-stratified split supports a like-for-like replication only; it does not establish brewery/site transfer. |
| UCI Wine Quality, dataset 186, DOI 10.24432/C56S3T | Simple chemistry-to-quality baseline over 4,898 red/white Vinho Verde samples; the repository states CC BY 4.0 | No brand, grape, vineyard, vintage, or process-time metadata. It cannot support vintage holdout, process control or terroir causality. Group leakage risk is unresolved because the required lot IDs are absent. |
| BIER 2025 Water, Energy, and Emissions Benchmarking | Facility-type-aware industry reference for normalized brewery, distillery, winery and bottling resource metrics | The 22 April 2026 release reports a 3% water-use-ratio reduction, 4% energy-use-ratio reduction and 11% emissions-ratio reduction across 2020/2022/2024; over 60% of participating facilities face at least medium water risk. These are aggregate sector results, not a single-site causal baseline or a freely reusable row-level training dataset. |
| Dryad CC0 wine-fermentation temperature study, DOI 10.5061/dryad.cfxpnvxp3 | First candidate for parsing time-indexed yeast/microbial measurements across 17, 23 and 25 °C, plus a freeze–thaw experiment | README states CC0, collected 2024–2025 in Adelaide. Table row counts 175/121/334 are not independent batch counts. Parse replicate and treatment structure before any split; exact raw bytes/headers/checksums still need download and audit. |
| Frontiers wine-fermentation kinetics (2026), DOI 10.3389/fsufs.2026.1821488 | Reproduce published Brix/ethanol time courses and kinetic calculations across selected inoculum, temperature, sugar and starter-propagation conditions | Data are in article/supplement; paper reports triplicate experiments and propagated uncertainty. Small experimental reference, not an industrial telemetry dataset or external-vintage/site qualification. Check supplement files and definitions before comparison. |
| NCBI GEO GSE79 lager fermentation transcriptome | Explore temporal biological mechanisms across 12 time points in a production-scale lager fermentation | Transcriptomics are not inline process sensors, product chemistry or sensory labels. Use only for a named biological-state question after source/rights audit. |
| Synthetic beer-fermentation fault-injection dataset (Zenodo record 21917783) | Smoke-test pipeline and anomaly/fault-injection reporting without consuming real plant data | Synthetic data validate software plumbing only, not biological accuracy, real anomaly prevalence, production savings or generalization. Check terms before redistribution. |
| Standardized industrial alcoholic-fermentation sensor time series | High-value partner-data gap: aligned sensor streams, batch identifiers, process events, lab targets and calibrated instruments | A 2026 review identifies gaps in standardized public sensor datasets and industrial validation, but public research time series do exist. Distinguish a controlled study from production telemetry. |

Recommended first experiments, in order:
1. Audit the three CC0 Dryad wine-temperature study files and README; pin SHA-256s and field dictionaries, preserve NA/ND, and map treatment, replicate, analyte, units and time. Start with grouped microbial-response/trajectory and fault-detection tasks, not an all-winery quality model.
2. Independently reconstruct the 2026 wine kinetic curves from the published article/supplement and compare interpolation, clearly declared kinetic baselines, and an appropriate nonlinear model. Preserve the reported uncertainty and triplicate structure; timepoints are not independent batches.
3. Replicate a published chemistry-to-trained-panel beer task from the Zenodo-associated data after checking file-level rights, columns, missingness and stable sample IDs. Exclude restricted consumer-review data by default.
4. Run UCI Wine Quality only as a separate static chemistry-to-quality baseline with its own split manifest and explicit claim ceiling. Do not pool different product targets into one leaderboard.
5. Use GEO GSE79 only for a separate temporal yeast-biology question; it is not a sensory or finished-quality target. Then seek partner production telemetry for the first genuine plant forecasting and intervention-value test.

Sources: https://datadryad.org/dataset/doi:10.5061/dryad.cfxpnvxp3 ; https://doi.org/10.3389/fsufs.2026.1821488 ; https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE79 ; https://zenodo.org/records/21917783 ; https://zenodo.org/records/10653704 ; https://archive.ics.uci.edu/dataset/186/wine%2Bquality ; https://www.bieroundtable.com/2025-water-energy-and-emissions-benchmarking-summary-bier ; https://doi.org/10.1007/s13197-025-06528-0

## 3. Evidence types and target separation

Every value must declare an evidence class:
- MEASURED: instrument or panel observation with method, units, timestamp and calibration context.
- MODELLED: output of a named model with version, inputs and uncertainty.
- SIMULATED: generated by an explicit simulation with assumptions and seed.
- DECLARED: human or producer assertion not independently measured.
- UNKNOWN: provenance or evidence class cannot be established.

Do not merge these classes into one unlabeled table. A model output must never be relabeled as a measurement. The digest of a file proves which bytes were evaluated; it does not prove the measurement is true.

Keep the following endpoints distinct:
- process state and future trajectory;
- laboratory chemistry or physicochemical properties;
- blinded expert-panel descriptors and repeatability;
- randomized blind consumer preference;
- ratings from public reviews;
- plant resource use and economics;
- safety or regulatory tests performed under an applicable validated method.

A sensory model trained on consumer review scores must be described as predicting those review scores, not as predicting universal flavor quality. A chemistry model may predict chemistry accurately without predicting preference.

## 4. Leakage-resistant experimental design

Primary split rules:
- Group all records from a fermentation batch or final product lot into one split.
- Use time-ordered cutoffs for trajectory forecasting. Features recorded after the cutoff, final-lab results, packaging results and post-hoc classifications are forbidden predictors.
- Use leave-one-batch-out as a first test; use leave-one-vintage, facility, yeast strain, instrument or sensor family out when coverage permits.
- Keep duplicate samples, replicate panels, bottle variants and derived records from the same parent lot together unless the explicit task is replicate-repeatability analysis.
- Fit imputation, normalization, feature selection, embedding, hyperparameter tuning and uncertainty calibration on training/validation groups only.
- Keep a source-, rights- and method-aware data ledger. Unknown licensing or data permission makes the data ineligible for redistribution/use, not silently open.
- Freeze the corpus, model version, split manifest and score definition before final evaluation. Preserve failed and missing cases in the denominator.

Random row splits may be reported only as a diagnostic comparison, never as the primary evidence for generalization.

## 5. Required comparators

Every forecast task with suitable inputs should compare:
1. Persistence / last-observation or fixed-rate baseline.
2. Simple linear or generalized-linear regression/classifier.
3. A tree ensemble suitable for tabular data.
4. A mechanistic or phenomenological process model, if the needed parameters and measurements are available.
5. A hybrid model that uses the process model and learns a bounded residual, if justified.
6. The current domain or operator baseline, when a plant partner can specify it.

Every candidate gets the same input information, held-out examples, target definition and hard gates. Do not compare a hybrid model with richer sensors against a baseline that was denied those sensors and attribute the entire gain to the algorithm.

## 6. Metrics and proposed targets

Report exact endpoint units, sample and batch counts, target missingness, metric denominator, group-level results and uncertainty. Choose metrics to fit each task:
- Continuous values: MAE and RMSE in physical units; NRMSE only with its scale denominator declared and frozen from training data.
- Completion time: absolute error and late/early decision rates under a declared operating window.
- Classification: per-class precision/recall, false-negative rate for the declared fault class, and calibration; accuracy alone is insufficient.
- Predictive uncertainty: empirical coverage and interval width together. For a nominal 90% interval, a proposed qualification band is 85–95% coverage on a sufficiently sized held-out set, with subgroup coverage reported. This is a proposed gate, not an observed result.
- Sensory outcomes: panel protocol, randomization, blinding, reference standards, inter-rater agreement and the independent test-set unit must be declared. Do not optimize against unblinded labels.
- Decision quality: expected regret or value of information under a fixed budget and identical candidate experiments. Prediction accuracy alone cannot establish that the chosen intervention improves a batch.
- Plant performance: product-conforming saleable output, water, energy, waste, cooling demand, downtime and cost are reported separately with consistent functional units.

Proposed model go/no-go after adequate empirical coverage: at least 10% lower MAE than the strongest pre-registered baseline on two independent batch/site/season holdouts, with a paired uncertainty interval supporting improvement and no worse calibration or hard-gate performance. If the available data cannot support this claim, report the result as inconclusive rather than lowering the bar after seeing scores.

Proposed experiment-selection target: at least 20% lower mean decision regret than the strongest simple fixed-budget baseline on held-out, preregistered scenarios, then replicate with real experiments before any operational claim. Synthetic-only wins qualify the oracle contract, not a real-world recommender.

Resource improvement must be measured against a matched baseline at comparable product mix, quality, output and facility type. BIER’s 2025 study, released 22 April 2026, reports sector-level reductions of 3% in water-use ratio, 4% in energy-use ratio, and 11% in emissions ratio across its 2020, 2022 and 2024 reporting years; it also reports that over 60% of participating facilities operate in areas with at least medium water risk. These aggregate trends are an industry reference, not directly comparable to a single-site intervention. A proposed first-pilot stretch goal is at least 10% reduction in one preregistered normalized water or energy metric, with no statistically supported degradation in the other resource metric, saleable yield, predefined quality constraints or safety gates. Meter boundary, functional unit, production mix, uncertainty and stopping rules must be frozen before the pilot. This is an ambition to test, not a result or universal engineering limit. Keep water scarcity and local context visible; do not convert a global median into a universal limit.

## 7. Minimum synthetic regression corpus

The companion JSON lists contract cases for positive controls and failure/indeterminate controls. These cases test whether the evaluation system rejects leakage, ambiguous labels and unsupported claims. They are not product measurements and contain no experimentally validated fermentation kinetics.

Required outcomes:
- invalid or leakage-contaminated experiments must not pass;
- valid-but-shifted data must retain a shift warning and group-level results;
- missing calibration, units, rights or required provenance must be indeterminate or ineligible, not optimistically imputed;
- measured/modelled/simulated evidence must remain distinct;
- hard quality/safety/authorization gates cannot be compensated by taste, yield or cost;
- candidate ordering and JSON field ordering must not alter the canonical receipt.

## 8. Cross-project architecture boundaries

- Symthaea: state forecasting, chemistry/feature learning where evidence supports it, uncertainty, active experiment selection and benchmark receipts. The existing symthaea-culinary foundation should be extended rather than duplicated.
- Symtropy: physically explicit process simulation, mass/energy balances, thermal dynamics, tank operations and multi-objective trade-offs. Predictions remain separate from measurement and actuator authority.
- Sol-Atlas: versioned, resolution-aware source-region, crop, water, climate, logistics and energy context. Maps and climate correlations do not prove beverage sensory outcomes.
- Mycelix: batch/lot identity, parent-child transformation provenance, rights/consent, custody and claim-specific verification. Hashes do not authenticate the underlying physical truth.

## 9. Executable synthetic-corpus receipt

The first Rust executable is intentionally hosted in the existing `symthaea-culinary` crate to reuse its workspace dependencies and avoid an overlapping standalone culinary package.

From the repository root, after checking out the exact commit you intend to qualify:

```bash
EXPECTED_HEAD="$(git rev-parse HEAD)"
cargo fmt --all -- --check
cargo test --manifest-path tools/beverage-benchmark-harness/Cargo.toml --bin beverage_benchmark_receipt
cargo run --manifest-path tools/beverage-benchmark-harness/Cargo.toml --bin beverage_benchmark_receipt -- \
  --corpus docs/research/beverage-engineering-scenarios-v1.json \
  --schema docs/research/beverage-engineering-scenarios-v1.schema.json \
  --expected-head "$EXPECTED_HEAD" \
  --out /tmp/beverage-benchmark-receipt.json
```

The receipt binds the raw corpus and schema bytes with SHA-256, includes the exact checked-out Git HEAD, rejects malformed input and unknown fields, checks unique scenario identifiers/names, verifies declared product/disposition values against schema enums, requires the corpus to remain explicitly synthetic, and emits a deterministic `PASS` or `FAIL` JSON report. The CLI exits non-zero on a failed contract.

The focused harness compiles the exact tracked Rust source files through `tools/beverage-benchmark-harness/Cargo.toml`. This avoids falsely attributing a monorepo manifest-enumeration failure to the beverage tools: the current root workspace glob also sees `crates/core/symthaea-observation-attestation/src/lib.rs`, but that directory has no `Cargo.toml`. The harness does not exclude or skip that source from the main workspace and does not establish full-workspace health.

**Qualification limit:** this first runner does not implement the full JSON Schema Draft 2020-12 specification. It checks a narrow set of safety-relevant schema invariants and then deserializes into strict Rust types. An independent standards-compliant JSON Schema validator remains required. Passing receipt checks establishes contract integrity only, not that any scientific result is valid. The test binary must be built, formatted, tested, and run against the exact intended head before the receipt can be called qualified.

## 10. UCI wine-quality source auditor

## Independent JSON Schema Draft 2020-12 validation

The isolated harness includes a local schema auditor backed by the pinned `jsonschema` 0.59.0 crate with default file/HTTP reference resolution disabled. It checks the declared Draft 2020-12 dialect, validates the schema against the Draft 2020-12 meta-schema, builds the validator with format checks enabled, and validates the instance. External `$ref`, `$dynamicRef`, and `$recursiveRef` targets are refused to prevent network or arbitrary file reads.

```bash
cargo test --manifest-path tools/beverage-benchmark-harness/Cargo.toml --bin schema_contract_audit
cargo run --manifest-path tools/beverage-benchmark-harness/Cargo.toml --bin schema_contract_audit -- \
  --schema docs/research/beverage-engineering-scenarios-v1.schema.json \
  --instance docs/research/beverage-engineering-scenarios-v1.json \
  --out /tmp/beverage-schema-receipt.json
```

This is independent of the Rust strict-deserialization and receipt-contract checks. The pinned library supports Draft 2020-12 but is not claimed to pass every case in the official JSON Schema conformance suite. A PASS receipt shows that this local schema and instance passed this validator path, not that domain semantics are scientifically true.

## Dryad fermentation time-series intake auditor

A third focused Rust executable supports the three long-form tab-separated files in the 2026 Dryad wine-temperature / yeast-ecology dataset (DOI 10.5061/dryad.cfxpnvxp3). It checks the declared columns, resolves columns by name, validates time/treatment/replicate fields, preserves `NA` and `ND` as distinct missingness, summarizes numeric analytes by unit, detects duplicate observation keys, hashes exact input bytes, and generates a JSON receipt.

After downloading the files and reviewing the source record:

```bash
cargo test --manifest-path tools/beverage-benchmark-harness/Cargo.toml --bin fermentation_time_series_audit
cargo run --manifest-path tools/beverage-benchmark-harness/Cargo.toml --bin fermentation_time_series_audit -- \
  --input /path/to/Onetto2026_Figure1_raw_data_final.txt \
  --dataset figure1 \
  --out /tmp/figure1-intake.json
```

Supported dataset kinds are `figure1`, `figure2`, and `figure-s1`. The auditor allows column reordering only when the exact required column set remains intact and reports that deviation. The receipt expressly marks source identity as caller-declared, not authenticated.

**Important design limit:** the source README describes `rep` as a technical replicate. Distinct `rep` values and timepoint rows are not evidence of independent fermentation batches. Model evaluation must group samples by the underlying biological/experimental unit and treatment, as justified by the study design. `AUDIT_OK` means raw-file contract integrity only, not experimental independence, causal inference, predictive performance or industrial transfer. Missingness is retained; it is never imputed to zero.

The second Rust executable audits a locally obtained red or white UCI Wine Quality CSV using the semicolon-delimited 12-column format. It computes the SHA-256 of the exact raw bytes; validates the declared header and column order; accounts for each parsed data row as valid or invalid; rejects missing, nonnumeric, nonfinite and semantically out-of-range values; and emits column summaries plus the quality-score distribution.

Example use after downloading and reviewing the current UCI terms:

```bash
cargo test -p symthaea-culinary --bin wine_quality_source_audit
cargo run -p symthaea-culinary --bin wine_quality_source_audit -- \
  --input /path/to/winequality-red.csv \
  --color red \
  --out /tmp/wine-quality-audit.json
```

The `color` is explicit metadata because the source files are separate and do not contain a color column. The tool labels source identity as caller-declared rather than authenticated. It does not bundle, redistribute, or download source data; it does not assert the file is genuinely from UCI merely because it parses. Review the dataset page and terms before any redistribution, preserve the raw input digest, and keep any future train/test split manifest separate from this intake receipt. The dataset supports a static chemistry-to-quality-score baseline only; absent batch/vintage/site identifiers prevent claims of production transfer.

## 11. Build sequence


Phase A — corpus and contract: parse the scenario corpus, verify schema and unique IDs, emit deterministic expected/actual receipts and fail-closed outcomes. The current PR proposes this specification and corpus only.

Phase B — first public-data adapter: select one dataset after checking access rights, schema, target units and batch identifiers. Pin source version and hashes; do not copy the dataset until license and terms permit.

Phase C — baselines: implement one process trajectory baseline and one tabular baseline under exactly the same group split. Store the split manifest and report.

Phase D — prospective validation: pre-register model, data cutoff, target, intervention and blind evaluation with a brewery, winery or distillery partner. No production deployment without process-engineering, food-safety and operator review.

Phase E — decision and plant twin integration: evaluate value of information and Pareto trade-offs with real, versioned cost/resource inputs.

## 12. Research anchors

- Theßeling et al. (2024), Predicting and improving complex beer flavor through machine learning, Nature Communications 15, 2368. https://www.nature.com/articles/s41467-024-46346-0
- Dazzarola et al. (2026), Toward a digital twin for beer quality control: development of a digital model integrating industrial process data and model-based fermentation descriptors, Journal of Food Engineering 403, 112726. https://doi.org/10.1016/j.jfoodeng.2025.112726
- Campo-Manzanares et al. (2026), Critical assessment of machine learning approaches for classification, dynamic prediction and surrogate modeling in food fermentation, Food Research International 229, 118403. https://doi.org/10.1016/j.foodres.2026.118403
- Horváth et al. (2026), Automated monitoring of alcoholic fermentation: trends and challenges, Journal of Food Science and Technology 63, 224–237. https://doi.org/10.1007/s13197-025-06528-0
- Liu et al. (2026), Machine Learning-Enabled Intelligent Technologies Across the Grape and Wine Value Chain, Comprehensive Reviews in Food Science and Food Safety 25(4), e70523. https://doi.org/10.1111/1541-4337.70523
- BIER (2025), Water, Energy, and Emissions Benchmarking Summary. https://www.bieroundtable.com/2025-water-energy-and-emissions-benchmarking-summary-bier
- Boulton, Nelson & Knoesen (2025), A Review of Sensors for the Monitoring, Modeling, and Control of Commercial Wine Fermentations. https://doi.org/10.3390/fermentation11060329

## Claim ceiling

This document specifies a benchmark intended to make future claims more rigorous. It does not report a passed model, a production trial, a sensory improvement, an energy or water saving, or a superiority result. Current CI/test state must be inspected on the exact PR head; skipped, queued, cancelled or absent checks are not passes.
