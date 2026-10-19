# MFG-VALUE-001: Paired Production-Outcome Evaluation Protocol

## Purpose

Create a reusable evaluator for the question that matters: does an engineering/manufacturing intervention improve conforming output on the same workload and disturbance conditions without hiding safety, quality, or accounting failures behind a scalar score?

This is an implementation slice for [MFG-VALUE-001](https://github.com/Luminous-Dynamics/symthaea/issues/7608). It complements domain-specific work such as the FDM quality benchmark (#7960/#7961), the household-goods factory and supply-chain benchmark (#8053), and explicit manufacturing baseline calibration (#8015). It does not establish that any candidate improves production.

## Paradigm shift

A model or optimizer is not the product; the measured outcome is the product. Compare baseline and candidate on identical, frozen scenario IDs and workload identities. Preserve hard gates as non-compensatory constraints. Report the primitive outcomes independently; do not blend yield, energy, speed, safety, cost, and repairability into a weighted score that allows one dimension to conceal a regression in another.

The unit of evaluation is a paired scenario, not a cherry-picked best run. Scenario IDs should include seed and scenario-defining choices. The workload identity should cover work order, part mix, material/lot identity, starting state, process route, measurement boundary, and acceptance criteria. Each pair carries the same workload identity, input-unit count, and input material quantity.

## Implemented evaluator scope

The Rust module `symthaea_fabrication_kernel::production_value` provides a deterministic qualification boundary:

- Every run partitions all input units exactly into accepted, scrap, rework, and held units.
- Every run closes the declared material mass in integer milligrams across accepted output, scrap, rework, held material, and an explicit residual.
- Energy and elapsed time must be finite; duration must be positive; energy must be non-negative.
- Baseline and candidate scenario sets must match exactly. Duplicate scenarios, mixed policy labels, workload mismatches, malformed identities, arithmetic overflow, invalid counts, and non-conserving records reject the comparison.
- Summaries preserve units and boundaries. There is no blended score and no automatic claim of economic, environmental, or safety improvement.
- Candidate eligibility is a separate, non-compensatory result. Any critical safety violation, escaped nonconforming unit, or unresolved held unit makes the candidate ineligible.

This module is an arithmetic/structure oracle over supplied records. It is not a sensor adapter, authenticated metrology pipeline, causal estimator, optimizer, digital twin, or manufacturing controller. A valid report neither authenticates inputs nor proves that the outcomes are physical. Callers must retain source data, measurement methods, provenance, uncertainty, the exact software revision, and inclusion/exclusion rules.

## Minimum experimental protocol

1. Freeze the product/workload, acceptance specification, route, scenario list, seeds, measurement boundary, and stopping rule before running either policy.
2. Name/version the baseline and candidate. Give them identical input lots and scenario sequences. Use paired repetitions and disposition every planned case, including aborts and failures.
3. Prefer independent metrology. Distinguish sensor telemetry, model estimates, operator observations, and measured outcomes.
4. Reserve chronological and/or product-family holdouts. Do not tune on the final evaluation set. Prevent related runs from leaking across splits.
5. Report first-pass yield, conforming yield, scrap/rework/held fractions, energy per conforming unit, accepted units per machine-hour, uncertainty, and maintenance/downtime only when those data have actually been collected.
6. Report safety, escaped defects, and conservation failures as hard gates—not costs that can be offset by productivity.
7. Compare with simple transparent baselines before claiming added value from Symthaea, Symtropy, or Mycelix. Repeat at an independently reviewed site before generalizing.

## Proposed benchmarks (not achieved)

- **100% paired coverage:** every candidate run has one baseline run on the same scenario/workload; every planned case is dispositioned.
- **100% accounting closure:** unit partitions and the stated material boundary reconcile exactly at the declared integer units; otherwise no comparison is emitted.
- **Zero hard-gate violations:** no critical safety violations, escaped nonconforming units, or unresolved held units for candidate eligibility.
- **Prediction value:** on a preregistered real-data holdout, target at least 10% lower MAE than a named conventional baseline for each claimed forecast target/horizon, with uncertainty and all horizons reported.
- **Production value:** investigate a 5-percentage-point yield increase or 10% lower energy per conforming unit only where a powered paired physical pilot supports it without hard-gate regression.
- **Reproducibility:** frozen inputs and software version produce byte-identical structural reports; performance measurements separately record environment and p50/p95/p99 latency.

These are investigation thresholds, not universal specifications or achieved results. A synthetic fixture qualifies arithmetic, not production benefit. Do not decide that a target is meaningful until baseline variance, measurement uncertainty, sample size, and effect size are estimated.

## Negative controls

Reject or explicitly fail eligibility for duplicate scenario IDs; missing/extra scenarios; mixed policy labels; changed workload identity; changed fixed input quantities; unit partition off by one; material balance off by one milligram; zero/negative/non-finite duration; negative/non-finite energy; integer overflow; zero accepted units causing division by zero; critical safety violations; escaped nonconforming units; or held output. Future forecasting comparisons should additionally include label-shuffle controls and report matched conventional baselines. Those statistical tests are not implemented by this arithmetic module.

## Typed evidence classes and provenance boundary

The outcome evaluator requires each run to declare one evidence class: `SyntheticFixture`, `SimulationOutput`, or `InstrumentObserved`. A comparison fails when a policy mixes classes internally or when the baseline and candidate classes differ. This prevents direct comparison of synthetic/modelled outcomes against instrument-observed outcomes from passing as a paired evaluation.

Each run also carries a workload SHA-256 identity and a raw-source-record SHA-256 field. The current implementation validates only lowercase 64-character hexadecimal syntax; it does **not** recompute hashes, fetch records, verify signatures, authenticate instruments, or establish that the declared class is truthful. Test fixtures use digest-shaped identifiers only. Production integration must compute digests from the canonical bytes, preserve the referenced evidence, authenticate acquisition where feasible, and record the measurement protocol. `InstrumentObserved` means the record is declared to originate from instruments; it is not synonymous with independently validated or causally identified.

## Paired uncertainty without pseudo-replication

The companion `paired_inference` module provides a deterministic percentile cluster-bootstrap interval for the mean paired difference. It requires at least 20 independently sampled clusters and at least 1000 resamples. It records the seed, resample count, confidence level, mean and median difference, interval bounds, and the number of clusters where the candidate is better/worse/tied under a declared metric direction.

**Cluster before bootstrap.** The caller must aggregate observations inside a genuinely independent unit (for example, material lot, production day, or independent site) and provide one paired value per unit. Repeated pieces, time samples, or multiple scenarios from one lot must not be passed as separate independent clusters. Cluster IDs must be unique and paired; the function rejects duplicates and non-finite values. The interval is over an equal-cluster-weighted mean difference.

This is a percentile bootstrap description of uncertainty, not a causal identification procedure or a statistical proof of superiority. It emits no p-value and does not decide eligibility by whether a confidence interval excludes zero. The independence assumption, study design, missingness, stopping rule, measurement uncertainty, and direction of each metric must be justified outside the helper. The minimum cluster count is a conservative project gate, not a universal theorem; small or highly heterogeneous studies may require substantially more data or another preregistered method.

`PairedBootstrapConfig::default()` uses 10,000 bootstrap replicates and a fixed reproducible seed. Always report the exact configuration and cluster definition.

### AM-Bench dataset-selection caveat

The current NIST index links AMB2025-02 tensile measurements, AMB2025-03 fatigue measurements, and AMB2025-06/07 laser-track calibration data. The AMB2025-02 description says eight miniature tensile specimens were excised from the same leg of one original AMB2022-01 artifact. Therefore those eight specimens must not automatically be treated as eight independent production clusters; the shared parent artifact is a potential dependence source. That dataset can support a narrowly defined material-property prediction exercise with the published split/target, but it cannot satisfy this protocol's 20-independent-cluster gate merely by counting specimens.

NIST publishes each dataset's citation instructions at the DOI/PDR landing page. Before ingestion, record the exact dataset revision, licence/terms, files and checksums, units, missingness, specimen/build/lot lineage, and permissible prediction target. The NIST page notes that AM-Bench 2025 journal articles are published after measurement activities complete; use the current source dataset and cited measurement description, not an inferred challenge score. If the PDR landing metadata cannot be read or provenance cannot be established, leave the adapter and evaluation blocked instead of fabricating the missing details.

## Cross-stack composition boundaries

Reuse the existing work rather than growing parallel owners:

- [Symtropy QDRIFT-SYNTH-001 PR #2066](https://github.com/Luminous-Dynamics/symtropy/pull/2066) supplies a deterministic synthetic drift/tool-wear/maintenance-policy challenge, including the biased-sensor negative control. It is useful for evaluator and falsification integration, but its toy-model outcomes are not physical measurements.
- [Symtropy MFG-DATA-001 PR #2080](https://github.com/Luminous-Dynamics/symtropy/pull/2080) reports a pinned, tool-disjoint Vicomtech machine-tool-wear feature evaluation with source-native label values. Its current feature-level results are not factory throughput, yield, energy-per-part, calibration, or a qualified maintenance policy. Do not map it into this production-outcome evaluator by inventing duration, mass, or energy fields.
- [Mycelix MFG-QDRIFT-001 #5218](https://github.com/Luminous-Dynamics/mycelix/issues/5218) owns the cross-repository process-run receipt and evidence-state contract; it already points to the versioned [ManufacturingEvidenceEntry PR #5197](https://github.com/Luminous-Dynamics/mycelix/pull/5197) and scientific benchmark manifest [PR #5180](https://github.com/Luminous-Dynamics/mycelix/pull/5180). This evaluator's report is a referenced artifact, not a provenance ledger or an attestation.
- Sol-Atlas can contribute site-bound material, energy, water, transport and supplier context only when each input has a source/version, spatial and temporal scope, units and uncertainty. Spatial context alone cannot qualify a manufacturing process model.

The first integration should use the synthetic QDRIFT negative control to test that evidence class never promotes from `SimulationOutput` to `InstrumentObserved`; the first external-data work should use the already source-pinned tool-disjoint split as a separate model-validation lane. Actual production benefit requires process-level observations with the required measurement boundaries and a controlled paired pilot.

## Machine-readable report artifacts

The production-outcome and paired-inference reports are serializable via Serde and include explicit `schema_version` fields. Structured inputs reject unknown fields where strict record deserialization is defined. JSON round-trip unit tests check the stability of report structures, not authenticity. Consumers should archive the serialized report with the exact source-code revision, input manifest, configuration, and an independently computed digest of the emitted bytes. Schema version changes must be deliberate and tested before Mycelix or another downstream evidence ledger consumes the format.

## Research anchors

- [NIST Smart Manufacturing Systems Test Bed](https://www.nist.gov/laboratories/tools-instruments/smart-manufacturing-systems-sms-test-bed): integrated CAD/CAM, machine tools, inspection, MTConnect data, and lifecycle information for manufacturing evaluation.
- [NIST Digital Twins for Advanced Manufacturing](https://www.nist.gov/programs-projects/digital-twins-advanced-manufacturing): interoperability, verification/validation, and quantified model uncertainty.
- [NASA-STD-7009B](https://standards.nasa.gov/standard/NASA/NASA-STD-7009): active modeling-and-simulation credibility standard covering verification, validation, and uncertainty.

- [ISO 22400-1:2014](https://www.iso.org/cms/%20render/live/en/sites/isoorg/contents/data/standard/05/68/56847.html): a manufacturing-operations KPI concepts and terminology framework; ISO reports the edition was reviewed and confirmed in 2025. This protocol borrows the discipline of explicit KPI scope and units but does not claim ISO conformance.
- [NIST AM-Bench source-data index](https://www.nist.gov/ambench/direct-am-bench-data-links-and-referencing-guidance) lists public measurement datasets, including AMB2025-02 tensile tests of PBF-LB IN718 (DOI [10.18434/mds2-3735](https://doi.org/10.18434/mds2-3735)), AMB2025-03 high-cycle fatigue tests of PBF-LB Ti-6Al-4V (DOI [10.18434/mds2-3734](https://doi.org/10.18434/mds2-3734)), and AMB2025-06/07 laser-track calibration data (DOI [10.18434/mds2-3707](https://doi.org/10.18434/mds2-3707)). These are possible source-bound model/material validation tasks, not factory yield, energy-per-part, or throughput evidence. Compose with the existing source-bound Symthaea data work (#8033/#8079 and #8058) rather than creating a duplicate dataset adapter.

These sources motivate the evaluation design; they do not validate this code or demonstrate a production improvement. Data selection still requires inspection of the DOI landing record, licence, schema, units, provenance, dependent-observation structure, and approved train/validation/test split.

## Local verification

```bash
cargo fmt --check
cargo test -p symthaea-fabrication-kernel production_value
cargo clippy -p symthaea-fabrication-kernel --all-targets --all-features -- -D warnings
```

A passing unit-test suite qualifies the evaluator's logic only. It does not constitute a physical manufacturing trial or external validation.
