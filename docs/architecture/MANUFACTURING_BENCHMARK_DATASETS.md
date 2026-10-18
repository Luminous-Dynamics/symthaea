# Manufacturing Benchmark Dataset Registry (Draft)

**Status:** Source registry and intake contract; not a claim that datasets have been downloaded, normalized, or benchmarked.  
**Owner:** Symthaea manufacturing workstream  
**Last reviewed:** 2026-10-18  
**Companion:** [Manufacturing Benchmark Protocol](MANUFACTURING_BENCHMARK_PROTOCOL.md)

## 1. Rules for admitting an external dataset

A source is a candidate until the exact artifact and usage terms have been reviewed. A URL or DOI is not a content digest; a digest identifies bytes but does not prove their truth or engineering suitability.

For every imported artifact, record:
- stable dataset ID, source organization, DOI or permanent locator, exact artifact URL, version/revision, retrieval timestamp, and licensing/attribution conditions;
- SHA-256 of the exact downloaded artifact, plus hashes of normalized outputs and transformation source;
- data dictionary, units, coordinate/frame conventions, timestamps/time zones, machine/part/material identities, and missing-value semantics;
- evidence class: public_measured_reference, public_operational_stream, published_simulation_instance, synthetic_fixture, or scenario_assumption;
- permitted benchmark families, prohibited inferences, inclusion/exclusion decisions, and independent review;
- train/calibration/test assignment before model development; split by entire part, material lot, procedure family, job, or time window whenever row-level splitting would leak information.

Never silently replace a dataset revision behind a stable case ID. New bytes or parsing logic create a new dataset revision and invalidate prior exact-head benchmark receipts. Dynamic endpoints must be captured as dated, content-addressed snapshots before using them in deterministic evaluation.

## 2. Candidate sources

| Dataset ID | Source and locator | Best-fit use | Boundaries and intake requirements |
|---|---|---|---|
| NIST-SMS-TDP-MTC | NIST SMS Test Bed Technical Data Package: https://smstestbed.nist.gov/tdp/mtc . Data catalog record: https://catalog.data.gov/dataset/design-manufacturing-and-inspection-data-for-a-box-assembly | C1 digital thread: connect design/CAD, NC code, MTConnect machine data, and QIF/first-article inspection across part instances. Useful for trace-link and schema-conformance fixtures. | Public catalog lists the NIST Open license and an August 2018 update date; still review the current artifact and attribution terms. Historical data is not a live-machine safety model, and a small assembly dataset is not a shipyard-wide capacity ground truth. |
| NIST-SMS-VDS | NIST SMS Test Bed data streams: https://smstestbed.nist.gov/vds/probe ; sample: https://smstestbed.nist.gov/vds/sample ; repository: https://smstestbed.nist.gov/qdr | C1 time-series intake and the temporal alignment between NC instructions and MTConnect controller observations; adapter and data-quality tests. | VDS is dynamic. Capture response bytes, observation time, agent/device identity, schema version and digest; do not use a live endpoint as a reproducible historical fixture without freezing a snapshot. A machine signal alone does not identify the part, route, or accepted quality result. |
| NIST-AMBENCH-2025-IN718-TENSILE | AMB2025-02 PBF-LB IN718 quasi-static tensile measurements, DOI: https://doi.org/10.18434/mds2-3735 | M1/C1 model-validation pathway for metal additive manufacturing material/process-property relationships, uncertainty reporting and calibration-vs-held-out comparisons. | Additive-manufacturing evidence, not welding-procedure qualification or CNC cutting ground truth. Pin the exact PDR files and metadata; preserve specimen, orientation, heat-treatment and measurement details. |
| NIST-AMBENCH-2025-TI64-FATIGUE | AMB2025-03 PBF-LB Ti-6Al-4V high-cycle rotating-bending fatigue measurements, DOI: https://doi.org/10.18434/mds2-3734 | Material/process model validation, fatigue-property prediction baselines, uncertainty and held-out evaluation. | Do not generalize the measured population to welded joints, other heat treatments, surface conditions or service environments without separate evidence. Pin dataset files, test conditions and specimen identifiers. |
| NIST-AMBENCH-2025-718-LASER-PADS | AMB2025-06/07 Alloy 718 laser-track/pad calibration data, DOI: https://doi.org/10.18434/mds2-3707 | Process-physics calibration exercises; sensitivity to laser-track turnaround time and powder-layer condition where those metadata apply. | Calibration data cannot also serve as a blind held-out test for a model trained on it. Validate the specific challenge description, measurements, and split before importing. |
| SHIP-BLOCK-SCHEDULING-2026 | Dataset of Ship Block Scheduling, Mendeley Data v1 (published 2026-03-03), DOI: https://doi.org/10.17632/4ds6t24mx2.1 | S1 scheduling-algorithm reproduction, multi-objective comparisons and stress scenarios around berth/block assignment. | Treat it as a published research dataset. Independently inspect raw inputs, assumptions, result files, parameter definitions and reproduction fidelity. Algorithm output is not proof of physical feasibility or industrial representativeness. |
| TUDELFT-RCPSP-MODULAR-PRODUCTION | Instances and file format for the Resource Constrained Project Scheduling Problem with Modular Production, DOI: https://doi.org/10.4121/21774530 | S1/SC2 solver validation on reproducible resource-constrained project scheduling instances; baseline comparison against exact/heuristic methods where practical. | Published scheduling instances, not measured factory or yard observations. Preserve the source file format and benchmark identity; do not map abstract resources to qualified real equipment without an independent capability model. |

## 3. First intake tranche

1. **Digital-thread fixture first:** inspect the NIST SMS assembly package. Build a small manifest that links one part instance to its NC program, machine/controller interval, inspection record and relevant design revision. Preserve every source key and record when a join is ambiguous.
2. **Scheduling benchmark second:** acquire the TU Delft RCPSP instances and the ship-block scheduling dataset. Reproduce source baselines, then test the Symtropy validator on converted schedules without treating infeasible schedules as valid just because a source paper reports an objective score.
3. **Material-model pathway third:** pin one AM-Bench property dataset with its measurement metadata and a predeclared train/calibration/test split. Begin with a transparent baseline and uncertainty metrics before adding HDC/FEP-based predictors.
4. **Cross-domain transfer last:** test whether a representation or tool improves on a held-out family. Never count success on an additive dataset as welding, shipbuilding or physical qualification evidence.

## 4. Acceptance checks for every intake

- Source locator and version resolve to the expected artifact; license and attribution are recorded.
- SHA-256 verifies after download; bytes are immutable once the case manifest is committed.
- Units and coordinate/time conventions are explicit; no silent coercion of missing values or units.
- Every normalized row retains a source-record locator and transformation revision.
- Duplicate identifiers, conflicting revisions, non-finite values, out-of-range quantities and broken links are rejected or surfaced as explicit unresolved issues.
- A negative control intentionally corrupts a digest, revision, unit, or key link and must cause intake/validation to fail closed.
- The expected result, baseline command, runtime profile and output digest are pinned before benchmark evaluation.
- No result is described as measured, certified, production-ready or an engineering approval unless its evidence level actually supports that statement.

## 5. Benchmark scorecard

For each dataset, publish counts of accepted/rejected/unresolved records, lineage completeness, parse and join coverage, unit-validation failures, duplicate/conflicting identities, deterministic normalized-output digest, and reproducibility status. Predictive evaluations additionally report baseline-relative error, worst-case error, per-case residuals, sample count and uncertainty calibration. Scheduling evaluations additionally report feasibility violations, makespan/objective values, resource conflicts, and gap to known optima or source baselines when those are available.

**Initial gate:** zero corrupted-evidence false accepts in the negative-control set; 100% provenance linkage for records included in an eligible benchmark case; and no silent unit/missing-value coercion. Model-quality and productivity claims remain unscored until the exact data split, metric, acceptance band and baseline are declared.
