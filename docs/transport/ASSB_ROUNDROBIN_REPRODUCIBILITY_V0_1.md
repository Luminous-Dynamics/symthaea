# ASSB round-robin reproducibility and fixture metrology v0.1

Status: protocol/data-contract and synthetic audit scaffold. It is not a physical test method, cell assembly instruction, safety protocol or statistical certification.

## Purpose

Make every ASSB performance claim carry a **reproducibility envelope**: exact material and cell lineage, laboratory, fixture, pressure calibration, protocol, raw data, rights, replicates and uncertainty.

This composes ASSB practical-cell translation #11759 / PR #11770, MAT-013 sample lineage, ION-001 transport evidence, MAT-INTERFACE and MAT-UQ. It does not create another battery material model or cell simulator.

## Evidence motivating the programme

- A 2024 *Nature Energy* interlaboratory study supplied common ASSB materials to **21 research groups**. Groups followed a common electrochemical protocol but used their own cell assembly procedures. The study found large variation in processing and electrochemical outcomes, including differences in processing pressure, pressing duration and In-to-Li ratio. It advocated standardized parameter reporting and triplicate data.
  - https://www.nature.com/articles/s41560-024-01634-3
  - Open manuscript: https://epub.uni-bayreuth.de/id/eprint/8685/1/s41560-024-01634-3.pdf
- A 2026 study of solid-state battery test fixtures proposes engineering targets for pressure homogeneity, setpoint accuracy, precision and available pressure range. The authors state that quantitative fixture benchmarks were not yet established and that their values are based on engineering judgement.
  - Daub (2026), DOI 10.1002/ese3.70581: https://doi.org/10.1002/ese3.70581
  - Open Fraunhofer manuscript: https://publica-rest.fraunhofer.de/server/api/core/bitstreams/558ed906-7c24-4d7e-b21c-980e741c93bc/content
- 2026 low-pressure ASSB research calls for standardized low-pressure benchmarking and operando chemo-mechanical diagnostics. Reported pressure, temperature, areal capacity, format and current density often vary together, making direct comparison difficult.
  - https://www.nature.com/articles/s41467-026-79393-w
  - https://doi.org/10.1016/j.jechem.2026.08.036

## Round-robin design

### 1. Freeze the reference

Freeze one exact architecture and material-lot set for the common-protocol arm. Record cell design/revision, electrolyte/cathode/anode lots, manufacturing process revision, cell format and test level. Use blinded sample codes that do not encode the laboratory or expected outcome.

### 2. Qualify the fixtures before interpreting electrochemistry

Each fixture records:
- pressure range and separate fabrication/operating-pressure use;
- pressure measurement method and instrument identity;
- calibration evidence and uncertainty;
- spatial pressure homogeneity;
- setpoint accuracy;
- repeatability;
- fixture revision and owning laboratory.

The 2026 paper's candidate engineering suggestions—pressure homogeneity <5%, setpoint MAPE <5%, pressure CV <5%, and a range spanning ±50% around the target—are **not formal standards**. Preserve the raw measured values and their metric definitions. Do not turn these numbers into universal pass/fail gates.

### 3. Separate three arms

- **COMMON_PROTOCOL:** same design, material lot, process revision, frozen protocol revision, test-condition fingerprint and pressure setpoint across laboratories.
- **LAB_NATIVE:** local protocol retained for descriptive within-lab comparison; it cannot silently join a common-protocol pool.
- **PROTOCOL_DEVIATION:** explicitly named protocol deviations; retain as separate evidence, never pool as if identical.

### 4. Replicate honestly

A replicate group counts distinct physical cell IDs, not repeated measurements of one cell. Triplicate cells per lab/condition is the minimum reporting floor. It is not a guarantee of statistical power; final cell counts are determined by prospective variance estimates and power analysis.

A group with fewer than three labs or fewer than three independent cells per lab is descriptive-only. At least three laboratories and triplicate independent cells per laboratory qualify a group for **candidate cross-lab variance analysis**, not for a claim that performance is reproducible.

### 5. Release a reproducibility envelope

A rights-compatible round-robin dataset should include raw current/voltage/time traces, pressure/force/displacement/time, temperature, sensor locations, calibration artifacts, fixture metadata, cell lineage, protocol version, deviations, all failed/invalid/incomplete/censored results and exclusions. Public lab labels may be pseudonymized, but the accountable custodian must retain the identity mapping for audit.

## Software contract v0.1

The read-only Rust auditor checks:
- exact lab, fixture, cell, blinded sample and evidence identities;
- fixture-to-laboratory ownership and calibration evidence;
- physical versus synthetic evidence classes;
- raw trace digests and data-rights references;
- distinct cell IDs for independent replicates within a group;
- exact design, material lot, process, protocol, condition and setpoint matching for COMMON_PROTOCOL;
- LAB_NATIVE and PROTOCOL_DEVIATION non-pooling;
- group counts by labs and independent cells per lab;
- EOL, right-censored, failed, invalid, incomplete and synthetic status preservation;
- deterministic report output under input permutation.

It does not calculate electrochemical performance, pool statistical results, certify pressure fixtures, establish interlaboratory reproducibility or infer cell safety.

## Benchmark outputs

For each group, report its mode, observation count, laboratory count, minimum independent cells per laboratory, and comparability state:
- KNOWN_ANSWER_FIXTURE_ONLY;
- DESCRIPTIVE_ONLY;
- WITHIN_LAB_ONLY;
- CANDIDATE_CROSS_LAB_VARIANCE_ANALYSIS.

A future statistical phase may report within-lab repeatability, between-lab reproducibility, variance components, ICC where appropriate, confidence intervals, survival/censoring analysis and protocol/batch effects. That phase must use a preregistered analysis plan and must never hide negative or invalid records.

## Safety boundary

No synthesis recipe, cell build procedure, pressure-test instruction, sulfide handling method, H₂S control method, equipment operating instructions or physical experiment authorization is included. Qualified laboratories own all physical work under institutional and regulatory controls.

## Claim ceiling

A software PASS establishes only identity, provenance, replicate-count, grouping and input-gate semantics on the authored synthetic corpus. It does not establish fixture accuracy, pressure uniformity, cell performance, electrochemical safety or physical reproducibility.
