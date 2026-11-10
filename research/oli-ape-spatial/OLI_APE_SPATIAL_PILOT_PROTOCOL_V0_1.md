# OLI-APE-SPATIAL-001 — source-pinned pilot protocol

**Status:** descriptive source audit completed; reproducible Rust adapter and subject-held-out model comparison not yet implemented.

## 1. Purpose

This is the first measured non-linguistic cognition reference track for OLI. It uses a spatial/object-tracking task from the EVApeCognition dataset, not a synthetic "intelligence" benchmark.

The task: a food item is hidden under one of three containers; the platform and/or participant is displaced; the participant chooses a container. Conditions are:
- no displacement;
- platform rotation;
- participant translocation ("slide");
- combined rotation/slide;
- combined slide/rotation.

The trial record preserves initial/final location, subject choice and correctness.

## 2. Source admission

Dataset: EVApeCognition, CC BY 4.0, https://doi.org/10.5281/zenodo.18846348.

Study: Albiach-Serrano, Call & Barth (2010), *Great apes track hidden objects after changes in the objects' position and in subject's orientation*, DOI 10.1002/ajp.20790.

Exact Git blob IDs are recorded in OLI_APE_SPATIAL_SOURCE_MANIFEST_V0_1.json. The dataset concept DOI is 10.5281/zenodo.18846348. The immutable dataset DOI cited in reference 193 of the Scientific Data article and indexed by the NLM Dataset Catalog is 10.5281/zenodo.18849382. The corresponding GitHub release tag v1.0.4 was checked by fetching the YAML, data CSV, glossary CSV, domain definitions and README at that ref; all five Git blob SHAs match the source manifest. Use the version DOI for citation and the Git tag/blob SHAs for source-tree reproduction.

## 3. Data minimization

The source CSV includes participant names, age and sex. A separate source roster includes birth dates, parentage and individual histories. The OLI pilot excludes those fields. Subject grouping is retained internally through a pseudonymous token solely to avoid trial leakage and support repeated-measures evaluation. The token-to-name mapping is not part of the public artifact.

Species is retained only as a coarse descriptive label; no species ranking is permitted. Do not ingest the separate ape roster.

## 4. Descriptive audit

The source file has 375 rows, 25 subjects, 5 conditions and 3 trials per subject in each condition. No malformed outcome was found. Subject-cluster bootstrap intervals are recorded in OLI_APE_SPATIAL_DESCRIPTIVE_AUDIT_V0_1.json.

The observed ordering matches the paper's descriptive interpretation: no displacement is easiest; single displacements are harder; combined displacements are hardest. This is one study and one task, not a universal cognition measure.

## 5. Benchmark baselines

1. Random choice (1/3 when choices are equiprobable).
2. Condition-independent outcome baseline.
3. Condition-only success-rate model.
4. Regularized logistic model with condition and block/trial context.
5. Hierarchical binomial model with subject-level grouping, if the small sample supports stable estimation.
6. Symthaea challenger with feature/representation ablations.

Report per-condition log loss, Brier score, calibration, accuracy, per-subject error and uncertainty. Do not collapse into a species-level IQ score.

## 6. Split law

Primary protocol: leave-one-subject-out. All trials from one subject remain together in the held-out fold. No trial-level random split. Model selection is nested within the training subjects. Because the pilot has only one study, no cross-study or cross-lab generalization claim is possible.

## 7. Acceptance gates

- exact Zenodo version DOI / release tag pinned;
- source blobs verified;
- CC BY 4.0 attribution preserved;
- raw identity/demographic fields omitted from public outputs;
- exact row/group validation;
- LOSO comparison against at least three baselines;
- deterministic preprocessing and result manifests;
- no cross-species ranking, consciousness or universal intelligence claim.

## 8. Current status

Only the source audit and descriptive subject-cluster bootstrap have been executed. Rust ingestion, model baselines, LOSO evaluation, calibration and independent replication are pending. No benchmark performance result is claimed.
