# PLASMA-MSA-001 — decision-qualified measurement-system capability

**Date:** 2026-11-14  
**Issue:** [#12188](https://github.com/Luminous-Dynamics/symthaea/issues/12188)  
**Status:** S0 research contract and known-answer benchmark  
**Branch:** `research/plasma-measurement-capability-v0-1`

## Executive thesis

A measurement is not decision-grade merely because it has a test-method label, a calibration certificate, or a valid raw-data digest. The Bond Qualification Passport must also establish that the complete measurement system can resolve the engineering decision being made.

This adds a missing qualification layer between method-scoped observations and transfer decisions:

`method validity` + `measurement-system capability` + `specimen/process evidence`
→ `decision-qualified observation`

The key paradigm shift is **qualify the ruler before trusting the result**.

## Why this matters now

The plasma programme already has distinct work on:
- exact method and claim scope (#12028, #12074);
- method-qualified evidence composition (#12051);
- cluster-aware equivalence design (#12070);
- full transfer composition (#12148);
- direct and derived endpoint typing (#12177);
- surface recovery, failure-mode composition and Passport currentness.

But a local instrument/fixture/operator/laboratory can be poorly suited to the decision even when the method is valid and calibration is nominally current. If measurement uncertainty is comparable to the equivalence margin, a transfer audit must not accept apparent similarity as evidence of equivalence.

## Research anchors and limits

### ISO 5725-2:2025

The ISO catalogue describes collaborative interlaboratory experiments for estimating repeatability and reproducibility of continuous measurement methods. It assumes a standardized method in routine use and a balanced uniform-level design. It is a precision-study framework, not a universal pass/fail rule for every adhesive joint or laboratory.

Source: https://committee.iso.org/cms/live/live/en/sites/isoorg/contents/data/standard/09/00/90054.html

### ISO 4587:2003

ISO lists ISO 4587:2003 as the published edition for tensile lap-shear strength of rigid-to-rigid bonded assemblies, with systematic review status in 2026. Its scope says the test does not provide design information. A method citation therefore does not establish design allowables or local measurement-system capability.

Source: https://www.iso.org/standard/34852.html

### Interlaboratory variability

Published work on bond-strength testing reports large repeatability and reproducibility dispersion in some tested systems, including coefficients of variation reaching tens of percent. The result is material- and method-specific, but it illustrates why the measurement system can dominate the decision margin.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC7040815/

### Current plasma comparator

The October 2026 PP/HIPS/PVC/PC/PET study reports three specimens per treatment category and demonstrates material/adhesive-dependent bond response. It is useful for identifying factors and designing a follow-up study, but it is not an interlaboratory capability study or a universal sample-size prescription.

Source: https://link.springer.com/article/10.1007/s44245-026-00396-4

## S0 evidence contract

A capability receipt must be scoped to one measurand and configuration. Minimum fields:

1. **Measurand:** endpoint, unit, method/edition, specimen geometry, material family and operating range.
2. **Equipment:** instrument identity, load cell/range, resolution, fixture/grip identity, firmware/software and analysis-pipeline digests.
3. **Calibration:** certificate identity, traceability chain, calibration validity, as-found/as-left status and relevant uncertainty.
4. **People and place:** operator identities/qualification state, laboratory/site, environmental conditions and date range.
5. **Study design:** crossed/nested structure, replication, independent units, randomization, reference artefacts and specimen allocation.
6. **Variance decomposition:** repeatability, operator, instrument, fixture, site/laboratory and specimen/lot effects, only where identifiable.
7. **Uncertainty:** bias, linearity, stability, reproducibility and combined/expanded uncertainty as applicable.
8. **Decision fitness:** uncertainty-to-margin ratio against the exact predeclared equivalence or acceptance margin.
9. **Lineage:** raw observations, exclusions, outlier policy, analysis version, source digests and specimen/cluster genealogy.
10. **Validity lifecycle:** expiry and invalidation events including calibration failure, drift, maintenance, relocation, fixture changes, software changes and operator qualification lapse.

Calibration traceability is necessary where applicable, but not sufficient. A published method precision is not a local instrument qualification.

## Decision policy

The S0 fixture uses a deliberately conservative illustrative policy:
- PASS only when method, edition, measurand, geometry and material scope match; calibration is current and traceable; the study design identifies the declared effects; no invalidation event exists; and the expanded uncertainty is no more than 25% of the preregistered decision margin.
- BLOCKED when evidence is missing, the study is underidentified, calibration is expired, the receipt is stale, or uncertainty is too large to resolve the decision.
- REJECTED when the receipt is internally contradictory, the source digest is malformed, the configuration demonstrably mismatches the claimed scope, or a forbidden inference is requested.

The 25% ratio is an S0 research target for this fixture, **not an ISO requirement or universal metrology rule**. The production policy must be risk- and endpoint-specific and approved by the responsible quality authority. Do not substitute generic %GRR cutoffs.

## Frozen synthetic fixture

File: `data/research/plasma_measurement_capability_fixture_v0_1.json`

The known-answer cases exercise:
- valid local capability with adequate decision resolution;
- expired/missing calibration and missing traceability;
- instrument, fixture, software and method-edition mismatches;
- wrong measurand, material family, geometry or operating range;
- non-identifiable operator/site/instrument effects;
- uncertainty above the decision-fit target;
- contradictory validity dates and invalidation events;
- malformed or duplicated source digests;
- attempts to reuse one capability receipt outside its scoped configuration.

The fixture is a contract test, not an empirical gauge R&R study.

## Physical promotion plan

1. Freeze one material–adhesive–joint geometry and one primary destructive endpoint.
2. Freeze method edition, specimen fabrication, bondline controls, conditioning, loading rate, fixture and failure classification.
3. Perform a pilot to estimate specimen heterogeneity and within-run repeatability.
4. Design a balanced crossed or justified nested study across operators and instruments; destructive specimens require independent specimen allocation, not fictitious repeated measures.
5. Conduct a blinded interlaboratory round robin with randomized specimen allocation, reference artefacts and complete raw-data lineage.
6. Estimate variance components and uncertainty using the model justified by the actual design; do not pool confounded effects.
7. Compare the resulting uncertainty with the actual equivalence/acceptance margin before spending on large process-transfer campaigns.

## Benchmarks

Prospective goals:
- 100% detection of stale, scope-mismatched and invalidated capability receipts in known-answer negatives;
- zero false PASS when uncertainty cannot resolve the declared decision margin;
- correct separation of repeatability, operator, instrument, fixture, laboratory and specimen effects when identifiable;
- empirical coverage of uncertainty intervals on independent reference artefacts and sites;
- report false acceptance, false rejection, abstention, uncertainty-to-margin ratio and cost/time per decision;
- no universal %GRR pass criterion detached from endpoint risk.

## Integration contracts

- Transfer (#12002): a transfer decision cannot PASS without a current capability receipt for each decisive endpoint.
- Method evidence (#12033/#12051): direct observations bind to the capability receipt matching method, edition, measurand, geometry and equipment configuration.
- Equivalence design (#12064/#12070): the decision margin and uncertainty budget are frozen together before observing candidate outcomes.
- Derived endpoints (#12166/#12177): propagate input measurement uncertainty; do not treat source observations as exact.
- Passport lifecycle (#12052): calibration, drift, maintenance, fixture, software, relocation and operator events invalidate affected receipts.

This workstream owns the measurement-system evidence contract and decision-fitness gate only. It does not implement a new metrology engine, variance-component estimator, universal statistical library, laboratory information system or product-release authority.

## Claim ceiling

S0 establishes only that the synthetic evidence contract can distinguish supported, blocked and rejected receipts. It does not establish capability of any physical instrument, lab, operator, test method or plasma-bonding process. No physical performance, transfer equivalence, standards compliance, design allowable or product release is claimed.
