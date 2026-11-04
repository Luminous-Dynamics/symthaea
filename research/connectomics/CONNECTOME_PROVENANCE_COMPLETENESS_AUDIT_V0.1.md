# Connectome Provenance-Completeness Audit V0.1

**Status:** research protocol; empirical and statistical qualification required  
**Issue:** [#10414 — SCI-CONNECTOMICS-072](https://github.com/Luminous-Dynamics/symthaea/issues/10414)  
**Parent:** [#10404 — Evidence dependence and false corroboration](https://github.com/Luminous-Dynamics/symthaea/issues/10404)  
**Date:** 2026-11-04  
**Scope:** independently sampled audit of claimed-independent evidence pairs

## 1. Purpose

A structurally valid provenance bundle can be incomplete. If a producer omits a shared image, segmentation revision, training corpus, preprocessing operation, registration transform, annotation history, reviewer exposure, or selection mechanism, a structural verifier cannot discover the omitted relation from the submitted bundle alone.

V0.1 defines an independent, finite-population audit of the independence assertions actually used by a declared scientific claim/fusion population. It is intentionally separate from:
- structural verification of declared provenance (#10406);
- probability sampling of unreviewed proofreading candidates (#9187);
- multi-rater agreement and reviewer calibration (#9186);
- sensitivity analysis for errors missed by discovery channels (#9248).

The audit estimates how often producer-declared independence is contradicted or unresolved within a frozen assertion frame and named source-of-record scope. It does not establish universal independence.

## 2. Provenance vocabulary and epistemic boundary

Use the W3C PROV conceptual distinction among entities, activities, agents, and derivation/influence relations as an interoperability foundation. A provenance statement remains an assertion about production history; it is not self-authenticating evidence that the account is complete.

The connectomics extension must capture, where applicable:
- source microscopy entities, acquisition activity and custodian;
- image rendering, crop, preprocessing and registration activities;
- segmentation and annotation revisions;
- model artifacts, training/fine-tuning corpora, and inference activities;
- reviewer/adjudicator agents and exposure activities;
- selection and candidate-generation activities;
- claim and query bundles generated from the evidence.

A digest establishes identity of bytes, not completeness of their history. A signature establishes signer authenticity under a trust policy, not the truth of every provenance assertion.

## 3. Audit estimand

Let the frozen finite frame contain N_h claimed-independent evidence-pair assertions in stratum h. Let Y_i=1 if an independent source-of-record audit finds a dependence that contradicts the assertion's declared independence scope; otherwise Y_i=0 only when the required audit records were accessible and the protocol found no such contradiction.

Unresolved source access, ambiguous lineage, and protocol violations are separate outcomes. They must not be silently dropped or coded as Y_i=0.

The principal estimands are:
1. finite-population contradiction count and proportion by predeclared stratum;
2. conservative contradiction bound when unresolved/ambiguous sampled units are counted as contradictions;
3. audit-source accessibility and unresolved fraction;
4. the fraction of claim/fusion assertions that remain eligible under a declared query-specific risk budget.

The interpretation of a zero finding is strictly bounded: **no shared ancestor was found within the named audited scope**. It is not a proof of universal statistical or biological independence.

## 4. Frozen finite-population frame

The frame consists only of evidence-pair assertions that an actual downstream claim or fusion procedure relies upon. Do not construct an unbounded all-pairs Cartesian product of every artifact in a dataset.

Each frame record binds:
- unique assertion ID;
- left and right evidence IDs;
- claim/query/fusion-operation IDs;
- evidence-dependence bundle digest and fusion-policy digest;
- declared independence protocol and scope;
- source/target release and evidence revision identities;
- query materiality and error-family strata;
- producer/model/training-family identifiers where known;
- source-record custodian and audit-access profile;
- frame inclusion status and reason;
- dependency graph revision and exact provenance references.

The frame manifest also binds:
- complete record count and canonical ordering;
- schema and frame-builder versions;
- stratum definitions and frame counts;
- pair-selection rules and exclusions;
- source revisions;
- frame digest and a commitment to the audit-seed procedure.

The frame must be frozen before audit selection and before audit outcomes are disclosed to claim producers. Excluded or inaccessible pairs remain represented with explicit status and reason; they do not disappear from the denominator.

## 5. V1 probability-sampling design

Use stratified simple random sampling without replacement (SRSWOR) as the first supported design. Every eligible frame unit must have a known, non-zero inclusion probability. Oversampling high-materiality or weak-provenance strata is allowed only with exact recorded inclusion probabilities and design weights.

Potential predeclared strata:
- query materiality / decision-loss tier;
- evidence-role pair (model/model, model/reviewer, reviewer/reviewer, modality/modality, pipeline/pipeline);
- provenance completeness class;
- producer/model/training lineage family;
- acquisition volume/session and annotation lineage;
- dependence assertion type.

Do not treat assertion pairs as IID merely because they are distinct rows. If shared models, source volumes, acquisition sessions, operations or review panels induce clustering that the design does not support, the report must be NOT_ESTIMABLE for the unsupported design or use a separately preregistered cluster-aware design.

Use exact hypergeometric finite-population bounds for binary contradiction outcomes under supported stratified SRSWOR. Allocate the error budget simultaneously over preregistered critical strata and query families. Repeated interim looks require a separately justified sequential error-spending or confidence-sequence design; unadjusted repeated 95% intervals are not acceptable.

## 6. Independent source-of-record examination

The auditor must be independent of the evidence producer and must not treat the producer's submitted provenance graph as the sole source of truth.

Depending on evidence type, inspect:
- acquisition and dataset-custodian manifests;
- raw image/volume identifiers and rendering logs;
- registration and coordinate-transform records;
- segmentation and annotation history;
- model training/fine-tuning data manifests and preprocessing configuration;
- experiment/training run records and checkpoint lineage;
- reviewer assignment, blinding, UI exposure and suggestion logs;
- candidate-selection and ranking records;
- source-controlled code/configuration and build attestations.

For each sampled assertion, record the exact sources examined, source revision/digest or privacy-preserving reference, access outcome, custodian, audit timestamp, and adjudicator disposition.

An inaccessible source is unresolved, not independent. If a source custodian cannot distinguish two repackaged dataset exports, retain that ambiguity.

## 7. Outcome vocabulary

Each sampled assertion receives exactly one primary disposition:

- CONTRADICTED_DEPENDENCE: an independently recovered shared ancestor/exposure contradicts the asserted independence scope.
- NO_SHARED_ANCESTOR_FOUND_WITHIN_AUDITED_SCOPE: all mandatory source classes for this protocol were accessible and no contradiction was found within scope.
- UNRESOLVED_SOURCE_ACCESS: one or more mandatory source records were inaccessible or incomplete.
- AMBIGUOUS_LINEAGE: competing ancestry interpretations remain.
- PROTOCOL_VIOLATION: blinding, sampling, source separation, custody or audit procedure was breached.

Keep supplementary finding codes for known shared raw images, segmentation, annotation, training examples, preprocessing, registration, acquisition session, reviewer exposure, selection mechanism, and other common ancestors.

For promotion accounting, unresolved, ambiguous and protocol-violating assertions are not independent. They remain in the sampled denominator and contribute to a separate conservative bound.

## 8. Statistical reporting

For every stratum and query family, report:
- frame population N_h;
- sample size n_h;
- exact first-order inclusion probability;
- design weights and any finite-population correction;
- counts of each outcome category;
- design-based contradiction estimate and exact one-sided upper bound;
- conservative upper bound treating unresolved/ambiguous outcomes as contradictions;
- multiplicity allocation and cumulative error budget;
- source-of-record coverage and missingness;
- exclusions, nonresponse, deviations, and cluster treatment;
- whether the declared risk budget is met.

A query-specific promotion requires the simultaneous upper bound on contradiction prevalence to be below its preregistered risk budget in every critical stratum, with minimum support and source-access completeness satisfied. A pooled aggregate cannot hide a failing high-impact stratum.

If a design does not support a valid bound, return NOT_ESTIMABLE. Do not substitute a naive IID binomial interval.

## 9. Challenge corpus

The synthetic/semi-synthetic challenge corpus must contain:
1. Complete provenance with shared ancestors correctly declared.
2. Omitted common raw EM volume in apparently distinct model lineages.
3. Omitted shared segmentation checkpoint.
4. Shared training examples hidden behind repackaged dataset IDs.
5. Shared preprocessing or registration artifact.
6. Shared annotations disguised as separate exports.
7. Reviewer suggestion exposure omitted from the provenance manifest.
8. Candidate selection derived from another ranker but declared random.
9. Inaccessible training records, which must become unresolved.
10. Source revision changed after frame freeze.
11. Selective producer exclusion of difficult pairs before frame commitment.
12. Audit seed chosen after pair identities or outcomes are exposed.
13. Oversampling with incorrect inclusion probabilities.
14. Clustered assertions analyzed with invalid IID assumptions.
15. A common ancestor discoverable only through an independent source-of-record.

Evaluator-only truth and injected hidden dependencies must remain separate from runtime audit inputs. The operator's audit report is frozen before evaluator truth is revealed.

## 10. Comparator matrix

Evaluate against identical finite frames and audit-source availability:
- self-declared provenance structural validation only (#10406);
- census audit of all pairs where feasible;
- simple random sample without stratification;
- preregistered stratified SRSWOR;
- producer-selected sample (negative control);
- naive IID binomial interval (negative control);
- exact design-based finite-population bounds;
- independent source-of-record audit by an external custodian.

Report cost, source-access coverage, contradiction detection, unresolved outcomes, bound width and false-promotion behavior. A zero-risk result obtained by refusing every assertion is not useful; coverage and audit cost are required endpoints.

## 11. Proposed gates (design goals, not results)

- 100% rejection/downgrade of seeded hidden dependencies when the relevant source records are available to the auditor.
- Zero promotion of inaccessible, ambiguous or unresolved ancestry as independent.
- Exact finite-population coverage under exhaustive simulation of the supported SRSWOR design.
- Zero false STOP / promote dispositions in enumerated adversarial sampling controls.
- Simultaneous upper bounds reported for every critical query stratum.
- Lower audit cost than census at the same preregistered upper-bound target on sufficiently large finite frames.
- Natural-data claims bounded to the sampled assertion population and named source-of-record coverage.

## 12. Integration

- #10406 validates the structure and declared semantics of the supplied dependence graph.
- #10414 defines the independent audit of whether claimed-independent assertions are contradicted by source-of-record lineage.
- #9187 samples unreviewed proofreading candidates; do not conflate its error estimand with provenance-assertion contradiction.
- #9186 preserves reviewer disagreement and measures review reliability.
- #9248 estimates errors potentially missed by all discovery channels; it is not an estimator of provenance completeness.
- #9462 may supply sequential finite-population methods only where its sampling assumptions match this design.
- #9255 consumes audit results as narrowly scoped evidence-dependence qualifications.

## 13. Claim ceiling

This protocol can support estimates of contradiction and unresolved lineage within a frozen finite assertion frame and named source-of-record audit scope. It cannot prove the absence of all unrecorded common causes, universal statistical independence, or biological truth. The output is a bounded audit result, never a universal independence certificate.
