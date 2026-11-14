# PLASMA-ENDPOINT-TRANSFORM-001 — typed endpoint evidence graphs

**Date:** 2026-11-14  
**Issue:** [#12112](https://github.com/Luminous-Dynamics/symthaea/issues/12112)  
**Stack:** #12028 → #12051 → #12070 → #12074 → this branch  
**Statistics dependency:** [#12117](https://github.com/Luminous-Dynamics/symthaea/issues/12117)

## Executive finding

The transfer programme's three required outputs are not all direct measurements:

1. initial_lap_shear_mpa is derived from direct joint-level apparent-shear observations.
2. aged_strength_retention is derived from distinct unaged and aged strength cohorts plus a qualified conditioning/exposure record.
3. failure_mode_fraction is derived from a population of failure-mode classifications plus a frozen taxonomy and a suitable cluster-aware proportion analysis.

Treating all three as if they were the same D3163 measurement would be a scientific category error. A generic “bond quality” scalar would hide which evidence supports each component and allow an excellent wetting result, constitutive curve or failure label to mask a poor joint-strength result.

This increment introduces a typed endpoint evidence graph: method-qualified observation receipts → registered transform and source roles → preregistered design/result receipt → bounded derived claim. It is an S0 evidence contract, not a numerical transformation engine.

## Four non-interchangeable evidence planes

| Evidence plane | What it observes | What it cannot establish by itself |
|---|---|---|
| Surface state / wettability | Contact angle, surface chemistry and activation state | Joint strength or durability |
| Adhesive constitutive response | Adhesive shear stress–strain response | Adhesion to a particular substrate or joint-level strength |
| Joint performance | Apparent shear / fracture / fatigue response for a specific assembly | Universal design allowables or service life |
| Failure-mode classification | Descriptive failure-surface category | Whether a failure mode is acceptable |

ASTM D5656/D5656M-26 is particularly important: the standard says it characterizes adhesive shear stress–strain behavior, proportional limit and shear strain at failure; it is not intended to determine adhesion to a particular substrate, and its validity for non-aluminum adherends cannot be assumed. D5573-99R19 classifies failure modes in the scoped SMC FRP joint family and explicitly does not determine acceptability. ASTM D3163 and D5868 remain joint-level comparative methods; their results must not be substituted by D5656 constitutive data or by D5573 classification.

Official sources:
- D5656/D5656M-26: https://store.astm.org/d5656_d5656m-26.html
- D5573-99R19: https://store.astm.org/standards/d5573
- D3163-01R23: https://store.astm.org/standards/d3163
- D5868-01R23: https://store.astm.org/standards/d5868

A recent 2026 five-polymer atmospheric-plasma study compared argon and nitrogen treatment across PP, HIPS, PVC, PC and PET. Its useful strategic point is that a lower water contact angle did not consistently produce a stronger adhesive joint: polymer, gas, adhesive and treatment severity interact. That is why the graph keeps wettability and downstream joint strength as separate claims.

Research source: https://link.springer.com/article/10.1007/s44245-026-00396-4

## Implemented transformation registry

Code: xtask/src/plasma_endpoint_transform.rs  
Fixture: data/research/plasma_endpoint_transform_fixture_v0_1.json  
Command:

    cargo run -p xtask -- plasma-endpoint-transform-audit

### 1. Direct initial joint strength

Transform: unit_preserving_strength_projection_v1  
Formula ID: identity_strength_unit_projection_v1  
Output: initial_lap_shear_mpa

The graph requires baseline and candidate D3163 or D5868 apparent-shear observations, exact method/edition/material/geometry/endpoint scope, distinct specimens, disjoint independent manufacturing clusters, and a design receipt whose raw observation endpoint is apparent_shear_strength.

D5656 constitutive-response data and water-contact-angle data are rejected as substitutes. A valid graph can still have an interval that does not establish equivalence; that remains a valid study with a negative outcome, not a positive qualification.

### 2. Aged-strength retention

Transform: aged_strength_retention_ratio_v1  
Formula ID: mean_aged_strength_over_mean_unaged_strength_v1  
Output: aged_strength_retention

The graph requires four distinct strength observation cohorts:
- baseline unaged;
- baseline aged;
- candidate unaged;
- candidate aged.

Each cohort binds specimen IDs to manufacturing clusters. Baseline aged/unaged cohorts must cover the same clusters; candidate aged/unaged cohorts must cover the same clusters; baseline and candidate clusters must be disjoint. Specimen IDs and raw-data artifacts cannot be reused across the four measurements.

Each aged cohort also binds a conditioning-method receipt (D1151 or D1183), exposure receipt ID/digest, protocol identity/digest, condition digest, temperature, humidity and duration. Baseline and candidate aging must use the same declared exposure conditions and method but distinct exposure records and raw artifacts. The conditioning-method receipt is included in the result's source-receipt lineage.

The formula identifier is frozen and versioned. S0 checks the formula identity and source graph; it does not execute the ratio calculation, select the denominator convention, fit a survival model or prove the conditioning protocol was physically performed. A production-quality transform will need the Statistics owner's explicit denominator, missingness/censoring and paired/cluster estimand contract.

### 3. Failure-mode fraction — deliberately blocked

Transform: predeclared_failure_mode_fraction_v1  
Formula ID: target_modes_over_classified_specimens_v1  
Output: failure_mode_fraction

The source evidence must be D5573-99R19 classification observations in the scoped SMC FRP family, with a frozen taxonomy digest and target failure-mode set. The claim ceiling remains descriptive and explicitly says “not acceptability”.

The graph always returns BLOCKED for this derived fraction until a qualified cluster-aware proportion-equivalence analysis method exists. A generic two-proportion score interval is not automatically valid for clustered manufacturing specimens. This is intentional abstention, not unfinished optimistic behaviour.

Statistics research issue [#12117](https://github.com/Luminous-Dynamics/symthaea/issues/12117) will compare naive score intervals, cluster-level analyses, small-sample corrected GEE/GLMM, beta-binomial/hierarchical models, and cluster bootstrap or randomization inference where design assumptions permit. It must measure type-I error at both equivalence boundaries, confidence coverage, power, ICC sensitivity, cluster-size imbalance and non-estimability. No approach is yet selected as universally best.

## Full design-receipt binding

Every endpoint graph now embeds the full EquivalenceDesignReceipt, not just a design ID or digest-shaped string.

The receipt declares:
- output endpoint and raw observation endpoint separately;
- exact transform ID and transform digest;
- frozen design/plan digest and result digest;
- independent unit and grouping scheme;
- equivalence margin, alpha and matching TOST confidence convention;
- planned specimen/cluster minima and actual manifests;
- power assumptions and a planning_only receipt;
- randomization, analysis, exclusion and time-source references;
- exact baseline/candidate method observation receipts.

S0 checks that the graph output/transform/raw observation mapping is registered and that the design embeds the exact same baseline/candidate method receipts. The graph analysis result must bind that design ID, full design receipt digest, plan digest, transform digest, all source receipt IDs and all source artifact digests. A design for another endpoint, cohort or transform cannot be substituted.

The design auditor's three-cluster structural floor is only an anti-pseudoreplication check. It is not a statistical adequacy, sample-size or power standard. A preregistered design's own minimum is checked separately; Statistics must establish whether that minimum is defensible.

## Adversarial fixture

Current fixture: **27 synthetic known-answer cases**.

| Disposition | Cases |
|---|---:|
| PASS | 4 |
| BLOCKED | 9 |
| REJECTED | 14 |

Challenge families include:
- valid D3163 direct strength and D5868 FRP-to-FRP direct strength;
- valid four-cohort aging/retention source graph;
- valid source D5573 classification whose derived fraction remains BLOCKED;
- valid structural design but interval outside the equivalence margin;
- missing unaged cohort or aging exposure;
- reused method receipt, raw artifact or specimen identity;
- baseline/aged/candidate cluster mismatch;
- D1151/D1183 protocol, conditioning-mode or method-receipt mismatch;
- D5656 constitutive response substituted for joint strength;
- D5573 generalized outside SMC FRP scope;
- failure-mode acceptability escalation;
- unregistered transform/formula, taxonomy or analysis profile;
- post-outcome transform modification;
- design receipt missing, mismatched or bound to the wrong transform;
- analysis result that omits or substitutes a source receipt;
- ten coupons per arm but one material lot per arm: BLOCKED, because within-lot coupons do not create independent lots.

Latest exact-branch check:
- the previous 26-case endpoint graph fixture independently replayed 26/26;
- the new ten-coupons/one-lot pseudo-replication case independently evaluates BLOCKED under the authored structural cluster floor and is now in the frozen corpus;
- the latest 27-case fixture is 26 previously replayed cases plus 1 individually verified negative control. A fresh full 27-case replay is still pending after the most recent fixture extension.

The fixture is a deterministic software-contract corpus, not a statistically powered physical benchmark.

## SOTA and industry positioning

Plasmatreat documents process control for gas flow, power, pressure, plasma stability, optical monitoring and recipe traceability. That is a mature incumbent capability and should be consumed, not copied as a dashboard.

The European Commission's Digital Product Passport registry became operational on 20 July 2026. The Commission describes it as a product-information container and registration layer; sector-specific obligations phase in through separate legal acts and timelines. This is a market signal for interoperable, lineage-preserving product records, not evidence that this plasma prototype satisfies any DPP legal requirement.

The differentiation to prove is an interface-level evidence passport that can say exactly:
- what each method actually measured;
- which material/adhesive/lot/process and specimens it covered;
- how a derived endpoint was produced;
- which independent design and analysis support the result;
- what remains blocked or unqualified;
- which future change or source retraction invalidates which claim.

Sources:
- Plasmatreat process control: https://www.plasmatreat.com/en/products-and-services/process-control
- European Commission DPP registry/timeline: https://single-market-economy.ec.europa.eu/single-market/digital-product-passport_en

## Benchmarks to outperform

Do not claim “better than SOTA” from synthetic fixture pass rates. Use matched challenge corpora and a preregistered physical cohort.

### Software/evidence benchmarks
- zero false PASS for wrong method, endpoint, material scope or transform;
- zero aged-retention PASS with missing unaged baseline/candidate references;
- zero aged-retention PASS with reused specimens, raw artifact or exposure record;
- zero design PASS for plan/transform/result/observation digest mismatch;
- zero coupon pseudo-replication accepted as independent lot count;
- 100% detection of known specimen/cluster overlap and cohort mismatch;
- 100% detection of post-outcome transform or taxonomy mutation;
- zero transfer PASS for failure-mode fraction until the clustered proportion method passes simulation qualification;
- deterministic replay and exact source-to-output receipt lineage.

### Physical/industrial benchmarks (prospective)
- hold out entire material lots, adhesive batches, plasma heads, production days, lines and sites—not random coupons from shared batches;
- report false acceptance, false block/abstention, interval coverage, failure-mode confusion, aging-retention error, requalification decisions and time-to-decision separately;
- compare against incumbent recipe alarm/monitoring, blanket requalification, one-factor-at-a-time bridge studies and pooled/domain-adaptation models under the same test budget and escaped-defect constraints;
- target at least 25% fewer destructive tests on eligible changes without increasing false acceptance, but only after safety/quality non-inferiority is demonstrated;
- compare lifecycle cost per accepted service-and-recovery outcome, not generator price or contact angle.

The 25% test-reduction figure is a programme target, not an achieved result or industry standard. A study with a narrow matched dataset cannot establish universal SOTA superiority.

## Next integration sequence

1. Merge/reconcile method scope → evidence composition → equivalence design → endpoint transforms in dependency order.
2. Qualify the cluster-aware proportion method in #12117 before enabling failure-mode fractions.
3. Integrate all three endpoint graphs into the actual qualification-transfer audit (#12006) so the legacy three-endpoint list cannot pass on interval names alone.
4. Integrate event-reconciled currentness (#12060) so calibration, supplier, process and field events selectively suspend/revoke downstream Passport claims.
5. Run Rust formatting, compile, tests and Clippy on the exact final head when a Rust environment is available; skipped CI jobs remain skipped, not passing.
6. Preregister one physical material/adhesive/product profile and hold out independent lots before any physical promotion claim.

## Validation ceiling

- Sixteen Rust tests are authored in plasma_endpoint_transform.rs, but not executed.
- Eight design-receipt Rust tests are authored, but not executed.
- Rust formatting, compilation, Clippy and exact-head hosted CI are not verified.
- SHA-256 fields are shape-checked and cross-bound, not recalculated/authenticated from source bytes.
- No numerical transform, TOST, power calculation, survival estimator or clustered proportion analysis is executed here.
- The data is synthetic; there is no physical plasma, bond, aging, failure classification, calibration, laboratory or production evidence in this increment.
- No design allowable, failure acceptability, product qualification, safety or release authority is claimed.


## Existing adjacent owners (no duplication)

The endpoint graph composes with other active plasma workstreams rather than creating a second version of their models:
- #12099 owns pre-bond surface-state trajectory, hydrophobic recovery, storage/handling and treatment-to-bond open-time. The aged-strength graph here is post-bond durability evidence; it does not substitute for pre-bond recovery evidence.
- #12086 owns substrate integrity and over-treatment constraints, including barrier-property or bulk-property degradation.
- #11907 owns decision-value-driven multifidelity experiment selection.
- #11795 owns the application-specific physical bondability-window benchmark.
- #12117 owns the still-unqualified cluster-aware failure-mode proportion inference.

The graph is the typed claim/provenance layer that composes these outputs. It does not introduce a parallel surface-recovery model, substrate-damage simulator, contact-angle-to-strength predictor or optimizer.
