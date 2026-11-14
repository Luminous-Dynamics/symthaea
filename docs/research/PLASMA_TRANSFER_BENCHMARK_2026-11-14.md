# PLASMA-TRANSFER-001 — Evidence-backed qualification transfer

**Date:** 2026-11-14  
**Status:** S0 benchmark contract and synthetic reference audit  
**Issue:** [#12002](https://github.com/Luminous-Dynamics/symthaea/issues/12002)

## Executive thesis

A plasma process is not qualified merely because a recipe name, generator setpoint, or nominal coupon result is unchanged. Qualification belongs to a material–adhesive–surface-treatment–geometry–environment–production configuration, with an explicit evidence envelope.

The proposed shift is from **recipe replication** to **bounded qualification transfer**:

- identify every relevant factor in the baseline and candidate configuration;
- distinguish a factor that is unchanged from one that changed but remains within a declared domain;
- require a bridge for every changed factor;
- require a dedicated bridge for the exact combination whenever multiple factors change together;
- compare measured downstream endpoints against predeclared equivalence margins using a qualified uncertainty calculation;
- abstain when evidence is missing, ambiguous, dependent, or out of scope.

This is designed to reduce unnecessary blanket requalification without silently transferring qualification across unknown changes.

## What the industry already does well

Plasma equipment incumbents have substantive process-control capabilities. Plasmatreat publishes optical emission monitoring, media-flow control, plasma power measurement, dynamic pressure monitoring, rotation / motion monitoring, and process data transport; it describes these controls as foundations for reproducible series production. Surfx describes real-time process monitoring, control and automated data logging / traceability on its integrated Atomflo-INT equipment. These are strong process-side baselines, not straw men.

The distinction is that **stable generator / jet telemetry is not a direct downstream equivalence certificate** for a changed material lot, adhesive batch, nozzle wear state, line, fixture, or ambient condition. The new benchmark is a qualification layer that consumes process telemetry plus independently sourced product outcomes; it does not replace the vendor control units.

Relevant sources:
- Plasmatreat Openair process monitoring: https://www.plasmatreat.com/en/products-and-services/openair-plasmar-process-monitoring
- Plasmatreat process control: https://www.plasmatreat.com/en/products-and-services/process-control
- Surfx Atomflo-INT: https://www.surfxtechnologies.com/products/atomflo-int/

## Why a process-space view is necessary

Péczi-Kovács and Kókai (2026) compare argon and nitrogen atmospheric plasma treatment for PP, HIPS, PVC, PC and PET. Their results show material- and adhesive-dependent response: both gases improved wetting and bond strength in the studied systems, nitrogen produced lower water contact angles, argon gave a wider processing window without visible degradation, and the lowest contact angle did not always correspond to the highest bond strength. This is a strong reason not to transfer a single scalar surrogate across material / adhesive changes.

Primc and Mozetič (2024) review the plasma modification of polymers and report that experimental setups and processing/adhesion correlations can scatter and sometimes conflict. Their discussion includes hydrophobic recovery and weakly bound low-molecular-weight surface layers as reasons why wetting and adhesion need not track one another.

References:
- Péczi-Kovács & Kókai, 2026, DOI [10.1007/s44245-026-00396-4](https://doi.org/10.1007/s44245-026-00396-4).
- Primc & Mozetič, 2024, DOI [10.3390/ma17071494](https://doi.org/10.3390/ma17071494).

Sika's surface-preparation guidance recommends requalification after material supplier, specification, environment, or production changes. NASA's JSC process-specification index lists PRC-1001 Rev H dated 2026-08-19; however, the linked publicly served "current" PDF still identifies itself as Rev G. This revision mismatch is recorded as a source-quality limitation. Do not infer Rev H requirements from the stale PDF; use the controlled current specification applicable to the customer and product.

## Relationship to the existing plasma work

| Existing owner | What it qualifies | Transfer benchmark relationship |
|---|---|---|
| #11795 | application-level durable surface activation and bondability | owns product and process acceptance endpoints |
| #11925 | delivered spatial exposure field and geometry | contributes spatial/process factors and exposure evidence |
| #11952 | measurement support and registration | prevents invalid joins between exposure maps and specimen outcomes |
| #11847 / #11839 | virtual metrology and generalization | supplies prediction/evidence, but does not grant qualification transfer |
| #11971 | bond-life, censoring, aging and mechanism evidence | contributes required durability endpoints and valid aged cohorts |
| #11907 | decision-focused experiment design | can propose economical bridge studies; this audit does not implement another optimizer |
| MAT-013 #4344 | material and sample lineage | source of identity/genealogy; no duplicate registry here |

## Factor model

The frozen v0.1 fixture includes these dimensions:

1. substrate grade;
2. substrate supplier;
3. substrate lot;
4. adhesive system;
5. adhesive batch;
6. plasma head;
7. process-gas mixture;
8. gas purity;
9. line speed;
10. treatment-to-bond delay;
11. ambient relative humidity;
12. fixture identity.

A production profile should extend this list where the physical application requires it: recycled fraction/additives, surface coating/passivation, material storage history, nozzle/electrode age, waveform, power, standoff, path overlap, geometry, facility identity, contamination controls, cure schedule, operator and calibration state.

A missing factor is not an implicit assertion that the factor is invariant. A factor change omitted from the controlled inventory must block.

## Transfer contract

### 1. Factor completeness

Both baseline and candidate profiles must identify every required factor. Any changed, untracked field blocks. The profile is a structured tuple, not a free-text recipe label.

### 2. Qualified scope

Every changed candidate value must be inside the frozen application-specific qualification scope. This is a hard boundary. S0 does not interpolate outside the envelope or infer a wider envelope from a model.

### 3. Single-factor bridge

Every changed factor requires a dedicated source-bound bridge record. A bridge contains:
- exact old and new factor values;
- test protocol identifier;
- SHA-256 artifact identity;
- distinct baseline and candidate specimen IDs;
- baseline and candidate independence groups;
- endpoint-specific equivalence margins;
- lower and upper absolute-difference confidence bounds;
- confidence level.

The bounds are generated by an external, qualified analysis method. The S0 audit checks structure and decision logic only; it does not calculate these statistics.

### 4. Joint-change bridge

If two or more factors change at once, the exact full changed-factor set requires an additional joint bridge. Pairwise tests, individual one-factor bridges, or an assumption of additivity do not demonstrate the full interaction. The strict v0.1 rule intentionally blocks unless the exact combination is represented.

This may be conservative. It is deliberately so until a separate, validated interaction-screening rule is established.

### 5. Equivalence decision

For every required endpoint, the externally calculated interval is interpreted against a positive, preregistered equivalence margin:

- upper bound at or below margin and confidence level meeting the profile threshold: admissible endpoint;
- interval crossing the margin or confidence level below threshold: BLOCKED;
- lower bound above the margin: REJECTED.

This is not a generic significance test. A non-significant difference is not accepted as evidence of equivalence. Margins must be engineered per endpoint and customer requirement, not copied from the synthetic fixture.

### 6. Evidence integrity

Missing evidence blocks. Duplicate specimen identities, overlapping baseline/candidate independence groups, malformed digests, stale factor mappings, duplicated bridge dimension sets, or reused artifact digests reject. This is a source-structure guard, not cryptographic verification of remote artifacts or laboratory independence.

## S0 fixture results

The fixture contains **21 synthetic known-answer challenges**:

| Disposition | Cases | Meaning |
|---|---:|---|
| PASS | 5 | structurally complete evidence within the synthetic contract |
| BLOCKED | 8 | insufficient evidence, missing exact interaction, or interval crossing |
| REJECTED | 8 | hard-scope violation, contradictory evidence, identity collision, or invalid digest |

Independent JavaScript replay of the constructed fixture matched all 21 expected dispositions. This source-side replay is not execution of the Rust implementation. Eight Rust unit tests are authored in the new module; compilation, formatting, Clippy and unit-test execution are not claimed.

## Benchmark versus current approaches

| Approach | Strength | Failure mode the benchmark exposes |
|---|---|---|
| Inline process control / recipe alarms | detects drift in power, gas, pressure, motion, optical signal and controller state | same telemetry may still produce different outcomes after material/adhesive/geometry changes |
| Blanket requalification on every change | conservative and simple to govern | can impose unnecessary coupons, time, cost and production delay |
| One-factor-at-a-time bridging | cheaper than full requalification for isolated changes | does not establish combined-change interaction or transportability |
| Generic pooled ML / domain adaptation | can exploit historical data across observed regimes | predictive similarity is not the same as a qualification decision; weak labels and held-out lot leakage can create false confidence |
| Evidence-carrying transfer | links each change to endpoints, uncertainty, specimens, independent groups and exact interactions | requires high-quality genealogy and preregistered physical bridge studies |

The differentiator is a controlled middle path between "everything requires a complete new qualification" and "the recipe looks similar, therefore it transfers."

## Prospective benchmark targets

These are development targets, not achieved results, industry standards, or universal acceptance limits.

| Metric | Prospective target | Evaluation rule |
|---|---:|---|
| Synthetic false accepts | 0 | all frozen hard negatives |
| Untracked-factor detection | 100% | adversarial change fixtures |
| Missing exact multi-factor bridge detection | 100% | pairwise-only and single-factor-only controls |
| Out-of-envelope transfer | 0 | reject every hard-limit negative |
| Interval calibration | 85–95% empirical coverage for nominal 90–95% intervals | whole-lot / whole-head / whole-line holdouts, adequately powered |
| Destructive test burden | at least 25% lower than blanket requalification on eligible cases | no increase in escaped-defect risk |
| Decision regret | at most 10% versus exact small-case oracle on ≥90% eligible cases | equal risk constraints and test budgets |
| Evidence integrity | 100% traceable source/specimen/protocol linkage | duplicate and common-source adversarial tests |

Always report the false-transfer rate and escaped-defect risk separately from the cost reduction. A system that saves coupons by accepting weak joints is a failure, even if the aggregate score is excellent.

## Physical validation design

Select one accessible material–adhesive profile before building a broad multi-material campaign. For the initial transfer study:
1. Freeze the reference profile and application-specific hard limits.
2. Choose candidate changes that represent lot, supplier, head/nozzle, gas, line-speed and storage/time changes.
3. Test each factor change individually and select at least one combined-change vector predicted to interact.
4. Randomize and block specimens by substrate lot, adhesive batch, treatment run, head and day; preserve true specimen/observation lineage.
5. Blind the downstream bond-strength and fracture-mode assessment where practicable.
6. Measure initial strength, aged retention and failure mode; add fatigue/service-life measurements only under #11971's valid event/censoring contract.
7. Hold out entire lots, heads, lines or sites for transfer evaluation. Never split coupons from the same physical lot randomly across train and test and call that cross-domain generalization.
8. Compare with blanket requalification and existing incumbent process-control logs under the same acceptance criteria.

A successful synthetic audit only proves that the evidence contract behaves as specified. A successful lab campaign is still not a customer process qualification; customer-controlled engineering and quality authorities retain that decision.

## Commercial strategy

Do not position this as a new plasma generator or a replacement for Plasmatreat / Surfx controllers. Position the first product as **independent process-to-product qualification and change-impact evidence**:

- integrate vendor telemetry rather than compete with it;
- connect process records to material genealogy, specimen support and downstream bond outcomes;
- automatically identify which changes are inside, outside or unknown relative to qualified scope;
- recommend only the minimum bridge evidence needed under a validated policy;
- produce an auditable change dossier for engineering and quality review.

The first commercial wedge is likely a high-cost bonded polymer/composite assembly with expensive destructive coupons and frequent supplier/lot changes. Revenue comes from reduced unnecessary requalification and fewer escaped weak bonds, not from a claim of higher plasma power or lower contact angle.

## Claim ceiling

S0 is a synthetic evidence-contract benchmark and offline audit. It does **not** demonstrate:
- physical equivalence between lots, suppliers, heads, lines or sites;
- a calibrated bond-strength or life predictor;
- improved adhesion, throughput, yield, cost or energy;
- standards compliance or customer approval;
- reduced laboratory testing in a real manufacturing programme;
- process control, automated requalification, actuation or product release.

Promotion requires exact-head Rust/CI evidence and independent physical results.