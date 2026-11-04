# HomeRenew repair recurrence and post-repair durability contract v0.1

**Status:** research design / no-runner tranche  
**Owner:** HOME-REPAIR-RECURRENCE-001 (#10431)  
**Parent:** HOME-REPAIR-QUAL-001 (#10358), HOME-SVC-001 (#10322)  
**Evidence class:** specification only; all examples and acceptance controls are synthetic until independently collected physical data exist.

## 1. Purpose

This document defines the household-repair event and follow-up profile needed to test whether a repair restores useful service and whether that restoration persists under observed use.

It composes, rather than replaces:

- SE-VV / R&M authority (#3697);
- lifecycle exposure, degradation, censoring and maintenance-segmentation semantics (#4920);
- household repeatability, production-variation and household-validity design (#10341);
- the household service receipt contract (#10403) and its implementation follow-on (#10407);
- the repairability and firmware-support profile (#10358).

It does **not** introduce a new evidence ledger, generic reliability estimator, survival-analysis library, spare-parts catalogue, repair workflow, diagnostic engine, or certification authority.

The central distinction is:

```
part installed
  != repair completed
  != service requalified
  != durable restoration demonstrated
```

## 2. Claim model

A repair episode may support several different claims. They must remain separate.

| Claim | Minimum evidence | Does not establish |
|---|---|---|
| Part replaced | exact removed/installed part identity and work record | compatibility, safety, restored function |
| Repair work completed | procedure completion and technician record | independent function |
| Service requalified | independent, product-specific post-repair acceptance evidence | future reliability |
| No recurrence observed | complete follow-up through a named time and exposure window | lifetime or zero recurrence probability |
| Cohort recurrence estimate | qualified event histories, stated estimand, censoring and clustering analysis | universal product lifetime |
| Lifetime projection | justified model, population, use conditions and validated assumptions | automatic transfer to other models or environments |

A receipt digest binds bytes, not the truth of a repair assertion. A technician statement, device self-test, or signed service record cannot substitute for the independent acceptance evidence required by the particular service.

## 3. Subject, configuration and episode identity

The minimum identity graph is:

```
appliance subject
  -> exact model / production lot / hardware + firmware revision
  -> installed module and part revisions
  -> fault episode
  -> repair episode
  -> post-repair qualification
  -> follow-up interval
  -> recurrence / competing event / censoring closure
```

Required identity fields:

- privacy-preserving stable appliance subject ID;
- model, serial identity or protected serial reference, production lot and manufacture revision;
- hardware and firmware revisions relevant to the failure and repair;
- critical module, removed part and installed part identifiers/revisions;
- repair procedure revision, repairer class and service location;
- exact parent fault and repair episode IDs;
- explicit configuration transition when a replacement, substitution, rework or firmware change alters the failure process.

Never silently continue a single lifecycle counter across a configuration change that invalidates comparability. Reuse the lifecycle segmentation semantics in #4920.

### Fault identity is not symptom identity

Record separately:

- observed symptom/fault family;
- component and interface implicated;
- diagnostic conclusion;
- confirmed mechanism (or unresolved mechanism);
- evidence authority for each conclusion.

A repeated symptom is not automatically the same causal fault. A repair recurrence analysis must declare whether its event means same symptom family, same failed module, or independently confirmed same mechanism. Report these definitions separately when the evidence permits.

## 4. Event timeline

Use event timestamps from the existing event-time/provenance owners. Preserve the source clock, timezone/offset, timestamp precision, monotonic ordering evidence where available, and uncertainty. Do not fabricate ordering when the source clocks cannot be reconciled.

A repair episode should expose, where applicable:

1. fault first observed;
2. service request opened;
3. diagnosis started and completed;
4. part ordered, dispatched and received;
5. repair work started and completed;
6. reassembly and required calibration completed;
7. independent post-repair qualification started and completed;
8. service restored to user;
9. follow-up checkpoints;
10. recurrence, new fault, maintenance, replacement, retirement or censoring closure.

The follow-up clock begins at **qualified restoration**, not at part installation or technician sign-off. If qualification is absent, failed, incomplete or stale, the episode remains unqualified and cannot enter a durable-restoration risk set.

A further repair after recurrence is a new repair episode linked to the original appliance and prior episode. It never deletes the earlier recurrence.

## 5. Exposure clocks

Preserve each available exposure clock independently:

- calendar time in service;
- powered/operating hours;
- starts and completed service cycles;
- workload-specific throughput (e.g. litres delivered, wash loads or accepted items);
- relevant environmental exposure (e.g. water hardness band, temperature, humidity) where measured and material.

Every exposure quantity needs a unit, observation boundary, start/end, source and uncertainty. A zero or missing exposure value is not an implicit zero-use period.

Do not convert cycles to calendar years, litres to operating hours, or one duty profile to another without a validated mapping. "90 days follow-up" and "90 days of daily use" are not equivalent exposure. When only calendar time is available, report that limitation rather than inventing an exposure-normalized reliability result.

## 6. Follow-up closure and event types

Each repair follow-up closes with exactly one declared observation-ending state, while preserving all events that occurred before closure.

| State | Meaning | Statistical treatment is not implied |
|---|---|---|
| Same-fault recurrence | predeclared recurrence definition observed after qualified restoration | event |
| Different-fault event | a different failure observed | event or competing event, depending on estimand |
| Re-repair | a new intervention after a recurrence or new fault | event plus new repair episode |
| Replacement | original appliance removed from service / replaced | competing terminal event for original-unit estimand |
| Retirement / irreparable termination | original unit leaves the target service population | competing terminal event where applicable |
| Administrative study end | observation ends at declared horizon with no observed event | right censoring |
| Owner withdrawal | consent/participation ends | reason-specific censoring; potential informativeness |
| Lost contact | follow-up cannot be obtained | reason-specific censoring; potential informativeness |
| Telemetry / instrument outage | observation path fails | observation failure, not automatically appliance success or failure |
| Unknown closure | reason cannot be established | unresolved; no silent promotion |

A replacement may be caused by a failed repair, cost, availability or user preference. Treating it as ordinary independent censoring can be misleading. The reason must be retained and sensitivity analysis may be required.

## 7. Estimands and analysis boundaries

No single estimator answers every repair question. Freeze the estimand before opening outcomes.

### 7.1 Time to first same-fault recurrence

Target: elapsed calendar time and, separately, declared use exposure from qualified restoration to first recurrence under a frozen definition.

Report:

- event count and at-risk population;
- competing events and closure reasons;
- follow-up distribution;
- cumulative incidence at preregistered horizons with uncertainty;
- strata by product revision, fault family, repair path, use intensity and production cohort where supportable.

When replacement or irreparable termination precludes observing recurrence on the original appliance, use a competing-risk estimand and an estimator appropriate to it. Do not present (1-widehat S_{KM}(t)) from a naive Kaplan–Meier calculation as the cause-specific cumulative incidence when competing events are present.

### 7.2 Recurrent repair burden

Target: number and direct cost of repairs accumulated by appliances over exposure.

A mean cumulative function (MCF) or a justified recurrent-event alternative is a useful descriptive view. Report both event count and cost burden; one inexpensive part replacement and one major module replacement are not equivalent. Preserve appliance-level histories and cluster uncertainty by appliance rather than treating each repair as an independent new appliance.

### 7.3 Time-to-qualified-restoration and downtime

Report diagnosis-to-qualified-restoration, parts waiting, labor, requalification and total time without accepted service separately. Report median and 90th percentile as well as the full distribution, with unresolved and abandoned repairs retained in the denominator.

### 7.4 Informative censoring and missingness

Complete-case follow-up is not an acceptable default when withdrawal, lost contact or replacement may depend on cost, severity, dissatisfaction, access to repair or service failure.

At minimum:
- publish closure reasons and follow-up completeness by cohort;
- compare observed characteristics of followed and unfollowed units;
- perform bounded sensitivity analyses for plausible missing outcomes;
- use censoring-weighted methods only with a declared model, positivity/overlap checks, model provenance and effective sample-size reporting;
- abstain from a population claim when the censoring mechanism is not defensibly addressed.

Generic estimators, parametric lifetime models, acceleration models and inferential guarantees remain under #3697. This document only fixes the household data and study requirements.

## 8. Competitive study design

Compare matched tasks on production-representative units through:

A. manufacturer-authorized service path;  
B. qualified independent-repair path;  
C. HomeRenew-assisted repair path.

Match or adjust for product revision, production lot, fault severity, baseline condition, use intensity, repair complexity and acceptance protocol. Record the reason for path selection; repair paths are not randomized merely because they are compared.

Where possible, use randomized allocation of repair tasks or blinded independent post-repair assessors. When randomization is impossible, preregister the matching/adjustment plan and state its residual confounding limits.

### Required metrics

| Dimension | Primary reporting |
|---|---|
| Restoration | independently qualified first-time restoration rate |
| Speed | diagnosis-to-restoration median / P90; unavailable-service duration |
| Parts | exact availability, actual purchase, delivered lead time, substitution and price |
| Repair burden | labor, parts, travel, consumables, repeat visit and total cost |
| Durability | same-fault cumulative incidence at fixed horizons; recurrence event count/cost MCF |
| Service | accepted service actually delivered after repair; downtime and unresolved demand |
| Safety / quality | requalification failure, hazard-related abort, leakage/contamination or other product-specific guardrails |
| Evidence quality | follow-up completeness, calibration status, missingness, source lineage and assessor independence |

No weighted repairability number may conceal an unqualified repair, a critical safety failure, or missing follow-up. Publish raw strata and uncertainty.

## 9. Synthetic contract controls

Synthetic fixtures must have multiple appliance subjects and unequal exposure. They should contain at least these known-answer controls:

1. zero recurrence after short follow-up;
2. zero recurrence after long follow-up;
3. same recurrence at a known calendar time and exposure;
4. different fault without same-fault recurrence;
5. repeated repair events on one appliance;
6. replacement before recurrence;
7. retirement / irreparable termination;
8. administrative right censoring;
9. owner withdrawal with reason;
10. lost contact plausibly related to cost/severity;
11. instrument outage that creates missing observation but not appliance failure;
12. part substitution that creates a new configuration segment;
13. repair record says "restored" but no independent qualification exists;
14. same event timestamps but unequal cycles / operating hours;
15. manufacturer and independent service cohorts with different case mix.

Semantic negative controls must reject:
- duplicate subject, repair or event IDs;
- impossible event order;
- follow-up before qualified restoration;
- recurrence from an unqualified restoration;
- negative or unitless exposure;
- exposure discontinuity without a documented reset/segment;
- incompatible part or configuration lineage;
- closure without a closure reason;
- event after terminal closure with no new subject/service episode;
- using fixture expected labels as observed outcomes.

The fixtures qualify schema and deterministic event reconstruction only. They do not qualify an estimator or product.

## 10. Qualification and evidence grades

- E1: within-unit repeatability of physical repair and requalification procedures.
- E2: independent-lab or independent-assessor reproducibility for the defined endpoint.
- E3: production-unit, lot and component-revision variation.
- E4: real-household validity under observed use, maintenance, access and follow-up.

These grades remain independent of the mere existence of a typed receipt. A dataset with many repeated repairs on a handful of serials is not E3 population evidence. A multi-lab study on prototypes is not E4 household evidence.

For laboratory measurements, select competent facilities for the specific methods; ISO/IEC 17025 is a laboratory competence reference, not a product certificate. Exact assay scope and accreditation must be checked for each measurement.

## 11. Industry comparison and defensible differentiation

The French repairability index uses producer-scored documentation, disassembly/access/tools/fasteners, spare-part availability, spare-part price and category-specific criteria, displayed on a ten-point scale. France also has a distinct durability index for categories including washing machines and televisions; do not confuse that index with the dishwasher repairability index.

Our additional claim should therefore be measured **repair outcome**, not another declared design score:
- was the part actually obtained in the target market?
- was the fault correctly diagnosed?
- was service independently requalified?
- how long did restoration take?
- did the same fault recur under recorded use?
- how much cumulative downtime, labor, cost and material replacement resulted?

ISO 14224 is a useful reliability/maintenance data-structure reference from petroleum, petrochemical and natural-gas operations. It is not a household-appliance requirement and its equipment taxonomy should not be imported wholesale.

## 12. Proposed programme gates (not industry standards)

1. **Contract completeness:** every repair attempt has a stable subject/episode identity and an explicit terminal/follow-up state.
2. **Qualification integrity:** no episode enters the durable-restoration cohort without independent service requalification.
3. **Denominator integrity:** all eligible repair episodes, repeat visits, withdrawals, replacements and missing follow-up are counted.
4. **Exposure integrity:** calendar time and physical use exposure remain separate, with no unvalidated conversion.
5. **Comparative integrity:** matched/adjusted arms and their case mix, uncertainty, and loss-to-follow-up are visible.
6. **Durability evidence:** no superiority claim until a preregistered cohort has sufficient observed events/follow-up and uncertainty to resolve the declared practical margin.
7. **Population transfer:** production-lot and household claims require E3/E4 evidence respectively.

Do not freeze arbitrary sample sizes such as "299 repairs" for durability. The relevant unit is the appliance/repair episode structure, with clustering and expected event rates; sample size must be chosen from the estimand, desired precision/power, competing-event prevalence, and clustering. The zero-failure binomial rule from an IID setting cannot be applied to repeated repairs or clustered appliance histories as if every repair were independent.

## 13. Stop / continue / pivot questions

After the pilot, answer separately:

- Which repair path achieves the highest qualified first-time restoration?
- Which path minimizes total household downtime and cost, including parts delay?
- Does HomeRenew improve restoration without increasing requalification failures or recurrence?
- Are apparent cohort differences explained by fault severity, product age, duty cycle, case mix or follow-up loss?
- Which module/interface dominates recurrent burden?
- Is the repair economically sensible compared with continued maintenance, refurbishment or replacement?
- What remains unmeasured or not identifiable?

A null result, a failed repair path or an unidentifiable comparison is a valid programme outcome.

## 14. References

- NIST/SEMATECH Engineering Statistics Handbook, reliability data analysis and censored life data: https://www.itl.nist.gov/div898/handbook/apr/section4/apr4.htm and https://www.itl.nist.gov/div898/handbook/apr/section4/apr41.htm
- NIST/SEMATECH competing-risk model assumptions: https://www.itl.nist.gov/div898/handbook/apr/section1/apr181.htm
- Geskus (2024), *Competing Risks: Concepts, Methods, and Software*, Annual Review of Statistics and Its Application: https://doi.org/10.1146/annurev-statistics-040522-094556
- *Recurrent Events Data Analysis for Product Repairs, Disease Recurrences, and Other Applications*, SIAM: https://epubs.siam.org/doi/book/10.1137/1.9780898718454
- ISO 14224:2016 (petroleum, petrochemical and natural-gas reliability/maintenance data collection): https://www.iso.org/standard/64076.html
- ISO/IEC 17025:2017 (testing and calibration laboratory competence): https://www.iso.org/standard/66912.html
- French Ministry, repairability index: https://www.ecologie.gouv.fr/politiques-publiques/indice-reparabilite
- French Ministry, durability index: https://www.ecologie.gouv.fr/politiques-publiques/indice-durabilite

## Claim ceiling

This is a research specification. It does not report physical repair trials, measured recurrence, product lifetime, market superiority, safety approval, certification, legal compliance, or a validated accelerated-life conversion.
