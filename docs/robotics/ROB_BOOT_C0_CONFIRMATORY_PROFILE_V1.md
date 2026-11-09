# ROB-BOOT C0 Confirmatory Profile v1

Status: **DRAFT PROFILE — CAMPAIGN NOT STARTED**

Authority: protocol/decision semantics only. This profile does not establish prerequisite qualification, physical safety, material truth, measurement capability, fabrication conformity, or an R2 result.

Related work:

- ROB-BOOT-000 #5078
- ROB-BOOT-000A #5107
- ROB-BOOT-000B #5148
- ROB-BOOT-000C #5203
- ROB-BOOT-001A #5204
- ROB-DESIGN-001C0 #5084
- ROB-DESIGN-001C0A #5219
- ROB-DESIGN-001C0B #5220
- ROB-DESIGN-001C0C #5221
- ROB-DESIGN-001C0D #5222
- ROB-DESIGN-001C0E #5248
- ROB-JOINTLAB-C0-FIX-001 #5207
- ROB-C0-BENCH-001 #5210
- ROB-REALIZE-001 #4859
- ENG-MEAS-001 #4919

This file specializes the generic preregistration template for the deliberately narrow C0 V1 campaign. Every campaign-specific value remains **UNSET** until readiness is established and the generation is sealed.

## 1. C0 V1 staging

The first physical C0 campaign is a passive, quasi-static structural experiment.

```text
C0 V1 static bench
!= motorized JointLab
!= actuator/system-identification campaign
```

C0 V1 requires no motor, servo, HAL authority, MuJoCo execution, multivariate system-identification fitter, or stochastic optimizer.

The campaign tests only whether one frozen design change can transfer to one bounded physical mass/deflection improvement under an exact evidence protocol.

## 2. Preferred physical fixture

Freeze one three-point-style profile:

```text
simply supported rectangular coupon
+ centered retained load
+ direct midspan incremental deflection
```

Required fixture identity/profile fields:

| Field | Value |
|---|---|
| fixture identity | **UNSET** |
| measured support span | **UNSET** |
| support geometry/radius | **UNSET** |
| loading-nose geometry/radius | **UNSET** |
| load-centering profile | **UNSET** |
| specimen alignment profile | **UNSET** |
| direct-deflection datum/frame | **UNSET** |
| fixture-compliance evidence root | **UNSET** |
| fixture applicability root | **UNSET** |

`LoadCase::SimplySupportedCenterPoint` in software is not evidence that the physical fixture realizes that boundary condition.

## 3. Passive retained-load profile

Preferred V1 uses retained reference masses rather than powered actuation.

Freeze:

| Field | Value |
|---|---|
| reference-mass stack identity | **UNSET** |
| hanger/nose mass identity | **UNSET** |
| retained-load hardware identity | **UNSET** |
| load-step sequence | **UNSET** |
| settle/hold rule | **UNSET** |
| unload/reset rule | **UNSET** |
| load-retention/guard profile | **UNSET** |

For the baseline/candidate non-inferiority comparison, use the **same exact retained load realization** where practical.

```text
same retained mass stack
```

is enough to define the operational comparison without requiring a high-precision local-gravity estimate merely to compare the two articles. Conversion to absolute newtons for an absolute analytical prediction is a separate stronger proposition.

## 4. Incremental deflection convention

Preferred V1 semantics:

```text
mount article
-> settle under self-weight
-> record/retain initial state
-> zero/reference incremental deflection
-> apply frozen additional load
-> measure incremental midspan response
```

This avoids silently mixing candidate-dependent self-weight sag into the protected incremental-load comparison.

If absolute sag including self-weight is later claimed, it requires a distinct model/measurement profile.

## 5. Exact design/search staging

C0 V1 source train:

```text
C0A exact template/parameter identity
-> C0B exact-design to CSG/mesh translation receipt
-> C0C exact normalized mechanics
-> C0E analytical applicability gate
-> C0D exhaustive deterministic selection
```

C0 V1 uses **complete enumeration**, not stochastic optimization.

```text
C0 V1 physical R2
!= optimizer superiority
```

The independent oracle remains qualification/reference truth only and must not be visible to production selection code.

## 6. Analytical applicability gate

Before candidate selection, bind one exact C0 analytical-applicability profile.

| Field | Value |
|---|---|
| applicability profile root | **UNSET** |
| shear/model-form budget | **UNSET** |
| lateral/out-of-plane stability rule | **UNSET** |
| section-aspect-ratio envelope | **UNSET** |
| small-deflection envelope | **UNSET** |
| fixture/contact compatibility rule | **UNSET** |
| elastic-response assumption/profile | **UNSET** |

Attractive Euler-Bernoulli ratios outside this profile remain visible as model-extrapolation candidates but cannot be selected for the simple C0 claim.

A physically successful out-of-profile candidate may motivate a later C0.x/C1 campaign; it does not retroactively validate the C0 model.

## 7. Candidate-selection reserve

Do not intentionally select a candidate sitting exactly on the predicted structural protection boundary.

Freeze separately:

| Field | Value |
|---|---|
| final physical deflection NI margin | **UNSET** |
| stricter predicted selection bound | **UNSET** |
| selection reserve | **UNSET** |
| reserve rationale/evidence root | **UNSET** |

Conceptually, for a deflection ratio where larger is worse:

```text
selection_limit <= physical_NI_limit - reserve
```

under the exact frozen decision semantics.

The reserve is an engineering selection rule, not proof that physical transfer will succeed.

## 8. Material/realization matching

Preferred first comparison uses baseline and candidate from the same parent stock/material lot where practical, with the same process/orientation.

Freeze:

| Field | Value |
|---|---|
| parent stock/material lot identity | **UNSET** |
| baseline article identity | **UNSET** |
| candidate article identity | **UNSET** |
| process/orientation profile | **UNSET** |
| pre-outcome dimensional acceptance rule | **UNSET** |
| pre-outcome rework/replacement rule | **UNSET** |

Article acceptance/rejection must occur before target-performance exposure according to the frozen rule.

```text
realized article improved
!= nominal selected design caused the improvement
```

If material/as-built deviation materially changes the selected design proposition, use an explicit design-transfer-indeterminate disposition rather than crediting the nominal design.

## 9. Stable reference-artifact surveillance

Use one stable reference artifact where practical to detect bench drift. A preferred block is conceptually:

```text
reference -> baseline -> reference -> candidate -> reference
```

or a frozen randomized/blocking equivalent.

Freeze:

| Field | Value |
|---|---|
| reference article identity | **UNSET** |
| historical/current response envelope | **UNSET** |
| surveillance decision rule | **UNSET** |
| action on surveillance failure | **UNSET** |

The reference is not a third competitor and does not enter the R2 target/protection score.

```text
reference stable
!= baseline/candidate result correct
```

but a reference excursion outside the frozen surveillance envelope requires measurement/session validity review.

Do not force a drifting reference back to its expected value with an outcome-dependent correction unless that correction was preregistered and qualified.

## 10. Exact-article confirmatory claim

C0 V1 is scoped to the exact tested baseline/candidate articles unless a later campaign explicitly preregisters independent fabricated-article population inference.

```text
within-run samples
!= runs
!= sessions/remounts
!= independent articles
```

Repeated runs/sessions inform repeatability, drift, and uncertainty for the exact articles. They do not manufacture a process-population sample size.

## 11. Guard-banded decision algebra

Freeze signed contrasts:

```text
Delta_m = m_candidate - m_baseline      // negative is better
Delta_d = d_candidate - d_baseline      // positive is worse
```

and supported uncertainty intervals:

```text
I_m = [L_m, U_m]
I_d = [L_d, U_d]
```

with exact method/profile identity.

Freeze:

```text
M_min > 0    minimum engineering-relevant mass reduction
D_NI >= 0    maximum admitted deflection regression
```

Recommended conservative V1 semantics:

```text
mass target supported
iff U_m <= -M_min

protected deflection supported
iff U_d <= D_NI

HeldOutImprovementSupportedExactArticles
iff both hold
   AND all other frozen hard gates hold
```

If an interval crosses its decision boundary, return an indeterminate/too-small disposition rather than choosing the favorable point estimate.

The uncertainty method must compose ENG-MEAS and must not derive false precision from high-rate raw-sample count.

A population p-value is not required for this exact-article V1 claim.

## 12. Exposure and sealing

Compose ROB-BOOT-000C #5203.

Before confirmation, seal:

- candidate identity;
- exact as-built article targets;
- search/model/memory snapshot;
- decision profile;
- analysis plan;
- confirmatory partition commitment;
- prediction artifact roots.

Outcome-bearing confirmatory data must not become visible to search, proposal memory, model fitting, threshold selection, or redesign under the same generation.

Where the qualified QUAL-INFRA sandbox is available, prefer reducer execution with sealed confirmatory inputs mounted only into the primary/independent analysis processes.

## 13. Independent reducer crosscheck

Compose ROB-BOOT-000B #5148.

Primary and independent reducers consume the same sealed raw evidence and frozen analysis plan, but neither receives the other's output before its own result is sealed.

Required agreement includes:

- input/protocol identities;
- run/article membership;
- exclusions;
- mass/deflection run-level estimates;
- uncertainty bounds;
- target/protection dispositions;
- final disposition.

Material disagreement blocks the confirmatory R2 disposition.

## 14. Synthetic pre-physical gate

Compose ROB-BOOT-001A #5204.

Before physical C0 `CampaignStarted`, the synthetic rehearsal must correctly handle at least:

- clean narrow success;
- null/no-effect result;
- engineering-tiny effect;
- protected deflection regression;
- measurement-dominated result;
- confirmatory contamination;
- optional-stopping trap;
- pseudo-replication trap;
- reducer disagreement;
- post-freeze mutation;
- safety abort;
- infrastructure abort;
- prediction-poor physical-surrogate win;
- memory/oracle leakage.

```text
synthetic rehearsal PASS
!= physical R2 PASS
```

but inability to reject known-invalid fixtures blocks the physical campaign.

## 15. C0 V1 disposition additions

In addition to the generic preregistration vocabulary, permit explicit outcomes such as:

```text
AnalyticalApplicabilityFailed
ReferenceSurveillanceFailed
SelectionReserveNotSatisfied
RealizationConformityFailed
RealizedArticleImprovedButDesignTransferIndeterminate
AnalysisCrosscheckFailed
ExactArticleMassTargetSupportedButDeflectionIndeterminate
ExactArticleDeflectionProtectedButMassTargetTooSmall
```

No weighted aggregate score may convert one of these hard-boundary outcomes into `HeldOutImprovementSupportedExactArticles`.

## 16. Start gate additions

C0 V1 may not start until every required field above is frozen and current, including:

- [ ] passive bench/fixture profile frozen;
- [ ] same retained-load identity/procedure frozen;
- [ ] incremental-deflection convention frozen;
- [ ] analytical applicability profile frozen;
- [ ] exhaustive domain/selection profile frozen;
- [ ] selection reserve frozen;
- [ ] matched-stock/material realization policy frozen;
- [ ] pre-outcome article acceptance/replacement rule frozen;
- [ ] reference surveillance profile frozen;
- [ ] exact-article uncertainty/guard-band decision rule frozen;
- [ ] exposure/sealing profile frozen;
- [ ] independent reducer profile frozen;
- [ ] synthetic hostile rehearsal PASS/current;
- [ ] all actually required source/evidence predecessors executable-qualified/current.

Queued qualification is not readiness.

## 17. Nonclaims

Even a successful C0 V1 campaign does not establish:

- manufacturing-population improvement;
- fatigue/lifetime reliability;
- absolute material-property truth;
- arbitrary structural analysis;
- general CAD competence;
- optimizer superiority;
- motor/actuator competence;
- system-identification competence;
- C1 JointCarrier competence;
- arm/humanoid competence;
- unrestricted recursive self-improvement.

It establishes only the exact narrow proposition frozen by the campaign generation and its evidence capsule.
