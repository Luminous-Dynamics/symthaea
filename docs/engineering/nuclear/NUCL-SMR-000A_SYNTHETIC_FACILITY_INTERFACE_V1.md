# NUCL-SMR-000A — Synthetic SMR Facility Interface Demonstrator V1

Issue: #6136  
Parent program: #6135  
Frozen parent: `fbd7a754ea8389ca93f7680d93ed8b48553e6376`  
Profile: `SyntheticSmrFacilityInterfaceDemonstratorV1`  
Corpus: `nucl-smr-000a-facility-interface-reference-v1`  
Canonical corpus SHA-256: `8fe2b7b7926a2f5cb219cd8642813c77fa248301670ffaf1826b415cb9842ee4`

## Purpose

Freeze the first fully synthetic, non-reactor-design subject for SMR facility integration. The subject exists only to test exact identity, interface generation, site/facility dependency, evidence/currentness, 3S routing, commissioning, digital-twin source-class, and change/requalification semantics.

The reactor nuclear island is represented only as an opaque externally owned interface subject. This contract does not model reactor physics and does not contain real plant operating values.

## Governing boundary

```text
SyntheticLicensedReactorInterfaceV1
+ synthetic site/facility mappings
+ external authority references
-> software integration subject only
```

and:

```text
synthetic mapping complete
!= reactor design verified
!= site approved
!= structure adequate
!= nuclear safety analysis
!= plant commissioned
!= licensed to construct or operate
```

## `SyntheticLicensedReactorInterfaceV1`

The synthetic reactor placeholder binds only:

- synthetic vendor/model identity;
- exact interface generation;
- synthetic jurisdiction/applicability ref;
- externally owned classification/requirement refs;
- facility-side role requirements;
- currentness/change-notice refs;
- claim ceiling.

The frozen V1 facility-side role vocabulary is:

```text
CivilFoundationInterface
ThermalServiceInterface
ElectricalPowerInterface
GridExportInterface
AuxiliaryUtilityInterface
EnvironmentalInterface
MaintenanceHandlingInterface
DataBoundaryInterface
WasteServiceInterface
EmergencySupportInterface
SafetyInterfaceRef
SecurityInterfaceRef
SafeguardsInterfaceRef
```

These are role names only. They contain no reactor setpoints, fuel parameters, nuclear material quantities, security layouts, protection-system logic, accident source terms, or operating instructions.

## Prohibited detail classes

The open synthetic profile must reject or omit:

- reactor-core geometry;
- fuel composition/enrichment;
- fuel-management strategy;
- criticality/neutronics inputs;
- accident source terms;
- protection setpoints/trip logic/bypass logic;
- physical-security vulnerabilities or response tactics;
- safeguards-sensitive material inventory, accountancy, inspection, or evasion detail;
- plant operating procedures or startup/power-ascension sequences.

Unavailable externally governed requirements are represented as unavailable/unknown, never guessed and never silently mapped to `not applicable`.

## Synthetic facility subject

The frozen reference subject contains exact synthetic generations for:

```text
site                    SYNTH_SITE_G1
civil/foundation        SYNTH_CIVIL_G1
thermal BOP             SYNTH_THERMAL_G1
electrical/grid BOP     SYNTH_ELECTRICAL_G1
common utilities        SYNTH_UTIL_G1
maintenance/logistics   SYNTH_MAINT_G1
commissioning graph     SYNTH_CX_G1
twin composition        SYNTH_TWIN_G1
reactor interface       SYNTH_IFACE_G1
```

Friendly labels are non-authoritative metadata. A claim-relevant generation change creates a distinct subject.

## Lifecycle stages

The demonstrator preserves this non-collapsing progression:

```text
VendorInterfacePublished
-> FacilityDesignMapped
-> SiteApplicabilityReviewed
-> FacilityImplementationObserved
-> FacilityCommissioningEvidence
-> ExternalIntegrationReview
```

No stage implies the next.

## 3S reference firewall

Safety, Security, and Safeguards remain three externally governed reference planes:

```text
SafetyInterfaceRef
!= SecurityInterfaceRef
!= SafeguardsInterfaceRef
```

Symthaea may identify that a facility change affects one or more of these external interfaces and route the dependency for review. It may not resolve an external 3S requirement locally or infer detailed protected/sensitive content.

## Digital-twin boundary

The synthetic facility may later compose:

```text
site/civil twin
+ thermal BOP twin
+ electrical/grid twin
+ utility-services twin
+ maintenance/logistics twin
+ commissioning/configuration twin
+ opaque reactor-interface federate
```

but:

```text
synthetic state
!= physical observation

virtual commissioning PASS
!= physical commissioning PASS
```

Every synthetic value retains its source class.

## Currentness and change

A reactor-interface generation change, site generation change, or claim-relevant facility configuration change may stale only the dependencies actually affected.

```text
change
-> impact/applicability review
-> selective reopening
```

Unrelated evidence may remain current only when an exact impact/non-impact theorem or owner-qualified mapping says so.

Historical failed evidence remains history after a corrected generation succeeds.

## Disposition vocabulary

The reference corpus uses categorical outcomes rather than scores:

```text
SyntheticMappingComplete
VendorInterfaceStale
SiteApplicabilityReviewRequired
SiteEvidenceIncomplete
CivilInterfaceUnresolved
ThermalInterfaceUnresolved
GridAuthorityMissing
UtilityCommonModeDetected
AsBuiltEvidenceIncomplete
AsBuiltContradiction
IntegratedCommissioningIncomplete
ExternalSafetyReviewRequired
ExternalSecurityReviewRequired
ExternalSafeguardsReviewRequired
ExternalClassificationUnknown
ClaimPromotionRejected
SyntheticSourceClassRejected
SelectiveRequalificationRequired
HistoricalFailureRetained
NoBuildOrOperateAuthority
```

No `plant_ready`, `safe`, `licensed`, `approved`, `commissioned`, or scalar readiness/health score is created by this contract.

## Adversarial reference corpus

The paired JSON corpus freezes 24 cases. It includes:

- interface-generation staleness;
- same-name/different-generation identity attacks;
- cross-site applicability attacks;
- missing site/foundation requirements;
- missing thermal/grid authority inputs;
- hidden common-mode utility dependencies;
- BIM/as-designed vs as-built evidence separation;
- integrated vs component commissioning separation;
- virtual-vs-physical source-class attacks;
- explicit Safety/Security/Safeguards external review routes;
- selective requalification;
- claim laundering from generic engineering into licensed nuclear safety analysis;
- browser/UI proposal authority laundering;
- append-only failed commissioning history;
- final full-synthetic closure with an explicit no-build/no-operate ceiling.

## Claim ceiling

This source subject may eventually support qualification of **software semantics only** for:

- synthetic interface representation;
- dependency/currentness reasoning;
- external 3S reference routing;
- virtual-commissioning planning semantics.

It explicitly does not establish:

- reactor design;
- nuclear safety analysis;
- site suitability;
- structural/foundation adequacy;
- radiological protection adequacy;
- physical-security effectiveness;
- safeguards compliance;
- emergency-plan adequacy;
- licensing or regulatory approval;
- construction readiness/permit;
- physical commissioning authorization;
- procurement authority;
- physical nuclear operation.

## Qualification plan

This tranche is intentionally docs/data only. A later independent stdlib oracle should:

1. bind the exact source head and both source blobs;
2. verify the canonical JSON digest;
3. independently derive all 24 dispositions without trusting a corpus rule table;
4. hostile-test prohibited detail/authority fields;
5. verify synthetic observations cannot satisfy physical observation classes;
6. verify external 3S unknowns fail closed;
7. require an exact two-source-file source diff and clean postflight.

Until that dedicated qualifier reaches exact-head terminal success, this source remains **frozen but unqualified**.
