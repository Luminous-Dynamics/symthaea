# BUILT-SITE-001A — terrestrial site/geotechnical evidence contract

Status: source/data freeze for review only
Parent: BUILT-SITE-001 #6034 / BUILT-ENV-001 #6032
Frozen base: `main@eae17187e199e3a53d108b437c0215b5ff812261`

## Purpose

Freeze a machine-checkable terrestrial site/geotechnical **evidence and applicability** profile before buildings, factories, foundations, yards, drainage works, roads, or utility corridors consume site assumptions.

This is not a geotechnical solver, foundation-design engine, code evaluator, professional approval system, or construction authority.

```text
site selected != site characterized
surface mapped != subsurface characterized
ground model assembled != ground verified everywhere
report exists != report applies to this exact site/configuration
site evidence admissible != foundation design != construction approval
```

## Boundary

The profile owns only integrity/applicability semantics for exact references to site/frame identity, investigation/sample/test identity and provenance, horizontal/depth coverage, classification context, sample-chain currentness, groundwater applicability/currentness, units/reference context, assumptions, interpolation, uncertainty, external-report applicability, site-configuration generation, post-work re-observation, consumer-design applicability review, history integrity, and claim ceilings.

Observations, engineering quantities, professional judgments, solver physics, designs, standards compliance, and physical work remain external owners.

## Ground-model rules

```text
one investigation point != homogeneous site
observation at depth z != condition below the declared investigated zone
interpolation != observation
nearby parcel report != current parcel evidence
historical groundwater != current groundwater
```

No missing property receives a favorable default.

Informative references only: ISO 14688-1:2017, ISO 14688-2:2017, ISO 14689:2017, ISO 22475-1:2021, and 2024 JRC second-generation Eurocode 7 ground-model/execution/service-life guidance. They create no compliance or professional acceptance.

Required groundwater evidence clears only with `CurrentObserved`; bounded profiles may explicitly state `NotApplicable`.

Interpolation states are `DeclaredBounded | NotUsed | Undeclared | Overextended`; only the first two clear. `CoercedFavorable` uncertainty is blocked.

A required external report clears only with exact identity, current revision, exact spatial match, exact consuming-configuration match, and retained limitations.

```text
applicable report != independently verified report
!= accepted recommendation != approved foundation != construction authority
```

## Change semantics

Site change and consumer-design change are distinct.

```text
fill/excavation/drainage/trenching completed
!= changed site re-observed

site unchanged + foundation footprint/loading changed
!= old evidence automatically applicable
```

The former requires affected post-work observation; the latter may require explicit applicability review.

Work-order/provenance events remain useful but cannot mint geotechnical truth.

## Dispositions

`SiteEvidenceAdmissible`, `SiteIdentityBlocked`, `ObservationIdentityBlocked`, `SpatialCoverageBlocked`, `DepthCoverageBlocked`, `ClassificationBoundaryBlocked`, `SampleChainBlocked`, `GroundwaterCurrentnessBlocked`, `UnitReferenceBlocked`, `AssumptionBindingBlocked`, `InterpolationBoundaryBlocked`, `UncertaintyBoundaryBlocked`, `ExternalReportBoundaryBlocked`, `ConfigurationCurrentnessBlocked`, `PostWorkObservationBlocked`, `ApplicabilityReviewRequired`, `HistoryIntegrityBlocked`, `SourceAuthorityBlocked`, `FavorableDefaultBlocked`, `DesignAuthorityBoundaryBlocked`, `PhysicalAuthorityBoundaryBlocked`.

These are categorical states, not a site-quality score.

## Fail-closed precedence

```text
history integrity
-> physical/permit authority promotion
-> design-authority promotion
-> work/provenance-only promotion
-> favorable-default attempt
-> site/frame identity
-> investigation/provenance identity
-> horizontal coverage
-> depth coverage
-> classification context
-> sample chain
-> groundwater currentness when required
-> unit/reference context
-> assumption binding
-> interpolation
-> uncertainty
-> external report applicability
-> site configuration currentness
-> post-work re-observation
-> changed-consumer applicability review
-> SiteEvidenceAdmissible
```

## Synthetic corpus

Schema: `built-site-001a-terrestrial-site-evidence-reference-v1`

Deterministic gzip packaging (`mtime=0`) contains canonical compact sorted-key JSON plus final newline.

- decompressed SHA-256: `aca82eaa1e3e09a834ad978a262a0e656a793cf461a502258997b79cf7de44a1`
- gzip SHA-256: `e7547ba293f362e5b8081232b426947fa6e0b5d297b6572e468907dd6a10d6f5`
- cases: **40**
- dispositions: **21**, every disposition exercised.

Coverage includes identity/provenance, spatial/depth overgeneralization, classification/sample chain, groundwater staleness, units, assumptions, interpolation, uncertainty, nearby/stale external reports, site changes, post-work observation, changed-foundation applicability, deleted negative history, work-order laundering, favorable-default coercion, and design/physical-authority promotion.

Positive controls include exact current reports, post-fill re-observation, explicit groundwater-not-applicable profiles, direct observation without interpolation, and completed consumer applicability review.

## Integration

BUILT-STRUCT may consume the exact receipt, but:

```text
SiteEvidenceAdmissible != foundation adequacy != support capacity != structural safety
```

BUILT-ENV should consume receipts rather than copy parameters. BUILT-ROB-PILOT-001 #6040 must not infer real-site suitability.

A separate qualifier must hard-bind this source head/blobs; verify gzip and canonical digests; reject unknown keys/enums; independently derive all 40 cases/21 dispositions; hostile-test nearby-parcel transfer, groundwater staleness, interpolation, favorable defaults, post-work currentness, consumer applicability, work-order laundering, and authority promotion; import no Symthaea production code; and require clean postflight.

## Claim ceiling

No actual site suitability, foundation design, bearing capacity, settlement prediction, slope stability, seismic adequacy, environmental clearance, code compliance, professional approval, permit/occupancy state, construction recommendation, procurement/resource allocation, or physical execution authority.
