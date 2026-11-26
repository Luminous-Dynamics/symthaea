# CIV-ENG-001A — Blocker → Engineering Demand → Capability Return Contract

Issue: #6230  
Parent: CIV-ENG-001 #6228  
Schema: `civ-eng-001a-reference-v1`

## Purpose

Freeze the first execution-free contract connecting a CIV productive-capability blocker to a bounded engineering investigation and, only after exact evidence is available, back to a new CIV closure generation.

This tranche is documentation/data only. It establishes no physical capability, material truth, manufacturing qualification, self-sufficiency, procurement authority, fabrication authority, or machine-operation authority.

## Core theorem

```text
CIV blocker
!= engineering requirement
!= material research demand
!= candidate intervention
!= qualified capability
!= new CIV closure
```

A capability may affect a later CIV closure generation only through an explicit capability-envelope return. Historical closure generations remain immutable.

## Ownership boundary

CIV-BOOT owns structural/import/generational closure. ENG-DESIGN and domain engineering own requirements. MAT-OPPORTUNITY-003 owns material/process/interface bottleneck attribution. MAT-ENG owns engineering↔materials translation. MAT-PSPP, MAT-UQ, MAT-INTERFACE and SCI-EXP own their respective scientific semantics. MFG-PROC/MFG-LIFE own manufacturing/lifecycle semantics. ENG-MEAS/ENG-TOL/calibration owners own measurement/precision. Domain bootstrap owners such as ROB-MFG, SENSE-BOOT and COMPUTE-MFG/SEMI-BOOT retain domain meaning.

CIV-ENG is only the projection/feedback boundary.

## Corpus encoding

The stored corpus is compact canonical JSON. To keep the fixture small, it uses abbreviated raw-input keys:

```text
vg  venue_generation
cg  closure_generation
ccg current_closure_generation
t   target_capability_ref
br  blocker_ref
bc  blocker_class
ss  structural_status
il  import_leverage_ref_present
env required_capability_envelope_present
mla material_leverage_asserted
ba  bottleneck_attribution
route declared_owner_route
er  engineering_requirement_ref_present
ev  candidate_evidence_level
pm  returned_profile_matches
gm  returned_generation_matches
mc  measurement_currentness
vc  venue_capacity_adequate
met metrology_available
intf interface_requirement_satisfied
g1  g1_supported
g2  g2_plus_supported
gd  generational_degradation_observed
pg  predicted_closure_gain
rg  realized_closure_gain
bm  bottleneck_migrated
pa  physical_execution_authority_present
pf  pilot_family
```

Each case starts from top-level `base`, applies its `o` override map, then derives the eight ordered expected axes in `e`.

The `axes` array and `codes` object are part of the frozen schema. An oracle must decode `e` using them; it must not branch on case ID.

## Frozen route map

The corpus freezes:

```text
MaterialPropertyGap        -> MAT_OPPORTUNITY_THEN_MAT_ENG
MaterialGradeGap           -> MAT_OPPORTUNITY_THEN_MAT_ENG
InterfaceOrSurfaceGap      -> MAT_INTERFACE_DOMAIN
ProcessCapabilityGap       -> MFG_PROC_DOMAIN
PrecisionOrToleranceGap    -> ENG_TOL_METROLOGY
MetrologyRenewalGap        -> ENG_MEAS_CALIBRATION
CalibrationRenewalGap      -> ENG_MEAS_CALIBRATION
ToolingRenewalGap          -> MFG_TOOLING
ComponentQualificationGap  -> DOMAIN_COMPONENT
ReliabilityOrLifecycleGap  -> MFG_LIFE_RELIABILITY
EnergyOrThermalGap         -> DOMAIN_ENERGY_THERMAL
SoftwareOrControlGap       -> SOFTWARE_CONTROL
SensorCalibrationGap       -> SENSE_BOOT
ComputeElectronicsGap      -> COMPUTE_SEMI_BOOT
RobotReplacementGap        -> ROB_MFG
EvidenceInsufficient       -> EVIDENCE_MEASUREMENT
MultiGenerationDegradation -> CIV_BOOT_003_DOMAIN
```

Unknown blocker classes derive `NeedsOwnerResolution`. A declared route differing from the frozen mapping derives `WrongOwnerRoute`.

## Frozen derivation rules

### Demand

```text
cg != ccg -> StaleOrMismatched
else env=false OR er=false -> Incomplete
else -> Represented
```

### Material-research eligibility

```text
mla=false -> NotMaterialQuestion
mla=true + ba=Missing  -> BottleneckEvidenceRequired
mla=true + ba=Negative -> MaterialVariableNotBottlenecking
mla=true + ba=Positive -> MaterialResearchEligibleUnderProfile
```

Import leverage does not alter this rule.

### Return evidence

```text
None                        -> EvidenceInsufficient
CandidateOnly               -> CandidateOnly
EngineeringEvidence         -> EngineeringEvidenceAvailable
ProcessOrComponentEvidence  -> ProcessOrComponentEvidenceAvailable
CapabilityEnvelopeSupported -> CapabilityEnvelopeSupportedUnderProfile
```

### CIV admission

`EligibleForNewClosureGeneration` requires all of:

1. Demand = `Represented`.
2. Routing = `CanonicalOwnerRoute`.
3. `ev=CapabilityEnvelopeSupported`.
4. `pm=true`.
5. `gm=true`.
6. `mc=Current`.
7. `vc=true`.
8. `met=true`.
9. `intf=true`.
10. If `mla=true`, material eligibility is `MaterialResearchEligibleUnderProfile`.

`cg != ccg`, `gm=false`, or `mc=Stale` derives `GenerationStale`. `pm=false` derives `ProfileMismatch`. Other failed prerequisites derive `NotEligible`.

### Generational preservation

```text
g1=false -> Unresolved
g1=true + g2=true -> G2PlusSupportedUnderProfile
g1=true + g2=false + gd=true -> GenerationalDegradation
otherwise g1=true -> G1Only
```

### Closure-effect calibration

Closure effect is derived only after CIV admission:

```text
bm=true -> BottleneckMigrated
rg=0 -> NoClosureGain
rg < pg -> PredictedGainOverestimated
rg > pg -> PredictedGainUnderestimated
otherwise -> ClosureGainObserved
```

This preserves predicted-versus-realized discrepancy instead of rewriting the earlier forecast.

### Authority

Every case derives exactly `NoPhysicalExecutionAuthorityFromThisContract`.

The raw adversarial input `pa` cannot grant, amplify, validate, consume, or transfer physical authority through CIV-ENG.

## Precision-bearing subset

C01, C14, C17, C18, C19, C20, C21, C23 and C24 carry `pf=PrecisionBearing`.

The subset exists to preserve:

```text
bulk steel available
!= bearing material state qualified
!= heat-treatment/microstructure established
!= precision races/elements producible
!= surface/contact state qualified
!= bearing article capability
!= G2+ bearing-production renewal
```

No fixture contains a real bearing manufacturing recipe, safety case, or product qualification.

## Ordered cases

The corpus contains exactly 32 cases, C01–C32, in array order. They cover demand completeness/currentness; canonical routing; material-bottleneck firewall; candidate/evidence promotion denial; profile/generation/currentness mismatches; capacity/metrology/interface blockers; G1 versus G2+ preservation; lower-performance substitution; bottleneck migration; predicted-versus-realized closure calibration; non-material redesign; external-service alternatives; zero CIV gain after valid engineering success; authority non-amplification; and one full synthetic positive chain.

## Canonical identity

Stored corpus:

`docs/engineering/data/civ_eng_001a_reference_v1.json`

Canonicalization is UTF-8 JSON with recursively sorted object keys, separators exactly `,` and `:`, no insignificant whitespace, and no trailing newline.

SHA-256:

`36aa384278fc16c9b64cf0861e98fc246645c31679099344709e24f0be806974`

The stored file uses exactly those canonical bytes.

## Qualification expectation

A later independent oracle must verify exact source/corpus identity, reconstruct each full input from `base + o`, derive every axis from the rules above, decode and compare `e` only after derivation, reject malformed/unknown required fields, include mutations for admission prerequisites and the authority boundary, and avoid importing production CIV/MAT/ENG decision code.

Qualification order is:

```text
001A source/docs/data freeze
-> 001A1 independent oracle
-> exact named executable qualification receipt
-> typed adapters only after qualified/current upstream owners exist
```

Source review, skipped CI, generic CI, or an upstream unexecuted qualifier is not PASS.

## Claim ceiling

A future exact-source PASS may establish only deterministic software semantics for blocker→demand routing, the material-demand firewall, capability-return admission, exact-generation currentness, multi-generation distinction, closure-delta calibration and non-authority over the frozen synthetic corpus.

It establishes no real material property, bearing capability, process capability, productive closure, self-sufficiency, economic merit, safety, procurement authority, fabrication authority or physical execution authority.