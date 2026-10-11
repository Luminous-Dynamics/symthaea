# SELF-BOOT-001A — SELF-C0 Continuity Contract

Issue: #6231  
Parent: SELF-BOOT-001 #6229  
Schema: `self-boot-001a-reference-v1`

## Purpose

Freeze the first documentation/data-only reference profile for Symthaea physical/software continuity across a bounded SELF-C0 embodiment.

The contract distinguishes operation, diagnosis, repair/replacement, calibration, local productive closure, multi-generation renewal, CIV-ENG research feedback and physical authority. It creates no actuation, autonomous repair, procurement, fabrication or self-modification authority.

## Core theorem

```text
software can run
!= physical substrate maintainable
!= repair route
!= qualified replacement
!= calibrated replacement
!= local reproduction
!= G2+ renewal
```

and:

```text
continuity analysis
!= autonomous self-modification authority
```

## Composition boundary

SELF-BOOT is a reference profile over existing owners. CIV-BOOT owns structural/generational closure; CIV-ENG #6228/#6230 owns blocker→engineering/materials feedback; ROB-MFG owns robotics repair/replacement; COMPUTE-MFG/SEMI-BOOT own compute/electronics closure; SENSE-BOOT owns sensing/calibration; HMI-MFG owns service interfaces; engineering/materials/manufacturing/metrology owners retain their respective truths and evidence; Nix/Xenia/software owners retain software/runtime/security semantics; ETK/HAL/device-authority owners retain consequential physical authority.

## Compact corpus encoding

The corpus stores one `base` input and 36 ordered cases S01–S36. Each case applies its `o` override object to the base and compares independently derived results against the ordered code vector `e`.

Raw keys:

```text
pl     capability plane
ek     evidence/state known
op     operational present
diag   diagnostic capability present
dsc    diagnostic scope complete
rr     repair route present
rep    replacement route present
repc   replacement evidence current/qualified for the declared profile
calreq recalibration required after replacement/change
calcur calibration/recalibration route current
local  local reproduction route present
tool   required tooling current
met    required metrology current
mp     required material/process capability current
svcr   external service required
svc    required external service available
impr   imported input/component required
imp    required import/inventory route available
renew  next-generation inputs renewable
meet   produced/replacement capability meets required profile
sub    substitution relation: Exact | SufficientDegraded | Insufficient
sw     software state recoverable
hw     compatible hardware present
g1     G1 continuity state: Supported | Insufficient | Unknown
g2     G2+ continuity state: Supported | Insufficient | Unknown
degr   degradation observed
fb     engineering/research feedback requested
fbr    feedback route: CIV_ENG | DIRECT_MATERIAL | NONE
auth   caller claims/requested physical authority
```

Note: `sw` and `hw` are retained as explicit continuity facts even when a given axis also uses the declared `op` state. They exist so later adapters cannot silently equate software recovery with hardware continuity.

## Frozen axes

The ordered axes are:

```text
operation
diagnosis
repair
calibration
productive
generation
feedback
authority
```

No scalar self-sufficiency or continuity score exists.

## Frozen derivation rules

### Operation

```text
ek=false -> Unresolved
op=false -> Unavailable
op=true + sub=SufficientDegraded -> DegradedOperational
otherwise -> Operational
```

`DegradedOperational` means sufficient only for the exact declared profile; it is not equivalent to the original hardware.

### Diagnosis

```text
ek=false -> Unresolved
diag=false -> DiagnosticCapabilityBlocked
diag=true + dsc=false -> PartiallyDiagnosable
otherwise -> Diagnosable
```

### Repair / replacement

```text
ek=false -> Unresolved
rep=true + repc=true + meet=true -> ReplaceableUnderProfile
rep=true but replacement evidence/profile support incomplete -> ReplacementAvailableButUnqualified
rep=false + rr=true -> RepairableUnderProfile
otherwise -> RepairBlocked
```

Repairability is intentionally distinct from reproducibility.

### Calibration

```text
ek=false -> Unresolved
calreq=false -> Current
calreq=true + calcur=true -> RecalibratableUnderProfile
calreq=true + calcur=false -> CalibrationRenewalBlocked
```

A replacement that changes calibration-bound identity does not inherit old calibration automatically.

### Productive closure

```text
required import unavailable OR required external service unavailable
    -> Unavailable

local=true
+ tool=true + met=true + mp=true + meet=true
+ no required import/service
    -> LocallyReproducibleUnderProfile

required import available OR required external service available
    -> ExternalServiceOrImportDependent

otherwise missing local/tool/metrology/material-process/profile support
    -> Unavailable
```

Useful external dependence is represented honestly; it is not treated as project failure.

### Multi-generation continuity

```text
g1=Unknown -> Unresolved
g1=Insufficient -> InsufficientForProfile

g1=Supported + g2=Supported + degr=true + meet=true
    -> DegradedButSufficientForProfile

g1=Supported + g2=Supported
    -> G2PlusPreserved

g1=Supported + g2=Insufficient + renew=false
    -> RenewalBlocked

g1=Supported + g2=Insufficient + renew=true
    -> InsufficientForProfile

g1=Supported + g2=Unknown
    -> G1Only
```

One repair never implies self-sustaining multi-generation closure.

### CIV-ENG feedback

```text
fb=false -> NotRequested
fb=true + fbr=CIV_ENG -> CanonicalCivEngRoute
fb=true + any other route -> WrongFeedbackRoute
```

SELF-BOOT must not directly launch material discovery or bypass canonical engineering/domain owners.

### Authority

Every case derives exactly:

`NoSelfModificationOrPhysicalExecutionAuthority`

The `auth` input is adversarial. It cannot alter this output.

## SELF-C0 scope

The 36 fixtures cover:

- exact Nix/software recovery vs unavailable hardware;
- slower-but-sufficient and insufficient compute substitutions;
- stocked MCU / PCB assembly / foundry and bare-PCB dependencies;
- board rework vs reproduction;
- local power-module replacement vs missing semiconductor dependency;
- fan/TIM replacement and lifecycle evidence;
- encoder replacement and recalibration;
- sensor/reference/ADC dependence;
- local mechanical parts vs imported actuator stack;
- precision-bearing profile dependence;
- HMI graceful degradation and proprietary maintainability blockers;
- diagnostic-access loss;
- cutting-tool, metrology and G2 precision-ratchet failures;
- external optical-metrology dependencies;
- canonical CIV-ENG feedback and a forbidden direct-material shortcut;
- one bounded synthetic G0→G3+ positive chain.

## Graceful degradation

A lower-performing substitute may preserve continuity when it still satisfies the exact required profile. The same substitute may be insufficient for another profile. No better/worse scalar is inferred.

Examples include lower compute throughput, lower sensor precision, lower bearing speed/precision, or simpler HMI capability.

## Canonical identity

Stored corpus:

`docs/engineering/data/self_boot_001a_reference_v1.json`

Canonicalization is UTF-8 JSON with recursively sorted object keys, separators exactly `,` and `:`, no insignificant whitespace and no trailing newline.

SHA-256:

`9b02598de651ad033f3c0f55fcb4184c8d1d0b456549bfceecccab35cd23773b`

The stored corpus uses exactly those bytes.

## Qualification expectation

A later independent oracle must reconstruct each raw case from `base + o`, derive all eight axes without case-ID branching, decode/compare expected `e` only after derivation, verify exact S01–S36 order/schema/codebook/digest, run independent adversarial mutations, and avoid importing production CIV/SELF/ROB/MAT/ENG decision code.

Qualification order:

```text
SELF-BOOT-001A source/docs/data freeze
-> SELF-BOOT-001A1 independent oracle
-> exact executable qualification receipt
-> adapters only after qualified/current owner surfaces exist
```

Skipped/generic CI, issue review, or an unexecuted upstream qualifier is not PASS.

## Claim ceiling

A future exact-source PASS may establish only deterministic software semantics for the frozen SELF-C0 operation/diagnosis/repair/calibration/productive/generational/feedback/authority distinctions.

It establishes no real hardware availability, autonomous self-maintenance, industrial independence, product safety, physical self-modification authority, procurement/fabrication permission, consciousness, personhood or survival claim.