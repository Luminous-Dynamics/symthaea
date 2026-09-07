# Physical Agency Blinded Confirmatory Measurement v1

Status: design contract only. This document introduces no HAL path, actuator API, execution permit, or physical authority.

## Motivation

PA-16 freezes a normalized-canonical outcome claim before execution, binds every judged metric name into the strict solver lineage, and evaluates returned evidence conservatively under uncertainty.

A stronger confirmatory architecture should also reduce unnecessary information disclosure to an untrusted simulator. A backend generally needs to know **what physical quantity to compute**, in **what unit**, with **what numerical inputs**, and with **what uncertainty strength**. It does not normally need the exact pass/fail threshold that the trusted evaluator will apply afterward, nor human/audit commentary that is irrelevant to the numerical problem.

The current lower-level request types mix computational fields with audit-facing text:

```text
SimulationRequest::objective              human-readable purpose
ModelParameter::provenance               audit/source label
SimulationContextRef::provenance_ref     audit/source reference
```

The strict canonical request transcript retains provenance for internal lineage and deliberately excludes `objective`, but a context-aware backend currently receives the complete request/context structures. Therefore simply withholding the typed decision claim is insufficient for a meaningful blinded profile.

The strongest strict-confirmatory profile should enforce:

```text
MeasurementPlanFrozenBeforeDecisionClaim
DecisionClaimFrozenBeforeExecution
BackendReceivesOnlyComputationalProjection
BackendVisibleMeasurementContractMinimized
```

and retain typed runtime receipts for every boundary.

## Core ordering invariant

The default blinded profile uses a two-stage preregistration sequence:

```text
SelectedCandidate
    + strict simulation request
    + measurement specification
        -> FrozenMeasurementPlan

FrozenMeasurementPlan
    + full decision claim
        -> PreparedBlindedConfirmatorySimulation

PreparedBlindedConfirmatorySimulation
        -> backend sees sanitized computational request
        -> backend sees measurement contract only
        -> result fixed
        -> trusted evaluator applies hidden decision claim
```

The machine request, model parameters, solver identity, contexts, requested metric identities, units, and required uncertainty strength are frozen before the decision claim is attached. The decision claim is then preregistered before solver execution.

This does not prove a human or caller did not know the intended threshold earlier. R1 structural evidence cannot establish that epistemic fact. It does prove that the trusted runtime did not mutate the frozen measurement plan after the decision claim entered the strict path.

## Internal lineage versus provider-visible lineage

PA-18 distinguishes two request identities because Symthaea should retain more audit information than an external provider needs to see.

```text
InternalCanonicalRequestTranscript
    full strict machine/audit lineage retained by Symthaea

CanonicalBackendRequestTranscript
    exact computational projection actually supplied to provider
```

These identities must not be conflated.

A provider/runtime attestation should be able to attest exactly what crossed the provider boundary without falsely claiming it consumed internal audit metadata that was deliberately withheld.

The final qualification retains both identities and proves that the backend projection was derived from the frozen internal request.

## Stage 1 — Frozen measurement plan

The first stage freezes only what is needed to define the numerical experiment and the required measurements.

Conceptual shape:

```rust
pub struct MeasurementRequirement {
    metric_name: String,
    unit: String,
    uncertainty: MeasurementUncertaintyRequirement,
}

pub enum MeasurementUncertaintyRequirement {
    AllowPointEstimate,
    RequireInterval,
}

pub struct FrozenMeasurementPlan {
    selected: SelectedCandidate,
    strict_request: PreparedSelectedSimulation,
    internal_request_transcript: CanonicalRequestTranscript,
    backend_request: BackendMeasurementRequest,
    backend_request_transcript: CanonicalBackendRequestTranscript,
    measurement_transcript: CanonicalMeasurementContractTranscript,
}
```

Construction is private/non-serializable.

The measurement plan MUST validate:

- exact selected proposal/transition/world lineage;
- strict request validity;
- one unit identity per measurement metric name;
- unique measurement metric names;
- unique `SimulationRequest::requested_metrics` names;
- exact set equality between confirmatory requested metric names and measurement requirements in the strongest profile;
- explicit uncertainty requirements;
- deterministic derivation of the backend-facing projection from the frozen strict request;
- byte-stable internal and backend request transcripts.

The exact-set rule prevents an ambiguous situation where a strict confirmatory request contains undeclared extra requested metrics with no typed measurement contract. Exploratory/auxiliary metrics should use a separate run or a future explicitly typed auxiliary-output contract rather than silently entering the strict confirmatory request.

## Sanitized backend request

The backend MUST NOT receive the legacy `SimulationRequest` or full `SimulationContextRef` objects directly in `BlindMeasurementOnly`.

Instead the trusted layer derives a computational projection.

Conceptual shape:

```rust
pub struct BackendModelParameter {
    name: String,
    value: f64,
    unit: String,
    uncertainty: Option<UncertaintyEstimate>,
}

pub struct BackendContextRef {
    kind: SimulationContextKind,
    context_id: String,
    digest_algorithm: ContextDigestAlgorithm,
    digest: String,
    frame_id: Option<String>,
}

pub struct BackendMeasurementRequest {
    request_id: String,
    domain: EngineeringDomain,
    solver: SolverKind,
    parameters: Vec<BackendModelParameter>,
    requested_metrics: Vec<String>,
    contexts: Vec<BackendContextRef>,
}
```

The provider-visible projection deliberately excludes:

```text
SimulationRequest::objective
ModelParameter::provenance
SimulationContextRef::provenance_ref
post-hoc evaluator annotations
free-form safety commentary
```

Those fields remain available inside Symthaea's internal evidence lineage where appropriate, but they are not numerical inputs and therefore should not cross the blinded provider boundary by default.

This gives the stronger relationship:

```text
internal provenance-bearing request
        -> trusted deterministic projection
        -> provider computational request
```

rather than:

```text
internal request == provider request
```

## Canonical backend-request transcript

The exact provider-visible projection MUST have its own canonical transcript.

Conceptual identity:

```text
CanonicalBackendRequestTranscript
    schema/domain separator
    request id
    engineering domain
    solver
    canonical parameter name/value/unit/uncertainty
    canonical requested metric names
    canonical backend context identities/digests/frame ids
```

It excludes all fields intentionally withheld from the provider.

The transcript is computed inside the trusted layer from the same `BackendMeasurementRequest` object that is passed to the adapter. The provider returns/echoes this exact transcript as structural consumption evidence.

A compact cryptographic digest may later be derived for indexing or attestation, but exact canonical bytes remain the source of truth for R1 equality.

## Model parameters and disclosure limits

Numerical model parameters remain provider-visible because they define the physical/numerical problem and are frozen before the decision claim is attached.

Parameter provenance strings do not cross the provider boundary in the blind profile; parameter values, units, names, and quantified uncertainty do.

Therefore the blinded profile does **not** claim that the backend cannot infer a decision threshold from legitimate model inputs, parameter names, domain knowledge, repeated experiments, external context, or caller behavior.

The structural guarantee is narrower:

- the trusted runtime does not pass the decision-claim transcript to the backend;
- the trusted runtime does not pass free-form request objective text;
- the trusted runtime does not pass parameter/context provenance commentary;
- the frozen computational request cannot be mutated after the decision claim is attached;
- any threshold genuinely required by the numerical problem must already exist in the frozen machine request before decision-claim preregistration.

A future stronger anti-leakage profile may add independent request construction, opaque provider-facing identifiers, or stronger role separation. PA-18 does not overclaim those properties.

## Stage 2 — preregistered decision claim

After the measurement plan is frozen, the full outcome claim may be attached.

Conceptual shape:

```rust
pub struct PreparedBlindedConfirmatorySimulation {
    measurement_plan: FrozenMeasurementPlan,
    decision_claim: SimulationOutcomeClaim,
    decision_transcript: CanonicalDecisionClaimTranscript,
    disclosure_profile: ConfirmatoryDisclosureProfile,
}
```

The decision claim contains the complete success semantics:

```text
claim schema
claim id
transition id
proposal id
metric name
metric unit
predicate kind
predicate threshold(s)
uncertainty policy
aggregation semantics
```

The preregistration step MUST prove:

- proposal/transition identity matches the frozen measurement plan;
- each claimed metric name maps to exactly one frozen measurement requirement;
- each claimed metric uses exactly the frozen unit;
- claim uncertainty requirements do not exceed the frozen measurement plan;
- the full claim remains normalized-canonical and scalar-satisfiable under PA-16 rules;
- the frozen internal request, backend projection, contexts, parameters, and measurement transcript are unchanged while attaching the decision claim.

For example:

```text
FrozenMeasurementPlan:
    stress / MPa / RequireInterval

DecisionClaim:
    stress / MPa <= 20
        -> admissible

DecisionClaim:
    stress / Pa <= 20_000_000
        -> reject
```

## Measurement transcript versus decision transcript

The strict path retains two different claim-related identities.

### CanonicalMeasurementContractTranscript

Provider-visible.

It contains only measurement semantics needed to produce evidence:

```text
measurement schema/domain separator
transition/proposal binding
metric name
exact unit identity
required uncertainty strength
```

It deliberately excludes:

```text
predicate kind
threshold values
pass/fail aggregation
number of predicates applied to one measurement when the same requirements result
```

### CanonicalDecisionClaimTranscript

Evaluator-only under the strongest default profile.

It contains the complete normalized-canonical decision claim, including predicate semantics and thresholds.

Two decision claims may share one measurement transcript while remaining different claims:

```text
stress / MPa <= 20
stress / MPa <= 30

same measurement transcript
different decision transcript
```

That is a desired property.

## Measurement-contract minimization

The provider-visible contract should disclose no more decision information than required to produce the evidence.

For the current scalar predicate algebra, one measurement requirement exists per metric name/unit pair. If multiple later criteria reference one frozen measurement, the claim may only demand evidence strength at or below the frozen measurement requirement.

If a frozen plan allows point-only evidence and a later claim requires an interval, preregistration fails before execution.

If a frozen plan requires an interval and the claim would permit a point estimate, the stricter frozen measurement requirement remains authoritative.

Adding a new provider-visible field is a contract change and requires review because it may increase decision leakage.

## Confirmatory disclosure profile

Conceptual policy:

```rust
pub enum ConfirmatoryDisclosureProfile {
    BlindMeasurementOnly,
    ThresholdVisible,
}
```

`BlindMeasurementOnly` is the strongest default profile.

Under `BlindMeasurementOnly`:

- decision transcript is not passed to the backend;
- legacy request objective is not passed to the backend;
- audit provenance strings are not passed to the backend;
- only the sanitized computational request and measurement transcript cross the boundary.

Some solvers genuinely require a threshold for event detection, stopping criteria, phase-boundary search, or threshold-triggered outputs. Those runs must not masquerade as blinded.

If a decision threshold must be intentionally revealed after the decision claim is attached, use `ThresholdVisible` and retain that fact in every downstream receipt. A policy requiring blinded evidence must reject threshold-visible receipts.

A threshold independently present in the frozen physical model/request before the decision claim is attached remains a machine parameter, not a post-hoc disclosure. PA-18 records this limitation rather than claiming the provider has no threshold information whatsoever.

## Proposed backend surface

The strongest adapter boundary is measurement-aware rather than decision-aware.

```rust
pub trait MeasurementAwareSimulationBackend: Debug + Send + Sync {
    fn name(&self) -> &'static str;
    fn supported_solvers(&self) -> &[SolverKind];

    fn run_measurement_bound(
        &self,
        request: &BackendMeasurementRequest,
        measurement: &CanonicalMeasurementContractTranscript,
    ) -> Result<MeasurementAwareSimulationResult, SimulationError>;
}

pub struct MeasurementAwareSimulationResult {
    pub result: SimulationResult,
    pub consumption: MeasurementConsumptionEvidence,
}

pub struct MeasurementConsumptionEvidence {
    pub backend_request_transcript: CanonicalBackendRequestTranscript,
    pub measurement_transcript: CanonicalMeasurementContractTranscript,
    pub consumed_contexts: Vec<BackendContextRef>,
}
```

Ordinary `SimulationBackend` and ordinary `ContextAwareSimulationBackend` implementations MUST NOT automatically satisfy this boundary.

## Measurement-consumption validation

The trusted registry should:

1. accept only a privately constructed `PreparedBlindedConfirmatorySimulation`;
2. select an explicit `MeasurementAwareSimulationBackend`;
3. pass the sanitized backend request and exact measurement transcript;
4. require the backend's returned backend-request transcript to equal the frozen provider-visible transcript byte-for-byte;
5. require the returned measurement transcript to equal the frozen measurement transcript byte-for-byte;
6. require consumed backend contexts to equal the frozen sanitized context set exactly;
7. validate result request id, engineering evidence mode, backend identity, solver/parser metadata, input/output digests, and other lower strict-result rules;
8. validate returned metric cardinality and units against the measurement plan;
9. validate PA-16 estimate/interval internal consistency;
10. only then fix the result as admissible confirmatory evidence;
11. apply the separately retained decision claim after the result is fixed.

The provider does not need to echo internal provenance strings it was deliberately not shown. Internal provenance remains linked through the `FrozenMeasurementPlan` and final receipt.

## Returned-metric uniqueness

Each frozen confirmatory metric name must map to exactly one returned metric with the exact frozen unit.

```text
measurement: stress / MPa
returned: stress / MPa                  -> structurally admissible
returned: stress / Pa                   -> reject
returned: stress / MPa + stress / Pa    -> reject
returned: stress / MPa + stress / MPa   -> reject
```

Unrelated solver metadata may remain present in non-metric evidence fields. Extra metric outputs are not part of the strongest strict confirmatory request; the frozen request metric set equals the measurement-specification metric set.

No implicit unit conversion is performed in v1.

```text
"Pa" != "kPa" != "MPa"
"m" != "meter"
"1" != "%"
```

A future separately qualified typed-unit profile may reuse the workspace's existing `uom` foundation but must not silently reinterpret historical string-valued evidence.

## Runtime receipts

Successful validation must mint typed non-serializable receipts rather than transient booleans.

### FrozenMeasurementPlan

Proves only that the trusted runtime froze the internal request, computational provider projection, and measurement requirements before decision-claim attachment.

### MeasurementConsumptionReceipt

Conceptual shape:

```rust
pub struct MeasurementConsumptionReceipt {
    backend: String,
    internal_request_transcript: CanonicalRequestTranscript,
    backend_request_transcript: CanonicalBackendRequestTranscript,
    measurement_transcript: CanonicalMeasurementContractTranscript,
    disclosure_profile: ConfirmatoryDisclosureProfile,
}
```

It proves only that this runtime path supplied the exact sanitized computational request and measurement contract to the adapter API and accepted the exact echoed provider-visible lineage under strict validation.

### BlindedConfirmatorySimulationQualification

Final runtime qualification retains both the measurement and decision sides:

```rust
pub struct BlindedConfirmatorySimulationQualification {
    pa16_lineage: ClaimBoundConfirmatorySimulationQualification,
    measurement_consumption: MeasurementConsumptionReceipt,
    decision_transcript: CanonicalDecisionClaimTranscript,
    disclosure_profile: ConfirmatoryDisclosureProfile,
}
```

The final receipt binds:

```text
exact internal request/context/result lineage
+ exact provider-visible computational request lineage
+ exact frozen measurement plan
+ exact measurement transcript
+ exact separately frozen decision transcript
+ disclosure profile
+ exact safety lineage
```

Serialized strings cannot recreate these receipts without rerunning trusted checks.

## Final identity and v5 profile

A human/audit reference may use a new domain such as:

```text
physical-agency-confirmatory:v5-blinded
```

but the string is never sufficient runtime authority.

Two runs with identical frozen measurement plans but different decision thresholds MUST have:

```text
same CanonicalBackendRequestTranscript
same CanonicalMeasurementContractTranscript
different CanonicalDecisionClaimTranscript
different final qualification identity
```

PA-16 v4 remains a valid historical R1 profile and must not be silently relabeled as blinded v5 evidence.

## Required adversarial tests

Before implementation may be promoted, PHYSIS must reject at least:

```text
measurement plan mutated after decision-claim attachment
backend receives legacy objective in BlindMeasurementOnly profile
backend receives parameter provenance in BlindMeasurementOnly profile
backend receives context provenance_ref in BlindMeasurementOnly profile
backend implements ContextAware but not MeasurementAware
missing backend-request consumption echo
substituted backend-request transcript
missing measurement-consumption echo
substituted measurement transcript
backend context set differs from sanitized frozen context set
same metric name assigned different units in one measurement plan
requested metric set differs from measurement-specification set
claim unit differs from frozen measurement unit
claim requires interval but frozen plan allows point only
same confirmatory metric returned in multiple units
duplicate returned metric with exact name/unit
returned metric unit differs from frozen measurement unit
reported estimate outside its own interval
serialized/fabricated FrozenMeasurementPlan construction
serialized/fabricated MeasurementConsumptionReceipt construction
v4-only qualification offered where blinded v5 is required
ThresholdVisible receipt relabeled as BlindMeasurementOnly
```

PHYSIS must positively establish:

```text
measurement plan frozen
    -> decision claim attached
    -> internal request unchanged
    -> backend projection unchanged
```

and:

```text
same frozen measurement requirements
+ different hidden thresholds
    -> same backend-request transcript
    -> same measurement transcript
    -> different decision transcripts
    -> different final qualification identities
```

and:

```text
legacy objective contains pass threshold
parameter provenance contains pass threshold
context provenance_ref contains pass threshold
+ BlindMeasurementOnly
    -> all three absent from provider-visible request
```

## Evidence tier

This entire profile remains **R1 StructuralBound** evidence.

It proves:

```text
machine measurement plan froze before decision-claim attachment
+
decision claim froze before execution
+
provider-visible request was deterministically projected from frozen internal lineage
+
free-form objective/provenance metadata were excluded from provider visibility
+
exact computational request + measurement requirements crossed the adapter API
+
backend echoed the exact provider-visible lineage it claims to have consumed
+
trusted registry validated provider-visible and internal lineages together
+
final typed receipt retained measurement + decision identities separately
```

It does NOT prove:

```text
the backend is honest
or
the external solver understood the measurement contract
or
the provider did not infer the hidden threshold from external information
or
the machine parameters were authored without prior knowledge of the threshold
or
the backend input/output digests are authentic
or
the physical model is correct
```

Those stronger properties belong to PA-17 R2/R3/R4 authenticated and independent evidence.

## Relationship to authenticated evidence

PA-17 remains the trust progression:

```text
R1 StructuralBound
    -> R2 SignatureVerified
    -> R3 LifecycleGoverned
    -> R4 IndependentQuorum
```

Future provider/runtime attestations should distinguish what the provider actually saw from what Symthaea retained internally.

### Pre-/during-run provider scope

```text
CanonicalBackendRequestTranscript
CanonicalMeasurementContractTranscript
ConfirmatoryDisclosureProfile
```

### Internal/final qualification scope

```text
CanonicalRequestTranscript
CanonicalBackendRequestTranscript
CanonicalMeasurementContractTranscript
CanonicalDecisionClaimTranscript
ConfirmatoryDisclosureProfile
exact output/safety lineage
```

That preserves blinding while still allowing the final trusted qualification to prove exactly which decision criterion was applied to which fixed result and which computational request the provider actually received.

## Public API boundary

Once implemented, the strongest public confirmatory path should require the two-stage measurement-first preparation and a measurement-aware registry.

The older context-only and PA-16 v4 paths may remain available as historical/lower-assurance profiles but must not silently satisfy policy requiring blinded v5 evidence.

```text
ContextAwareSimulationBackend
    -> strict context evidence

MeasurementAwareSimulationBackend
    -> sanitized computational request
    + exact measurement contract
    -> MeasurementConsumptionReceipt
    -> eligible for blinded v5 confirmatory qualification
```

## Execution boundary

Nothing in this contract grants physical authority.

```text
BlindedConfirmatorySimulationQualification
    != ExecutionPermit
    != HALCapability
    != ActuatorCommand
```

The HAL/interlock remains independently authoritative below cognition. Physical execution remains out of scope until strict simulation qualification, authenticated evidence, and runtime authority are separately qualified.