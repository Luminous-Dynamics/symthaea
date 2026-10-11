# Symthaea Systems Engineering Semantic Hardening Profile v1

Status: architecture profile only. No runtime authority is established by this document.

Tracking: #3677, #3692

## Purpose

SE-001 establishes the first typed descriptive systems-engineering graph. This profile freezes the semantic work that should follow qualification of that substrate without expanding SE-002 into a catch-all layer.

The central rule is:

```text
semantic model state
!= engineering evidence
!= evidence currentness
!= requirement satisfaction
!= qualification
```

This profile therefore strengthens what the model can *describe* while preserving ETK as the authority boundary for evidence admission, applicability/currentness, assurance consequences, and downstream decisions.

## Scope boundary

This profile does not authorize changes to the frozen SE-001 subject while its exact-subject qualification remains pending.

SE-002 should remain narrowly focused on stakeholder-need / requirement decomposition and read-only composition with existing ETK requirement/obligation facts.

The semantic hardening work described here belongs in follow-on PRs.

## Standards alignment

Prefer established interoperable semantics where they fit.

### SysML v2 / KerML

Use SysML v2 / KerML as the primary external systems-modeling semantic reference rather than inventing a proprietary MBSE language.

### SysML v2 Quantities and Units

Align physical quantity semantics with the SysML v2 Quantities and Units Domain Library where practical.

Internal Rust types may be optimized for correctness and ergonomics, but external mappings should preserve quantity kind, unit identity, source lexical representation, and conversion provenance.

### ReqIF and OSLC Requirements Management

Treat ReqIF as file-oriented requirements interchange.

Treat OSLC Requirements Management as a complementary live-resource/API interoperability target for organizations whose requirements live in lifecycle-management tools.

Neither external representation implies that a requirement is accepted by Symthaea or ETK.

### W3C PROV

W3C PROV may be used as an outward projection for generic provenance interoperability.

It does not replace richer internal identities, evidence lineage, configuration scope, or ETK authority semantics.

## Semantic layers

Keep at least these categories distinct:

```text
source representation
        ↓
semantic model object
        ↓
candidate engineering interpretation
        ↓
analysis / observation
        ↓
candidate evidence
        ↓
ETK admission/currentness/assurance
```

A projection or analysis may reference multiple layers but must not collapse them.

## Quantities, units, frames, and time bases

A systems-engineering model needs more than free-form numeric values.

Future typed semantics should support at least:

- quantity kind / physical dimension;
- scalar, vector, tensor, or structured quantity shape where needed;
- unit identity;
- exact value representation;
- uncertainty association where relevant;
- source lexical representation;
- conversion identity and provenance;
- coordinate/reference frame identity;
- clock/time-base identity for temporal data;
- validity envelope / operating range.

Freeze:

```text
unit convertible
!= measurement equivalent
```

Examples include measurements expressed in compatible units but taken in different coordinate frames, reference temperatures, clock domains, calibration contexts, or operating conditions.

No implicit unit guessing is allowed on an authority-bearing path.

No implicit frame transformation is allowed without a declared transform and provenance.

## Interface contracts

An interface node should eventually become an explicit contract rather than a label connecting components.

A typed interface contract may include:

- endpoint / port identities;
- directionality;
- quantity and unit contract;
- value/range envelope;
- rate, latency, jitter, deadline, or synchronization constraints;
- reference frame;
- clock/time base;
- schema/protocol identity;
- error model;
- failure/degraded behavior;
- initialization/shutdown behavior;
- ownership/responsibility boundary;
- compatibility-analysis result references.

Representative interface classes include:

- electrical power/data bus;
- fluid port;
- mechanical mounting interface;
- thermal interface;
- RF/signal interface;
- software API;
- network link;
- sensor/telemetry stream.

Freeze:

```text
interface compatibility calculation
!= successful integration
```

## Requirement quality and verification intent

The semantic layer should support deterministic diagnostics for requirement quality without becoming the requirement-acceptance authority.

Candidate diagnostics include:

- ambiguous or subjective wording;
- missing quantity/unit;
- unspecified tolerance;
- undefined reference;
- unbounded term;
- compound requirement containing multiple obligations;
- unverifiable formulation;
- conflicting requirement pair;
- inconsistent modal language;
- missing applicability scope;
- missing verification intent.

Represent verification intent explicitly using at least:

```text
Analysis
Inspection
Demonstration
Test
```

with optional method/tool/fixture intent.

Freeze:

```text
quality lint passes
!= requirement accepted

verification activity planned
!= verification passed
```

## Assumption registry

Assumptions should become first-class, scoped and testable.

An assumption should be able to reference:

- source/rationale;
- affected subject/configuration;
- operating envelope;
- effective time window where applicable;
- review/expiry trigger;
- verification/test intent;
- dependent model objects;
- superseding assumption;
- external source identity;
- uncertainty/unknown marker.

A violated, expired, superseded, or changed assumption may generate a candidate applicability review.

It must not directly mark ETK evidence invalid.

## Configuration semantics

A generic `IncludedInConfiguration` relationship is useful but insufficient for real lifecycle work.

Future semantics should distinguish:

- membership;
- baseline;
- variant;
- option/feature selection;
- inheritance/derivation;
- supersession;
- as-specified;
- as-designed;
- as-built;
- as-tested;
- as-operated;
- external configuration-management source identity.

Configuration identity must be content/lineage aware enough that analysis and evidence can state exactly which subject they apply to.

Freeze:

```text
configuration membership
!= baseline approval
!= release
!= deployment
!= manufacturing authority
```

## Lifecycle transition policy

Descriptive lifecycle states should remain separate from acceptance/currentness/qualification vocabulary.

A later versioned `LifecycleTransitionPolicy` should define:

- permitted transitions;
- reversible vs irreversible transitions;
- transitions requiring new semantic identity;
- transition rationale;
- policy/profile version;
- migration behavior when policy versions change.

A generic node replacement operation must not silently define lifecycle policy.

## Change-impact propagation policy

SE-001 proves that deterministic descriptive impact paths are possible. The next layer should move propagation meaning into an explicit policy object instead of allowing relation implementation details to become permanent semantics.

A versioned `ImpactPropagationPolicy` should define, per relation class:

- forward propagation;
- reverse propagation;
- bidirectional propagation;
- no propagation;
- direct vs transitive classification;
- stop conditions;
- scope filters;
- rationale;
- policy version.

Impact outputs should distinguish at least:

```text
DirectSemanticImpact
TransitiveSemanticImpact
ApplicabilityReviewRequired
CandidateEvidenceStaleness
```

`EstablishedApplicabilityLoss` remains an ETK/currentness conclusion.

Freeze:

```text
semantic impact path
!= evidence invalidation
```

## Review of authority-adjacent vocabulary

After SE-001 exact-subject qualification, review relation names that could be misread as authority-bearing.

In particular, `InvalidatesCandidate` must either be demonstrated to apply only to explicitly non-authoritative candidate semantics or be renamed/replaced with less authority-adjacent vocabulary such as a review/staleness candidate relation.

Do not mutate the frozen SE-001 subject solely to make this naming improvement before its qualification result exists.

## External source and projection provenance

Imported model objects should preserve both their projected semantic identity and their source identity.

A future projection record should carry at least:

- source standard/tool family;
- source standard/tool version;
- source document/model identity;
- source revision;
- external object ID;
- lexical/source artifact digest;
- projection adapter identity/version;
- supported semantics set;
- unsupported semantics set;
- lossy-projection report;
- round-trip capability class;
- normalized internal object identity.

Freeze:

```text
parsed successfully
!= imported completely
!= semantically faithful
```

## Digital twin boundary

Operational/digital-twin standards such as Asset Administration Shell and OPC UA should project into the semantic/configuration/observation layers rather than becoming alternate authority models.

The semantic core should provide enough stable configuration, quantity, interface, source and time identities that those integrations do not need to invent parallel concepts.

## Adversarial semantic corpus

Qualification should include deliberately difficult fixtures such as:

- dimensionally incompatible values;
- compatible units but incompatible reference frames;
- clock-domain mismatch;
- range mismatch across an interface;
- timing contract mismatch;
- stale external model revision;
- assumption expiry;
- circular variant/baseline derivation;
- forbidden lifecycle transition;
- propagation-policy version drift;
- changed model with stale impact result;
- lossy projection presented as complete;
- semantic impact presented incorrectly as ETK invalidation.

Failures should be typed and reproducible.

## Proposed implementation sequence

```text
SE-SEM-000  semantic hardening profile (this document)
SE-SEM-001  quantities / units / frames / time bases
SE-SEM-002  typed interface contracts
SE-SEM-003  requirement-quality + verification-intent semantics
SE-SEM-004  assumption registry
SE-SEM-005  configuration / baseline / variant semantics
SE-SEM-006  lifecycle transition policy
SE-SEM-007  versioned impact-propagation policy
SE-SEM-008  external source/projection provenance
SE-SEM-009  adversarial semantic corpus
```

Implementation PRs remain blocked on the lower semantic contracts they require.

## Integration consequences

These semantics should become shared substrate for later integrations:

- FMI/SSP: units, clocks, variables, configurations and source identity;
- AADL/OSATE: interfaces, timing, modes and allocations;
- OpenMDAO/Dakota/CasADi: quantities, constraints and exact configurations;
- CAD/mesh/EDA: geometry frames, units and configuration lineage;
- digital twins: as-built/as-operated configuration plus time/observation identity;
- fabrication/metrology: dimensional quantities, frames and as-built divergence;
- HDC/LTC cognition: proposal targets that remain separate from model truth.

## Non-goals

- no second ETK;
- no universal confidence scalar;
- no automatic requirement acceptance;
- no implicit unit or coordinate-frame guessing;
- no graph-change-to-evidence-invalidation shortcut;
- no configuration-to-release shortcut;
- no external mapping presented as lossless without evidence;
- no lifecycle state that silently encodes deployment or certification authority.

## Closure criterion

This tranche is mature when Symthaea can represent an exact engineered configuration with dimensionally meaningful quantities, explicit interface contracts, scoped assumptions, baseline/variant lineage, versioned lifecycle and impact semantics, and external projection provenance while preserving the distinction between:

```text
what the model says
what a tool calculated
what was observed
what may need review
what evidence remains current
what ETK permits us to rely upon
```
