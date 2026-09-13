# ASSURE-001 — Immutable Multi-Surface AI Subject Manifest

## Governing theorem

```text
same product name
    != same model
    != same prompt
    != same tool authority
    != same policy
    != same runtime
    != same qualified subject

same provider alias
    != immutable revision

unknown surface
    != unavailable surface
    != not-applicable surface
    != omitted surface

known commitment
    != retrievable artifact
    != replayable system

same SHA-256 digest
    + different commitment method/schema/namespace
    != same subject identity
```

ASSURE-001 defines the exact system-under-test identity used by later external AI/agent qualification campaigns. It remains a separate crate so richer AI-specific identity semantics do not broaden the qualified ASSURE-000 claim/evidence kernel.

## Stable subject key

`AiSubjectManifest::subject_key` is semantic and identity-bearing. It is a stable logical key for the system under test, not a UI/display name. Presentation-only labels belong outside the canonical manifest.

Changing the subject key changes both the ASSURE-001 manifest identity and its ASSURE-000 bridged subject identity.

## Registered surface profile

A `SurfaceProfile` defines the complete set of material surfaces a manifest must address. The initial external-AI vocabulary includes source tree, immutable image/artifact, model, system prompt/instructions, tool/MCP authority, policy/authorization, runtime/environment, declared deployment envelope, named external dependencies, and domain-specific custom surfaces.

Profiles are canonical and order-independent. Duplicate registered surfaces fail closed. Canonical profile surfaces and manifest bindings are ordered by lexicographic UTF-8 byte order of their canonical wire names, not Rust enum declaration order.

The golden-vector corpus freezes a profile digest plus a full manifest digest so independent implementations can detect byte-level canonicalization drift.

Evaluator implementation, corpus, intervention schedule, verifier identity, acceptance gates, and analysis plan are intentionally excluded: those describe a qualification campaign, not the system under test.

## Completeness and applicability

Every registered surface must have exactly one binding. Omission fails closed.

Each binding is exactly one of:

```text
Known(material-commitment)
Unknown
Unavailable(reason-class)
NotApplicable
```

Applicable states (`Known`, `Unknown`, `Unavailable`) require a `SurfaceLocator`. `NotApplicable` requires no locator and has one canonical locator-free representation. Therefore arbitrary provider/name/version bytes cannot mint different identities for the same declared absence.

`Known` means an exact typed material commitment is available. It does **not** prove provider truthfulness, artifact retrievability, or provider non-equivocation.

`Unknown` means the surface applies but exact material identity is not known. `Unavailable` means it applies but cannot currently be obtained for a declared reason. `NotApplicable` means it genuinely does not apply; it is not a substitute for missing information.

`CompletenessSummary::has_complete_material_identity()` is true only when every applicable registered surface is `Known`. This is an identity-completeness predicate, not replayability. Replayability additionally requires evidence that committed artifacts/configuration remain obtainable and executable.

## Commitment method and interpretation are identity

A SHA-256 digest is not self-describing. ASSURE-001 binds the preimage interpretation into `MaterialCommitment`:

```text
ArtifactBytesSha256
CanonicalDescriptorSha256 { schema }
ProviderRevisionTokenSha256 { namespace }
CustomSha256(method-id)
```

`ArtifactBytesSha256` means SHA-256 over exact artifact bytes.

`CanonicalDescriptorSha256 { schema }` means SHA-256 over a descriptor under an explicitly identified descriptor/canonicalization schema. The schema ID is part of identity; the same digest under two schema IDs is not the same material commitment.

`ProviderRevisionTokenSha256 { namespace }` binds an exact provider revision token under an explicitly identified provider token namespace/profile. The namespace is identity-bearing. This does not prove that the provider token is immutable, content-addressed, or non-equivocating.

`CustomSha256(method-id)` binds a domain-specific SHA-256 preimage/canonicalization method. Consumers must understand that method before inferring equivalence.

This prevents hashes of artifact bytes, descriptors, provider revision strings, or custom canonical forms from being silently equated merely because all are 32-byte SHA-256 digests.

## Provider aliases and dynamic dependencies

A provider/model/service name, declared version, or rolling endpoint is locator metadata, not an immutable revision.

```text
model = "latest"
    != immutable model commitment

https://tool.example/api
    != immutable tool implementation
```

When an immutable commitment is unavailable, the surface remains `Unknown` or `Unavailable`. Locator metadata is identity-bound because routing can itself be behaviorally material.

Material remote tools, MCP servers, hosted models, retrieval indices, policy services, and other dependencies get explicit registered surfaces. Adding/removing a dependency or changing its locator, state, commitment method/schema, or digest changes subject identity.

## Secret boundary

Raw secrets are not subject identity material. Do not include API keys, bearer tokens, private credential bytes, or hashes of low-entropy secrets merely to make a manifest change when credentials rotate.

Behaviorally material effects of credentials should instead be represented through non-secret commitments such as effective capability/permission scope, role identity where safe to disclose, policy/authorization identity, remote-service identity, or a safe secret-class/version handle.

## ASSURE-000 bridge

ASSURE-001 does not modify `symthaea-assurance-core`.

`AiSubjectManifest::as_core_subject()` commits the complete ASSURE-001 manifest ID into one domain-separated `SubjectComponentKind::Custom("assure-001-ai-subject-manifest")` using the semantic subject key. Therefore:

```text
same ASSURE-001 manifest
    -> same ASSURE-000 subject identity

material ASSURE-001 change
    -> different ASSURE-000 subject identity
```

## Deployment envelope is identity, not authority

A declared deployment envelope may be a subject surface. Its presence records which deployment conditions are being identified; it does not establish deployment eligibility or authorization.

## Deliberate nonclaims

ASSURE-001 does not establish provider truthfulness, artifact availability, replayability, remote-service immutability, prompt secrecy, credential safety beyond the representation boundary, campaign preregistration, successful replication, common-cause verifier independence, certification, compliance, or deployment authority.

Those require later evidence, campaign, lifecycle, policy, or standards layers.
