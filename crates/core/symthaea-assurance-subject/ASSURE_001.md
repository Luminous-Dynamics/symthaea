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
    != uniquely fixed underlying bytes
    != retrievable artifact
    != replayable system

same SHA-256 digest
    + different commitment method/schema/namespace
    != same subject identity
```

ASSURE-001 defines the exact system-under-test identity used by later external AI/agent qualification campaigns. It remains a separate crate so richer AI-specific identity semantics do not broaden the qualified ASSURE-000 claim/evidence kernel.

## Stable subject key

`AiSubjectManifest::subject_key` is semantic and identity-bearing. It is a stable logical key for the system under test, not a UI/display name. Presentation-only labels belong outside the canonical manifest.

## Registered surface profile

A `SurfaceProfile` defines the complete set of material surfaces a manifest must address: source tree, immutable image/artifact, model, system prompt/instructions, tool/MCP authority, policy/authorization, runtime/environment, declared deployment envelope, named external dependencies, and domain-specific custom surfaces.

Profiles are canonical and order-independent. Duplicate registered surfaces fail closed. Canonical surfaces and bindings are ordered by lexicographic UTF-8 byte order of explicit canonical wire names, not implementation-language enum order. Golden vectors freeze a profile digest and full manifest digest for independent implementations.

Evaluator implementation, corpus, intervention schedule, verifier identity, acceptance gates, and analysis plan are intentionally excluded because they describe the campaign rather than the system under test.

## Completeness and applicability

Every registered surface must have exactly one binding. Omission fails closed.

Each binding is exactly one of:

```text
Known(material-commitment)
Unknown
Unavailable(reason-class)
NotApplicable
```

Applicable states (`Known`, `Unknown`, `Unavailable`) require a `SurfaceLocator`. `NotApplicable` requires no locator and has one canonical locator-free representation. Arbitrary provider/name/version bytes therefore cannot mint different identities for the same declared absence.

`Known` means an exact typed commitment is available for the declared surface identity. It does **not** prove provider truthfulness, non-equivocation, content-addressedness, artifact retrievability, or that the underlying bytes are uniquely fixed by the provider.

`Unknown` means the surface applies but no exact commitment is known. `Unavailable` means it applies but cannot currently be obtained for a declared reason. `NotApplicable` means it genuinely does not apply.

`CompletenessSummary::has_complete_committed_identity()` is true only when every applicable registered surface has a typed commitment. This proves commitment completeness only. It does not establish uniquely fixed underlying bytes, artifact availability, provider cooperation, or replayability.

## Commitment method and interpretation are identity

A SHA-256 digest is not self-describing. ASSURE-001 binds the preimage interpretation into `MaterialCommitment`:

```text
ArtifactBytesSha256
CanonicalDescriptorSha256 { schema }
ProviderRevisionTokenSha256 { namespace }
CustomSha256(method-id)
```

`ArtifactBytesSha256` means SHA-256 over exact artifact bytes.

`CanonicalDescriptorSha256 { schema }` means SHA-256 over a descriptor under an explicitly identified descriptor/canonicalization schema. The schema ID is part of identity.

`ProviderRevisionTokenSha256 { namespace }` binds the exact provider token bytes under an explicitly identified provider token namespace/profile. This records the token exactly; it does not establish that the provider maps the token immutably or non-equivocally to one artifact.

`CustomSha256(method-id)` binds a domain-specific SHA-256 preimage/canonicalization method. Consumers must understand that method before inferring equivalence.

The same digest under different methods, schema IDs, or token namespaces is therefore a different material commitment.

## Provider aliases and dynamic dependencies

A provider/model/service name, declared version, or rolling endpoint is locator metadata, not an immutable revision. When an immutable commitment is unavailable, the surface remains `Unknown` or `Unavailable`. Locator metadata is identity-bound because routing can itself be behaviorally material.

Material remote tools, MCP servers, hosted models, retrieval indices, policy services, and other dependencies get explicit registered surfaces. Adding/removing a dependency or changing its locator, state, commitment method/schema/namespace, or digest changes subject identity.

## Secret boundary

Raw secrets are not subject identity material. Do not include API keys, bearer tokens, private credential bytes, or hashes of low-entropy secrets merely to make a manifest change when credentials rotate.

Behaviorally material effects of credentials should instead be represented through non-secret commitments such as effective capability/permission scope, role identity where safe to disclose, policy/authorization identity, remote-service identity, or a safe secret-class/version handle.

## ASSURE-000 bridge

ASSURE-001 does not modify `symthaea-assurance-core`.

`AiSubjectManifest::as_core_subject()` commits the complete ASSURE-001 manifest ID into one domain-separated `SubjectComponentKind::Custom("assure-001-ai-subject-manifest")` using the semantic subject key. The qualified ASSURE-000 binding rules therefore remain unchanged while richer subject identity lives in this separate layer.

## Deployment envelope is identity, not authority

A declared deployment envelope may be a subject surface. Its presence records which deployment conditions are identified; it does not establish deployment eligibility or authorization.

## Deliberate nonclaims

ASSURE-001 does not establish provider truthfulness, non-equivocation, content-addressedness of provider tokens, artifact availability, replayability, remote-service immutability, prompt secrecy, credential safety beyond the representation boundary, campaign preregistration, successful replication, common-cause verifier independence, certification, compliance, or deployment authority.
