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

same digest algorithm
    != same commitment semantics
```

ASSURE-001 defines the system-under-test identity used by later external AI/agent qualification campaigns. It is deliberately separate from ASSURE-000 so richer subject semantics cannot destabilize the qualified generic claim/evidence kernel.

## Registered surface profile

A `SurfaceProfile` defines the complete set of material surfaces a manifest is expected to address. The initial external-AI vocabulary includes:

- source tree;
- immutable image/artifact;
- model;
- system prompt/instructions;
- tool/MCP authority;
- policy/authorization;
- runtime/environment;
- declared deployment envelope;
- named external dependencies;
- domain-specific custom surfaces.

Profiles are canonical and order-independent. Duplicate registered surfaces fail closed.

Canonical profile surfaces and manifest bindings are ordered by lexicographic UTF-8 byte order of their canonical wire names (for example `custom:aaa` sorts before `model`). Rust enum declaration order has no wire authority. The `canonical_subject_golden_vector_v1` test freezes both a profile digest and full manifest digest so independent implementations can detect byte-level canonicalization drift.

Evaluator implementation, corpus, intervention schedule, verifier identity, acceptance gates, and analysis plan are intentionally excluded: those describe the qualification campaign, not the system under test.

## Completeness is identity

Every registered surface must have exactly one binding. Omission fails closed rather than being silently interpreted as uncertainty.

Each binding carries one of four states:

```text
Known(material-commitment)
Unknown
Unavailable(reason-class)
NotApplicable
```

These states are identity-distinct.

`Known` means only that an exact typed material commitment is available for the identity being represented. It does **not** prove that a provider's description is truthful, that the committed artifact is retrievable, or that the upstream provider cannot equivocate. Those are evidence/attestation/availability questions for later assurance layers.

`Unknown` means the surface applies but the exact material identity is not known.

`Unavailable` means the surface applies but the exact identity cannot currently be obtained for a declared reason such as provider non-disclosure, access denial, or missing measurement.

`NotApplicable` means the registered surface genuinely does not apply to this subject. It is not a substitute for missing information.

`CompletenessSummary::has_complete_material_identity()` is true only when every applicable registered surface is `Known`. This is an identity-completeness predicate, **not replayability**. Replayability additionally requires evidence that the committed artifacts/configuration can actually be obtained and executed under the relevant environment.

## Commitment method is part of identity

A SHA-256 digest is not self-describing. ASSURE-001 therefore binds the commitment method and digest together as `MaterialCommitment`.

Initial methods are:

```text
ArtifactBytesSha256
CanonicalDescriptorSha256
ProviderRevisionTokenSha256
Custom(method-id)
```

The same 32-byte digest under two different methods yields different manifest identity.

`ArtifactBytesSha256` means SHA-256 over the exact artifact bytes.

`CanonicalDescriptorSha256` is only cross-implementation meaningful when the descriptor schema and canonicalization algorithm are specified by the surrounding assurance contract. The enum name alone does not invent a canonical descriptor format.

`ProviderRevisionTokenSha256` binds the exact provider revision token being claimed. It does **not** prove that the provider's token is immutable, content-addressed, or non-equivocating.

`Custom(method-id)` is domain-separated by method identity; consumers must understand that method before inferring equivalence.

This distinction prevents a hash of model bytes from being silently equated with a hash of a provider revision string merely because both use SHA-256.

## Provider aliases are not revisions

A provider/model/service name or rolling endpoint is locator metadata, not an immutable revision.

```text
model = "latest"
    != immutable model commitment

https://tool.example/api
    != immutable tool implementation
```

When an immutable commitment is unavailable, the surface must remain `Unknown` or `Unavailable`. ASSURE-001 never hashes a rolling alias and then calls that hash an immutable model revision.

Locator metadata is bound into manifest identity because provider/name/version routing can itself be behaviorally material. A future explicitly non-semantic metadata layer may carry presentation-only labels; ASSURE-001 does not silently guess that a locator is harmless metadata.

## Dynamic external dependencies

Remote tools, MCP servers, hosted models, retrieval indices, policy services, and other remote dependencies can mutate behind stable names. Material external services therefore get their own registered surface identities.

Adding or removing a registered dependency changes the profile identity. Changing its locator, completeness state, commitment method, or digest changes the manifest identity.

## Secret boundary

Raw secrets are not subject identity material.

Do not include API keys, bearer tokens, private credential bytes, or other recoverable secrets in locator fields or commitments merely to make a manifest change when credentials rotate.

Behaviorally material effects of credentials should instead be represented by non-secret surfaces such as:

- effective capability/permission scope;
- account/tenant role where safe to disclose;
- policy/authorization identity;
- remote-service identity;
- a safe secret-class/version handle when rotation materially changes behavior.

A digest of a low-entropy secret is still a secret-leak risk and is not an acceptable substitute for this boundary.

## ASSURE-000 bridge

ASSURE-001 does not modify `symthaea-assurance-core`.

`AiSubjectManifest::as_core_subject()` commits the complete ASSURE-001 manifest ID into a domain-specific `SubjectComponentKind::Custom("assure-001-ai-subject-manifest")`. Therefore:

```text
same ASSURE-001 manifest
    -> same ASSURE-000 subject identity

material ASSURE-001 manifest change
    -> different ASSURE-000 subject identity
```

This gives later claims and evidence the stable ASSURE-000 binding semantics while preserving the richer subject model in its own versioned layer.

## Deployment envelope is identity, not authority

A declared deployment envelope may be part of the subject profile. Its presence records which deployment conditions are being identified; it does not establish deployment eligibility or authorization.

## Deliberate nonclaims

ASSURE-001 does not establish:

- provider truthfulness;
- artifact retrievability or system replayability;
- remote-service immutability when no immutable revision is available;
- equivalence between differently specified commitment methods;
- prompt secrecy;
- credential safety beyond excluding raw-secret representation from the intended model;
- evaluator/corpus identity;
- preregistration;
- successful replication;
- common-cause verifier independence;
- certification, compliance, or deployment authority.

Those claims require later evidence, campaign, availability, policy, or standards layers.
