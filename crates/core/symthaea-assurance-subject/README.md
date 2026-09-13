# symthaea-assurance-subject

ASSURE-001 defines exact multi-surface system-under-test identity for external AI and agent qualification.

A product/provider label is not a sufficient subject identity. The manifest uses a semantic `subject_key`, a canonical `SurfaceProfile`, and exactly one `SurfaceBinding` per registered material surface. Presentation/display labels stay outside canonical identity.

Every surface is `Known(material-commitment)`, `Unknown`, `Unavailable(reason)`, or `NotApplicable`. Omission fails closed. Applicable states require a locator; `NotApplicable` is locator-free so irrelevant routing metadata cannot alter identity for a declared absence.

Known commitments bind both SHA-256 and the preimage interpretation:

- `ArtifactBytesSha256`
- `CanonicalDescriptorSha256 { schema }`
- `ProviderRevisionTokenSha256 { namespace }`
- `CustomSha256(method-id)`

The same digest under different methods, descriptor schemas, or provider-token namespaces is a different material commitment.

`CompletenessSummary::has_complete_committed_identity()` means every applicable registered surface has a typed commitment. It does **not** establish that provider tokens uniquely fix underlying bytes, artifacts are retrievable, or the system is replayable.

The standard external-AI profile covers source/image identity, model, system prompt, tool authority, policy, runtime, deployment envelope, and explicitly named external dependencies. Evaluator/corpus/campaign identity is intentionally excluded.

Provider aliases and stable URLs are locators, not immutable revisions. Raw secrets and credential bytes are not manifest material; hashing low-entropy secrets is not a safe substitute.

Canonical surface ordering is language-neutral lexicographic UTF-8 ordering of explicit wire names. Golden vectors freeze cross-implementation profile and manifest digests.

`AiSubjectManifest::as_core_subject()` bridges the complete ASSURE-001 manifest commitment into the qualified ASSURE-000 kernel without modifying that semantic waist.

See `ASSURE_001.md` for the complete theorem and nonclaims.
