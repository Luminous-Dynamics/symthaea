# SCI-002 Review Checklist — Canonical Scientific Artifact Identity v1

**Status:** review aid only; non-authorizing; non-qualifying.

Use this checklist to review `SCIENTIFIC_ARTIFACT_IDENTITY_V1.md` without expanding SCI-002 into execution, evidence adjudication, replication, or action authority.

## A. Core identity separation

- [ ] Human-readable labels/locators are explicitly non-canonical.
- [ ] Digest syntax is distinct from bytes-to-digest verification.
- [ ] Raw byte content identity is distinct from canonical semantic identity.
- [ ] Canonical semantic identity is distinct from scientific truth/validity.
- [ ] Identity is distinct from provenance.
- [ ] Identity is distinct from freshness/currentness/lifecycle eligibility.
- [ ] Same content is not treated as independent evidence.
- [ ] Different content is not treated as independent evidence.
- [ ] Scientific identity cannot become action authority.

## B. Raw content identity

- [ ] Digest algorithm identity is stable and exact.
- [ ] Digest length is exact.
- [ ] Textual representation has one canonical spelling.
- [ ] Uppercase, whitespace, malformed length, and unknown algorithms fail closed.
- [ ] Byte length is bound where required.
- [ ] Positive verified-content state requires recomputing the digest over exact bytes.
- [ ] Caller-supplied digest metadata cannot mint positive verification.
- [ ] The design does not use Rust `DefaultHasher`, ordinary `Hash`, Debug output, or incidental serializer bytes as the scientific identity definition.

## C. Semantic identity

- [ ] Domain namespace is bound.
- [ ] Semantic schema/profile identity is bound.
- [ ] Canonicalization profile identity is bound.
- [ ] The shared layer does not impose one universal scientific payload schema.
- [ ] Serializer output is not accidentally normative unless an explicit canonical serializer profile says it is.
- [ ] Float semantics remain domain/profile-owned.
- [ ] Same raw bytes under different interpretation profiles can yield different semantic identities.
- [ ] Different raw bytes may yield the same semantic identity only through an explicit canonicalization profile.

## D. Composite identity

- [ ] Composite kind/profile is domain separated.
- [ ] Child identities and relation semantics are explicitly bound.
- [ ] Ordered vs order-insensitive collections are specified, never inferred.
- [ ] Graph node/edge semantics and generation scope are explicit.
- [ ] Raw observation roots remain distinguishable from normalized semantic roots.
- [ ] Adding/removing unrelated graph nodes changes identity exactly when the declared generation semantics say it should.

## E. Transformation/equivalence

- [ ] A transformation never silently implies identity preservation.
- [ ] Source and target artifact identities are retained.
- [ ] Transformation implementation/config/profile identity is retained.
- [ ] Execution/provenance is retained separately from content identity.
- [ ] `SemanticsPreserving` cannot be self-asserted by an unqualified caller.
- [ ] Information-reducing/model-derived transformations remain distinguishable from lossless canonicalization.
- [ ] Cross-schema or cross-algorithm equivalence is explicit and receipt-derived rather than historical identity rewriting.

## F. Persistence and authority

- [ ] Positive verified wrappers are private-fielded.
- [ ] Positive verified wrappers are not directly recreatable through ordinary deserialization.
- [ ] Archived bytes remain audit evidence rather than self-restoring current authority.
- [ ] No identity type provides convenience authority transitions such as `trusted`, `scientifically_valid`, `independent`, `causal`, `novel`, or `authorized`.
- [ ] Content identity does not imply source authenticity.
- [ ] Content identity does not imply experimental status or replication.

## G. Migration safety

- [ ] Existing RCA identities are not silently rewritten.
- [ ] Existing NeuroBridge roots are not silently rewritten.
- [ ] Existing Matter evidence references are not silently upgraded from `ReferenceOnly`.
- [ ] Existing Earth-observation identities retain their stronger domain semantics.
- [ ] Existing proposition/evidence IDs are not reinterpreted without explicit reconstruction/admission.
- [ ] Migration is adapter/correspondence-receipt based.
- [ ] No qualification inheritance occurs from referenced draft PRs.

## H. SCI dependency discipline

- [ ] SCI-002 raw identity does not require SCI-003 execution semantics.
- [ ] SCI-003 may consume SCI-002 identities without making them execution evidence.
- [ ] #786 verification receipts bind SCI-002 identities but still require independent verifier-soundness gates.
- [ ] SCI-006 dependency/replication semantics remain separate.
- [ ] SCI-014 proposition/disposition semantics remain separate.
- [ ] No `config_hash()` / `DefaultHasher` diagnostic identifier becomes a canonical scientific root.

## I. First implementation gate

The first Rust tranche should remain limited to:

```text
ContentDigestV1
RawArtifactContentIdentityV1
VerifiedArtifactContentV1
```

Reviewers should reject first-tranche expansion into:

- universal semantic canonicalization;
- arbitrary graph identity;
- execution receipts;
- verification receipts;
- source trust;
- scientific qualification;
- evidence independence;
- replication;
- action authority.

## Review question

> Does SCI-002 provide enough exact identity machinery for downstream scientific receipts to bind the right artifacts while preserving the fact that identity, interpretation, provenance, verification, scientific authority, and action authority are different things?
