# Scientific Proposition Identity v1 — profile and schema hardening

Status: architecture hardening companion to `SCIENTIFIC_PROPOSITION_IDENTITY_V1.md`.

This note closes several identity loopholes that become visible once the shared proposition envelope is treated as a long-lived Theory Atlas primitive.

## 1. Core hardening theorem

```text
schema label
    != schema identity

payload digest
    != retained semantic payload

same domain label
    != same domain namespace authority

new schema version
    != automatically same proposition identity

canonical identity
    != semantic correctness
```

The proposition identity layer must therefore bind immutable profile/schema identity and retain enough material to reconstruct what was actually committed.

## 2. Semantic schema identity must be content-addressed

A field such as:

```text
domain_semantic_schema_id = "economics-causal-v1"
```

is useful for display but insufficient as normative identity if the bytes/contract behind that label can drift.

A future proposition envelope should bind at least conceptually:

```text
ScientificPropositionIdentityV1 {
    proposition_identity_profile_digest,
    domain_namespace_identity,
    domain_semantic_schema_profile_digest,
    semantic_payload_digest,
}
```

The human schema name may also be retained, but the profile digest is normative.

Changing canonical field meanings, tags, normalization rules, ordering rules, omitted/required fields, or semantic interpretation requires a new schema profile identity.

```text
same schema name + changed contract
    -> fail / new profile
```

not silent reuse.

## 3. The semantic payload must remain auditable

A digest alone proves content identity only if the committed content can actually be recovered from a retained artifact or deterministic source.

The proposition registry should therefore retain either:

```text
canonical semantic payload bytes
```

or a content-addressed artifact reference whose exact bytes are durably retrievable under the scientific artifact identity layer.

Conceptually:

```text
semantic_payload_artifact
    -> verified canonical bytes
    -> semantic_payload_digest
    -> proposition identity
```

not:

```text
opaque digest string
    -> trust that someone remembers what it meant
```

A proposition identity with permanently unavailable semantic bytes may remain a historical opaque reference, but should not be treated as a reconstructible shared-kernel semantic target.

## 4. Digest computation is not semantic qualification

Even perfect hashing does not prove that the canonicalizer captured the intended scientific semantics.

Therefore:

```text
verified canonical bytes
    != correct scientific semantic model
```

A domain semantic profile requires its own review/qualification lineage establishing at least:

- which fields are constitutive of proposition semantics;
- which fields are assessment/evidence metadata instead;
- canonical ordering and encoding;
- treatment of optional/default values;
- string/identifier normalization rules when strings are normative;
- versioning policy;
- adversarial examples showing semantically distinct targets cannot collapse through omitted fields.

The proposition identity primitive should not self-certify that a domain profile is scientifically adequate.

## 5. Domain namespace identity must not be a mutable label

The shared kernel needs a stable namespace boundary so two unrelated producers cannot both claim:

```text
domain = "economics"
```

while assigning incompatible semantic schemas under one apparent domain identity.

A future `domain_namespace_identity` may be content-addressed governance/registry identity or another exact registered namespace capability, but it must not be inferred solely from a display string.

The namespace identity grants naming scope only. It does not grant scientific truth or disposition authority.

## 6. Schema upgrades do not silently preserve proposition identity

Suppose schema v1 and schema v2 encode the same intended scientific proposition differently.

Because their schema profile identities differ, their exact proposition identities should normally differ too:

```text
P[v1] != P[v2]
```

This is safer than pretending the new canonicalizer is automatically equivalent to the old one.

A migration may issue an explicit receipt such as:

```text
SemanticTargetEquivalenceReceipt {
    source_proposition_id,
    destination_proposition_id,
    source_schema_profile,
    destination_schema_profile,
    equivalence_method,
    assumptions,
    qualification_artifact,
}
```

Only that receipt may support cross-schema target equivalence.

This preserves historical identity while allowing the Atlas to evolve its representation.

## 7. Equivalence receipt is not identity rewriting

Even if a qualified migration establishes semantic equivalence:

```text
P[v1] ~= P[v2]
```

we should not rewrite old evidence records from `P[v1]` to `P[v2]`.

Instead the Atlas retains:

```text
Evidence E -> P[v1]
P[v1] --QualifiedEquivalentTarget--> P[v2]
```

and derives any current cross-version view through the receipt.

That keeps historical evidence addressing immutable.

## 8. Human renderings and translations remain secondary artifacts

Renderings may be multilingual or revised for clarity without changing proposition semantic identity when the domain semantic payload remains unchanged.

A rendering should therefore retain:

```text
rendering_id
proposition_id
language / notation profile
rendering bytes/content digest
```

but rendering identity should not define proposition identity.

If a translation changes scientific semantics, it is not merely another rendering; it becomes either a corrected rendering or a candidate new proposition requiring explicit semantic relation.

## 9. Normative strings inside canonical payload require exact rules

If a domain semantic payload includes normative strings or external identifiers, its profile must define exact canonical treatment.

Possible choices include:

- raw UTF-8 bytes under an exact normalization policy;
- registered external identifier bytes;
- typed content-addressed references instead of free-form strings.

The shared kernel should not silently apply locale-dependent case folding, Unicode normalization, whitespace normalization, or synonym expansion.

```text
presentation normalization
    != scientific semantic equivalence
```

## 10. No caller-supplied proposition digest authority

A future constructor should not accept:

```text
proposition_id: Digest
```

from the caller and trust it as authoritative identity.

The trusted path should be:

```text
registered semantic schema/profile
    + domain semantic object
    -> canonical semantic bytes
    -> verified digest computation
    -> issued ScientificPropositionIdentityV1
```

Callers may reference an already issued proposition ID, but cannot mint one by typing a digest into a struct.

## 11. Identity issuance remains non-authorizing

An issued proposition identity establishes only:

```text
this exact canonical semantic target is registered under this exact profile
```

It does not establish:

```text
well-formed theory
measurability
testability
truth
falsifiability
causality
importance
safety
support
current disposition
```

Those are separate layers.

## 12. Suggested implementation split

The first implementation should remain narrower than a full proposition registry.

Recommended sequence:

```text
SCI-014a.1a  immutable PropositionSemanticProfile identity
SCI-014a.1b  canonical semantic payload artifact
SCI-014a.1c  proposition identity issuance from verified profile + payload
SCI-014a.1d  persistence/replay verification
SCI-014a.1e  cross-profile semantic-equivalence receipt
```

Only afterward should a broader registry/revision graph be introduced.

## 13. Qualification adversarial cases

A future focused qualification should include at least:

1. changing only schema display name does not define identity;
2. changing schema profile contract changes schema profile identity;
3. same payload bytes under different schema profiles do not automatically share proposition identity;
4. same schema profile + same canonical payload yields the same identity independent of caller order/presentation;
5. semantic payload digest cannot be admitted without retained/recoverable canonical payload evidence under the selected storage policy;
6. caller-supplied fake proposition IDs cannot bypass issuance;
7. cross-schema equivalence requires an explicit receipt;
8. an equivalence receipt does not rewrite predecessor evidence addresses;
9. rendering/translation changes do not define semantic identity unless the semantic payload changes;
10. proposition issuance exposes no truth/disposition/action authority.

## 14. Important non-claims

This note does not choose a cryptographic hash algorithm, implement a domain namespace registry, define universal semantic correctness, guarantee archival availability, or establish cross-schema equivalence for any existing Symthaea proposition ID.

Its only purpose is to ensure that the future proposition identity object remains reconstructible, profile-bound, migration-safe, and honest about what identity can and cannot prove.
