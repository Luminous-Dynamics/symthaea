# ASSURE-002A — Self-Describing Semantic Definition Commitments

ASSURE-002A is a narrow hardening tranche stacked on ASSURE-002. It exists because an exact SHA-256 digest does not, by itself, identify the canonical preimage convention whose bytes were hashed.

## Governing theorem

```text
same semantic id
+ same SHA-256 digest
!= same semantic commitment
unless definition schema also matches
```

Likewise:

```text
definition digest known
    != definition bytes available
    != definition semantics adequate
    != definition semantics trusted
```

Availability/replayability remains outside this tranche.

## Staged primitive

`src/semantic_schema.rs` introduces an independently executable migration primitive:

```text
SelfDescribingSemanticCommitmentV1 {
    semantic_id,
    definition_schema,
    definition_digest,
}
```

Its canonical wire commitment is domain separated and explicitly orders:

1. commitment schema;
2. semantic ID;
3. definition schema;
4. definition digest.

`definition_schema` is identity-bearing. Changing only the schema changes the commitment digest even when semantic ID and definition digest stay fixed.

## Canonical set semantics

`canonical_semantic_set()` sorts only by semantic identifier and rejects duplicate semantic IDs even when callers attach different schemas or definition digests.

This preserves the ASSURE-002 rule that one semantic ID names one slot in one declared semantic set. Schema/digest are identity-bearing contents of the slot, not a way to create multiple meanings under the same label.

## Ordering lineage semantics

`OrderingLineageIdentityV1` defines comparability by:

```text
ordering source
+ full self-describing validation-profile commitment
+ epoch
```

Sequence remains event data. A validation-profile schema change therefore makes two ordering lineages incomparable rather than silently reinterpreting one sequence space under different validation rules.

## Conformance corpus

The initial ASSURE-002A corpus proves:

- definition schema changes commitment identity;
- definition digest changes commitment identity;
- duplicate semantic IDs fail even when schema differs;
- semantic sets canonicalize by semantic ID only;
- validation-profile schema drift makes ordering lineage incomparable;
- source/profile/epoch identity is required for ordering comparability;
- an independently calculated canonical commitment golden vector is stable.

The current golden fixture is:

```text
semantic-id: matched-sham
definition-schema: symthaea.assurance.semantic-definition.canonical-text-v1
definition-digest: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
commitment-digest: ccf57b5d5fbc762fc072eaa7b885a2eff8e8de78a56329e4999b403da2011ac4
```

## Migration boundary

This branch deliberately does **not** rewrite ASSURE-002's large campaign kernel while PR #2788 is awaiting its own exact-head qualification.

The integration sequence is:

```text
ASSURE-002 exact-head qualification
        +
ASSURE-002A primitive qualification
        ↓
replace label+digest campaign semantics
with schema+digest commitments
        ↓
update campaign/support/control/custom-evidence wire fields
        ↓
recompute full campaign golden vectors independently
        ↓
qualify integrated exact head
```

Until integration completes, ASSURE-002's documented limitation remains authoritative: its current `SemanticCommitmentV1` binds semantic ID + definition digest under an externally agreed definition-preimage convention.

## Deliberate nonclaims

ASSURE-002A does not establish:

- authenticity of definition bytes;
- availability of definition bytes;
- adequacy or correctness of a criterion/control definition;
- criterion satisfaction;
- ordering-provider authenticity;
- experimental quality;
- claim support;
- successful replication;
- verifier independence;
- compliance/certification;
- deployment or action authority.
