# ASSURE-002A — Self-Describing Semantic Definition Commitments

ASSURE-002A is a narrow hardening tranche stacked on ASSURE-002. It extracts semantic-definition identity into a reusable core crate because campaign planning, evidence resolution, imported evidence, and future standards projections may all need the same theorem.

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

## Reusable primitive

`symthaea-assurance-semantics` defines:

```text
SemanticCommitmentV1 {
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

## Canonical semantic-set semantics

`canonical_semantic_set()` sorts only by semantic identifier and rejects duplicate semantic IDs even when callers attach different schemas or definition digests.

This preserves the ASSURE-002 rule that one semantic ID names one slot in one declared semantic set. Schema/digest are identity-bearing contents of that slot, not a mechanism for creating multiple meanings under the same label.

## Conformance corpus

The initial reusable-crate corpus proves:

- definition schema changes commitment identity;
- definition digest changes commitment identity;
- exact semantic ID/schema/digest components are preserved;
- duplicate semantic IDs fail even when schema differs;
- semantic sets canonicalize by semantic ID only;
- an independently calculated canonical commitment golden vector is stable.

The frozen primitive-level golden fixture is:

```text
semantic-id: matched-sham
definition-schema: symthaea.assurance.semantic-definition.canonical-text-v1
definition-digest: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
commitment-digest: ccf57b5d5fbc762fc072eaa7b885a2eff8e8de78a56329e4999b403da2011ac4
```

This freezes only the generic semantic-commitment wire format. Full campaign/registration/ordering/admission golden vectors remain deferred until integration into ASSURE-002.

## Integration sequence

The parent ASSURE-002 candidate remains frozen while independently qualified. ASSURE-002A is qualified as a separate core primitive first.

Then integrate from those frozen inputs:

```text
qualified ASSURE-002 campaign kernel
        +
qualified semantic commitment primitive
        ↓
replace campaign-local label+digest semantics
with SemanticCommitmentV1
        ↓
make schema identity-bearing in support criteria,
controls, custom evidence, conditions, and
ordering-validation profiles
        ↓
add validation-profile schema incomparability theorem
        ↓
recompute full campaign golden vectors independently
        ↓
qualify integrated exact head
```

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
