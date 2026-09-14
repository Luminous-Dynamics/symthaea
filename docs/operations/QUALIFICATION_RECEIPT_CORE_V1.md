# Qualification Receipt Core V1 — canonical semantic framing

This branch defines the first provider/retry-neutral `QualificationReceiptId` framing candidate for #905. It is deliberately downstream of `PassSelectionV2` and `QualificationReceiptCandidateGateV1(disposition = WitnessRequired)`.

The receipt core is a **portable immutable claim object**, not trusted qualification authority. It may be created before the base-owned witness so that the witness has one exact semantic receipt identity to bind to provider/run/recipe evidence.

This branch owns only the lock-neutral semantic framing theorem: the normative framing specification, the implementation-independent Python reference, and its golden-vector regressions. The Rust conformance implementation is intentionally a separate child branch so workspace enrollment, generated `Cargo.lock`, and Cargo qualification cannot masquerade as part of the semantic framing theorem.

## Identity split

The semantic identity commits only the selected qualification proposition:

- `QualificationSubjectId`
- `QualificationProfileId`
- one common `InputClosureId`
- one common `QualificationEnvironmentId`
- the exact sorted required recipe set
- the provider/retry-neutral `QualificationAttemptSubjectId` selected for each recipe
- fixed V1 selection/cross-cutting disposition tags
- fixed V1 non-claim tags

It deliberately excludes `PassSelectionId`, `AttemptHistoryId`, registration/observation IDs, provider references, evidence-content IDs, witness identity, detached attestation and current admission. Those remain mandatory occurrence/evidence records outside the semantic ID.

Therefore:

```text
same exact selected theorem reproduced by another provider
    -> may share QualificationReceiptId
    -> must retain different occurrence evidence

later retry/failure
    -> historical QualificationReceiptId is unchanged
    -> AttemptHistoryId / PassSelectionId change

subject/profile/recipe/closure/environment drift
    -> QualificationReceiptId changes
```

## Selection disposition is intentionally below trusted PASS

V1 freezes:

```text
selection_disposition = SelectedRequiredRecipesReportedPassedV1
non_claim              = NoTrustedPassEstablished
```

This means the upstream structural selection contains an executed observation whose terminal value is `Passed` for each required recipe. It does **not** mean a trusted witness has verified the provider/run binding, a detached signer has authenticated it, or current admission has accepted it.

## Canonical framing

V1 uses the same framing style as #830:

1. frame the domain `symthaea.qualification-receipt-core.v1`;
2. frame the schema string;
3. frame subject/profile/closure/environment IDs in the fixed order above;
4. write recipe count as `u64` little-endian;
5. for each recipe sorted by `(recipe_id, attempt_subject_id)`, frame both UTF-8 strings;
6. frame `SelectedRequiredRecipesReportedPassedV1`;
7. frame `NoGenericCrossCuttingRulesV1`;
8. write non-claim count as `u64` little-endian;
9. frame the fixed sorted V1 non-claim tags.

`frame(bytes)` is:

```text
u64_le(byte_length) || exact_bytes
```

The semantic ID is:

```text
QualificationReceiptId = "blake3:" || hex(BLAKE3(canonical_frame))
```

JSON is transport/display only and is not hashed as the semantic identity. Parsed `QualificationReceiptId` strings are canonical lowercase only.

## Golden vector

The normative framing and independent Python reference freeze the published vector:

- canonical frame length: `1036` bytes
- `QualificationReceiptId`: `blake3:10b18b9ab46ae0c75b4cfb322ff6dfed48f7b45541cf3de2a4128812e28cf69d`

The receipt vector itself crosses BLAKE3's 1024-byte chunk boundary. The Python reference additionally reproduces the official BLAKE3 empty-string, `abc`, and 4096-byte tree-hash vectors.

A separate Rust conformance child is expected to reproduce the same canonical frame and ID. Rust conformance is not a prerequisite for the identity *definition* itself, and its compile/test/lock status must remain separately visible.

## Composition

The intended authority chain is:

```text
PassSelectionV2
    -> exact evidence byte closure
    -> ReceiptCandidate(WitnessRequired)
    -> QualificationReceiptCoreV1 / QualificationReceiptId
       (portable immutable claim; still untrusted)
    -> base-owned witness assessment (#1157 specialization)
       (provider/run/subject/recipe/receipt binding only)
    -> detached attestation (#955)
    -> current admission / withdrawal / freshness (#931)
```

Do not use the legacy `FocusedSoftwareContractWitnessed` vocabulary as a second permanent qualification authority family.

## Authority ceiling

This lock-neutral tranche establishes the semantic framing/identity candidate only. It does **not** establish:

- that a provider really executed the selected recipe;
- that a reported `Passed` terminal is trustworthy;
- evidence correctness or scientific validity;
- trusted acquisition/provenance;
- Rust conformance;
- generated workspace-lock correctness;
- detached signer/witness authenticity;
- current admission;
- merge or execution authority.

The independent Python reference has been authoring-checked against the frozen vectors, but that is not hosted qualification.

## Conformance-layer split

Keep the four meanings separate:

```text
canonical semantic framing theorem
    != Rust conformance implementation
    != generated Cargo.lock transition
    != Cargo/hosted qualification
```

The Rust implementation belongs on a child branch containing only its `Cargo.toml` and `src/lib.rs`. Because `crates/core/*` is workspace-globbed, that child is not `--locked`-ready until a real Cargo environment generates and verifies the corresponding local-package lock entry. The lock-neutral semantic parent must not absorb that generated-build obligation.
