# KAMA-PRIV-001C — Intimate Privacy Audit Receipts

Status: source-design candidate
Issue: #4915
Parent: KAMA-PRIV-001B / #4905
Authority: privacy metadata/evidence only; **no consent, clinical, content, somatic, motor, or deletion-outside-domain authority**

## Purpose

Make intimate-memory retraction and recomputation auditable without retaining intimate payloads inside the receipt layer.

## Retraction receipt

`IntimatePrivacyAuditReceiptV1` binds:

- operation ID;
- privacy policy/version ID;
- requested and completed timestamps;
- source artifact ID;
- applied/already-retracted status;
- canonicalized descendant action set;
- whether deletion of an external export remains unproven;
- domain-separated deterministic BLAKE3 commitment.

Event order is canonicalized before hashing. Duplicate artifact events are rejected. An `Applied` receipt must contain at least one event; an `AlreadyRetracted` receipt must not manufacture fresh events.

The external-deletion flag must exactly agree with the presence of an `ExternalExportNotice` event.

## Recompute lineage

`IntimateRecomputeAuditReceiptV1` binds:

- recomputation operation ID;
- privacy policy/version;
- originating invalidation receipt commitment;
- retracted source artifact ID;
- retired/prior artifact ID;
- fresh replacement artifact ID;
- exact remaining source IDs;
- `MustRecomputeWithoutRetractedSource` derivation policy;
- completion timestamp;
- deterministic receipt commitment.

The retired artifact ID cannot be reused as the replacement identity. The retracted source cannot remain among the replacement sources. Source order and duplicate source entries are canonicalized through a sorted set before commitment.

## Privacy boundary

Receipts contain only opaque operation, policy, artifact, lineage, timing, action, and commitment metadata. They do not retain:

- transcript text;
- fantasy content;
- private preference values;
- psychology values;
- embeddings;
- reconstructive summaries;
- raw source payloads.

## What the receipt proves

A valid receipt can establish that the governed software reported a specific privacy operation under a specific policy and committed to its declared lineage/action metadata.

It does **not** independently prove that a third-party export, offline backup, user-made copy, or previously authorized external record was erased. `external_deletion_unproven` remains explicit for that reason.

A recomputation receipt establishes lineage, not semantic quality. Separate tests/evidence are required to show that a recomputed memory is correct, appropriately minimized, and non-reconstructive.

## Tests

Source and integration tests cover:

- deterministic event canonicalization;
- event-order independence;
- semantic/policy change sensitivity;
- invalid timestamp rejection;
- duplicate artifact-event rejection;
- external deletion limitation preservation;
- recompute source canonicalization/order independence;
- retired artifact-ID reuse rejection;
- retracted-source reuse rejection.

## Nonclaims

This tranche establishes no global-erasure proof, third-party deletion proof, backup deletion proof, clinical validity, consent inference, content permission, human-contact authority, or motor authority.
