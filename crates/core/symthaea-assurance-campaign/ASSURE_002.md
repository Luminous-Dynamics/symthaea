# ASSURE-002 — Preregistered Qualification Campaigns

## Governing theorem

```text
plan content-addressed later
    != preregistered earlier

self-declared timestamp
    != durable ordering evidence

registration receipt exists
    != registration is current

post-registration evidence
    != evidence bound to that exact registration

same ordering source
    + different validation profile
    != comparable ordering lineage
```

ASSURE-002 adds campaign identity, preregistration ordering, registration currentness, and preregistered evidence admission above ASSURE-000/001 without modifying either qualified parent.

## Campaign plan

`CampaignPlanV1` binds:

- semantic `plan_key`;
- campaign nonce;
- exact ASSURE-000 claim digest;
- exact ASSURE-001 manifest identity;
- exact bridged ASSURE-000 subject identity;
- the derived ASSURE-000 `QualificationPlan` digest;
- maximum support tier;
- reproduction-evidence requirement separately from support strength;
- registered evidence kinds;
- registered controls;
- explicit failure, contradiction, inconclusive, and invalidation conditions.

Ordering of set-like inputs has no authority. Evidence kinds use lexicographic UTF-8 ordering of canonical wire names rather than Rust enum declaration order. Duplicate semantic entries fail closed.

`CampaignPlanV1::core_plan()` derives the ASSURE-000 qualification plan from the richer ASSURE-002 plan instead of maintaining two independently editable plan authorities.

## Registration is an ordered event

A registration is constructed in three stages:

```text
exact CampaignPlanV1
    + exact predecessor, if any
    + registrar identity
        ↓
RegistrationStatementV1
        ↓
external ordering system observes exact statement digest
        ↓
OrderingReceiptV1
        ↓
PreregistrationReceiptV1
```

`OrderingReceiptV1` binds:

- ordering source identity;
- ordering-validation profile identity;
- non-zero epoch;
- non-zero monotonic sequence;
- exact statement digest;
- exact external receipt/attestation digest.

Two receipts are order-comparable only when source, validation profile, and epoch all match. This prevents the same external receipt bytes from being interpreted under different validation rules while silently retaining one ordering meaning.

ASSURE-002 verifies statement binding and relative ordering. It does **not** authenticate the external ordering authority or prove that the external receipt is genuine. The validation-profile identifier records which external verification contract is intended; the adapter/verifier is still responsible for actually enforcing that contract before supplying an accepted receipt.

A local wall clock is not used as preregistration authority.

## Empty pre-evidence root

Every registration statement binds an exact campaign-specific empty evidence root and `pre-evidence-count = 0`.

The empty root commits:

- campaign nonce;
- exact plan digest;
- predecessor registration, if any.

A successor plan therefore starts a new admitted-evidence lineage even when it retains the same campaign nonce. Evidence admitted under the predecessor is not silently inherited by the successor.

This is a canonical campaign-evidence boundary. It is not, by itself, proof that some unrelated external data store contained no files; the ordering/registration integration is responsible for making this root the authoritative campaign evidence state.

## Currentness and forks

`resolve_current_registration()` treats the registration history as an append-only single-successor chain.

It fails closed on:

- no registrations;
- multiple roots;
- missing predecessors;
- multiple distinct successors of one predecessor;
- disconnected registrations;
- cross-campaign lineage;
- non-increasing or incomparable source/profile/epoch ordering lineage;
- conflicting withdrawals;
- successor registration after withdrawal of its predecessor;
- withdrawal of the unique leaf, which leaves no current registration.

Byte-identical duplicate receipts are idempotent.

A registration with a valid successor is superseded automatically. Only the unique unwithdrawn leaf is current.

## Evidence production is distinct from evidence admission

Each admitted evidence item must satisfy:

```text
registration < production[i] < admission[i]
```

while the authoritative ledger itself requires:

```text
registration < admission[1] < admission[2] < admission[3] < ...
```

All comparable events use one exact ordering source + validation profile + epoch lineage.

Evidence production does **not** have to occur after the previous evidence admission. Multiple workers may legitimately produce evidence concurrently or in batches after preregistration, and those already-produced artifacts may later be admitted one by one. What must never regress is the authoritative admission history.

The production ordering statement binds:

- exact current registration digest;
- campaign nonce;
- current plan digest;
- exact ASSURE-000 evidence-artifact digest.

Evidence produced before or at the registration sequence is classified `ProducedBeforeOrAtRegistration`. Evidence from another source/profile/epoch is `IncomparableOrderingLineage`. Neither can enter the preregistered ledger.

These records may still be retained elsewhere as post-hoc evidence; ASSURE-002 merely refuses to label them preregistered.

## Append-only evidence admission

`CampaignEvidenceLedgerV1` begins from the exact empty root bound into the current registration. The registration ordering receipt is also the initial admission-ordering head.

Each admitted item requires:

1. the exact current registration;
2. the exact current campaign plan;
3. an evidence kind registered by the plan;
4. no duplicate evidence ID or exact evidence digest;
5. a production statement ordered after the current registration;
6. an admission statement binding the current evidence root, next ordinal, exact evidence digest, previous admission-ordering head, and production-ordering receipt;
7. admission ordering strictly later than the evidence's production ordering;
8. admission ordering strictly later than the ledger's previous admission ordering.

Successful admission produces a new content-addressed root that commits the previous root, ordinal, evidence digest, production-ordering digest, and admission-ordering digest. The admission ordering becomes the ledger's new ordering head.

This permits concurrent production but prevents two successful admissions from occupying the same or regressing ordering position.

An evidence ledger created under a superseded registration cannot be reused with a successor registration.

## Important evidence boundary

ASSURE-002 establishes campaign timing/admission semantics. ASSURE-000 remains authoritative for validating that evidence artifacts bind the correct claim and bridged subject when a qualification result is constructed.

Therefore:

```text
ASSURE-002 preregistered admission
    != ASSURE-000 claim support
    != ASSURE-003 evidence resolution
```

## Support and reproduction remain separate

The campaign plan records a maximum support tier and a separate reproduction-evidence policy.

A preregistered plan does not make an experiment well-designed. A distinct-verifier evidence requirement does not prove common-cause independence. Successful replication remains an ASSURE-003 theorem; common-cause independence remains ASSURE-015/#2633.

## Deliberate nonclaims

ASSURE-002 does not establish:

- authenticity or trustworthiness of an external ordering authority;
- that a validation profile was actually enforced merely because its identifier is bound;
- trusted wall-clock time;
- quality or adequacy of the experimental design;
- claim support from evidence;
- successful replication;
- common-cause verifier independence;
- artifact availability or replayability;
- certification/compliance;
- deployment eligibility or action authority.
