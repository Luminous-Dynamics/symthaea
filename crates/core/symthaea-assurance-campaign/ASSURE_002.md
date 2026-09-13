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

Ordering of set-like inputs has no authority. Duplicate semantic entries fail closed.

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
- non-zero epoch;
- non-zero monotonic sequence;
- exact statement digest;
- exact external receipt/attestation digest.

ASSURE-002 verifies statement binding and relative ordering. It does **not** authenticate the external ordering authority or prove that the external receipt is genuine. That trust must be established by the adapter/verifier supplying the receipt.

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
- non-increasing or incomparable ordering lineage;
- conflicting withdrawals;
- successor registration after withdrawal of its predecessor;
- withdrawal of the unique leaf, which leaves no current registration.

Byte-identical duplicate receipts are idempotent.

A registration with a valid successor is superseded automatically. Only the unique unwithdrawn leaf is current.

## Evidence production is distinct from evidence admission

ASSURE-002 separates:

```text
registration
    < evidence production
    < evidence admission
```

under one exact ordering source + epoch.

The production ordering statement binds:

- exact current registration digest;
- campaign nonce;
- current plan digest;
- exact ASSURE-000 evidence-artifact digest.

Evidence produced before or at the registration sequence is classified `ProducedBeforeOrAtRegistration`. Evidence from another source/epoch is `IncomparableOrderingLineage`. Neither can enter the preregistered ledger.

These records may still be retained elsewhere as post-hoc evidence; ASSURE-002 merely refuses to label them preregistered.

## Append-only evidence admission

`CampaignEvidenceLedgerV1` begins from the exact empty root bound into the current registration.

Each admitted item requires:

1. the exact current registration;
2. the exact current campaign plan;
3. an evidence kind registered by the plan;
4. a production statement ordered after registration;
5. an admission statement binding the current evidence root, next ordinal, exact evidence digest, and production-ordering receipt;
6. admission ordering strictly later than production ordering;
7. no duplicate evidence digest.

Successful admission produces a new content-addressed root that commits the previous root, ordinal, evidence digest, production-ordering digest, and admission-ordering digest.

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
- trusted wall-clock time;
- quality or adequacy of the experimental design;
- claim support from evidence;
- successful replication;
- common-cause verifier independence;
- artifact availability or replayability;
- certification/compliance;
- deployment eligibility or action authority.
