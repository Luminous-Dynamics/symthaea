# Engineering Trust Kernel — Discharge Receipt / Currentness Oracle V1

**Status:** independent reference semantics; exact checked-in execution pending  
**Branch:** `engineering/etk-2c-discharge-receipt-oracle`  
**Base:** `main` at `ada0d92fce3f0ff10883061b0608edfd37b16317`

## Theorem boundary

```text
admitted evidence
!= discharge receipt
!= present-tense discharged obligation
!= qualified design
!= deployment/manufacturing/actuation authority
```

This reference oracle freezes the next Engineering Trust Kernel boundary after simulation-evidence admission. It distinguishes a historically valid receipt from whether that receipt is still applicable to the *current* obligation and engineering context.

A receipt does not disappear when it becomes stale. It remains an immutable historical record; only its present applicability changes.

## Independence

`scripts/etk-obligation-discharge-oracle.py` is standard-library Python and imports no Symthaea code. It is based directly on current `main`, independently of production ETK-2B PR #1730.

A production implementation must reproduce the semantics independently rather than invoking this oracle as an authority source.

## Issuance-time obligation vs current obligation

V1 models two separate obligation states:

- `issued_obligation`: the exact proposition against which the admitted evidence originally crossed the discharge-receipt boundary;
- `current_obligation`: the proposition being evaluated now.

This distinction is necessary for correct historical semantics. If the obligation claim changes later, the old receipt must not be considered current, but it also must not be rewritten as though it had never existed.

## Content-addressed obligation revision

The obligation snapshot ID uses the same intended v1 domain separation as ETK-2B:

```text
symthaea.etk-proof-obligation-snapshot.v1\0
```

The canonical preimage contains exactly:

```text
schema = symthaea.etk-proof-obligation-snapshot.v1
obligation_id
claim
expected_evidence_kind
```

Lifecycle status and attached evidence references are excluded because they are consequences/history rather than the proposition being justified.

The synthetic fixture freezes this obligation snapshot vector:

```text
sha256:dc75ef2b334f23bab3a50ce984058dd36f396a736d27b2f397bfe494ac7da4d4
```

## Receipt identity

Receipt issuance is permitted only when:

1. the issuance-time obligation is a well-formed `Simulation` obligation;
2. the admitted evidence targets the exact same obligation ID;
3. the admitted evidence carries the exact content-addressed issuance-time obligation snapshot.

The immutable receipt preimage contains exactly:

- admitted evidence ID;
- candidate artifact ID;
- currentness-proof ID at admission/issuance;
- evidence-policy ID;
- obligation ID;
- obligation revision;
- accepted requirement revision;
- subject ID;
- twin/design revision;
- validity-domain ID.

The fixture freezes this receipt vector:

```text
sha256:cbd1a2effbcb4dd72c9b9ebd0e2b4319fdd250a7b3b6b3e658c780ad4fdb3f66
```

Changing present context does **not** rewrite this receipt identity.

## Present-tense applicability

A valid historical receipt yields `CurrentDischarge` only when the following still match:

- obligation ID;
- content-addressed obligation revision;
- evidence kind remains `Simulation`;
- subject ID;
- twin/design revision;
- accepted requirement revision;
- validity-domain ID;
- currentness-proof ID.

If issuance itself is invalid, the oracle returns:

```text
DenyReceiptIssuance { ordered_reasons[] }
```

If issuance was valid but current context has drifted, it returns:

```text
HistoricalReceipt { receipt_id, obligation_id, obligation_revision,
                    admitted_evidence_id, ordered_reasons[] }
```

If both issuance and present applicability hold, it returns:

```text
CurrentDischarge { receipt_id, obligation_id, obligation_revision,
                   admitted_evidence_id }
```

There is no scalar trust score and no automatic downgrade from one evidence class to another.

## Adversarial coverage

The self-test checks:

- admitted evidence bound to the wrong obligation ID;
- admitted evidence carrying a stale obligation revision;
- an issuance-time obligation that is not `Simulation` evidence;
- post-issuance claim mutation;
- post-issuance obligation-ID replacement;
- current evidence-kind change;
- subject change;
- twin/design revision change;
- accepted requirement revision change;
- validity-domain change;
- currentness-proof refresh/change;
- unknown/shadow admitted-evidence fields;
- malformed current context;
- deterministic ordering for simultaneous twin/currentness changes;
- invariant receipt identity across later staleness/currentness changes.

The critical theorem tested is:

```text
historical receipt identity is immutable
AND
historical receipt applicability is revocable by semantic/context drift
```

## Local candidate execution evidence

A local candidate of the oracle semantics was executed with its built-in self-test and `py_compile`, producing:

```text
ok obligation_snapshot=sha256:dc75ef2b334f23bab3a50ce984058dd36f396a736d27b2f397bfe494ac7da4d4
ok discharge_receipt=sha256:cbd1a2effbcb4dd72c9b9ebd0e2b4319fdd250a7b3b6b3e658c780ad4fdb3f66
```

The locally executed candidate SHA-256 was:

```text
9676131ca0176d45a581793f0b267543e66d13daddcc1e2a18eed72bced37108
```

The checked-in script currently has Git blob identity:

```text
c1ae72321463c4e927cfdc6191734940c5723bbd
```

The connector write changed source-text bytes relative to the locally executed candidate, so this document deliberately does **not** claim exact-byte execution of the checked-in blob. Repository CI or a separately recorded exact-byte run must establish that evidence.

## Deliberate nonclaims

This oracle does not:

- prove that the admitted evidence was correctly admitted;
- prove solver correctness or physical truth;
- authenticate `currentness_proof_id`;
- prove evidence independence;
- prove the completeness of the validity domain;
- establish safety-case closure across multiple obligations;
- qualify a design;
- authorize deployment, fabrication, or actuation;
- define native/analytical evidence admission.

It consumes an admitted-evidence identity as an already-established premise and freezes only the receipt/current-applicability theorem.

## Production parity target

ETK-2B should match both frozen vectors and the three-way decision semantics:

```text
DenyReceiptIssuance
HistoricalReceipt
CurrentDischarge
```

The production Rust type system may represent these differently, but the semantic theorem must remain equivalent. In particular, currentness refresh and obligation semantic mutation must make old receipts inapplicable without deleting or rewriting their historical identity.
