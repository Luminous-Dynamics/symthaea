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

This reference oracle freezes the ETK boundary after evidence admission. A receipt is an immutable historical record of a valid issuance-time transition; whether it still discharges an obligation is separately derived from the *current* obligation and current engineering context.

## Independence

`scripts/etk-obligation-discharge-oracle.py` is standard-library Python and imports no Symthaea code. It is based directly on current `main`, independently of production ETK-2B PR #1730. Production Rust must reproduce the semantics without invoking this script as an authority source.

## One coherent synthetic lineage

The reference fixture now uses a single internally consistent lineage that can be reproduced through #1730's public production API:

```text
fixed obligation definition
  -> content-addressed obligation snapshot
  -> independently computed admitted simulation evidence identity
  -> discharge receipt identity
  -> current applicability decision
```

Frozen vectors:

```text
obligation snapshot
sha256:dc75ef2b334f23bab3a50ce984058dd36f396a736d27b2f397bfe494ac7da4d4

admitted simulation evidence
sha256:794475a988dbecf945306f051a54e000e03fe918c886ee1cc13a6f28b7ad9b10

discharge receipt
sha256:2d97cb2ecfd2fbaf49da360c0c659b2a5f733a8ad01990533297f7af20fed830
```

The admitted-evidence vector is computed with the previously frozen ETK simulation-admission canonicalization/domain-separation rule, but ETK-2C treats that admitted evidence as an already-established premise rather than re-running the admission theorem.

## Issuance-time obligation vs current obligation

V1 carries two separate obligation states:

- `issued_obligation`: the exact proposition for which the receipt was originally issued;
- `current_obligation`: the proposition being evaluated now.

If the claim or expected evidence kind changes later, the old receipt becomes historical/inapplicable. Its identity is not rewritten or deleted.

## Content-addressed obligation revision

The snapshot domain is:

```text
symthaea.etk-proof-obligation-snapshot.v1\0
```

Its canonical preimage contains exactly:

```text
schema = symthaea.etk-proof-obligation-snapshot.v1
obligation_id
claim
expected_evidence_kind
```

Lifecycle status and attached evidence references are deliberately excluded because they are consequences/history, not the proposition being justified.

## Receipt issuance

Receipt issuance is permitted only when the issuance-time obligation is a well-formed `Simulation` obligation and the admitted evidence targets its exact obligation ID and exact content-addressed revision.

The receipt preimage binds:

- admitted-evidence identity;
- candidate-artifact identity;
- currentness-proof identity at issuance;
- evidence-policy identity;
- obligation ID and snapshot;
- accepted-requirement revision;
- subject;
- twin/design revision;
- validity domain.

Changing later context does not change the receipt hash.

## Present-tense applicability

A valid receipt yields `CurrentDischarge` only while all of the following still match:

- current obligation ID;
- current content-addressed obligation revision;
- current obligation still expects `Simulation` evidence;
- subject;
- twin/design revision;
- accepted-requirement revision;
- validity domain;
- currentness-proof identity.

The oracle has three semantically distinct outcomes:

```text
DenyReceiptIssuance { ordered_reasons[] }
HistoricalReceipt { immutable receipt identity, ordered staleness reasons[] }
CurrentDischarge { immutable receipt identity }
```

There is no scalar trust score or implicit evidence-class conversion.

## Adversarial coverage

The self-test covers wrong admitted obligation/revision, a non-simulation issuance obligation, post-issuance claim mutation, obligation replacement, current evidence-kind change, subject/twin/requirement/validity/currentness drift, shadow fields, malformed current context, deterministic multi-fault ordering, and immutable receipt identity across later staleness.

The central theorem is:

```text
historical receipt identity is immutable
AND
present applicability is revocable by semantic/context drift
```

## Local candidate execution evidence

The latest local candidate passed its built-in self-test and `py_compile`:

```text
ok obligation_snapshot=sha256:dc75ef2b334f23bab3a50ce984058dd36f396a736d27b2f397bfe494ac7da4d4
ok discharge_receipt=sha256:2d97cb2ecfd2fbaf49da360c0c659b2a5f733a8ad01990533297f7af20fed830
```

Local candidate SHA-256:

```text
5b7b0a895f2875ea9aba521c69e556be4d8768619435e2e84b31c52694500721
```

The checked-in script after connector write has Git blob identity:

```text
5a3c3ab258168d3968acab2f2b6a034f4fb5b660
```

The checked-in source bytes are not claimed to be the exact locally executed bytes. Exact-head repository execution remains a separate evidence gate.

## Production composition check

#1730 now contains a public-API integration test that must reproduce the same three frozen identities in order:

```text
ProofObligation
 -> proof_obligation_snapshot_id_v1
 -> admit_simulation_evidence_v1
 -> issue_obligation_discharge_receipt_v1
 -> is_obligation_discharged_v1
```

It also requires currentness refresh and semantic obligation mutation to revoke applicability while preserving the immutable historical receipt ID.

## Deliberate nonclaims

ETK-2C does not prove solver correctness, physical truth, the correctness of evidence admission itself, authenticated currentness, evidence independence, multi-obligation safety-case closure, design qualification, native/analytical evidence validity, deployment approval, fabrication approval, or actuation authority.

Its scope is deliberately narrow: **already-admitted evidence -> historical receipt -> present applicability**.
