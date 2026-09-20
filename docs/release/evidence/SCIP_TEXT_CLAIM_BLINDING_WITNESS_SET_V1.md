# SCIP Text-Claim Blinding Witness Set V1 — V23 Evidence Note

## Boundary

V22 validates one role-session witness at a time. V23 composes the complete required witness census for one exact V20 annotation bundle and one common V21 case-admission receipt reference.

```text
exact V20 bundle
  -> derive actual human role slots
exact V22 role-session witnesses
  -> validate each against the bundle
  -> exact role census + common bindings + cross-role uniqueness
  -> case-level process-evidence completeness receipt
```

V23 does not authenticate the retained evidence and does not recompute V21 case admission.

## Required role census is bundle-derived

The following role slots always exist:

```text
extraction-annotator-0
extraction-annotator-1
alignment-annotator-0
alignment-annotator-1
```

`extraction-adjudicator` is required if and only if the exact V20 extraction bundle contains adjudication. `alignment-adjudicator` is required if and only if the exact V20 alignment bundle contains adjudication.

The validator requires exactly one witness for every required role slot, in canonical phase/role order, and rejects missing, duplicate, extra, or reordered sessions.

The hostile suite exercises both a six-role disagreement case and a four-role exact-agreement case.

## Common case/schema bindings

Every individually valid V22 witness in one V23 set must bind the same:

- V20 annotation receipt;
- V21 case-admission receipt reference;
- public claim-schema SHA-256 actually used by the sessions.

This closes the case-level composition gap without inventing a global public-schema digest.

## Cross-role uniqueness

Across the required witness set V23 requires distinct:

- session IDs;
- challenge nonces;
- V22 witness receipts;
- session-manifest evidence roots;
- access-control snapshot roots;
- audit-capture roots.

A single retained log/session therefore cannot be duplicated across multiple human-role slots to simulate independent process coverage.

## Frozen V23 policy

Semantic policy SHA-256:

`f50117a1dbe55574f65608194943595b298e66fde6e7dea727e57a51267f1a8f`

It binds:

```text
V22 blinding-witness policy
edb837213b56f4423bc096d3bb11c90dd07de03cf5b672398b1a17de2a6b6cc0

V21 case-admission policy
452f4201c230ca3278d1ddc744fb70bca5944fa0d77b89e494096cfbf9c3dfb9
```

The composed receipt uses:

`symthaea-scip-text-claim-blinding-witness-set-v1\0`

and commits the common annotation/case/schema identities plus ordered V22 role records and their witness/access-profile receipts.

## Local execution

Before Git-object construction:

```text
PYTHONPATH=scripts python3 -m py_compile \
  scripts/scip_text_claim_blinding_witness_set.py \
  scripts/test_scip_text_claim_blinding_witness_set.py
PASS

PYTHONPATH=scripts python3 -B scripts/test_scip_text_claim_blinding_witness_set.py
PASS_BLINDING_WITNESS_SET_ADVERSARIAL
```

The suite rejects:

- missing role sessions;
- duplicated/extra role sessions;
- role-order drift;
- differing case-admission receipt references across roles;
- differing public claim-schema identities across roles while each V22 witness remains individually valid;
- reused session IDs;
- reused challenge nonces;
- reused session-manifest roots;
- reused access-control roots;
- reused audit-capture roots;
- invented adjudicator sessions when the V20 bundle has no adjudicator;
- unknown authority fields;
- duplicate JSON keys.

It also proves a common case-admission context change changes the composed receipt without promoting semantic authority.

Exact SHA-256 values:

```text
validator
394c9170eaf90fda68c1f764b88497e89adb40f0d3581255d76564a991c65c27

harness
16d941578598dd4eef2ed4ab389c00a8c8b1fb446dd8d8d831959be38f883ff4

policy
1b9dc4f9100c46f763e6b0b71a07b22a80b0c48bac93aaa657ae24257e5fe527
```

Local Git blobs:

```text
validator 6076e570aaa0062fe12697543a3a14374094bedf
harness   3ccd50e9f0566d4693efc57ee8b8607725324451
policy    c6ff51f282f032acd2d45e85687095cc6042146b
```

## Positive claim and ceiling

A valid result may state only:

```text
process_evidence_set_structurally_complete=true
all_required_role_slots_present_exactly_once=true
common_case_binding_established=true
```

It simultaneously states:

```text
case_admission_receipt_recomputed=false
evidence_authenticity_established=false
actor_authentication_established=false
audit_truthfulness_established=false
audit_completeness_in_reality_established=false
human_independence_established=false
human_expertise_established=false
human_correctness_established=false
surface_fidelity_established=false
confirmatory_execution_authorized=false
```

V23 therefore establishes process-evidence *completeness of supplied structural witnesses*, not truth/authenticity of those witnesses.

## Next boundary

V24 should bind the V23 complete-set receipt to authenticated retained evidence. The cleanest architecture is a separate Xenia-backed evidence-authenticity envelope over exact session-manifest/access-control/audit roots and witness issuer identity. That layer must authenticate evidence provenance without granting Broca truth, human-correctness, semantic-fidelity, or action authority.
