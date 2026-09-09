# Trusted Main Root Receipt V2 — independent evidence selection

V2 closes the evidence-selection gap left intentionally visible by V1.

V1 answers:

```text
Are these supplied policy/readback/root/enforcement objects internally valid,
content-addressed, mutually bound, and sufficient for the historical-root
policy theorem?
```

V2 additionally answers:

```text
Are these the exact evidence objects independently selected by the trusted
phase for this admission event?
```

Those are different questions.

## Core invariant

```text
self-consistent evidence bundle
!=
trusted evidence selection
```

A V2 admitted receipt requires all five identities below to be supplied
independently of the evidence bundle being composed:

- `expected_protection_policy_id`
- `expected_protection_verification_id`
- `expected_effective_rules_verification_id`
- `expected_root_subject_id`
- `expected_enforcement_evidence_id`

The V2 composer re-runs V1 validation, re-derives each identity from the supplied
bytes, and rejects if any re-derived identity disagrees with the independently
expected value.

## Trusted source for each selector

The trusted phase should obtain the selectors from the evidence lineage that
produced them, not by reading the corresponding ID field from the bundle just
before invoking V2.

Conceptually:

```text
reviewed P0 policy bytes
    -> independently recorded ProtectionPolicyId

trusted structural readback execution/output
    -> independently recorded ProtectionVerificationId

trusted effective-rules execution/output
    -> independently recorded EffectiveRulesVerificationId

GitHub commit/ref readback selected by trusted root-admission logic
    -> independently recorded RootSubjectId

trusted enforcement-evidence producer / administrative evidence lineage
    -> independently recorded EnforcementEvidenceId
```

Then and only then:

```text
expected IDs selected above
+ evidence bytes supplied for composition
        ↓
V2 exact identity comparisons
        ↓
AdmittedHistoricalRoot
```

## Forbidden operator pattern

Do **not** do this:

```text
read evidence bundle
extract its five ID fields
pass those same fields as --expected-*
```

That is self-selection and defeats the purpose of V2.

The trusted orchestrator should already possess the expected identities before
it accepts the evidence bundle for composition. If it does not, the admission
step is not ready.

## V1 relationship

V1 remains useful for:

- closed-schema parsing;
- derived content-ID recomputation;
- repository/ref/ruleset cross-binding;
- root SHA binding;
- enforcement semantic validation;
- conformance and debugging.

But an authority-bearing post-P0 historical root should be represented by the
V2 receipt, not by treating a standalone V1 receipt as sufficient evidence
selection.

V2 retains the V1 receipt ID only as provenance:

```text
v1_root_receipt_id
```

It is not the V2 trust decision.

## V2 identities

V2 adds a domain-separated content identity for the trusted selection itself:

```text
TrustedEvidenceSelectionId = H(
    ProtectionPolicyId,
    ProtectionVerificationId,
    EffectiveRulesVerificationId,
    RootSubjectId,
    EnforcementEvidenceId
)
```

The final V2 root-receipt identity also binds the exact root/harness and V1
provenance coordinates.

Changing any independently selected evidence coordinate therefore changes the
selection ID and final root receipt ID.

## Authority ceiling

Even a valid V2 receipt says only:

```text
this exact root was historically admitted under these exact selected evidence
objects
```

It still explicitly does **not** establish:

- current admission (`#931`);
- detached signer/witness authentication (`#955`);
- scientific, Butlin, consciousness, or benchmark authority;
- self-hosted runner activation;
- bootstrap authorization for `#1119`;
- future-root authorization.

Thus:

```text
content-addressed selection
!=
authenticated producer identity
!=
current admission
```

## CLI contract

The V2 command requires all trusted selectors explicitly:

```text
python3 scripts/trusted_main_root_receipt_v2.py \
  POLICY.json \
  STRUCTURAL.json \
  EFFECTIVE.json \
  ROOT_SUBJECT.json \
  ENFORCEMENT.json \
  --root-harness-identity "$ROOT_HARNESS_ID" \
  --expected-protection-policy-id "$EXPECTED_POLICY_ID" \
  --expected-protection-verification-id "$EXPECTED_STRUCTURAL_ID" \
  --expected-effective-rules-verification-id "$EXPECTED_EFFECTIVE_ID" \
  --expected-root-subject-id "$EXPECTED_ROOT_SUBJECT_ID" \
  --expected-enforcement-evidence-id "$EXPECTED_ENFORCEMENT_ID"
```

The command exits successfully only after V1 itself yields
`AdmittedHistoricalRoot` and every independently expected identity matches the
re-derived evidence bytes.

## Bootstrap

V2 must not be used to fabricate a root receipt for the pre-protection #1119
bootstrap. The existing bootstrap remains `BootstrapNoPredecessor`.

The intended order remains:

```text
P0 server protection applied
    ↓
trusted structural/effective/enforcement evidence produced
    ↓
trusted phase records expected IDs
    ↓
TrustedMainRootReceiptV2
    ↓
future root-receipt-bound base-owned witness
    ↓
detached attestation
    ↓
current admission
```
