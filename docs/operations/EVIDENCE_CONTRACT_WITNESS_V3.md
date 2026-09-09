# Evidence Contract Witness V3 — independently selected root binding

V3 is the intended post-bootstrap positive witness path.

The layers are deliberately cumulative:

```text
V1
  hostile candidate receipt/provider/recipe verification

V2
  independently expected trusted-root receipt ID

V3
  the independently expected root receipt must itself be
  TrustedMainRootReceiptV2 with independently selected P0 evidence
```

V3 exists because this is not sufficient:

```text
predecessor SHA/tree
+ trusted recipe bytes
+ any self-consistent admitted-looking root receipt
```

The required chain is:

```text
reviewed P0 policy
+ independently selected structural/effective/enforcement/root evidence
        ↓
TrustedMainRootReceiptV2
        ↓
independently expected TrustedMainRootReceiptV2 ID
        ↓
V3 base-owned witness
```

## Ordinary post-bootstrap requirements

V3 requires all V1 candidate/provider inputs plus:

- `trusted_root_receipt_v2` bytes;
- an independently supplied `expected_trusted_root_receipt_id`;
- independently observed predecessor SHA and tree;
- independently hashed trusted predecessor qualifier/workflow bytes.

It then verifies:

1. the root receipt is exactly schema
   `symthaea.github-trusted-main-root-receipt.v2`;
2. the root receipt has the closed V2 field set;
3. all typed/content-addressed fields have canonical forms;
4. the V2 `trusted_evidence_selection_id` recomputes from the five selected
   evidence IDs;
5. the V2 root-receipt content ID recomputes from the exact receipt bytes;
6. that content ID equals the independently expected root-receipt ID;
7. repository/ref/root SHA/tree agree with trusted provider observations;
8. the root harness identity equals the trusted recipe derived from independently
   hashed predecessor verifier/workflow bytes;
9. the low-level V1 witness still accepts the candidate/provider/recipe evidence.

Only then can a candidate with an unchanged recipe retain
`FocusedSoftwareContractWitnessed`.

## Old root-receipt V1 is not sufficient

Witness V2 remains useful historical conformance infrastructure, but its
ordinary positive path accepts `TrustedMainRootReceiptV1`. V3 intentionally
rejects that schema.

Thus:

```text
TrustedMainRootReceiptV1
!=
post-bootstrap positive trust root
```

Once the V2 root receipt is the admitted root primitive, production admission
should target V3 (or a later strict successor), not stop at witness V2.

## Bootstrap

The #1119 bootstrap remains predecessor-free:

```text
trusted_root_receipt_id       = null
trusted_evidence_selection_id = null
relation                      = BootstrapNoPredecessor
```

V3 explicitly rejects attempts to smuggle a trusted-root receipt or predecessor
recipe/root inputs into bootstrap mode.

## Authority ceiling

A successful V3 witness still means only that the focused **software contract**
was witnessed against the exact admitted predecessor root and recipe.

It does not establish:

- scientific or consciousness evidence;
- full repository CI;
- reproducible execution environment;
- detached signer/witness authentication (#955);
- current admission (#931);
- future-root authority.

```text
historical root admission
+ software-contract witness
!=
current admission
```

## Operational selection rule

The trusted phase must possess the expected V2 root-receipt ID from its trusted
root-admission lineage before consuming the candidate/root bundle. Do not read a
root receipt, copy its `root_receipt_id`, and immediately pass that same value as
`--expected-trusted-root-receipt-id`; that recreates self-selection.

The expected root ID should be supplied from the independently recorded #1240
root-admission output or its later authenticated/attested representation.
