# SCIP Text-Claim Case Admission V1 — V21 Evidence Note

## Boundary

V19 defines the canonical corpus case and whole-corpus split/leakage theorems. V20 defines the human annotation/adjudication bundle and its content receipt. V21 is the narrow composition bridge between them.

```text
exact V20 bundle
  -> V20 structural validation + annotation receipt
exact V19 case candidate
  -> independent V19 single-case identity reconstruction
cross-layer equality checks
  -> case admission receipt
```

The bridge does not run V19 whole-corpus validation and therefore cannot establish split assignment or cross-split leakage safety.

## Critical expected-inventory correction

V19's `expected_inventory_sha256` is the human gold **surface extraction inventory**. It must therefore equal V20's:

`frozen_surface_inventory_sha256`

and not V20's later source-alignment inventory.

This preserves the benchmark sequence:

```text
surface -> expected extracted claims -> exact extraction scoring
                                  \
                                   -> hidden source alignment -> V17 semantic comparison
```

V20's annotation receipt still commits the hidden alignment evidence, but the alignment representation may not replace the extraction target in a V19 case. V21 has an explicit hostile regression for that substitution.

## Exact binding

A positive V21 result requires exact equality between the V19 case and the V20 validation result for:

- `surface_sha256`;
- `source_inventory_sha256`;
- `expected_inventory_sha256 == frozen_surface_inventory_sha256`;
- `annotation_receipt_sha256`.

V21 independently reconstructs the V19 `case_id` using the exact V19 case-identity domain and field set.

The semantic `case_id` continues to exclude `assignment_key_sha256` and `split`, as required by V19. The V21 admission receipt deliberately includes both so assignment/split context changes produce a different admission receipt without manufacturing a different semantic case.

## Frozen V21 policy

Semantic policy SHA-256:

`452f4201c230ca3278d1ddc744fb70bca5944fa0d77b89e494096cfbf9c3dfb9`

It binds:

- V19 manifest policy `9a30ba36b3e872aaec349647f71433579bc0932f24c96335bb32afe77498ffc0`;
- V20 annotation policy `12a8431fd8a93dd8ba158758bb4089ef951960240fcab11a5178159e67f19bb9`.

Admission receipts use the NUL-terminated domain:

`symthaea-scip-text-claim-case-admission-v1\0`

and bind V19 case identity, assignment key, split, V20 annotation receipt, and the hidden V20 alignment inventory digest.

## Local execution

Before Git object construction:

```text
python3 -m py_compile scripts/scip_text_claim_case_admission.py \
  scripts/test_scip_text_claim_case_admission.py
PASS

python3 -B scripts/test_scip_text_claim_case_admission.py
PASS_CASE_ADMISSION_ADVERSARIAL
```

The suite verifies:

- a correctly bound V20/V19 case is admitted;
- surface substitution fails even with a recomputed V19 case ID;
- source substitution fails even with a recomputed case ID;
- annotation-receipt substitution fails;
- hidden aligned inventory cannot replace the frozen surface inventory as V19 expected gold;
- forged V19 case identity fails;
- assignment-key changes preserve semantic case ID but change admission receipt;
- split changes preserve semantic case ID but change admission receipt and never authorize confirmatory execution;
- leakage-metadata changes alter semantic case ID and admission receipt when the case ID is correctly recomputed;
- an invalid V20 bundle fails through the inherited V20 validator;
- V19 policy substitution fails;
- V21 authority-boundary relaxation fails.

Exact final SHA-256 values:

```text
bridge
52fe894f3822cd0f892a1b31fab4ff1dbe08346f2be887bfbf65585d8121f678

harness
fc79282229748fa911888e44eeb938942410295ab4cc793995ac89d49dda3892

policy
fd7250f061b6ba97eab8979993b5d622dc6ad572d1dd22b070e0440866186439
```

Local Git blobs:

```text
bridge  537bdda1403ef426b6ea653e2f1759b8893970a8
harness 13d0def16bc596529ff62e4e5ef5fc0b60d53961
policy  26cde8cef449abdf2dea56ba3b9d905330aee7da
```

## Positive claim and ceiling

V21 may emit only the narrow positive state:

`eligible_for_v19_manifest_membership_subject_to_global_validation=true`

That means the single case's human-reference identities are coherently bound. It does not mean the case's split is correct or the corpus is safe as a whole.

The same result explicitly retains:

```text
split_assignment_verified=false
cross_split_leakage_verified=false
v19_manifest_qualified=false
human_correctness_established=false
surface_fidelity_established=false
confirmatory_execution_authorized=false
```

V19 remains the only layer that can validate deterministic split reconstruction and cross-split leakage over the complete sealed corpus.
