# Evidence Contract Witness V2 — admitted-root-bound contract

V2 is the intended **post-bootstrap** authority path for the focused evidence
contract. It composes the low-level V1 candidate/provider/recipe verifier with
the immutable trusted-main root receipt defined by #1240.

V1 remains useful as a conformance primitive and for #1119's explicit
`BootstrapNoPredecessor` evaluation. After bootstrap, however, a standalone V1
`FocusedSoftwareContractWitnessed` result is **not sufficient** for trusted-root
qualification because predecessor SHA/tree + recipe identity do not prove that
the predecessor was an admitted protected root.

## Invariant

```text
candidate/provider/recipe bindings (V1)
+
independently expected TrustedMainRootReceiptId
+
root receipt bytes matching that ID
+
independently observed predecessor SHA/tree
+
independently observed trusted qualifier/workflow bytes
=
V2 admitted-root-bound witness evaluation
```

Candidate-controlled data must not choose any of the `expected_*` inputs.
Content-addressed root receipt bytes are not self-authenticating: the trusted
phase obtains the **expected root receipt ID independently** from the protected
root/evidence lineage and then verifies that the supplied receipt bytes hash to
that exact ID.

## Reuse V1 data-only collection

Use `EVIDENCE_CONTRACT_WITNESS_V1.md` steps 1–6 to independently obtain:

- repository ID/name;
- candidate exact SHA/tree;
- run ID/attempt/conclusion/workflow ID;
- canonical artifact ID/name/archive digest;
- candidate qualifier/workflow SHA-256 values;
- complete authority-test target set;
- canonical receipt bytes.

Do not execute candidate workflow/script bytes.

## Additional post-bootstrap trusted-root inputs

Ordinary V2 witnessing also requires:

1. an immutable `TrustedMainRootReceiptV1` whose
   `admission_disposition=AdmittedHistoricalRoot`;
2. the expected `TrustedMainRootReceiptId`, obtained independently rather than
   learned from the supplied root receipt bytes;
3. exact predecessor SHA/tree independently read from the historical root;
4. exact trusted predecessor qualifier/workflow bytes independently hashed;
5. a root receipt `root_harness_identity` equal to the recipe ID independently
   computed from those trusted predecessor bytes.

The root receipt must also retain the narrow non-authority fields from #1240:

```text
historical_scope              = exact-root-only
current_admission             = not-evaluated
receipt_attestation           = none
self_hosted_runner_activation = not-authorized
scientific_authority          = none
bootstrap_authority           = none
```

A root receipt using `root_harness_identity=none-pre-bootstrap` cannot authorize
an ordinary V2 witness.

## Invocation

```bash
python3 scripts/evidence_contract_witness_v2.py "$RECEIPT" \
  --expected-subject-sha "$HEAD_SHA" \
  --expected-subject-tree "$HEAD_TREE" \
  --expected-run-id "$RUN_ID" \
  --expected-run-attempt "$RUN_ATTEMPT" \
  --expected-run-conclusion "$RUN_CONCLUSION" \
  --expected-workflow-id "$WORKFLOW_ID" \
  --expected-artifact-id "$ARTIFACT_ID" \
  --expected-artifact-name "$ARTIFACT_NAME" \
  --expected-artifact-archive-sha256 "$ARCHIVE_SHA256" \
  --expected-repository "$REPOSITORY_NAME" \
  --expected-repository-id "$REPOSITORY_ID" \
  --expected-candidate-qualifier-sha256 "$CANDIDATE_QUALIFIER_SHA256" \
  --expected-candidate-workflow-sha256 "$CANDIDATE_WORKFLOW_SHA256" \
  --expected-authority-integration-targets "$EXPECTED_TARGETS" \
  --trusted-root-receipt "$TRUSTED_ROOT_RECEIPT" \
  --expected-trusted-root-receipt-id "$EXPECTED_TRUSTED_ROOT_RECEIPT_ID" \
  --trusted-predecessor-sha "$TRUSTED_PREDECESSOR_SHA" \
  --trusted-predecessor-tree "$TRUSTED_PREDECESSOR_TREE" \
  --trusted-qualifier-sha256 "$TRUSTED_QUALIFIER_SHA256" \
  --trusted-workflow-sha256 "$TRUSTED_WORKFLOW_SHA256"
```

The trusted phase must fail closed if the expected root receipt ID is absent.
Do not derive it from `$TRUSTED_ROOT_RECEIPT` inside the same authority step.

## Bootstrap

#1119 remains different. It must use only:

```text
--bootstrap-no-predecessor
```

and must not supply either a root receipt or expected root receipt ID. V2 then
preserves `BootstrapNoPredecessor`; it does not manufacture an admitted root or
ordinary witness.

## Dispositions

V2 preserves V1's candidate/recipe dispositions while adding the admitted-root
binding:

- `BootstrapNoPredecessor` — explicit first-install conformance only;
- `CandidateFailed` / `ProviderNonSuccess` — execution/provider facts;
- `RecipeChangedConformanceOnly` — candidate recipe differs from trusted recipe;
- `FocusedSoftwareContractWitnessed` — only possible post-bootstrap after the
  admitted root receipt, independently expected receipt ID, predecessor
  coordinates and trusted recipe bytes all agree.

Even `FocusedSoftwareContractWitnessed` remains narrow software-contract
historical evidence. It does not establish current admission, detached signer
identity, full repository health, environmental reproducibility, or any
scientific/Butlin/consciousness claim. #955 and #931 remain separate layers.

Related: #330, #905, #931, #955, #1119, #1157, #1240.
