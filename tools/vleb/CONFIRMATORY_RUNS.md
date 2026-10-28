# VLEB confirmatory-run provenance

A run-manifest schema is necessary but not sufficient evidence. The evaluator must pin the exact manifest bytes out of band (for example through protected CI configuration or a separately verified attestation), and confirmatory manifests must link the evidence bundle to an approved GitHub Actions signer/workflow.

## Confirmatory checklist

1. Freeze and hash the task contract, protocol, candidate patch, permitted receipt-auditor source, task/receipt schemas, mutation catalog/results, evidence receipt, claim-auditor source/schema, witness-receipt schema and recorder, evaluator-pinned atomic claim inventory and witness execution receipts, and the evidence bundle. All referenced audit inputs must be members of the evidence ZIP; loose files outside the sealed ZIP must never be passed to either auditor.
2. Create a `vleb-run-manifest-v1` manifest that records the repository's immutable numeric ID; workflow identity/ref/SHA; source-repository digest/ref for the actual run; run ID/attempt; intended candidate checkout revision; immutable runner image ID; observed network policy; toolchain digests; two-sided task controls; active mutation IDs; and evidence digests. The candidate checkout revision must equal the evaluator-pinned expected head. Keep three Git identities distinct: (a) candidate checkout revision, (b) source-repository digest/ref from the run, and (c) workflow identity SHA/ref. For pull requests, the source/run SHA may be a merge commit while the candidate head is the PR's head commit; do not collapse them into one field.
3. Configure a dedicated, trusted workflow to produce the evidence bundle. For confirmatory runs the workflow must create a GitHub Actions/Sigstore attestation for the **evidence bundle**, not merely a source file. Pin the signer repository and workflow. Do not let the candidate agent modify the trusted manifest or the signer workflow.
4. Independently obtain the SHA-256 of the manifest from the evaluator's protected run record. The manifest must not be allowed to supply its own trusted pin.
5. Independently pin the manifest-verifier source/schema, permitted receipt-auditor source and binary, claim-auditor source/schema, witness-receipt schema and producer, and each admitted witness-verifier kind. Obtain these from protected evaluator configuration, never from the candidate manifest. For repository identity, retrieve the numeric ID independently (for example `gh api repos/Luminous-Dynamics/symthaea --jq .id`), rather than relying only on a mutable `owner/repo` name. Then run the validator from a trusted checkout:

```sh
python tools/vleb/verify_run_manifest.py \
  --manifest /evidence/run-manifest.json \
  --artifact-root /evidence \
  --expected-manifest-sha256 "$TRUSTED_RUN_MANIFEST_SHA256" \
  --expected-verifier-sha256 "$TRUSTED_MANIFEST_VERIFIER_SHA256" \
  --expected-schema-sha256 "$TRUSTED_MANIFEST_SCHEMA_SHA256" \
  --expected-auditor-sha256 "$TRUSTED_RECEIPT_AUDITOR_SHA256" \
  --expected-verifier-binary-sha256 "$TRUSTED_RECEIPT_VERIFIER_BINARY_SHA256" \
  --expected-claim-auditor-sha256 "$TRUSTED_CLAIM_AUDITOR_SHA256" \
  --expected-claim-schema-sha256 "$TRUSTED_CLAIM_SCHEMA_SHA256" \
  --expected-witness-receipt-schema-sha256 "$TRUSTED_WITNESS_RECEIPT_SCHEMA_SHA256" \
  --expected-witness-receipt-producer-sha256 "$TRUSTED_WITNESS_RECEIPT_PRODUCER_SHA256" \
  --trusted-claim-verifier "source_index=$TRUSTED_SOURCE_INDEX_VERIFIER_SHA256" \
  --trusted-claim-verifier "independent_execution=$TRUSTED_EXECUTION_VERIFIER_SHA256"
```

For a confirmatory manifest, the validator fails closed unless `gh attestation verify --format json` succeeds for the declared evidence bundle, repository, signer workflow, source ref/digest and predicate type. It then matches immutable certificate claims (repository ID, workflow identity/SHA/ref, source-repository digest/ref, invocation run ID/attempt) and the subject's SHA-256 to the manifest. It safely extracts the bundle, verifies all referenced file digests *inside the ZIP*, and invokes the independently pinned receipt auditor. The auditor source and receipt-verifier binary hashes must match caller-supplied pins, not merely manifest fields. Confirmatory scoring also requires the claim-evidence gate to validate every manifest-pinned claim and each witness execution receipt. The evaluator checks receipt digests and schema, source input/output bytes, producer/verifier identity pins, and task/claim/repository/workflow/candidate/run-attempt/runtime bindings. Missing verifier or producer pins are unresolved and make a run ineligible. The receipt producer must be invoked from a protected workflow with independently obtained run context; the candidate must not supply the observed-result classification. The validator cannot make a public-development task score-eligible; it reports `score_eligible=false` for development smoke manifests.

The recorder requires a protected-run context and an externally computed `--expected-invocation-sha256`. Compute that digest over the exact `argv` array, working directory and timeout using canonical JSON (sorted keys, compact separators, UTF-8, no ASCII escaping), then store the same digest in the evaluator-pinned claim witness as `invocation_sha256`. The claim witness inventory/claim hash must be pinned by the evaluator, not inferred from candidate-authored output. The verifier independently recomputes the digest from the receipt and rejects a mismatch even if the receipt and bundle are rehashed. Do not compute this expected hash from a candidate-generated receipt: it must come from the trusted task/run policy.

The recorder's `--observed-result` is still a policy input, not an inferred truth. A protected evaluator must choose it according to the pinned verifier's expected contract; for a verifier without an unambiguous result on exit status alone, use a trusted result-classifier and record inconclusive when classification is uncertain. The candidate must not choose the receipt, classification or expected invocation hash.

## Security boundary

- SHA-256 checks prove only that the bytes match the expected digest.
- A signed GitHub artifact attestation establishes a provenance relationship to the workflow/signer. It does not prove the workflow did correct work or that the code is semantically correct.
- The receipt auditor checks task behavior evidence, test inventory, mutation kills, differential observables and runtime identity, but its correctness still depends on the trusted verifier/source and the validity of the underlying oracle.
- Runtime image and toolchain identity fields must be collected by trusted runner logic, not copied from untrusted receipt text.
- A confirmatory result is score-eligible only when the manifest pin, bundle attestation, artifact hashes, task admission controls, independent receipt audit and independent claim-evidence audit all pass. Otherwise retain the outcome as fail/unresolved and do not emit a headline score.
- Witness receipts bind recorded execution metadata and output bytes to the exact attested run; they are not a signature or standalone proof of semantic truth. The protected evaluator must invoke the pinned recorder, independently classify outcomes, and ensure the command is the intended verifier. Timeouts are inconclusive and cannot support a claim. Validate the receipt schema/producer pins and actual Sigstore output at the exact head before any confirmatory score is published; see [issue #8702](https://github.com/Luminous-Dynamics/symthaea/issues/8702).

Official references: [GitHub artifact-attestation security model](https://docs.github.com/en/actions/concepts/security/artifact-attestations), [attestation verification command](https://cli.github.com/manual/gh_attestation_verify), [actions/attest](https://github.com/actions/attest).
