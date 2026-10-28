# VLEB confirmatory-run provenance

A run-manifest schema is necessary but not sufficient evidence. The evaluator must pin the exact manifest bytes out of band (for example through protected CI configuration or a separately verified attestation), and confirmatory manifests must link the evidence bundle to an approved GitHub Actions signer/workflow.

## Confirmatory checklist

1. Freeze and hash the task contract, protocol, candidate patch, permitted auditor source, task/receipt schemas, mutation catalog, mutation results, evidence receipt and evidence bundle. All referenced audit inputs must be members of the evidence ZIP; loose files outside the sealed ZIP must never be passed to the receipt auditor.
2. Create a `vleb-run-manifest-v1` manifest that records the repository's immutable numeric ID, workflow identity, exact workflow SHA/run ID/attempt, intended checkout revision, immutable runner image ID, observed network policy, toolchain digests, two-sided task controls, active mutation IDs and evidence digests. The candidate checkout revision must equal the evaluator-pinned expected head. The attestation's source digest instead identifies the trusted workflow source commit.
3. Configure a dedicated, trusted workflow to produce the evidence bundle. For confirmatory runs the workflow must create a GitHub Actions/Sigstore attestation for the **evidence bundle**, not merely a source file. Pin the signer repository and workflow. Do not let the candidate agent modify the trusted manifest or the signer workflow.
4. Independently obtain the SHA-256 of the manifest from the evaluator's protected run record. The manifest must not be allowed to supply its own trusted pin.
5. Independently pin the manifest-verifier source, its schema, the permitted receipt-auditor source, and the receipt-verifier binary digest. For repository identity, retrieve the numeric ID independently (for example `gh api repos/Luminous-Dynamics/symthaea --jq .id`), rather than relying only on a mutable `owner/repo` name. Then run the validator from a trusted checkout:

```sh
python tools/vleb/verify_run_manifest.py \
  --manifest /evidence/run-manifest.json \
  --artifact-root /evidence \
  --expected-manifest-sha256 "$TRUSTED_RUN_MANIFEST_SHA256" \
  --expected-verifier-sha256 "$TRUSTED_MANIFEST_VERIFIER_SHA256" \
  --expected-schema-sha256 "$TRUSTED_MANIFEST_SCHEMA_SHA256" \
  --expected-auditor-sha256 "$TRUSTED_RECEIPT_AUDITOR_SHA256" \
  --expected-verifier-binary-sha256 "$TRUSTED_RECEIPT_VERIFIER_BINARY_SHA256"
```

For a confirmatory manifest, the validator fails closed unless `gh attestation verify --format json` succeeds for the declared evidence bundle, repository, signer workflow, source ref/digest and predicate type. It then matches immutable certificate claims (repository ID, workflow identity/SHA/ref, invocation run ID/attempt) and the subject's SHA-256 to the manifest. It safely extracts the bundle, verifies all referenced file digests *inside the ZIP*, and invokes the independently pinned receipt auditor. The auditor source and receipt-verifier binary hashes must match caller-supplied pins, not merely manifest fields. The validator cannot make a public-development task score-eligible; it reports `score_eligible=false` for development smoke manifests.

## Security boundary

- SHA-256 checks prove only that the bytes match the expected digest.
- A signed GitHub artifact attestation establishes a provenance relationship to the workflow/signer. It does not prove the workflow did correct work or that the code is semantically correct.
- The receipt auditor checks task behavior evidence, test inventory, mutation kills, differential observables and runtime identity, but its correctness still depends on the trusted verifier/source and the validity of the underlying oracle.
- Runtime image and toolchain identity fields must be collected by trusted runner logic, not copied from untrusted receipt text.
- A confirmatory result is score-eligible only when the manifest pin, bundle attestation, artifact hashes, task admission controls and independent receipt audit all pass. Otherwise retain the outcome as fail/unresolved and do not emit a headline score.

Official references: [GitHub artifact-attestation security model](https://docs.github.com/en/actions/concepts/security/artifact-attestations), [attestation verification command](https://cli.github.com/manual/gh_attestation_verify), [actions/attest](https://github.com/actions/attest).
