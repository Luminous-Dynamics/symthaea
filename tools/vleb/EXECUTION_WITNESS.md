# VLEB evaluator-observed execution witness receipts

This prototype records a process invocation as a content-addressed witness. It complements the atomic legacy-claim contract and is intended to close one gap between “a claim contains a hash” and “a trusted evaluator observed the verifier run.”

## Files

- `run_execution_witness.py` runs a pinned invocation from an evaluator-authored plan without a shell, captures stdout/stderr into the evidence root, hashes inputs before/after execution, and writes a receipt containing task/claim identity, repository ID, run ID/attempt, candidate revision, source digest, runtime identity, verifier executable digest, command/environment digests and timing.
- `execution-witness-plan.schema.json` defines the evaluator-authored plan contract.
- `execution-witness-receipt.schema.json` defines the recorded receipt contract.
- `audit_execution_witness.py` independently checks the receipt against out-of-band pins, the exact plan, input/output bytes, executable digest and evaluator-supplied task/run/source/runtime identity.
- `test_execution_witness.py` covers positive execution, replay across run attempts, plan/input/output tampering, wrong verifier pins, path traversal, and the distinction between an auditable command with an unexpected exit code and a passing execution contract.

## Trust model

The plan is evaluator-authored and its SHA-256 must come from protected evaluator configuration, not from the candidate or receipt. The runner source, receipt auditor source/schema, verifier executable, task/source identity, run attempt and immutable runtime identity are independently pinned. The enclosing evidence ZIP should be attested by the trusted workflow and include the plan, receipt, inputs, stdout and stderr. Audit paths are resolved from the sealed evidence root; loose files outside the bundle are not evidence.

The runner records actual process metadata, but does not decide semantic truth. `execution_contract_passed` means only that the observed exit code matched the plan's expected exit code and the process did not time out. It does **not** establish that the program's behavior is correct. Semantic claims still need a pinned verifier and an independently justified oracle. The audit result separates `evidence_valid` from `execution_contract_passed` so a real but failing execution can remain visible instead of being mislabeled as a successful witness.

The receipt's own fields are not a cryptographic signature. Trust comes from running this pinned recorder in evaluator-controlled CI, binding the receipt and all referenced artifacts into the attested evidence bundle, and auditing them against protected out-of-band pins. Do not let an agent choose the plan or the trusted identities. Do not pass secrets in argv or environment variables; the runner rejects environment key names that look secret-bearing.

## Local protocol

Plan paths and artifact paths are relative to a dedicated evidence root. A plan has `schema_version: "vleb-execution-witness-plan-v1"` and the exact field set validated by the runner:

`task_id`, `claim_id`, `repository`, `run`, `candidate_revision`, `source_sha256`, `runtime`, `verifier`, `invocation`, `source_path`, `inputs`, and `outputs`.

The `invocation` contains the exact `argv`, working directory, explicit environment allowlist, timeout, and expected exit code. The executable path in `argv[0]` must match the pinned verifier executable path. Each input has a predeclared SHA-256; the recorder rejects changed inputs, records before/after digests, and stores captured output files without overwriting existing evidence.

Example invocation (all values shown as placeholders must be supplied by protected evaluator configuration):

```sh
python tools/vleb/run_execution_witness.py \
  --plan plan.json \
  --evidence-root /evidence \
  --receipt-path receipts/claim-01.json \
  --expected-plan-sha256 "$TRUSTED_PLAN_SHA256" \
  --expected-runner-sha256 "$TRUSTED_WITNESS_RUNNER_SHA256" \
  --expected-verifier-sha256 "$TRUSTED_WITNESS_VERIFIER_SHA256" \
  --expected-task-id "$TASK_ID" \
  --expected-claim-id "$CLAIM_ID" \
  --expected-repository "Luminous-Dynamics/symthaea" \
  --expected-repository-id "$TRUSTED_REPOSITORY_ID" \
  --expected-run-id "$GITHUB_RUN_ID" \
  --expected-run-attempt "$GITHUB_RUN_ATTEMPT" \
  --expected-candidate-revision "$CANDIDATE_HEAD_SHA" \
  --expected-source-sha256 "$TRUSTED_SOURCE_SHA256" \
  --expected-runtime-image-id "$TRUSTED_RUNTIME_IMAGE_ID" \
  --expected-platform "$TRUSTED_PLATFORM" \
  --expected-network-mode "none"
```

Audit in a trusted checkout, using the receipt and plan inside the attested bundle and the receipt/auditor/schema/runner/plan/verifier digests stored out of band:

```sh
python tools/vleb/audit_execution_witness.py \
  --receipt receipts/claim-01.json \
  --plan plan.json \
  --schema tools/vleb/execution-witness-receipt.schema.json \
  --evidence-root /evidence \
  --expected-receipt-sha256 "$TRUSTED_RECEIPT_SHA256" \
  --expected-auditor-sha256 "$TRUSTED_WITNESS_AUDITOR_SHA256" \
  --expected-schema-sha256 "$TRUSTED_WITNESS_SCHEMA_SHA256" \
  --expected-runner-sha256 "$TRUSTED_WITNESS_RUNNER_SHA256" \
  --expected-plan-sha256 "$TRUSTED_PLAN_SHA256" \
  --expected-verifier-sha256 "$TRUSTED_WITNESS_VERIFIER_SHA256" \
  --expected-task-id "$TASK_ID" \
  --expected-claim-id "$CLAIM_ID" \
  --expected-repository "Luminous-Dynamics/symthaea" \
  --expected-repository-id "$TRUSTED_REPOSITORY_ID" \
  --expected-run-id "$GITHUB_RUN_ID" \
  --expected-run-attempt "$GITHUB_RUN_ATTEMPT" \
  --expected-candidate-revision "$CANDIDATE_HEAD_SHA" \
  --expected-source-sha256 "$TRUSTED_SOURCE_SHA256" \
  --expected-runtime-image-id "$TRUSTED_RUNTIME_IMAGE_ID" \
  --expected-platform "$TRUSTED_PLATFORM" \
  --expected-network-mode "none"
```

## Relationship to the VLEB score gate

This is a new execution-evidence primitive. It is not yet wired into confirmatory score eligibility or the atomic claim auditor. That integration must make every `independent_execution` claim reference an audited receipt and evaluator-pinned plan, and must keep missing, contradicted and unresolved claims in the assigned-task denominator. Until that integration and the exact-head test suite pass, do not count a receipt as benchmark evidence for a published score.

## Standards and benchmark design

The structure follows the general separation in the [in-toto Attestation Framework](https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md): artifact identity/digests and the meaning of an attestation are separate concerns. Software provenance can establish where an artifact came from, not that the execution was semantically correct ([SLSA provenance](https://slsa.dev/spec/v1.1/provenance), [GitHub artifact attestations](https://docs.github.com/en/actions/concepts/security/artifact-attestations)). For AI-system evaluation, the emerging [NIST TEVV-Athlon framework](https://www.nist.gov/artificial-intelligence/ai-research/tevv-athlon-framework-evaluating-ai-systems) likewise motivates explicit verification and validation evidence rather than a score alone.
