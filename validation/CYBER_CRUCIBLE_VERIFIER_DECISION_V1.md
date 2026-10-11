# Cyber Crucible verifier-decision receipt v1

## Purpose and security boundary

This contract makes a verifier's decision independently consumable. It is deliberately separate from the evaluator's signed receipt. The decision receipt authenticates a decision under a specified signing policy; it does **not** prove that the verifier binary was uncompromised, that source evidence is true, that a held-out oracle was correctly implemented, that an incident diagnosis is correct, or that an action is safe.

The receipt is always `non_authorizing: true`. A consumer MUST NOT use the `accept` value alone as operational permission. It means only that the verifier signed an acceptance decision for the exact bound input and context.

## Canonicalization and signature

The detached Ed25519 signature covers:

```text
UTF8("symthaea-cyber-crucible-verifier-decision.v1") || 0x00
    || RFC8785_JCS(payload)
```

The envelope contains exactly `payload` and `signature`. The signature is unpadded base64url. Unknown fields, duplicate JSON members, unknown policy fields, noncanonical encodings, and invalid context are rejected.

## Bindings

The payload binds the exact campaign, subject, attempt, scenario ID/revision/digest, input-receipt digest, complete diagnostic-report digest, evaluator-policy digest, expected context digest, verifier-build digest, verifier-provenance-policy digest, decision-signing-policy digest, signer identity, decision, reason classification, and validity interval.

`input_receipt_digest` is SHA-256 over RFC 8785 JCS of the complete upstream receipt envelope, including its detached signature. The integrated consumer requires that exact envelope as an input, recomputes the digest, and checks campaign, subject, attempt, and scenario fields against trusted expected context. This proves which input artifact the decision names; the signed verifier decision attests that the verifier accepted or rejected it under the stated policy.

`report_digest` is SHA-256 over RFC 8785 JCS of the complete diagnostic report object. The integrated consumer recomputes it and checks report subject, scenario identity, and run ID against the trusted expected context. For this contract, `attempt_id` must equal `report.run.run_id`.

`context_digest` is SHA-256 over RFC 8785 JCS of the complete externally supplied expected-context object with exactly the fields defined in the verifier. That expected context MUST be independently supplied by a trusted consumer/operator; it MUST NOT be copied from the decision receipt itself. The required context fields are `campaign_id`, `subject_id`, `attempt_id`, `scenario_id`, `scenario_revision`, `scenario_digest`, `input_receipt_digest`, `report_digest`, `evaluator_policy_digest`, `verifier_build_digest`, `verifier_provenance_policy_digest`, and `decision_nonce`. The provenance policy digest is SHA-256 over the deterministic JSON representation used by the provenance verifier; the integrated consumer recomputes it from the operator-controlled provenance policy and rejects a mismatch before consuming the signed decision.

The verifier build digest is also externally supplied. A signed self-assertion inside a receipt does not establish the executable that actually ran. Production provenance therefore belongs to a separate trusted release/attestation channel; this verifier checks only the digest binding. Issued/expiry timestamps are signer claims checked against the verifier's local clock; this contract does not establish a trusted timestamping service or a rollback-resistant clock.

## Trust policy and replay

The decision-signing trust policy must be obtained through an operator-controlled trust-root path, not from the subject under evaluation. Key IDs and public-key material must be unique; key validity windows, revocation state, and the verifier identity are checked. Receipts must be valid for no longer than one hour, must be currently valid, and are only accepted when a persistent replay ledger is supplied.

The SQLite ledger provides atomic single-host uniqueness for decision IDs and one-time challenge nonces. It is not distributed replay protection and it does not provide rollback protection against a host administrator who can restore the database. A distributed deployment needs a shared transactional uniqueness service with availability, consistency, and rollback properties appropriate to its threat model. The ledger directory and its parent hierarchy must be controlled by the verifier operator; checking the final path alone cannot secure an attacker-controlled parent directory against filesystem races.

## Decision semantics

The `accept`/`reject` value applies only to the verifier's acceptance of the exact receipt/report artifacts under the stated integrity and trust policy. It is not an evaluator competence score, not a statement that an incident diagnosis is correct, and not an operational authorization. A valid signed `reject` receipt is a successfully authenticated decision receipt, but it is not an acceptance. The CLI returns exit code zero when the receipt is authenticated, regardless of whether its decision is `accept` or `reject`; consumers must inspect the returned `decision`. Verification errors return exit code 2. Neither result grants action authority.

Reason codes are bounded, redacted classifications rather than free-form attacker-controlled diagnostics. For an `accept` decision, the reason code must be `authenticated_under_policy`; reject decisions cannot use that code.

## CLI

The integrated consumer takes the report, the exact upstream signed receipt, the signed decision receipt, operator-controlled decision trust policy, and independently supplied expected context:

```sh
python scripts/verify_cyber_crucible_report_decision.py \
  --report diagnostic-loop-report.json \
  --input-receipt upstream-signed-receipt.json \
  --decision-receipt decision-receipt.json \
  --trust-policy decision-trust-policy.json \
  --expected-context trusted-expected-context.json \
  --verifier-artifact /opt/symthaea/releases/cyber-crucible-verifier-v1.tar.gz \
  --provenance-policy /etc/symthaea/trust/verifier-provenance-policy.json \
  --replay-db /var/lib/symthaea/cyber-crucible-decisions.sqlite3
```

The consumer validates the existing diagnostic-loop contract, checks the exact input and report digests plus stable run/scenario identity against expected context, verifies the provenance of the exact verifier bundle, validates the bundle's deterministic inventory and per-file digests, checks that the bundle SHA-256 equals the `verifier_build_digest` and the provenance-policy digest equals the signed `verifier_provenance_policy_digest`, and only then authenticates the signed decision receipt. The provenance policy must be supplied through an operator-controlled path; it is not read from the report or the subject. The lower-level `verify_cyber_crucible_decision_receipt.py` CLI remains available when only a decision receipt needs to be authenticated, but it does not provide the integrated provenance or bundle-inventory gate.

Do not place signing private keys in the subject workspace or source repository. The signing service should construct receipts only after it has independently computed the decision and bound context. This repository contains a verifier, not a production decision-signing key or signing service.

## Deterministic verifier bundle

`scripts/package_cyber_crucible_verifier.py` builds `cyber-crucible-verifier-v1.tar.gz` deterministically. The archive includes the source verifier, diagnostic-loop validator, all decision/report/provenance checks, strict schemas, public calibration fixture, locked Python requirements, the regression suites, and the workflows/docs needed to audit the contract.

The embedded `BUNDLE-MANIFEST.json` records the declared source commit and an exact sorted inventory with SHA-256 and byte length for every included file. The packager fixes gzip timestamp, tar entry ordering, ownership, timestamps, and modes; rejects symlinked source files and output traversal; writes atomically; then verifies its own output. The contract workflow builds the bundle twice from the same checked-out commit, byte-compares the archives, and validates inventory/manifest digests.

Build and independently re-check an archive:

```sh
SOURCE_COMMIT="$(git rev-parse HEAD)"
python scripts/package_cyber_crucible_verifier.py build \
  --root . \
  --output dist/cyber-crucible-verifier-v1.tar.gz \
  --source-commit "$SOURCE_COMMIT"
python scripts/package_cyber_crucible_verifier.py verify \
  --bundle dist/cyber-crucible-verifier-v1.tar.gz \
  --expected-source-commit "$SOURCE_COMMIT"
sha256sum dist/cyber-crucible-verifier-v1.tar.gz
```

The declared commit in the manifest is descriptive metadata, not trusted provenance by itself. The verifier's independent provenance policy and verified attestation must bind the archive digest to the source ref/commit and trusted builder. The bundle validator checks the inventory but does not establish that the live process is running the contents extracted from that archive.
## Verifier build-provenance gate

This tranche adds `scripts/verify_cyber_crucible_verifier_provenance.py` and the strict `cyber_crucible_verifier_provenance_policy_v1.schema.json`. The verifier delegates cryptographic signature, certificate/issuer, signer-workflow, source-ref/digest, SLSA predicate-type, and self-hosted-runner checks to an operator-pinned GitHub CLI running `gh attestation verify`. It independently hashes the artifact and then applies exact SLSA policy checks to the CLI-verified statement.

Policy binds the artifact SHA-256 and subject name; artifact source repository/ref/commit and workflow path; the separate attestation signer repository/workflow/commit; OIDC issuer; SLSA `builder.id` and `buildType`; and the entire expected `externalParameters` object. For the currently pinned `actions/attest` v4.2.2 producer, the contract fixtures expect GitHub's native `buildType` URI `https://actions.github.io/buildtypes/workflow/v1` and builder ID `https://github.com/actions/runner` (not the similarly named community-maintained SLSA GitHub Actions build type). A real release attestation still must confirm the output shape before this can be called operationally qualified. Validation rejects multiple verified statements, multiple subjects, missing verified transparency timestamps, subject mismatch, changed source commit, changed builder/build type, or extra/unexpected external parameters. A signed self-assertion in a decision receipt is not a substitute for this verification.

Example invocation (the policy must be independently provisioned, not accepted from the subject):

```sh
python scripts/verify_cyber_crucible_verifier_provenance.py \
  --artifact /opt/symthaea/releases/cyber-crucible-verifier-v1.tar.gz \
  --policy /etc/symthaea/trust/verifier-provenance-policy.json
```

The policy also pins the absolute GitHub CLI executable path and its SHA-256. Operators must obtain that CLI binary and expected digest from a separate trusted installation channel and protect the executable, policy file, and their parent directories from subject-controlled writes. This script checks a supplied hash; it does not bootstrap trust in the hash itself. Hash comparisons before/after verification reduce accidental or simple replacement races, but do not eliminate TOCTOU risk if an attacker controls those paths.

The attestation signer workflow is intentionally separate from the source build workflow: `signer_workflow_path` and `signer_workflow_commit` pin the trusted workflow identity, while `workflow_path`, `source_ref`, and `source_commit` bind the build inputs encoded in the SLSA statement. The operator should use a protected, independently reviewed signer workflow whose execution cannot be influenced by untrusted caller inputs. GitHub documents that predicate content can be workflow-controlled even after the signature has been verified, so this verifier compares those claims against a frozen policy instead of treating signature validity as proof that every claim is true.

The artifact bundle itself is independently checked by `package_cyber_crucible_verifier.py verify`. The attestation wrapper currently uses online GitHub API retrieval; offline verification of a saved Sigstore/GitHub attestation bundle through `gh attestation verify --bundle` is not exposed by this wrapper yet. It also does not verify the attestation signer workflow's source-code review status or prove the running verifier process is uncompromised. Its output is non-authorizing and cannot approve a containment action.

Research basis:

- [GitHub CLI `gh attestation verify`](https://cli.github.com/manual/gh_attestation_verify) documents repository/workflow/source identity constraints and the JSON verification result, including certificate identity and verified timestamps.
- [GitHub artifact attestations](https://docs.github.com/en/actions/concepts/security/artifact-attestations) describes the provenance guarantees and explicitly notes that attestations are not a guarantee that an artifact is secure.
- [SLSA v1.2 artifact-verification guidance](https://slsa.dev/spec/v1.2/verifying-artifacts) recommends checking the trusted builder, provenance signature, `buildType`, and expected `externalParameters`.
- The [SLSA GitHub Generator repository](https://github.com/slsa-framework/slsa-github-generator) notes that the project is no longer actively maintained and points users toward GitHub artifact attestations for new use cases.

## Trusted release and attestation workflow

`.github/workflows/cyber-crucible-verifier-release.yml` is a tag-only caller for refs matching `cyber-crucible-verifier-v*`. It invokes the trusted reusable workflow at the full immutable commit `d64e0907a3d12aa9f1e467bb169eae9ff91d92a3`; that reusable workflow does not depend on mutable `@main`.

The reusable workflow receives the release ref and GitHub event commit as separate inputs, checks out the exact ref without persisting checkout credentials, and fails unless `git rev-parse HEAD` equals the supplied 40-character commit. Its build job has only `contents: read`; it installs pinned Python dependencies, meta-validates schemas, runs all receipt/report/provenance/bundle regressions, constructs the archive twice, byte-compares the results, and verifies the exact source commit and bundle inventory.

The `release-approval` job waits at the `cyber-crucible-verifier-release` GitHub Environment after the bundle has passed testing. Repository administrators must configure this Environment with required reviewers (ideally disallowing self-review and administrator bypass) and an allowlist restricted to approved release tags. If the environment does not have protection rules, the workflow will not pause for approval. The separate attestation job depends on that approval job but does not itself declare an Environment; this preserves the tag-bound workflow/ref identity in the signing job's OIDC context. The attestation job runs on a fresh runner and does not check out the release source or execute packaged scripts. It downloads only the named build artifact, rejects missing/symlink/extra files, copies the archive to a fixed subject path, and invokes SHA-pinned `actions/attest` to create GitHub's signed SLSA provenance attestation. Its permissions are limited to the content lookup, OIDC signing, attestation persistence, and artifact metadata needed by GitHub's attestation service.

GitHub's official guidance recommends reusable trusted build workflows precisely because a trusted builder must not be influenced by arbitrary caller inputs; it also notes that a valid signature does not make every user-controlled predicate claim truthful. The verifier therefore separately pins the caller/source and reusable signer identities and compares sensitive SLSA predicate values to the operator policy. See [GitHub's reusable-workflow artifact-attestation guidance](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/increase-security-rating) and [`actions/attest` documentation](https://github.com/actions/attest).

To create a release attestation, a maintainer must first merge this PR, configure the protected `cyber-crucible-verifier-release` Environment and tag rules so only the approved release authority can create `cyber-crucible-verifier-v*` tags, and push an approved tag to the exact reviewed commit. This PR does not create or push a release tag. Until that workflow has actually completed and the artifact has been independently verified using an operator-provisioned policy, there is no production attestation result to claim.
## Status and non-claims

The implementation and regression suite are a contract tranche, not a qualification result. Test files use deterministic test-only key material. The public fixture is not a hidden oracle. Passing schema/unit tests does not establish cross-domain cybersecurity competence or authorization for live containment.
