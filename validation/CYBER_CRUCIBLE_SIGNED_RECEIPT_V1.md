# Cyber Crucible authenticated receipt contract v1

This is a narrow upstream trust-boundary component for the diagnostic-loop contract. It verifies detached Ed25519 signatures over a domain-separated RFC 8785 canonical JSON payload, resolves evaluator identity and roles through an operator-controlled trust policy, binds each receipt to a trusted expected campaign/subject/attempt/scenario context, checks validity windows and key revocation, and atomically consumes receipt IDs and receipt nonces in a persistent replay ledger.

It is **not** a production authorization service, incident-response runner, or proof that the evidence is truthful. A verified evaluator signature proves that the corresponding private key signed this specific payload; whether the key belongs to an approved evaluator depends on the trust policy and how that policy was provisioned.

## Run the checks

Install the pinned dependencies used by the path-scoped workflow:

    python3 -m pip install "cryptography==50.0.2" "rfc8785==0.1.4"
    python3 -m unittest discover -s tests -p "test_cyber_crucible_signed_receipt.py" -v

The test uses the Ed25519 known-answer vector from RFC 8032, an RFC 8785 number-serialization vector, and a public, test-only key fixture for end-to-end mutation tests. The corresponding private seed is intentionally a public test vector and MUST NEVER be used for operational signing.

The signed-receipt cases also run through the existing diagnostic-loop workflow when this stacked PR is part of its merge candidate. The dedicated workflow in this branch is retained as an independently scoped entry point, but new-workflow-only path filters are not treated as sufficient evidence: the parent workflow owns activation for the stacked diff.

## Receipt envelope and signature input

The envelope contains exactly:

- \`payload\`: a versioned object whose complete field set is defined by \`cyber_crucible_signed_receipt_v1.schema.json\`.
- \`signature\`: unpadded base64url encoding of the 64-byte Ed25519 signature.

The signed bytes are:

    ASCII("symthaea-cyber-crucible-signed-receipt.v1") || 0x00 || JCS(payload)

where JCS is the JSON Canonicalization Scheme specified by [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html). The detached signature is not included in the signed payload. The algorithm is fixed to Ed25519; the receipt cannot negotiate a weaker algorithm. Duplicate JSON member names are rejected before objects are used.

The required signed bindings include the receipt ID, key and evaluator IDs, evaluator role, campaign ID, exact subject ID and attempt ID, a 32-byte random one-time per-receipt challenge nonce, strict UTC-second issuance/expiry timestamps, exact scenario ID/revision/digest, and evidence digest. The verifier requires the caller to supply the expected subject and attempt independently, preventing a receipt from being relabeled to a different candidate or run. Authorization-decision receipts additionally require the exact proposed action ID and immutable proposal digest. The expected context is supplied independently by the caller; it must never be reconstructed from the untrusted receipt being verified.

## Trust policy

The trust policy must be provisioned by an operator-controlled mechanism and supplied separately from the subject. Its v1 shape is:

    {
      "schema_version": "cyber-crucible-trust-policy.v1",
      "policy_id": "operator-policy-version",
      "keys": [{
        "key_id": "evaluator-security-2026-q4",
        "evaluator_id": "approved-independent-evaluator",
        "public_key": "BASE64URL_RAW_32_BYTE_ED25519_PUBLIC_KEY",
        "roles": ["security_verdict"],
        "not_before": "2026-10-01T00:00:00Z",
        "not_after": "2027-01-01T00:00:00Z",
        "revoked_at": null
      }]
    }

Every key has an explicit evaluator identity, a finite validity interval, and a closed role allowlist. Key IDs and underlying public-key bytes must be unique within one policy; the same key cannot be aliased under another evaluator identity. An evaluator identity may not hold both functional and security verdict roles, and may not hold an evaluation role and `authorization_decision`. These checks prevent the trust configuration from nominally defining independent evaluators while allowing one principal/key to satisfy both sides. Unknown identities/roles, expired receipts, future issuance times, invalid timestamps, revoked keys, context mismatches, non-canonical base64url, and invalid signatures fail closed. Revocation is deliberately conservative: once \`revoked_at\` has been reached, the key is rejected even for otherwise valid historical receipts. A separate archival verifier would be needed to preserve historical signature validity under a formally specified revocation policy.

## Replay and operational boundary

The verifier uses SQLite uniqueness constraints inside a \`BEGIN IMMEDIATE\` transaction to consume both receipt IDs and receipt nonces atomically on one host. The ledger file is created with restrictive permissions and chmodded to owner-only. The path and parent directory must remain protected from untrusted writers. For distributed verifiers, replace this local ledger with a shared, durable store offering equivalent linearizable uniqueness semantics; a local SQLite file is not distributed replay protection.

The caller must issue an unpredictable, unique 32-byte challenge nonce for every receipt and provide independently trusted values for campaign, subject, attempt, scenario identity, evidence digest, and role. Passing \`--expected-context\` values from the untrusted receipt itself defeats context binding. The replay database prevents repeated receipt-ID/receipt-nonce consumption only within that database's protection and lifecycle. Every independently issued receipt—including functional, security, authorization, outcome, or recovery receipts—must get its own fresh challenge nonce; subject_id and attempt_id provide the stable linkage across receipts for the same attempt.

CLI example (paths are operator-controlled where noted):

    python3 scripts/verify_cyber_crucible_signed_receipt.py \
      --receipt /protected/inbox/receipt.json \
      --trust-policy /protected/policy/trust-policy.json \
      --expected-context /protected/run/expected-context.json \
      --replay-db /protected/state/receipt-replay.sqlite3

The JSON response includes \`signature_verified: true\`, policy identity and digest, receipt bindings, \`replay_checked: true\`, and \`non_authorizing: true\`. The library API rejects receipt acceptance when no persistent replay ledger is supplied. The CLI always requires an explicit ledger path. The output is a local verifier result, not itself a signed verifier receipt, and must not be consumed as operational authorization. An upstream verifier service should sign and audit its own redacted verification decision.

## Security and evidence limits

- SHA-256 fields bind the receipt to named evidence and scenario content only if the producer's digest calculation and source content are independently checked. A signature does not establish the truth or provenance of the underlying evidence by itself.
- The supplied trust policy is a trust anchor. Its provenance, version history, access controls, rollback protection, key rotation, and emergency revocation procedures are outside this function and must be enforced by the deployment.
- The SQLite replay ledger is stateful and host-local. Restoring an old database snapshot can re-enable replay; production deployments need protected durable state and snapshot/rollback controls.
- Independent evaluator identities must not be controlled by the subject under evaluation, and a single key should not be authorized for roles that policy separation requires to remain independent.
- A signed \`pass\` is only the evaluator's signed assertion. It does not itself prove that a live action is safe, authorize containment, establish root cause, or qualify a simulated fixture as hardware or production evidence.
- This contract does not implement a held-out oracle, trusted timestamping, hardware-backed key custody, or a distributed replay service. Those remain explicit future requirements.

## Research basis

- [RFC 8785 — JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785.html) defines deterministic JSON representation for cryptographic operations and advises input validation before signature verification.
- [RFC 8032 — Edwards-Curve Digital Signature Algorithm (EdDSA)](https://www.rfc-editor.org/rfc/rfc8032.html) specifies Ed25519 and provides known-answer test vectors used by this suite.
- The existing diagnostic-loop contract remains the source for lifecycle, evidence-reference, claim-maturity, and non-authorizing process checks. This receipt verifier adds authenticity and trust-policy checks without duplicating that process validator.
