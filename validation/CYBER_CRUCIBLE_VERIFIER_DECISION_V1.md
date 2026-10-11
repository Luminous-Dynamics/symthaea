# Cyber Crucible verifier-decision receipt v1

## Purpose and security boundary

This contract makes a verifier's decision independently consumable. It is deliberately separate from the evaluator's signed receipt. The decision receipt authenticates a decision under a specified signing policy; it does **not** prove that the verifier binary was uncompromised, that source evidence is true, that a held-out oracle was correctly implemented, that an incident diagnosis is correct, or that an action is safe.

The receipt is always \`non_authorizing: true\`. A consumer MUST NOT use the \`accept\` value alone as operational permission. It means only that the verifier signed an acceptance decision for the exact bound input and context.

## Canonicalization and signature

The detached Ed25519 signature covers:

\`\`\`text
UTF8("symthaea-cyber-crucible-verifier-decision.v1") || 0x00
    || RFC8785_JCS(payload)
\`\`\`

The envelope contains exactly \`payload\` and \`signature\`. The signature is unpadded base64url. Unknown fields, duplicate JSON members, unknown policy fields, noncanonical encodings, and invalid context are rejected.

## Bindings

The payload binds the exact campaign, subject, attempt, scenario ID/revision/digest, input-receipt digest, complete diagnostic-report digest, evaluator-policy digest, expected context digest, verifier-build digest, decision-signing-policy digest, signer identity, decision, reason classification, and validity interval.

\`context_digest\` is SHA-256 over RFC 8785 JCS of the complete externally supplied expected-context object, with exactly the fields defined in the verifier. That expected context MUST be independently supplied by a trusted consumer/operator; it MUST NOT be copied from the decision receipt itself. \`input_receipt_digest\` is the SHA-256 digest of the RFC 8785 JCS representation of the complete upstream receipt envelope, including its signature. The caller computing the expected context must compute these values independently.

The report digest is SHA-256 over RFC 8785 JCS of the complete report object. The integrated consumer recomputes it and checks report subject, scenario identity, and run ID against the trusted expected context; for this contract, `attempt_id` must equal `report.run.run_id`. The verifier build digest is also externally supplied. A signed self-assertion inside a receipt does not establish the executable that actually ran. Production provenance therefore belongs to a separate trusted release/attestation channel; this verifier checks only the digest binding.

## Trust policy and replay

The decision-signing trust policy must be obtained through an operator-controlled trust-root path, not from the subject under evaluation. Key IDs and public-key material must be unique; key validity windows, revocation state, and the verifier identity are checked. Receipts must be valid for no longer than one hour, must be currently valid, and are only accepted when a persistent replay ledger is supplied.

The SQLite ledger provides atomic single-host uniqueness for decision IDs and one-time challenge nonces. It is not distributed replay protection and it does not provide rollback protection against a host administrator who can restore the database. A distributed deployment needs a shared transactional uniqueness service with availability, consistency, and rollback properties appropriate to its threat model. The ledger directory and its parent hierarchy must be controlled by the verifier operator; checking the final path alone cannot secure an attacker-controlled parent directory against filesystem races.

## Decision semantics

A valid signed \`reject\` receipt is a successfully authenticated decision receipt, but it is not an acceptance. The CLI returns exit code zero when the receipt is authenticated, regardless of whether its decision is \`accept\` or \`reject\`; consumers must inspect the returned \`decision\`. Verification errors return exit code 2. Neither result grants action authority.

Reason codes are bounded, redacted classifications rather than free-form attacker-controlled diagnostics. For an \`accept\` decision, the reason code must be \`authenticated_under_policy\`; reject decisions cannot use that code.

## CLI

\`\`\`sh
python scripts/verify_cyber_crucible_report_decision.py \\
  --report diagnostic-loop-report.json \\
  --decision-receipt decision-receipt.json \\
  --trust-policy decision-trust-policy.json \\
  --expected-context trusted-expected-context.json \\
  --replay-db /var/lib/symthaea/cyber-crucible-decisions.sqlite3

The integrated consumer validates the existing diagnostic-loop contract, verifies the report digest and stable run/scenario identity against the independent context, then verifies the signed decision receipt. The lower-level verifier CLI remains available when only a decision receipt needs to be authenticated.
\`\`\`

Do not place signing private keys in the subject workspace or source repository. The signing service should construct receipts only after it has independently computed the decision and bound context. This repository contains a verifier, not a production decision-signing key or signing service.

## Status and non-claims

The implementation and regression suite are a contract tranche, not a qualification result. Test files use deterministic test-only key material. The public fixture is not a hidden oracle. Passing schema/unit tests does not establish cross-domain cybersecurity competence or authorization for live containment.
