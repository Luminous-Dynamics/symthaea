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

The payload binds the exact campaign, subject, attempt, scenario ID/revision/digest, input-receipt digest, complete diagnostic-report digest, evaluator-policy digest, expected context digest, verifier-build digest, decision-signing-policy digest, signer identity, decision, reason classification, and validity interval.

`input_receipt_digest` is SHA-256 over RFC 8785 JCS of the complete upstream receipt envelope, including its detached signature. The integrated consumer requires that exact envelope as an input, recomputes the digest, and checks campaign, subject, attempt, and scenario fields against trusted expected context. This proves which input artifact the decision names; the signed verifier decision attests that the verifier accepted or rejected it under the stated policy.

`report_digest` is SHA-256 over RFC 8785 JCS of the complete diagnostic report object. The integrated consumer recomputes it and checks report subject, scenario identity, and run ID against the trusted expected context. For this contract, `attempt_id` must equal `report.run.run_id`.

`context_digest` is SHA-256 over RFC 8785 JCS of the complete externally supplied expected-context object with exactly the fields defined in the verifier. That expected context MUST be independently supplied by a trusted consumer/operator; it MUST NOT be copied from the decision receipt itself.

The verifier build digest is also externally supplied. A signed self-assertion inside a receipt does not establish the executable that actually ran. Production provenance therefore belongs to a separate trusted release/attestation channel; this verifier checks only the digest binding.

## Trust policy and replay

The decision-signing trust policy must be obtained through an operator-controlled trust-root path, not from the subject under evaluation. Key IDs and public-key material must be unique; key validity windows, revocation state, and the verifier identity are checked. Receipts must be valid for no longer than one hour, must be currently valid, and are only accepted when a persistent replay ledger is supplied.

The SQLite ledger provides atomic single-host uniqueness for decision IDs and one-time challenge nonces. It is not distributed replay protection and it does not provide rollback protection against a host administrator who can restore the database. A distributed deployment needs a shared transactional uniqueness service with availability, consistency, and rollback properties appropriate to its threat model. The ledger directory and its parent hierarchy must be controlled by the verifier operator; checking the final path alone cannot secure an attacker-controlled parent directory against filesystem races.

## Decision semantics

A valid signed `reject` receipt is a successfully authenticated decision receipt, but it is not an acceptance. The CLI returns exit code zero when the receipt is authenticated, regardless of whether its decision is `accept` or `reject`; consumers must inspect the returned `decision`. Verification errors return exit code 2. Neither result grants action authority.

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
  --replay-db /var/lib/symthaea/cyber-crucible-decisions.sqlite3
```

The consumer validates the existing diagnostic-loop contract, checks the exact input and report digests plus stable run/scenario identity against expected context, then authenticates the signed decision receipt. The lower-level `verify_cyber_crucible_decision_receipt.py` CLI remains available when only a decision receipt needs to be authenticated.

Do not place signing private keys in the subject workspace or source repository. The signing service should construct receipts only after it has independently computed the decision and bound context. This repository contains a verifier, not a production decision-signing key or signing service.

## Status and non-claims

The implementation and regression suite are a contract tranche, not a qualification result. Test files use deterministic test-only key material. The public fixture is not a hidden oracle. Passing schema/unit tests does not establish cross-domain cybersecurity competence or authorization for live containment.
