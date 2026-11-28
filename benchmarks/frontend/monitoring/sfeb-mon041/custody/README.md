# MON-041 Custody Verifier v0.1

A dependency-free Go prototype that verifies Ed25519-signed, claim-bound evidence receipts. This is an offline semantic/custody preflight, not a production trust service.

## Checks

- Domain-separated Ed25519 signatures over deterministic receipt JSON.
- Claim, trial, recovery-boundary, evidence-class and expected-source binding.
- 256-bit challenge nonce format, issuance, uniqueness and reuse against a supplied consumed-nonce snapshot.
- Capture freshness, expiry and bounded future clock skew.
- Signer-to-capture-principal binding, key validity window and revocation.
- Required evidence coverage, duplicate classes, session sequence and receipt hash-chain linkage.
- Strict CLI JSON decoding; unknown fields and trailing documents are rejected.

## Safety boundary

`custody_verdict=qualified` means receipts are internally authenticated against the trust-key and nonce snapshots supplied to the evaluator. It does not authenticate those snapshots, prove artifact availability/content beyond the supplied digest, establish physical independence, prove capture completeness, or establish evidence truth.

Trust-store bootstrap and durable atomic nonce consumption are external and unimplemented. Therefore `campaign_verdict` is always `indeterminate` and `production_action_permitted` is always `false`. The verifier never connects to Kafka/PostgreSQL or performs mutations. The deterministic fixture key is public test material and MUST NOT be reused operationally.

## CLI

`go run ./cmd/qualify --input INPUT.json`

Exit codes: 0 custody preflight qualified (campaign still indeterminate); 2 blocked; 3 indeterminate; 64 usage; 65 invalid input; 66 input unavailable; 74 output failure.

Run `bash CHECKS.sh` for formatting, vet, unit, race, CLI and benchmark checks.
