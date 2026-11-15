# Request-Bound Balance Proof V1

**Status:** Incremental implementation candidate. Source-level changes are authored on a research branch; no build, test, guest execution or proof verification has been run in this environment.

**Tracking:** Symthaea #12269; Mycelix PPAC #6230.

## 1. Purpose

Add a distinct RISC Zero guest package/image that binds the balance-threshold computation to one exact request context.

The existing guest package is preserved byte-for-byte. The new request-bound guest is a separate package with its own generated ELF and ImageID, so existing tag-0/tag-1 receipts continue to verify against the original guest identity when the old verifier binary is retained. The new guest commits a domain-separated request digest into its public journal.

This is a first integration step, not the completed issuer-bound private-claim theorem.

## 2. Request structure

`BalanceProofRequestV1` contains:

| Field | Width | Meaning |
|---|---:|---|
| `request_id` | 16 bytes | Unique request identity in the verifier's namespace |
| `source_statement_reference_digest` | 32 bytes | Reference to a source statement; not authenticated by this guest |
| `claim_schema_digest` | 32 bytes | Claim schema, field and unit semantics |
| `predicate_digest` | 32 bytes | Exact predicate semantics, including comparison boundary and rounding |
| `issuer_key_reference_digest` | 32 bytes | Reference to an issuer/key; not authenticated or authorized by this guest |
| `audience_digest` | 32 bytes | Pairwise/purpose-scoped verifier audience |
| `verifier_digest` | 32 bytes | Verifier/request authority context |
| `purpose_digest` | 32 bytes | Declared purpose |
| `policy_digest` | 32 bytes | Policy/profile reference |
| `verifier_challenge` | 32 bytes | Challenge supplied by the verifier-side request owner |
| `required_minimum` | 8 bytes in digest | Public threshold, encoded big-endian |

Structural validation rejects all-zero identifiers/digests/challenges. A nonzero challenge is not proof that the verifier issued it, nor that it has not already been consumed.

## 3. Canonical request digest

The digest is:

`SHA-256(domain || request_id || source_statement_reference_digest || claim_schema_digest || predicate_digest || issuer_key_reference_digest || audience_digest || verifier_digest || purpose_digest || policy_digest || verifier_challenge || required_minimum_be)`

Domain:

`symthaea-balance-proof-request-v1\0`

All context fields are fixed width and concatenated in the order above. The threshold is an unsigned 64-bit big-endian integer. Serde/JSON serialization bytes are not hashed.

A fixed known-answer vector is included in the isolated request-bound core tests:

- request ID: 16 bytes of `0x01`
- source statement reference: 32 bytes of `0x02`
- claim schema: 32 bytes of `0x04`
- predicate: 32 bytes of `0x05`
- issuer-key reference: 32 bytes of `0x03`
- audience: 32 bytes of `0x06`
- verifier: 32 bytes of `0x07`
- purpose: 32 bytes of `0x08`
- policy: 32 bytes of `0x09`
- challenge: 32 bytes of `0x0a`
- threshold: 500

Expected digest:

`f173037a1126d495e2b228fa79d6132cee083d97613498322b5a7d2f3fda538d`

## 4. Guest and host behavior

The separate request-bound guest receives:

- private `balance: u64`;
- public `BalanceProofRequestV1`.

The new guest rejects structurally malformed request contexts, computes the threshold predicate, and commits only:

- `sufficient`;
- `required_minimum`;
- `request_digest`.

The host API:

- `BalanceProofRequestBoundProverV1::prove(balance, request)`
- `BalanceProofRequestBoundProverV1::verify_for_request(prove_info, expected_request)`

The verifier checks the receipt against the dedicated `SYMTHAEA_ZKPROOF_REQUEST_BOUND_GUEST_ID`, decodes the request-bound output, recomputes the expected request digest and rejects a mismatch.

A receipt for a different request, source reference, issuer-key reference, audience, verifier, purpose, policy, challenge or threshold must not be accepted by `verify_for_request`.

A proof with `sufficient == false` is still a valid proof of the declared computation. The domain policy decides how insufficiency affects eligibility.

## 5. Compatibility

- The existing `guest` package remains byte-for-byte unchanged; its generated ImageID remains the authority for legacy tag-0/tag-1 receipts.
- `guest-request-bound` is an additional package listed in the `risc0-build` methods manifest and receives its own ImageID.
- `BalanceProofRequestBoundProverV1` proves/verifies only against the dedicated request-bound guest ELF/ImageID.
- Legacy APIs continue to use the original `SYMTHAEA_ZKPROOF_GUEST_ELF` and `SYMTHAEA_ZKPROOF_GUEST_ID` constants.
- The new `symthaea-zkproof-request-bound-core` crate owns the request types and SHA-256 0.10 digest implementation with default features disabled; both request-bound guest and host depend on it.
- The existing `symthaea-zkproof-core` crate and its dependency graph remain unchanged for legacy guest use.

## 5a. Image identity and receipt compatibility

A RISC Zero ImageID identifies the exact guest program. Adding code to an existing guest changes its ELF and therefore its ImageID, even when old dispatch branches are logically unchanged.

This implementation therefore keeps the original guest package untouched and adds a second guest package. The build manifest lists both packages, and the generated host constants are intentionally separate:

- legacy: `SYMTHAEA_ZKPROOF_GUEST_ELF` / `SYMTHAEA_ZKPROOF_GUEST_ID`;
- request-bound: `SYMTHAEA_ZKPROOF_REQUEST_BOUND_GUEST_ELF` / `SYMTHAEA_ZKPROOF_REQUEST_BOUND_GUEST_ID`.

The expected generated symbols are derived from the `symthaea-zkproof-request-bound-guest` binary target name. They still require confirmation by the methods build. Existing historical receipts still require the historical ELF/ImageID and verifier policy that produced them; this branch does not claim that old receipt bytes can be verified by the new request-bound ImageID.

## 6. Security and privacy claim ceiling

This tranche establishes only a candidate relation of the form:

`public_journal = (balance >= threshold, threshold, digest(exact_request_context))`

under the exact RISC Zero guest image, after the corresponding receipt verifier succeeds.

It does **not** establish:

1. that the source statement was issued or signed by the referenced issuer;
2. that the referenced issuer/key is authorized for the claim;
3. that the private balance opens or equals a value in the referenced source statement;
4. that the verifier challenge was issued by the named verifier or consumed exactly once;
5. that credential status, source provenance or policy remains current;
6. that the audience/purpose is institutionally applicable;
7. that the participant consented or that an action is authorized;
8. that the public request digest is unlinkable if callers reuse request fields across contexts;
9. that the RISC Zero proof is a substitute for independent program-image and verifier qualification.

The source and issuer digests are **references**, not cryptographic source bindings. The guest does not verify an issuer signature or commitment opening. Never label this guest as a completed issuer-bound PPAC proof.

## 7. Required next tranche

The next source-bound step must use a reviewed existing signature and commitment construction, not an ad hoc scheme. It must prove that:

1. the issuer-authenticated canonical statement contains the exact value commitment;
2. the private balance and hiding commitment opening satisfy the same statement;
3. the issuer key selected by the proof is bound to an independently qualified issuer profile;
4. the public journal binds the statement digest and request digest;
5. the verifier checks the expected guest image and its own fresh, single-use request challenge;
6. no stable global subject or low-entropy deterministic value hash is exposed.

The issuer's institutional legitimacy, key authority, credential revocation, policy applicability and LISE dependency closure remain external qualification requirements even when signature verification occurs inside the guest.

## 8. Authored tests

Shared-core tests cover:
- deterministic canonical digest;
- fixed known-answer digest;
- mutation of every digest/request field changes the digest;
- all-zero required fields reject;
- output matches only the exact expected request.

Host tests cover:
- correct request receipt verifies;
- request-context substitutions reject;
- threshold substitution rejects;
- insufficient balance is represented as a valid negative predicate result.

These tests are authored, not executed. Exact-head cargo, guest build, native host proof and RISC Zero receipt verification remain pending.
