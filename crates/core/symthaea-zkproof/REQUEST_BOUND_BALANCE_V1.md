# Request-Bound Balance Proof V1

**Status:** Incremental implementation candidate. Source-level changes are authored on a research branch; no build, test, guest execution or proof verification has been run in this environment.

**Tracking:** Symthaea #12269; Mycelix PPAC #6230.

## 1. Purpose

Add a distinct RISC Zero guest path that binds the balance-threshold computation to one exact request context.

The legacy tag-1 path remains unchanged. The new tag-2 path is deliberately versioned and commits a domain-separated request digest into the public journal.

This is a first integration step, not the completed issuer-bound private-claim theorem.

## 2. Request structure

`BalanceProofRequestV1` contains:

| Field | Width | Meaning |
|---|---:|---|
| `request_id` | 16 bytes | Unique request identity in the verifier's namespace |
| `source_statement_reference_digest` | 32 bytes | Reference to a source statement; not authenticated by this guest |
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

`SHA-256(domain || request_id || source_statement_reference_digest || issuer_key_reference_digest || audience_digest || verifier_digest || purpose_digest || policy_digest || verifier_challenge || required_minimum_be)`

Domain:

`symthaea-balance-proof-request-v1\0`

All context fields are fixed width and concatenated in the order above. The threshold is an unsigned 64-bit big-endian integer. Serde/JSON serialization bytes are not hashed.

A fixed known-answer vector is included in the shared core tests:

- request ID: 16 bytes of `0x01`
- source statement reference: 32 bytes of `0x02`
- issuer-key reference: 32 bytes of `0x03`
- audience: 32 bytes of `0x04`
- verifier: 32 bytes of `0x05`
- purpose: 32 bytes of `0x06`
- policy: 32 bytes of `0x07`
- challenge: 32 bytes of `0x08`
- threshold: 500

Expected digest:

`679d120fd3a036abb517e0d37e8aebfcdcabdeeda34247f89297c6f95bed96cb`

## 4. Guest and host behavior

Guest tag 2 receives:

- private `balance: u64`;
- public `BalanceProofRequestV1`.

The guest rejects structurally malformed request contexts, computes the threshold predicate, and commits only:

- `sufficient`;
- `required_minimum`;
- `request_digest`.

The host API:

- `BalanceProofRequestBoundProverV1::prove(balance, request)`
- `BalanceProofRequestBoundProverV1::verify_for_request(prove_info, expected_request)`

The verifier checks the receipt against the compiled `SYMTHAEA_ZKPROOF_GUEST_ID`, decodes the tag-2 output, recomputes the expected request digest and rejects a mismatch.

A receipt for a different request, source reference, issuer-key reference, audience, verifier, purpose, policy, challenge or threshold must not be accepted by `verify_for_request`.

A proof with `sufficient == false` is still a valid proof of the declared computation. The domain policy decides how insufficiency affects eligibility.

## 5. Compatibility

- Guest tag 0 remains the existing evolution computation.
- Guest tag 1 and the existing `BalanceProofInput` / `BalanceProofOutput` API remain unchanged.
- Guest tag 2 and `BalanceProofRequestBoundProverV1` are additive.
- The shared core uses SHA-256 0.10 with default features disabled so the same digest function is available to the no-std guest and host.

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

The source and issuer digests are **references**, not cryptographic source bindings. The guest does not verify an issuer signature or commitment opening. Never label this tag as a completed issuer-bound PPAC proof.

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
