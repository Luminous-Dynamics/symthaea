# Claim Plan IR v1 — research specification

**Status:** research specification; not a production format  
**Owner:** Symthaea — SYM-CLAIM-IR-001 / #12393  
**Integration:** Mycelix PPAC #6230 / WEB-CAPTURE-TLS #6245  
**Date:** 2026-11-15

## 1. Purpose

Claim Plan IR (CPIR) is a verifier-authorized, deterministic intermediate representation for a single private claim. It prevents separately correct components from silently composing different sources, fields, units, predicates, audiences, policies, or trust profiles.

The relying verifier or domain policy owner supplies the plan. The claimant may provide witness material and choose among explicitly permitted source sessions, but may not widen the requested field, parser, policy, audience, proof program, uniqueness scope, or trust profile.

CPIR is a compilation contract, not a credential, proof system, trust root, consent record, authorization, or execution engine.

## 2. The compositional failure it addresses

The source-to-decision pipeline contains several distinct relations:

```text
request authority
  -> source/session profile
  -> authenticated transcript
  -> selected commitment opening
  -> HTTP message and body framing
  -> parser + schema + exact field path
  -> typed interpretation (units / time / currency)
  -> private predicate
  -> proof statement + expected program identity
  -> credential/currentness qualification
  -> domain authority / consent / action boundary
  -> independent effect observation
```

A proof may be valid over the wrong field. A parser may be deterministic over bytes that were never authenticated. A normalized amount may be correct arithmetic over a caller-supplied number. A valid source proof may be presented to the wrong audience or under a different policy. A signed state root may not prove a correct state transition.

CPIR assigns one canonical plan identity to the proof-relevant prefix and emits explicit dependency edges for the qualification lifecycle. Each adapter must consume that identity and its typed outputs; it must not reconstruct policy from duplicated strings.

## 3. Non-goals and ownership boundaries

CPIR MUST NOT:

- implement TLS, TLSNotary, DECO, a notary, a TEE, a JSON parser, a credential issuer, a ZKP backend, a status-list service, a nullifier ledger, or an action executor;
- replace Mycelix PEC/PPAC semantics, LISE dependency invalidation, Xenia consent/session capabilities, or domain-owned authorization;
- infer provider truth, account ownership, legal-person uniqueness, human consent, institutional authority, or external effect from source/proof validity;
- treat host-side parser output, a source URL, a provider name, a backend ID, a caller-supplied boolean, or a digest string as cryptographic evidence;
- collapse structural validity, source authenticity, proof validity, currentness, authority, consent, and effect observation into one `verified` flag;
- create a global stable subject pseudonym or cross-application nullifier.

Ownership remains split:
- Symthaea #12301: transcript commitment opening and same-witness predicate relation.
- Symthaea #12328 / PR #12336: exact HTTP/JSON field semantics and bounded structural selector.
- Symthaea #12373: TLSNotary typed verified-commitment API boundary.
- Symthaea #12269: issuer/source and verifier-request binding.
- Mycelix #6230 / #6245 / #6251 / #6219: PPAC composition, source profile, domain-scoped uniqueness, and LISE requalification.
- Xenia: session authentication and exact human consent/action capabilities.
- The domain authority owner: whether a qualified claim may authorize a particular effect.

## 4. IR layers

### 4.1 Authority and request envelope

The plan fixes:
- opaque plan identifier and version;
- policy owner/issuer reference and policy digest;
- verifier identity, audience, purpose and intended claim;
- verifier-issued challenge reference, nonce commitment and expiry;
- request digest and the allowed public inputs.

The plan is created before source acquisition. A claimant cannot self-issue a new audience, freshness window, or challenge and have it treated as equivalent.

### 4.2 Source profile

The source profile fixes:
- exact protocol family and mode: direct MPC-TLS, delegated notary, or proxy;
- TLSNotary release/tag and full source commit when TLSNotary is used;
- crate/feature/toolchain closure and the relevant commitment strategy;
- server identity policy and trust-root/key epoch;
- HTTP request method, origin, path/query policy and request binding;
- received response ordinal and direction;
- allowed HTTP framing/content-encoding profile.

These modes are not substitutable. In particular, proxy mode must not inherit the identity privacy or network-path assumptions of MPC-TLS. A notary-signed portable artifact inherits trust in the notary's correct in-session participation. A direct verifier result is designated-verifier evidence unless a separate portable contract is qualified.

TLSNotary's authenticated transcript ranges establish bytes and direction/range authenticity under the selected protocol profile. They do not by themselves establish JSON field meaning or provider truth.

### 4.3 Commitment opening

The opening contract identifies:
- authenticated session/header commitment root;
- commitment profile and hash/domain version;
- exact direction, response ordinal, leaf/range selector and byte interval;
- the typed opening or upstream-selected-field certificate expected from the qualified adapter;
- the trusted verification key/profile and release identity.

Do not reproduce private TLSNotary serialization, internal field IDs, or Merkle domains in a downstream adapter. A typed upstream API is preferred. Full Presentation verification and a minimal selected-field certificate are separate implementation alternatives whose assurance and proof cost must be measured.

### 4.4 Parser and field semantics

The field contract fixes:
- HTTP parser profile and version/digest;
- response framing restrictions and body offset mapping;
- media type and content-encoding policy;
- JSON parser/schema profile and digest;
- exact RFC 6901 JSON Pointer;
- duplicate-key behavior (v1 requires rejection);
- Unicode comparison/normalization policy (v1 defaults to exact scalar sequence; no implicit normalization);
- lexical rules for the selected scalar and resource limits.

A body-relative JSON span is not a TLS transcript offset. The mapping must be derived through the frozen HTTP framing profile and then authenticated by the selected TLSNotary opening. A parser result cannot authenticate itself.

A hidden JSON string requires a proof of the relevant lexical grammar as well as its structural path. The first field-selector slice's single-hidden-string construction is not a generic proof for arbitrary JSON scalar types, HTTP/2, chunked bodies, compressed bodies, or multiple hidden fields.

### 4.5 Typed interpretation and predicate

The plan fixes the conversion profile from raw field bytes to a typed semantic value. Examples:
- money: currency, minor-unit scale, decimal grammar, sign and rounding policy;
- time: timezone, calendar/epoch representation and freshness boundary;
- quantities: dimension, unit and conversion version;
- status: closed enum and source-specific mapping.

Avoid floating-point money and unspecified locale parsing. Conversion MUST be deterministic and versioned. Any normalized value must be derived from the exact authenticated field under the declared conversion profile.

The predicate AST is typed and versioned. Public constants and private witness inputs are explicit. The proof relation consumes the exact value derived by the field contract; it cannot accept a parallel caller-provided scalar.

### 4.6 Privacy-preserving uniqueness

The plan identifies the domain owner's uniqueness subject and scope:
- credential;
- provider account;
- entitlement;
- application participant;
- legal/natural person (only with separately qualified person-binding evidence).

A commitment salt is not a uniqueness subject. Freshly reissued commitments must not reset a same-scope entitlement when the domain's qualified subject binding is stable, but different applications must not receive a globally linkable subject handle. Recovery and issuer-key rotation need explicit continuity policy.

The IR specifies a nullifier derivation profile reference and scope, not a new nullifier engine. Mycelix's existing owner and consume-once boundary remains authoritative.

### 4.7 Proof and qualification identity

The proof contract fixes:
- backend and proof-format profile;
- expected guest ImageID/program digest;
- verifier implementation/build identity;
- exact public statement encoding/version;
- source, parser, schema and predicate profile digests.

The verifier must check the expected program identity from trusted configuration. A caller-selected ImageID is not a trust anchor. An envelope's syntactic validity or a non-empty proof byte string is not proof validity.

The output is a proof/qualification receipt candidate bound to the plan digest. It is not consent or action authority.

### 4.8 LISE dependency manifest

The compiler emits owner-qualified dependencies for at least:
- source TLS release/profile and server identity trust root;
- notary/verifier key epoch and revocation status;
- parser, framing, schema, field path and conversion profile;
- predicate/policy digest;
- proof backend, guest ImageID and verifier build;
- credential issuer and status evidence;
- evidence frontier/freshness policy.

Dependencies must carry owner identity and change semantics. LISE determines which claim receipts require requalification; CPIR does not create a global `current=true` service. Historical validity at an earlier evidence cutoff must remain reconstructable when current qualification changes.

## 5. Canonical encoding and digest

Version 1 proposes RFC 8785 JSON Canonicalization Scheme (JCS) for the plan envelope, with these additional admission constraints:

1. Reject duplicate JSON member names before canonicalization.
2. Reject unknown critical members and unsupported enum values.
3. Disallow floating-point values in policy or claim semantics; encode exact integers as bounded integers and decimals as grammar-constrained strings or integer minor units.
4. Do not Unicode-normalize identifiers implicitly. Any normalization must be an explicit field-level profile.
5. Require lowercase hexadecimal for fixed-width digests and validate exact digest length for the declared algorithm.
6. Keep the plan digest outside the canonicalized object to avoid self-reference.

Proposed digest:

```text
plan_digest = SHA-256(
  "LUMINOUS/CLAIM-PLAN-IR/V1\0" || JCS(plan)
)
```

The final algorithm and canonicalization profile must be reviewed before implementation. Domain separation and deterministic encoding are mandatory; the example above is a candidate, not a cryptographic approval.

## 6. Typed adapter pipeline

The API design should make invalid transitions difficult to express:

```text
VerifierApprovedPlan
  -> VerifiedSourceSession
  -> AuthenticatedTranscript
  -> OpenedTranscriptRange
  -> ParsedClaimField
  -> TypedClaimValue
  -> PredicateStatement
  -> ProofReceipt
  -> MycelixQualificationInput
```

Use opaque constructors and keep raw deserialization separate from trusted states. A value parsed from untrusted bytes must begin as an untrusted candidate. Only a verifier/guest that checks the required cryptographic relation can promote it to an authenticated type. Rust phantom types can prevent accidental mixing in one process, but cannot substitute for cryptographic verification across process or serialization boundaries.

Adapters must be monotonic in evidence: each stage may add a precisely named proposition; no stage may silently promote a different proposition.

## 7. Compiler validation

A deterministic compiler has two outputs:
1. a canonical public plan and plan digest;
2. a compiled obligation/dependency manifest naming the exact verification steps and owner for each.

Compilation fails closed on missing trust roots, unsupported source mode, unresolved parser/schema, ambiguous field encoding, unspecified units, floating-point policy values, stale/revoked key snapshot, missing proof program identity, unsupported uniqueness scope, or an incomplete dependency owner.

Compilation is not execution. A compiled plan does not prove that any TLS session occurred, that an opening verifies, that a credential is current, or that an action is authorized.

## 8. Conformance and adversarial tests

The first frozen fixture set should contain:
- valid positive end-to-end plan;
- wrong source profile, origin, method, path, query, response ordinal or direction;
- transcript A with private witness B;
- valid opening for the wrong field/sibling/path;
- duplicate and escaped JSON keys, Unicode edge cases and JSON Pointer escaping;
- HTTP framing, content-length and unsupported transfer/content-encoding substitutions;
- parser/schema/conversion digest substitution;
- money scale/currency/sign/rounding and time-zone/freshness boundary errors;
- stale/revoked notary key, release, trust-root or TCB snapshot;
- request/audience/purpose/challenge/policy/ImageID substitution;
- replay, expiry and concurrent same-scope duplicate claim;
- fresh-salt re-issuance under the same qualified subject;
- nullifier scope substitution and cross-context linkability;
- missing LISE dependency edge or incomplete change feed;
- proof-valid but issuer-unauthorized, consent-denied or institutionally unauthorized result;
- accepted request but failed, partial, unknown or unobserved external effect.

Every suite needs positive, negative and indeterminate controls. Reject-all behavior is not a pass. Each fixture records its expected disposition, oracle provenance and adjudication status; unadjudicated cases remain provisional.

## 9. Benchmark plan and prospective gates

Benchmark plan compilation separately from TLS, proving and institutional decision latency.

| Dimension | Measurement | Initial target |
|---|---|---|
| Canonicalization | byte-identical output across independent implementations | 100% |
| Mutation resistance | wrong-plan/source/path/schema/predicate/request/program bindings accepted | 0 |
| Conformance | positive controls accepted; negative controls rejected; indeterminate controls remain indeterminate | 100% of adjudicated fixtures |
| Determinism | same plan/version yields same digest and dependency manifest | 100% |
| Privacy | extra fields disclosed or new cross-scope stable identifiers emitted | 0 |
| Requalification | affected receipts selected after a relevant dependency change | 100% of frozen dependency-change controls |
| Performance | plan compile/validate p50/p95/p99, CPU, memory, bytes, manifest size | baseline first; optimize after matched measurement |

For an eventual matched system benchmark, report source/session time, notary/TEE work, opening verification, parser semantics, proof generation/verification, nullifier consumption, LISE qualification and time-to-qualified-private-decision separately. Keep assurance properties as a vector/Pareto frontier. Never trade a weaker witness model or more disclosure for a faster headline number without showing that trade-off.

## 10. Implementation sequence

1. Review this schema and threat model against #12301, #12328, #12373 and Mycelix #6230.
2. Add independent canonicalization and schema conformance fixtures.
3. Implement the compiler as a standalone Rust research workspace, outside Symthaea's default Cargo workspace.
4. Add typed adapters only when the upstream TLSNotary commitment API and exact opening contract are available.
5. Prove the same authenticated value is parsed, typed and consumed by the predicate.
6. Emit the LISE dependency manifest and verify targeted invalidation.
7. Integrate Xenia consent and domain action only after Mycelix independently qualifies the claim.

## 11. Current external baseline notes

- TLSNotary v0.1.0-beta.1 was announced on 2026-11-14 and is the current integration candidate in the related research issues; it is a prerelease, not a production qualification.
- TLSNotary's commit strategies include custom ranges, HTTP objects and per-byte commitments. HTTP-object commitment is the documented default trade-off; each additional range costs storage linearly (the current documentation estimates about 250 bytes of additional Secret artefact per BLAKE3 range).
- Scribe (published 2026-09-02) combines TLS provenance, an attestation-node signature on a salted commitment, a transparent STARK policy proof, application-scoped spend tags, a Merkle spent-tag aggregator and Falcon-signed root publication. It explicitly assumes honest attestation-node provenance behavior; fresh salts can create distinct commitments; a signed root does not independently constrain a correct append-only transition. These are separate benchmark dimensions, not reasons to discount the system's stated theorem.

## 12. Claim ceiling

A future qualified implementation may establish that a verifier-authorized, versioned predicate was evaluated over a value opened from a specific TLS-authenticated transcript range, under a specified source mode, parser/schema/conversion profile, request/audience/purpose and proof-program identity, with explicit qualification dependencies.

It still does not establish that the provider's statement is true, that the claimant legally owns the account, that the subject is unique as a natural person, that an issuer is authorized for every use, that the credential remains current without a fresh LISE decision, that the user understood or freely gave consent, that an institution authorized a consequential action, or that the external effect occurred.
