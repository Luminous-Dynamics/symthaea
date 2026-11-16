# Claim Plan IR v1 — research specification

**Status:** research specification; not a production format  
**Owner:** Symthaea — SYM-CLAIM-IR-001 / #12393  
**Integration:** Mycelix PPAC #6230 / WEB-CAPTURE-TLS #6245  
**Date:** 2026-11-15

## 1. Purpose

Claim Plan IR (CPIR) is a verifier-authorized, deterministic intermediate representation for a single private claim that may depend on multiple typed fields from one authenticated HTTP response. It prevents separately correct components from silently composing different sources, fields, units, predicates, audiences, policies, or trust profiles.

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
- Mycelix #5401 / #5404 / #5405 / #5411 / #5466 / #5468: cross-session linkability, cumulative disclosure inference, proof-shape correlation, and federated disclosure-budget conservation. CPIR references these owners rather than implementing a duplicate privacy ledger.
- Mycelix #5489: compile and optimize the full multi-round presentation transcript, rather than evaluating each presentation independently.
- Mycelix #5477: distinguish record-, person-, group- and relational-graph privacy units; per-person controls must not be misrepresented as household/cooperative protection.
- Mycelix #5447: standards-pinned SD-JWT / BBS / OpenID4VP credential interoperability and independent conformance; CPIR binds the resulting credential/presentation profile as a dependency rather than reimplementing those formats or claiming their conformance.
- Xenia: session authentication and exact human consent/action capabilities.
- The domain authority owner: whether a qualified claim may authorize a particular effect.

## 4. IR layers

### 4.1 Authority and request envelope

The plan fixes:
- opaque plan identifier and version;
- policy owner/issuer reference and policy digest;
- verifier identity, audience, purpose and intended claim;
- verifier-issued challenge reference, nonce commitment, request issue time (`issued_at`) and expiry (`expires_at`); v1 encodes both as fixed-width UTC seconds and requires `expires_at > issued_at`;
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

### 4.2.1 Session witness, transferability and plaintext boundary

A zkTLS result is not automatically a universally portable proof. The source session requires an in-session witness; later parties verify either a designated-verifier result, a delegated witness's signed attestation, a public proof layered over that attestation, or a hardware-rooted result. The trust profile must distinguish these propositions.

The required top-level `trust` block binds:
- a digest-addressed trust profile;
- witness kind and identity (direct verifier, proxy verifier, delegated notary, protocol attestor or TEE attestor);
- evidence artifact kinds actually present (live-session result, signed transcript attestation, signed value commitment, transparent ZK proof, TEE quote, state-root signature or membership/consistency proof);
- post-session verification scope (same session participant, delegated attestation verifier, public cryptographic verifier or TCB attestation verifier);
- owner-defined plaintext-visibility and collusion-assumption profiles.

The semantic preflight maps the selected source mode to its minimum witness and verification scope. Examples:
- TLSNotary direct MPC and three-party direct verifier profiles are session-participant evidence; they are not portable to arbitrary verifiers merely because an opening proof is present.
- TLSNotary delegated-notary evidence is portable only under the delegated notary's identity, key and trust policy. Later verifiers are attestation verifiers and inherit that notary trust.
- Proxy profiles name the proxy as the in-session verifier. A proxy result does not become a public proof unless a separate qualified artifact/profile establishes that.
- Reclaim-style TEE evidence depends on the TCB, hardware vendor, measured image, firmware and attestation-root qualification. The plaintext-visibility profile must say which enclave code sees the token and response.
- Scribe combines an attestor-signed value commitment with a transparent STARK policy proof. Public verification of the STARK does not remove the attestation-node trust required for the TLS provenance statement.

A TEE can improve operator-integrity assurance while changing the confidentiality claim: if credentials and decrypted responses enter the enclave, plaintext is visible to enclave code and privacy depends on attestation, TCB and vendor assumptions. This is not equivalent to a cryptographic design in which plaintext remains on the holder's device and the witness sees only ciphertext. The profile must not call these guarantees identical.

The `plaintext_visibility_profile` is an owner-qualified profile reference, not a caller-authored `private=true` assertion. The schema and preflight also constrain v1 HTTPS origins to lowercase ASCII DNS A-labels, omit default port 443, require numeric ports 1..=65535, and require a canonical DNS server identity to exactly match the origin host. IPv6 literals and internationalized U-labels are out of scope for v1; callers must use an approved A-label profile. It records which principals can observe authentication secrets, decrypted response bytes, session key material, selected disclosure and telemetry. The `collusion_assumptions_profile` separately records which principals may collude and what the proof remains sound/private against. A generic "zero knowledge" label is not a substitute.

Source-specific reading:
- TLSNotary describes zkTLS as designated-verifier: a witness must participate during the TLS session; portable downstream verification relies on a notary/attestor signature. https://tlsnotary.org/blog/2026/06/17/public-verifiability/
- TLSNotary's TEE comparison says moving decryption into an enclave changes the privacy root from prover-side cryptographic confidentiality to hardware attestation and TCB assumptions. https://tlsnotary.org/blog/2026/06/23/where-trust-lives/
- zkPass documents hybrid proxy/MPC modes; proxy mode has verifier-in-the-path and network assumptions that must not be inherited by MPC mode. https://docs.zkpass.org/overview/technical-overview
- Scribe's STARK verifies the attestation signature, commitment and policy, while its security analysis explicitly assumes an honest attestation node for TLS provenance. https://link.springer.com/article/10.1007/s12083-026-02246-2

A syntactically valid trust block does not authenticate these identities. The referenced profile digests must be resolved by their owner and checked against fresh, independently qualified trust-root evidence before any decision.


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
- one or more exact RFC 6901 JSON Pointers (v1 admits 1–64 typed fields from the same authenticated response ordinal);
- a unique stable `field_id` for each field, referenced by the predicate AST;
- duplicate-key behavior (v1 requires rejection);
- Unicode comparison/normalization policy (v1 defaults to exact scalar sequence; no implicit normalization);
- lexical rules for each selected scalar and resource limits.

All field openings in a v1 plan must come from the same authenticated response ordinal/direction and share one HTTP parser and application schema profile. Duplicate field IDs and duplicate JSON Pointers reject. The predicate compiler must bind every selected field; unused field selections reject rather than silently increasing disclosure.

A body-relative JSON span is not a TLS transcript offset. The mapping must be derived through the frozen HTTP framing profile and then authenticated by the selected TLSNotary opening. A parser result cannot authenticate itself.

A hidden JSON string requires a proof of the relevant lexical grammar as well as its structural path. The first field-selector slice's single-hidden-string construction is not a generic proof for arbitrary JSON scalar types, HTTP/2, chunked bodies, compressed bodies, or multiple hidden fields.

### 4.5 Typed interpretation and predicate

The plan fixes a conversion profile for each field from raw field bytes to a typed semantic value. The predicate AST addresses fields by `field_id`, never by a host-supplied value or positional index. Examples:
- money: currency, minor-unit scale, decimal grammar, sign and rounding policy;
- time: timezone, calendar/epoch representation and freshness boundary;
- quantities: dimension, unit and conversion version;
- status: closed enum and source-specific mapping.

Avoid floating-point money and unspecified locale parsing. Conversion MUST be deterministic and versioned. Any normalized value must be derived from the exact authenticated field under the declared conversion profile.

The predicate AST is typed and versioned. It supports boolean composition (`all`, `any`, `not`), typed comparisons, and a timestamp freshness leaf anchored to the verifier request's `issued_at`. Public constants and private witness inputs are explicit. The compiler resolves every `field_id`, checks operand types and verifies that all selected fields participate in the predicate. The proof relation consumes the exact values derived by the field contracts; it cannot accept a parallel caller-provided scalar. A freshness leaf is admitted only for a field parsed under the RFC 3339 timestamp profile and binds maximum age plus permitted future skew to the verifier request issue time.

### 4.6 Privacy-preserving uniqueness

The plan identifies the domain owner's uniqueness subject and scope:
- credential;
- provider account;
- entitlement;
- application participant;
- legal/natural person (only with separately qualified person-binding evidence).

A commitment salt is not a uniqueness subject. Freshly reissued commitments must not reset a same-scope entitlement when the domain's qualified subject binding is stable, but different applications must not receive a globally linkable subject handle. Recovery and issuer-key rotation need explicit continuity policy.

The IR specifies a nullifier derivation profile reference and scope, not a new nullifier engine. Mycelix's existing owner and consume-once boundary remains authoritative.

### 4.7 Credential-status lookup privacy and freshness

A randomized or unlinkable credential proof does not hide a status lookup that exposes a stable list URL, entry ID, status index, fetch cadence, cache behavior, update time or response class. The current W3C status-list maturity must be part of the profile identity:

- W3C Bitstring Status List v1.0 is a Recommendation (15 May 2025).
- W3C Bitstring Status List v1.1 is a Working Draft (11 November 2026), not a Recommendation. Its privacy guidance discusses global identifiers such as `id`, `statusListIndex` and `statusListCredential`, list-size/population effects, verifier caching, CDNs, decoys, monitoring and status-message correlation/alteration.

The CPIR `privacy.status_lookup` profile therefore binds exact specification version and maturity, retrieval mode, index allocation, cache policy, observer-visible channels, freshness interval and anti-rollback requirement. A proposed target of 131,072 bits is an explicit profile parameter informed by v1.1's privacy discussion; it is not a blanket anonymity guarantee or an assertion about every deployment.

Caching/CDN can reduce issuer-visible fetch correlation, but also create stale-status and rollback risk. Status privacy and revocation freshness must be measured together. A cache hit from an old list must never be treated as current merely because the cache is faster.

Mycelix #6272 owns the status-list lookup correlation benchmark. It compares direct fetch, verifier cache, CDN, proxy and batched profiles, with status-index, identifier, timing, cache, update-cadence and response-class observer channels. CPIR only binds its owner-qualified profile. It does not implement a new status service.

### 4.8 Adaptive disclosure and observer-visible privacy

A zero-knowledge predicate result can still leak the private value through repeated queries. For example, repeated verifier-authorized threshold checks `x >= t` can binary-search a bounded amount or age, even when every individual response reveals only one bit. Fresh challenges prevent replay; they do not prevent inference from an adaptive sequence.

The plan therefore references, but does not reimplement, the relevant Mycelix-owned privacy profiles:
- **Multi-round transcript optimization** — Mycelix #5489 evaluates candidate-set intersections over complete presentation sequences, with task-answer preservation and a global tail floor. A per-presentation privacy result is not a transcript privacy result.
- **Relational/group privacy** — Mycelix #5477 differentiates record, person, group and relational-graph adjacency. Per-person disclosure controls do not imply household/cooperative protection.
- **Cumulative disclosure composition** — Mycelix #5404 / benchmark #5405 measures candidate-set reduction and inference after a sequence of minimized outputs.
- **Cross-session / recipient linkability** — #5401 measures content, identifier, proof, envelope and behavioral correlation.
- **Proof-shape correlation** — #5411 measures statement-count, revealed-index, schema/profile and structural fingerprints even when proof bytes are fresh.
- **Federated disclosure-budget conservation** — #5466 / #5468 measures concurrent spend, stale replicas, partitions, retries, policy epochs and escrow double-spend.
- **Observer-visible traces** — #5460 / #5462 handle trace distinguishability and side channels.

CPIR must bind the exact profile IDs and digests for: allowed predicate/query families; cumulative disclosure model; per-subject/scope query-budget policy and epoch; presentation-shape policy; status/credential lookup privacy; and metadata/observer surface. The owner of each profile remains authoritative. If a scope or owner is absent, query history is incomplete, a stale policy epoch is presented, or the budget state is indeterminate, the claim is not silently treated as a fresh allowance.

The query policy must address adaptive composition, not merely each query in isolation. It should specify the permitted result granularity, query-family constraints, request/subject scope, concurrency/retry semantics and denial/indeterminate behavior. Any budget reservation/consumption must use the existing Mycelix owner boundary and consume-once semantics; a fresh nonce, new salt, new presentation or new verifier replica must not reset a same-scope limit.

Do not label abstract disclosure-risk units as differential-privacy epsilon/delta. #5466 explicitly models abstract risk costs, not a calibrated formal DP guarantee. A ZK proof establishes a relation; it does not automatically establish anonymity, unlinkability, non-inference or protection against adaptive querying.

Benchmark privacy as a vector: candidate-set reduction/inference accuracy by query depth; reconstruction success for adaptive predicates; cross-verifier linkability; proof-shape correlation; status-fetch correlation; observer-visible timing/size/error differences; and budget overspend under concurrency/replay. Preserve task utility across controls—denying optional disclosure must not silently change the required task result.

### 4.9 Proof and qualification identity

The proof contract fixes:
- backend and proof-format profile;
- expected guest ImageID/program digest;
- verifier implementation/build identity;
- exact public statement encoding/version;
- source, parser, schema and predicate profile digests.

The verifier must check the expected program identity from trusted configuration. A caller-selected ImageID is not a trust anchor. An envelope's syntactic validity or a non-empty proof byte string is not proof validity.

The output is a proof/qualification receipt candidate bound to the plan digest. It is not consent or action authority.

### 4.10 LISE dependency manifest

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

Version 1 proposes RFC 8785 JSON Canonicalization Scheme (JCS) for the plan envelope. JCS builds on the I-JSON subset and ECMAScript primitive serialization; the input must satisfy these constraints:

1. Reject duplicate JSON member names **before** conversion to an ordinary map/object or canonicalization. A lossy parse that keeps only the last duplicate is not an acceptable preflight.
2. Reject unknown critical members and unsupported enum values. The v1 schema uses closed object shapes; extension points require a future version.
3. JCS numbers must be representable as IEEE 754 binary64 values. For integer semantics, v1 further restricts numeric literals to the exact safe-integer range (±(2^53−1)); larger integers and precise decimals are strings with a typed conversion profile. Floating-point business values are not admitted.
4. Preserve Unicode strings as-is; JCS does not perform Unicode normalization. Any normalization must be an explicit field-level profile and must not be introduced by a serializer.
5. Require lowercase hexadecimal for fixed-width digests and validate exact digest length for the declared algorithm.
6. Keep the plan digest outside the canonicalized object to avoid self-reference.
7. Validate URI/HTTPS-origin semantics, cross-field version/maturity consistency, source-mode/commitment compatibility and typed predicate semantics in an explicit semantic preflight. JSON Schema shape validation alone does not establish these relations.
8. Use a validator that asserts JSON Schema formats; some validators treat `format` as annotation unless format assertion is enabled.

Proposed digest:

```text
plan_digest = SHA-256(
  UTF8("LUMINOUS/CLAIM-PLAN-IR/V1") || 0x00 || JCS(plan)
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
  -> ParsedClaimFieldSet
  -> TypedClaimValueSet
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

Compilation fails closed on missing trust roots, unsupported source mode, unresolved parser/schema, ambiguous field encoding, duplicate or unreferenced field IDs, unresolved predicate field references, operand/type mismatch, invalid freshness semantics, unspecified units, floating-point policy values, stale/revoked key snapshot, missing proof program identity, unsupported uniqueness scope, or an incomplete dependency owner.

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
- adaptive threshold/binary-search query sequences; repeated identical and near-identical predicates; cross-verifier composition; fresh-challenge and fresh-salt budget resets; stale disclosure-budget epochs; concurrent budget double-spend; proof-shape and status-fetch correlation; timing/size/error-channel leakage;
- missing LISE dependency edge or incomplete change feed;
- proof-valid but issuer-unauthorized, consent-denied or institutionally unauthorized result;
- accepted request but failed, partial, unknown or unobserved external effect.

Every suite needs positive, negative and indeterminate controls. Reject-all behavior is not a pass. Each fixture records its expected disposition, oracle provenance and adjudication status; unadjudicated cases remain provisional.

## 9. Benchmark plan and prospective gates

Benchmark plan compilation separately from TLS, proving and institutional decision latency. CPIR's privacy references are bindings to the Mycelix owners above, not a replacement implementation or a claim of differential privacy.

| Dimension | Measurement | Initial target |
|---|---|---|
| Canonicalization | byte-identical output across independent implementations | 100% |
| Mutation resistance | wrong-plan/source/path/schema/predicate/request/program bindings accepted | 0 |
| Conformance | positive controls accepted; negative controls rejected; indeterminate controls remain indeterminate | 100% of adjudicated fixtures |
| Determinism | same plan/version yields same digest and dependency manifest | 100% |
| Privacy | extra fields disclosed or new cross-scope stable identifiers emitted | 0 |
| Adaptive disclosure | unauthorized query-sequence inference and same-scope budget overspend | 0 accepted beyond the owner-defined policy |
| Linkability | cross-scope presentation correlation and proof-shape/status-fetch false negatives | 0 for the frozen declared observer model |
| Requalification | affected receipts selected after a relevant dependency change | 100% of frozen dependency-change controls |
| Performance | plan compile/validate p50/p95/p99, CPU, memory, bytes, manifest size | baseline first; optimize after matched measurement |
| End-to-end competitiveness | p50/p95/p99 time-to-qualified-private-decision under matched claim, trust, privacy and hardware profiles | prospective target: at least 20% lower p95 than the best matched baseline, with no increase in leakage or false accepts |
| Sample quality | raw per-run samples, confidence intervals and reproducible environment manifest | at least 1,000 measured samples per primary cell before making a p95/p99 superiority claim |

For an eventual matched system benchmark, report source/session time, notary/TEE work, opening verification, parser semantics, proof generation/verification, nullifier consumption, LISE qualification and time-to-qualified-private-decision separately. Keep assurance properties as a vector/Pareto frontier. Never trade a weaker witness model or more disclosure for a faster headline number without showing that trade-off.

## 10. Initial fixture corpus

The positive fixture now demonstrates one multi-field ZAR eligibility decision from one authenticated response: `currency == ZAR`, `available_minor_units >= 100000`, and `as_of` fresh within 300 seconds of verifier `issued_at` (with 30 seconds future skew). This avoids relying on host-supplied pre-normalized composites.

The standalone fixture directory now contains:
- one synthetic ZAR balance plan positive fixture;
- unsafe integer literal (JCS/I-JSON admission);
- status specification / maturity mismatch;
- URL path substituted into the HTTPS origin;
- missing status anti-rollback requirement;
- duplicate JSON member name in a raw pre-canonicalization fixture (two duplicate `plan_id` member names are preserved in the raw bytes);
- invalid RFC 6901 JSON Pointer escape (`~2`);
- request expiry equal to issue time;
- delegated TLSNotary witness mislabeled as public cryptographic verification;
- HTTPS origin port outside 1..=65535;
- explicit default HTTPS port 443;
- DNS server identity mismatch with the canonical origin host;
- a machine-readable manifest describing the expected rejection stage.

The fixtures are structural/semantic test inputs only. They contain zero digests and synthetic profile IDs, so they do not represent a valid cryptographic trust profile, issued credential, source session, or proof. The duplicate-member fixture must be rejected by the raw JSON admission parser before a normal JSON object is formed.

### Standalone Rust preflight prototype

An isolated, research-only Rust workspace now exists at `research-tools/claim-plan-ir/validator/`. Its intended admission pipeline is:

1. bounded raw JSON input;
2. custom duplicate-aware deserialization (reject decoded duplicate keys before ordinary object construction);
3. integer/floating-point admission before JCS;
4. JSON Schema 2020-12 validation with format assertions enabled;
5. cross-field semantic checks for multi-field binding, unique pointers and IDs, RFC 6901 `~0`/`~1` escape validity, typed predicate references, freshness, status maturity, canonical lowercase DNS A-label HTTPS origin/port/server continuity, path-template shape and source-mode/witness/verification-scope/commitment consistency;
6. RFC 8785 serialization and domain-separated SHA-256 plan digest.

Direct crate versions are pinned in `Cargo.toml`: `jsonschema 0.61.2`, `serde_json_canonicalizer 0.4.1`, `serde_json 1.0.151`, `serde 1.0.229`, `sha2 0.10.9` and `hex 0.4.3`. JSON Schema format assertion is explicitly enabled. Canonicalization follows the duplicate-aware parser because canonicalization cannot repair ambiguous duplicate-key input after an ordinary map parser has discarded a duplicate.

The Rust source contains 17 authored unit tests for the three-field positive fixture, duplicate keys, unsafe integers, status maturity, origin paths, missing anti-rollback, unknown field references, freshness type mismatch, unused fields, floating-point inputs, invalid RFC 6901 pointer escapes, RFC 8785 UTF-16 object-key ordering, nonpositive request lifetime, source-mode/trust-scope consistency, HTTPS port range, explicit default-port rejection and DNS server/origin identity matching. **The crate has not been compiled or tested**: rustc/Cargo are absent in the authoring environment. A Cargo.lock is also not present, so the transitive dependency closure is not frozen. Authored tests remain unrun.

## 11. Implementation sequence

1. Review this schema and threat model against #12301, #12328, #12373 and Mycelix #6230.
2. Add independent canonicalization and schema conformance fixtures.
3. Implement the compiler as a standalone Rust research workspace, outside Symthaea's default Cargo workspace.
4. Add typed adapters only when the upstream TLSNotary commitment API and exact opening contract are available.
5. Prove the same authenticated value is parsed, typed and consumed by the predicate.
6. Emit the LISE dependency manifest and verify targeted invalidation.
7. Integrate Xenia consent and domain action only after Mycelix independently qualifies the claim.

## 12. Current external baseline notes

- TLSNotary v0.1.0-beta.1 was announced on 2026-11-14 and is the current integration candidate in the related research issues; it is a prerelease, not a production qualification.
- W3C Bitstring Status List v1.1 is a Working Draft published 2026-11-11; v1.0 remains the Recommendation. The v1.1 draft explicitly discusses global identifier/index correlation, small-list population effects, verifier caching/CDNs, decoy distinguishability and status-message correlation. CPIR status profiles must encode maturity rather than silently promoting a Working Draft to a Recommendation. Source: https://www.w3.org/TR/2026/WD-vc-bitstring-status-list-1.1-20261111/.
- Credential/presentation interoperability is a separate standards track, not a substitute for Web2 source provenance. IETF RFC 9901 defines SD-JWT selective disclosure. W3C Data Integrity BBS Cryptosuites v1.0 is a Candidate Recommendation Draft dated 2026-09-10, not a Recommendation; its CRD calls for two independent implementations for every feature to exit Candidate Recommendation. BBS derived proofs are designed to be unlinkable to the original signature, but the spec also discusses residual linkage from revealed-index patterns and global JSON-LD identifiers. Mycelix #5447 owns exact version-pinned SD-JWT/BBS/OpenID4VP comparison and conformance; CPIR binds the relevant profile rather than implementing a replacement. Sources: https://datatracker.ietf.org/doc/rfc9901/ ; https://www.w3.org/TR/2026/CRD-vc-di-bbs-20260910/.
- OpenID Foundation announced on 2026-08-07 that OpenID4VP and OpenID4VCI conformance suites with HAIP 1.0 were complete and open for self-certification. Its announcement reports 47 tested pairs with 87% passing in the event it describes. Treat this as interoperability/conformance evidence, not a cryptographic privacy or source-provenance guarantee. Source: https://openid.net/openid4vp-and-openid4vci-conformance-tests-are-complete-and-open-for-self-certification/.
- By 24 September 2026, the OpenID Foundation reported the first 14 organizations had self-certified OpenID4VP / OpenID4VCI with HAIP 1.0 against the official conformance suites. This is stronger implementation-interoperability evidence than a synthetic adapter test, but it still does not qualify TLS source authenticity, privacy against correlated presentation metadata, or institutional authority. Source: https://openid.net/first-implementers-certify-to-openid4vp-and-openid4vci-with-haip/.
TLSNotary Beta.1's own analysis estimates approximately 48 serialized round trips for native MPC, 56 for browser MPC and 21 for proxy mode in its latency model. This makes round-trip count and p95 network latency first-class benchmark outputs, not just bytes or proof time. The MPC default run transfers 30.08 MB total; verifier-to-prover traffic is 0.60 MB after Beta.1 optimizations. The benchmark's default cable/fiber/mobile profiles and 10 samples per point remain the scope of these estimates. - TLSNotary's official Beta.1 benchmark (published 2026-11-14) reports 8.2–9.3% MPC native improvement and 13.1–20.9% MPC browser improvement over alpha.15; proxy mode improved 21.0–24.3% native and 35.5–39.4% browser over a separate May 2026 proxy benchmark. These are within-project release improvements, not evidence that proxy beats MPC under equal trust assumptions. It uses a 1 KiB request / 2 KiB response. Reported MPC native proving times are 13.35 s (cable), 3.43 s (fiber), 9.63 s (mobile 5G); MPC browser/WASM is 14.15 s, 4.17 s, 10.45 s. Proxy native is 1.30 s, 0.83 s, 1.29 s; proxy browser/WASM is 1.57 s, 1.12 s, 1.55 s. MPC comparisons are against alpha.15; proxy comparisons are against a separate May 2026 report. The benchmark uses Azure Standard_F8als_v6 / AMD EPYC 9004, 8 vCPU, 16 GB RAM, Debian 12, with 10 measured samples and 3 warmups per point. It reports 30.08 MB total MPC traffic and 0.60 MB verifier-to-prover traffic in the default run. The official post says its measurements prove statements over the entire server response; do not call them selected-field ZK proof timings or compare proxy and MPC as equal trust profiles. TLS 1.3 is listed as a priority for a subsequent release, not a Beta.1 capability. Source: https://tlsnotary.org/blog/beta1-performance-next-steps/
- TLSNotary's full-reveal fast path skips response-side ciphertext/plaintext ZK only when the entire response direction is revealed; one private byte keeps that direction on the per-byte ZK path. The reported 1–51 KiB response sweep is essentially flat in full-reveal mode, but this is not a selective-disclosure speedup and must not be benchmarked as equivalent to hidden-field proofs. The path still proves TLS key derivation. Source: https://tlsnotary.org/blog/2026/05/19/fast-reveal/.
- Scribe's paper reports policy-only STARK proving from 6 ms to 346 ms, verification from 0.4 ms to 9.7 ms, and proof sizes from 32.1 KiB to 77.8 KiB for repeated 16-bit threshold checks. The reported experiment uses an Apple M4 Pro / 24 GB system and small sample counts (10 STARK repetitions). It separately reports a TLSNotary workflow average of 856 ms proving and 749 ms verification. These workloads and trust assumptions are not matched to TLSNotary Beta.1 or this system; use them as independent reference points, not a leaderboard. Source: https://doi.org/10.1007/s12083-026-02246-2
- Before declaring SOTA superiority, reproduce each competitor's exact release, protocol mode, request/response bytes, parser, predicate, hardware/network profile and disclosure policy. Report source-session, proof generation, verification, parser, uniqueness consumption, LISE and final decision latency separately. A lower local proof time cannot compensate for a weaker witness assumption, more plaintext exposure, or an easier claim.
- TLSNotary's commit strategies include custom ranges, HTTP objects and per-byte commitments. HTTP-object commitment is the documented default trade-off; each additional range costs storage linearly (the current documentation estimates about 250 bytes of additional Secret artefact per BLAKE3 range).
- Scribe (published 2026-09-02) combines TLS provenance, an attestation-node signature on a salted commitment, a transparent STARK policy proof, application-scoped spend tags, a Merkle spent-tag aggregator and Falcon-signed root publication. It explicitly assumes honest attestation-node provenance behavior; fresh salts can create distinct commitments; a signed root does not independently constrain a correct append-only transition. These are separate benchmark dimensions, not reasons to discount the system's stated theorem.

## 13. Claim ceiling

A future qualified implementation may establish that a verifier-authorized, versioned predicate was evaluated over a value opened from a specific TLS-authenticated transcript range, under a specified source mode, parser/schema/conversion profile, request/audience/purpose and proof-program identity, with explicit qualification dependencies.

It still does not establish that the provider's statement is true, that the claimant legally owns the account, that the subject is unique as a natural person, that an issuer is authorized for every use, that the credential remains current without a fresh LISE decision, that the user understood or freely gave consent, that an institution authorized a consequential action, or that the external effect occurred.
