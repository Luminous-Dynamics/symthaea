# TLSNotary-to-RISC Zero Composition V1

**Status:** Architecture and feasibility study only. No TLSNotary verifier has been ported into a RISC Zero guest; no guest build, receipt, cycle count or proof is claimed.

**Pinned source:** TLSNotary v0.1.0-beta.1, commit `40a069565a2eaecbb5e691dbc634be20e8498daf`.

**Owners:** Symthaea #12301 (source opening and same-witness predicate), #12328 (application field semantics), #12373 (verified commitment API), and draft PR #12336 (host-side sparse disclosure / HTTP offset candidate).

## 1. The actual composition to prove

The source-bound claim must be one relation over one private value:

```text
notary-authenticated TLS session
  + exact transcript commitment record
  + exact received-direction byte range
  + exact HTTP response/body framing
  + verifier-selected JSON Pointer and parser profile
  + private witness that opens that same commitment
  + request-bound claim schema / unit / predicate
  -> one source-bound ZK receipt
```

A valid proof of `balance >= threshold` over an arbitrary host-supplied `u64` is insufficient. The guest must prove that the *same* private witness is the value committed by the notary-authenticated source and that it occupies the intended field in the authenticated response.

Keep the following relations distinct and compose them explicitly:
1. notary/session authenticity;
2. selected transcript commitment authenticity;
3. exact range and HTTP message correspondence;
4. application field/path semantics;
5. private value opening;
6. predicate correctness;
7. exact verifier request/audience/purpose/policy binding;
8. external issuer legitimacy, currentness, consent and action authority.

## 2. Why the whole TLS MPC implementation should not enter the guest

TLSNotary Beta.1's workspace contains a large live protocol stack:
- `tlsn` prover/verifier protocol;
- MPC-TLS, MPZ circuits/garbling/OT/VM dependencies;
- mux/transport and proxy;
- attestation and core verification libraries.

The RISC Zero guest is not the place to run an interactive TLS connection or reproduce the MPC session. The TLS session is completed by the prover and notary outside the zkVM. The guest should consume a bounded, self-contained evidence object and verify only the cryptographic source relation that a relying party needs.

**Preferred initial dependency boundary:** use the narrow `tlsn-attestation` + `tlsn-core` verification surface, not the top-level `tlsn` protocol crate and its MPZ/MPC dependencies.

This is a design hypothesis, not a validated Cargo feature graph. Exact target compilation and dependency closure must be checked before adopting it.

## 3. Beta.1 verification APIs: what is available and what is not

### Useful public surface

At the pinned commit:
- `tlsn_attestation::presentation::Presentation::verify(&CryptoProvider)` verifies the notary attestation proof, optional server-identity proof and optional transcript proof.
- It returns `PresentationOutput`, which exposes the verified `Attestation`, optional `server_name`, `ConnectionInfo`, optional authenticated `PartialTranscript` and extensions.
- `Presentation::verifying_key()` exposes the notary verifying key from the attestation proof.
- `PartialTranscript::received_authed()` exposes authenticated received-direction ranges.
- `PartialTranscript::received_unsafe(expected)` expands the sparse transcript into a full-length byte vector and fills unauthenticated positions with zero. It must never be mistaken for authenticated data outside the returned range set.
- `TranscriptCommitment` / `PlaintextHash` are public types, with direction, range, algorithm and typed hash metadata.
- `CryptoProvider` exposes hash, certificate, signing and signature-verifier providers.
- TLSN documents that presentation verification does not invoke runtime randomness; attestation creation does.

### Blocking API gap

The commitment data required by the guest is not externally readable after `Presentation::verify()`:
- `Attestation.body` is public as a field, but the fields of `Body` are private.
- `Body::transcript_commitments()`, `Body::connection_info()`, `Body::server_ephemeral_key()` and `Body::cert_commitment()` are `pub(crate)`.
- Therefore an external Symthaea caller can access the verified opened transcript, but cannot enumerate the transcript commitments authenticated by the notary-signed body root.

This is tracked in [SYM-ZK-TLSN-API-001 / #12373](https://github.com/Luminous-Dynamics/symthaea/issues/12373).

Do not mirror internal BCS/bincode serialization, reconstruct private field IDs, or independently reimplement TLSNotary's body Merkle/domain-separation semantics. A field-order or encoding drift would break the exact attestation guarantee.

## 4. Two guest-side verification designs

### Route A — verify the full Presentation in the guest

Input:
- serialized TLSNotary `Presentation`;
- exact expected notary verifying key/profile;
- exact expected server identity and allowed TLS version;
- trusted certificate roots/profile;
- verifier request context;
- private value and commitment opening;
- selected JSON field/path structural disclosure view.

Guest:
1. calls `Presentation::verify()`;
2. rejects an unexpected notary key;
3. checks server name, TLS version, connection metadata and transcript bounds against verifier-owned expectations;
4. obtains the authenticated `PartialTranscript`;
5. requires the verified commitment metadata API from #12373 to identify the exact received-direction `PlaintextHash` commitment whose range equals the hidden string contents (or an explicitly defined exact range);
6. recomputes TLSN's exact `H(message || blinder)` opening under the commitment's recorded algorithm;
7. applies the field/path relation from #12336/#12328 to the authenticated structural context and received stream offsets;
8. runs the exact claim predicate on those same private bytes;
9. commits only the selected request digest, claim-schema/predicate digests, source root/commitment reference, result and verifier challenge.

Advantages:
- uses TLSNotary's public verification semantics rather than a local copy;
- easiest initial correctness reference;
- no full live TLS/MPC stack is required.

Costs / risks:
- may verify more attestation body and transcript material than the claim requires;
- RISC Zero guest support for the required `std`/crypto dependencies is a feasibility gate;
- certificate path and notary signature verification may dominate guest cycles.

### Route B — verify a minimal selected-field certificate (optimization target)

Ask TLSNotary for a first-class, versioned proof object containing only the exact data needed for one field claim:
- notary-signed attestation header;
- attestation verifying key and signature;
- typed transcript commitment record (direction, range, algorithm, hash);
- Merkle inclusion proof for that exact commitment record against the signed body root;
- the exact server-identity / connection fields needed for the admitted server profile, with authenticated inclusion proofs;
- any transcript-range opening metadata required for the selected structural context.

A verifier inside the guest checks:
1. notary signature over the exact canonical header;
2. typed field commitment inclusion under the signed attestation body root;
3. server identity commitment and connection profile;
4. exact transcript range/hash opening;
5. structural JSON field relation;
6. private predicate and request binding.

This could reduce receipt size and guest cycles because it verifies only the fields required by the claim. The certificate **must be provided by a supported TLSNotary API**, not assembled from private serialization or guessed body field IDs.

The required Merkle field-ID / domain separation and canonical signature encoding are currently internal to `tlsn-attestation`. The upstream API must expose a typed proof and verification method, not just raw hashes.

### Rejected Route C — trust host verification

Host verifies TLSNotary, then sends a host-supplied `verified=true`, parsed value or commitment digest to the RISC Zero guest.

This proves only that the guest computed a predicate over untrusted host input. It does not bind the value to the source transcript. It is rejected.

## 5. RISC Zero guest feasibility

The current Symthaea guest is `#![no_std]` and depends on the dedicated `symthaea-zkproof-core` types. TLSNotary Beta.1's attestation/core crates use `std` APIs (for example collections and time), and `tlsn-attestation` also depends on `rand` for attestation creation.

Important distinction:
- TLSNotary's presentation-verification path is documented not to invoke runtime randomness.
- That does not mean its crate graph is currently `no_std`.
- RISC Zero's guest `std` feature is available but described as experimental; its target has bounded memory and a bare-metal execution model.
- The TLS MPC/MPZ protocol crate should not be included in this guest.

Required feasibility experiments:
1. minimal guest package containing only pinned `tlsn-core` and `tlsn-attestation` verification dependencies;
2. build with `risc0-zkvm = 3.0` exact feature configuration and `getrandom` disabled/unsupported where appropriate;
3. verify a fixed synthetic Beta.1 Presentation fixture;
4. measure guest ELF size, memory high-water mark, cycle count, host proof generation time, receipt size and receipt verification time;
5. repeat with a custom one-root certificate profile instead of Mozilla's full root bundle;
6. compare full Presentation verification with the minimal selected-field certificate once the upstream API exists.

Do not modify the legacy guest tag-0/tag-1 code. Use a separate versioned guest package and a distinct ImageID, as already done in the request-bound guest work (#12295).

## 6. Identity and trust checks in the guest

The guest must receive verifier-owned expected trust configuration, not trust roots selected by the prover.

Minimum checks:
- expected notary verifying key or an explicitly versioned notary key-set root;
- exact TLSNotary protocol/attestation version;
- expected server DNS name / origin;
- allowed TLS version profile (Beta.1 support must be checked against the exact release; do not infer TLS 1.3 from roadmap statements);
- exact received transcript length and bounded selected ranges;
- exact request-response pairing / response ordinal;
- source statement/session root, selected commitment, parser profile, JSON Pointer, claim schema and predicate digest;
- exact verifier challenge and audience;
- expected RISC Zero guest ImageID.

A valid notary signature authenticates what that notary attested. It does not prove the notary key is trusted by the institution, nor that the provider data is true, current or legally authoritative. Notary trust, provider authority and Mycelix's institutional decision remain separate.

## 7. Composing with the existing request-bound guest

Symthaea PR #12295 already adds a separate request-bound RISC Zero guest and exact request digest. It explicitly does not authenticate issuer signatures or open source commitments.

Do not create a second request-binding scheme. Extend the existing request context with exact source bindings only after the TLSNotary source object is defined:
- TLSNotary attestation/session header root and version;
- exact commitment-record digest and direction/range;
- exact server identity / source profile;
- parser profile and JSON Pointer;
- exact claim field schema and predicate semantics;
- challenge, audience, verifier and policy.

The resulting guest relation should use the **same private value bytes** for:
- TLSNotary commitment opening;
- JSON field selection;
- numeric/date/unit parser;
- predicate.

No public journal should contain the raw hidden value, blinder, complete JSON response, bearer token, account number or global stable subject identifier.

## 8. Benchmark protocol

Compare on the same synthetic HTTPS response and exact claim:
1. RISC Zero guest verifies a full TLSNotary Presentation + field predicate.
2. RISC Zero guest verifies a minimal selected-field certificate + field predicate (if the upstream API is available).
3. TLSNotary's own Noir/UltraHonk example baseline, upgraded so the verifier binds the exact field path.
4. Scribe-style attestor-extracted commitment + STARK baseline (explicitly mark attestor extraction as a trust assumption).
5. TEE plaintext processor baseline (explicitly mark plaintext visibility inside the enclave).
6. Attested MPC-TLS verifier with prover-side plaintext (the #6254 integrity/privacy profile).

Keep the following workloads identical:
- response sizes: 2 KiB, 10 KiB, 100 KiB, 1 MiB;
- JSON nesting: depth 2 / 8 / 16 / configured maximum;
- number of same-shaped distractor fields: 0 / 1 / 10 / 100;
- selected HTTP field commitment strategy: HTTP-object / custom range / per-byte;
- selected ranges: 1 / 4 / 16 / 64;
- private value grammar: fixed-width date, scaled integer/currency, boolean, bounded decimal.

Measure separately:
- TLS/MPC session establishment and online time;
- TLSNotary commitment/attestation generation;
- transcript proof and selective disclosure;
- host parsing/field selection;
- guest cycles, guest ELF size and memory high-water mark;
- proving p50/p95/p99 and verification p50/p95/p99;
- receipt/proof bytes, network bytes and number of interaction rounds;
- CPU, memory and energy per successful qualified claim;
- visible response bytes / total response bytes;
- structural metadata revealed and sensitive field leakage;
- false acceptance, positive-control acceptance and parser disagreement;
- trusted parties, keys, hardware/TCB and plaintext visibility.

Reported Scribe paper baseline (2 September 2026) includes STARK proving approximately 6–346 ms depending on policy repetitions, verification approximately 0.4–9.7 ms, and proof sizes approximately 32.1–77.8 KB. Its reported TLSNotary workflow measurement is approximately 856 ms proving / 749 ms verifying. These are component/paper baselines, not matched Mycelix results; reproduce on common hardware and identical statements before claiming superiority.

### Prospective targets (not achieved results)

- Zero false accepts across the expanded adversarial corpus.
- 100% acceptance of valid positive controls under the exact frozen parser/source profiles.
- Zero acceptance of a wrong commitment, wrong direction, wrong range, wrong field, wrong request or wrong ImageID.
- Zero bytes of application plaintext or TLS decryption key material entering the attested-verifier enclave profile.
- At least 20% lower p95 end-to-end time-to-qualified-private-decision than a matched baseline only if the trust, privacy, proof soundness, hardware and disclosure budgets are identical.
- Minimal-field certificate target: at least 20% fewer guest cycles and 20% smaller receipt than full-Presentation verification, with identical claim and trust assumptions. This is a design target to validate, not a current performance claim.
- Report Pareto frontiers for privacy, trust assumptions, p95 latency, proof size, energy and availability; do not hide them in one composite score.

## 9. Paradigm shift

**Move from proving a value to proving a value's complete evidence lineage.**

A conventional balance proof proves arithmetic over a private integer. A zkTLS presentation proves some data came through a TLS session. The stronger composition proves the one hidden field was:
- received from the expected server;
- included in a notary-authenticated transcript commitment;
- selected by the exact requested parser/schema path;
- used by the same witness in the predicate;
- tied to the exact verifier request and challenge.

This is the difference between a privacy-preserving calculation and a privacy-preserving *source-bound claim*.

The performance opportunity is to prove only the exact authenticated fields needed for the claim, rather than exposing or verifying the entire response or running the live TLS MPC protocol inside the zkVM.

## 10. Nonclaims

This document is a design and source feasibility study. It does not establish:
- a compiled TLSNotary/RISC Zero adapter;
- a verified TLSNotary proof inside a RISC Zero receipt;
- a successful build of TLSN's `std` dependencies on RISC Zero;
- provider truth, account ownership, issuer legitimacy, legal entitlement, freshness, consent or action authority;
- a performance advantage over Scribe, DECO, zkPass, Reclaim, Noir or TLSNotary.

All benchmarks and targets remain unexecuted until the dedicated verification slice is built and independently reproduced.
