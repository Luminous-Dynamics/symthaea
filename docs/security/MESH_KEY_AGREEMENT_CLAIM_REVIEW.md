# Mesh Key-Agreement Claim-Integrity Review

**Date:** 2026-10-22  
**Status:** source-level design review; security claims corrected, protocol remediation pending  
**Scope:** `PeerKeyStore` (feature `mesh-key-exchange`) and `PqcHandshakeManager` (feature `pqc-handshake`)  
**Deployment statement:** this review documents properties of the source and API. It does not establish that either feature is enabled in a deployment or that an attack has occurred.

## Executive summary

The current code conflates several distinct properties: pairwise key separation, authentication, forward secrecy, post-quantum confidentiality, and hybrid key-combiner security. These are not interchangeable.

Three key findings:

1. **Static X25519 is not forward secrecy.** `PeerKeyStore` accepts a long-term static X25519 secret, performs static-static DH, and derives keys with fixed HKDF-SHA-256 info labels. The same local secret and peer public key reproduce the same pairwise keys. A later compromise of the long-term private key plus a recorded peer public key can reconstruct the derived key. This can provide pairwise key separation, but not forward secrecy.
2. **Peer identity is not bound by the `agree` KDF.** `peer_source_id` is used as a map key, not as HKDF salt/info. The method accepts a public key without authenticating its owner. The caller must ensure the key is bound to the claimed peer identity in an authenticated transcript before the result is used.
3. **The current PQC integration is not a proven hybrid key exchange.** `PqcHandshakeManager` derives a key by applying BLAKE3's `derive_key` context to the concatenation of a public classical challenge nonce and an ML-KEM shared secret. A public nonce is context, not an independent classical secret. The shown API does not itself enforce completion of the classical handshake or bind both peer identities, the KEM public key and the KEM ciphertext to an authenticated transcript. The manager reuses one ML-KEM keypair across sessions, so later compromise of that private key can expose recorded KEM sessions.

The immediate work corrects overclaims and adds deterministic witnesses for the static-DH limitations. It does not replace the protocols with a newly invented handshake.

## K-1 — Static X25519 pairwise derivation does not provide forward secrecy

`PeerKeyStore::new(secret_bytes)` stores a supplied `StaticSecret`. `agree(peer_source_id, peer_public)` computes X25519 against a supplied public key, then derives an encryption key and an authentication key via HKDF-SHA-256 and fixed info strings:

- `symthaea-mesh-chacha20-v1`
- `symthaea-mesh-auth-v1`

No per-session ephemeral private input or ratchet contributes to the DH result. Repeating the same static key pair repeats the shared secret and the derived keys. Consequently, a later attacker who obtains a long-term private key and has the corresponding recorded peer public key can recompute that pairwise key. This is the opposite of forward secrecy as defined for ephemeral DH exchange; forward secrecy requires ephemeral contributions and appropriate erasure. See RFC 9678, which explicitly describes ECDH as ephemeral for forward secrecy: https://www.rfc-editor.org/rfc/rfc9678.html.

The new test `legacy_static_x25519_agreement_repeats_key_after_long_term_secret_reuse` records this property using fixed test secrets. The companion test `legacy_static_x25519_derivation_does_not_bind_peer_source_id` shows the same static peer key derives the same traffic key even when stored under different peer-ID labels. These are known limitation witnesses, not security-passing tests.

## K-2 — Peer identity is a lookup label, not KDF-bound context

The `peer_source_id` argument selects the `HashMap` slot in which keys are stored, but it is not mixed into the DH secret or HKDF `info`. With an unchanged local static secret and the same supplied public key, calling `agree` under two different source IDs gives the same encryption and authentication keys. The new test `legacy_static_x25519_derivation_does_not_bind_peer_source_id` reproduces this behavior.

This is not, on its own, proof of an active network attack: the caller may authenticate the public key and maintain a valid identity mapping elsewhere. It does mean this API does not enforce that mapping. Key authentication and derivation context need to be explicit and checked at the protocol boundary.

## K-3 — The public challenge nonce is not a classical secret contribution

In `PqcHandshakeManager`, both initiator and responder derive a session key from:

```text
BLAKE3 derive_key(
  context = "symthaea-pqc-session-v1",
  input_key_material = public_classical_challenge || ML-KEM_shared_secret
)
```

The source now names this function `blake3_derive_session_key`, matching the implementation; the earlier `hkdf_derive` name and HKDF-SHA-256 comments were inaccurate.

A challenge nonce can bind context and freshness if handled correctly, but it is transmitted as protocol data and is not an independent secret. It therefore cannot be counted as a classical DH secret contribution to a hybrid combiner. In the present design, confidentiality is derived from the ML-KEM shared secret. The security argument is not “at least as strong as the strongest component” simply because two values are concatenated and passed to a KDF.

A proper hybrid key exchange that aims to remain confidential if either classical or post-quantum *key agreement* component fails must combine independent secret outputs using a specified, reviewed combiner and bind the whole exchange transcript. Ed25519 signatures provide authentication, not an independent DH shared secret.

## K-4 — The API does not enforce KEM transcript authentication

`receive_kem_public_key(peer_id, bytes)` stores caller-provided KEM public key bytes. `encapsulate_for_peer` uses those bytes and accepts a caller-supplied `peer_id` and `classical_nonce`; `decapsulate_and_derive` accepts a caller-supplied peer label, ciphertext and nonce. The manager exposes the classical handshake object separately but does not itself enforce that it has completed before storing a session key.

The current API does not bind all of the following into one authenticated transcript:

- protocol name and version, negotiated algorithms, and initiator/responder roles;
- both stable peer identities and their authenticated public keys;
- the ML-KEM public key and ciphertext;
- the classical handshake challenge and evidence that it was verified;
- session/boot context, key epoch, and downgrade policy.

This review does not claim that a higher layer cannot perform some of these checks. It states that this module's public interface does not enforce them, so production callers can misuse it without the type system preventing that state. The test `legacy_pqc_session_derivation_does_not_bind_peer_id` deterministically derives the same key under two peer-ID labels from the same KEM ciphertext and public challenge. The former derivation smoke test is now explicitly named `legacy_pqc_derivation_succeeds_without_verified_classical_handshake_state` to make clear that no classical verification is performed by that test.

## K-5 — Static ML-KEM manager keypair does not provide KEM forward secrecy

`PqcHandshakeManager::new` generates one `MlKem768KeyPair` and keeps it on the manager while deriving multiple peer sessions. A captured KEM ciphertext can be decapsulated by whoever later obtains the manager's private KEM key. Fresh encapsulation randomness means session secrets may differ, but it does not prevent historical recovery after compromise of the static decapsulation key.

A production design needs a documented KEM-key rotation and erasure lifecycle, and must decide whether ephemeral/session-specific KEM keys are feasible for its deployment. It must also bind the KEM key to the intended peer identity. An Ed25519 signature that is assumed to become forgeable in a quantum threat model is not sufficient authentication for claiming post-quantum protection against an active attacker; use a suitable post-quantum authentication mechanism or another explicit trust anchor for that threat model.

## K-6 — Implicit rejection means decapsulation success is not peer authentication

The existing invalid-ciphertext test notes ML-KEM implicit rejection: an invalid ciphertext may yield a pseudorandom shared secret rather than a distinguishable decapsulation failure. Accordingly, a successful return from `decapsulate_and_derive` does not prove the sender used a valid encapsulation, that the other side holds the same key, or that the peer is authenticated. The old test name `test_wrong_ciphertext_fails` was misleading because it asserted the result was `Ok`. It is renamed to `test_invalid_ciphertext_uses_implicit_rejection_and_needs_key_confirmation` and now makes explicit that the API stores a candidate session key.

Production protocols need **key confirmation** bound to the authenticated transcript before marking the session established. Treat derived keys as unconfirmed until the peer proves possession of the same key over the exact transcript. Negative cases must include bad KEM ciphertexts, mismatched key shares and identities, and key-confirmation failure. Do not try to detect invalid KEM ciphertext by relying on a distinguishable decapsulation error; implicit rejection is a deliberate KEM property.

## Standards-based remediation baseline (current as of October 2026)

The appropriate next baseline is not an ad-hoc “X25519 + ML-KEM secret concatenation.” As of August 2026, **RFC 10024** defines the TLS 1.3 `X25519MLKEM768` hybrid group, combining ML-KEM-768 with **ephemeral** X25519/ECDHE. It applies the hybrid framework in RFC 9954, where component shared secrets are combined through the existing TLS 1.3 key schedule. RFC 9954 explicitly warns that its security analysis relies crucially on the TLS 1.3 transcript; implementers cannot assume the same hybridization is secure merely by copying the concatenation into a different protocol.

References:
- RFC 10024, *Post-Quantum Traditional (PQ/T) Hybrid Key Agreement Mechanisms for TLS 1.3*: https://www.rfc-editor.org/rfc/rfc10024.html
- RFC 9954, *Hybrid Key Exchange in TLS 1.3*: https://www.rfc-editor.org/rfc/rfc9954.html
- NIST FIPS 203, ML-KEM: https://csrc.nist.gov/pubs/fips/203/final
- NIST SP 800-227, Recommendations for Key-Encapsulation Mechanisms: https://csrc.nist.gov/pubs/sp/800/227/final

**Recommendation:** prefer a reviewed standard implementation for the existing online/IP transports. For the off-grid LoRa transport, treat protocol adaptation as a separate design and review task. Do not claim RFC 10024 compliance unless the implementation actually conforms to TLS 1.3 and its exact construction. Use the standard as a security-model baseline, not permission to improvise a novel transcript or key schedule.

### Constrained-link cost is part of cryptographic correctness

The current LoRa fragment payload is 214 bytes (222-byte MTU minus the 8-byte fragment header), plus one XOR parity frame per payload. ML-KEM-768 has a 1,184-byte public key and a 1,088-byte ciphertext. Under the current framing, each one requires six data fragments plus a parity fragment. A combined server share containing the ciphertext and a 32-byte X25519 ephemeral public share is still about 1,120 bytes, also six data fragments plus parity.

Therefore, the two hybrid key-share directions alone imply at least **14 fragment transmissions** under this simple framing—before transcript/authentication metadata, retransmissions, key confirmation, link-layer overhead or packet loss. This is a capacity estimate from serialized sizes, not a measured radio benchmark. The protocol must define canonical lengths, FEC over the correct authenticated representation, per-fragment authenticity/reassembly, retry and timeouts, identity binding, and a downgrade-proof transition. Never squeeze key-share bytes into the existing packet identifier or MTU by truncating them.

Benchmark the real trade-off: setup latency and completion probability across loss rates; bytes/fragments per successful session; radio airtime and energy; peak memory; CPU and wall-clock cost on the intended embedded hardware; and session recovery after loss/restart. Compare the secure protocol against itself under link conditions, and against a reviewed implementation on a comparable platform.

## K-8 — The service was marking a one-sided KEM exchange as complete

The service implementation of `run_pqc_exchange` originally sent the local ML-KEM public key and attempted to read the peer's public key, then called `encapsulate_for_peer`. It explicitly discarded the resulting KEM ciphertext in `_ciphertext`, left peer-side decapsulation as a TODO, and nevertheless returned `Ok(())`. The outer handshake then logged `PQC key exchange complete — hybrid session key derived`. That is not a completed shared-key exchange: the peer has not received the encapsulation ciphertext, decapsulated it, or confirmed possession of the same key.

The partial service path also treated PQC failure as non-fatal and continued using classical-only trust when the `pqc-handshake` feature was enabled. That allows an attacker who can reliably disrupt the PQC phase to induce a downgrade unless the caller has an explicit, authenticated policy permitting fallback.

**Fail-closed changes in this branch:**
- `run_pqc_exchange` now returns a specific error instead of performing a one-sided exchange whose key cannot be confirmed.
- If PQC establishment fails while the feature is enabled, `run_handshake_for_peer` removes the just-verified classical trust entry and returns the error rather than silently falling back. The `TrustChanged::Verified` event is emitted only after enabled handshake requirements pass.
- `PqcHandshakeManager` stores KEM-derived values in `pending_session_keys`; `session_key()` and `session_count()` now report established sessions only. There is currently no activation/key-confirmation API, so a candidate remains pending and is not exposed as an active session.
- `decapsulate_and_derive` returns `()`, not raw key bytes. A future protocol must implement canonical authenticated transcript binding and key confirmation before a session can be activated.
- The candidate timestamp is named `created_at`, not `established_at`, to avoid misrepresenting its lifecycle.

These are intentional API behavior changes: with `pqc-handshake` enabled, the service will now fail closed until the complete bidirectional exchange, transcript binding and key confirmation are implemented. That is safer than claiming a protection it does not provide; deployments requiring uninterrupted service must make an explicit policy decision before enabling the incomplete feature.

## K-7 — X25519 low-order input can produce a publicly known traffic key (mitigated in this branch)

The original `PeerKeyStore::agree` derived encryption/authentication keys from the X25519 shared secret without checking `SharedSecret::was_contributory()`. RFC 7748 §6.1 explains that a small-order input can produce an all-zero shared secret and describes checking for that result; §7 explains that non-contributory behavior can eliminate the other party's private-key contribution. The `x25519-dalek` API exposes `SharedSecret::was_contributory()`. References:
- RFC 7748: https://www.rfc-editor.org/rfc/rfc7748.html
- `x25519-dalek::SharedSecret` API: https://docs.rs/x25519-dalek/latest/x25519_dalek/struct.SharedSecret.html

Because the key derivation uses fixed HKDF-SHA-256 info `symthaea-mesh-chacha20-v1`, an all-zero DH result makes the encryption key reproducible from public source code and constants. If a malformed/untrusted public key reaches that original implementation, ciphertext under the derived key is not confidential.

**Fix now present in this branch:** `PeerKeyStore::agree` calls `was_contributory()` before KDF work or key-store mutation and returns a `TrustVerificationError` on non-contributory results. Its return type is now `Result<[u8; 32], SwarmError>`; `ContinuousMind::agree_with_peer` propagates rejection as `Result<Option<[u8; 32]>, SwarmError>`. A new regression test, `x25519_low_order_peer_keys_are_rejected_without_storing_keys`, requires an error and verifies neither encryption nor authentication keys are stored. Valid test callers now handle the fallible result.

This is a source-level mitigation and regression-test addition; it has **not** been compiled or run locally. The API change may require downstream caller updates. Also, this check does not authenticate the peer public key, bind identity/transcript, or provide forward secrecy; those remain separate requirements. Extend the regression corpus to known low-order encodings beyond all-zero and keep any unsupported raw public-key entrypoints fail-closed.

## Immediate disposition

- The X25519 store now rejects non-contributory shared secrets before deriving or storing keys; exact-head tests and review are still required.
- The service no longer presents the incomplete one-sided PQC exchange as complete and no longer silently falls back to classical-only when the enabled PQC phase fails.
- KEM outputs remain pending until transcript-bound key confirmation is implemented; the present manager cannot establish an active session.
- The existing X25519 helper should be described as **static pairwise key derivation**, not forward-secret key exchange.
- The public challenge nonce must be described as public transcript context, not a secret classical input.
- The KDF name and comments must match the implemented BLAKE3 derive-key function.
- The PQC helper is **research-only** until key establishment, transcript binding, identity authentication, downgrade behavior, session lifecycle and security claims receive independent review.
- Do not patch in ad-hoc ephemeral fields or invent an unaudited key combiner as a quick fix. Prefer a well-reviewed handshake framework or a protocol with a clear, reviewable security analysis.

## Protocol-v2/v3 direction

A replacement should define the threat model first, then choose a reviewed handshake and binding scheme. At minimum:

1. Authenticate peer identities and bind identity keys to both the X25519 and ML-KEM public keys.
2. Combine independent secret outputs only under a specified construction with a documented failure model; public nonces and signatures are not secret key material.
3. Bind the complete canonical transcript (roles, identities, algorithm choices, both key shares/public keys, KEM ciphertext, nonces and version) into the session-key schedule and confirmation messages.
4. Use fresh ephemeral contributions where forward secrecy is claimed, rotate KEM keys where appropriate, erase transient secret material, and define compromise/recovery behavior.
5. Derive separate direction-specific encryption and authentication keys with explicit domain separation. Do not rely on a truncated source identity to partition traffic keys.
6. Enforce state transitions in types/API so a session key cannot be created until required authentication and transcript checks pass.
7. Test active substitution, reflection, downgrade, replay, unknown/zero/invalid public inputs, duplicate session IDs, key rollback, static-key compromise, ciphertext capture, and mismatched transcript fields.

Relevant baselines: Noise patterns make authentication and forward-secrecy properties explicit (https://noiseprotocol.org/noise.html); RFC 7748 describes X25519 key agreement and KDF inputs (https://www.rfc-editor.org/rfc/rfc7748.html); RFC 9678 explicitly adds ephemeral DH for forward secrecy (https://www.rfc-editor.org/rfc/rfc9678.html).

## Benchmark targets

These are proposed gates, not achieved results.

| Metric | Target | Evidence |
|---|---:|---|
| Historical key recovery under claimed FS | 0 | Recorded transcript cannot reconstruct prior traffic keys after long-term compromise under the defined threat model |
| Identity/key binding | 100% of negative cases rejected | Mismatched peer ID, public key and transcript tests fail closed |
| X25519 contributory behavior | 100% of known low-order inputs rejected | All-zero/low-order test vectors return an error and insert no traffic keys |
| Transcript binding | 100% of single-field mutations rejected | Independent implementation/test vectors |
| KEM ciphertext/public-key substitution | 100% of seeded active MITM cases prevented under stated auth assumptions | Executable protocol traces |
| Downgrade and role/reflection cases | 100% rejected | Frozen negative suite |
| Secret lifecycle | Zero retained session keys after expiry/removal | Tests and memory-handling review, with limitations stated |
| Positive compatibility | 100% | Canonical handshake vectors and interop cases |
| Cost | p50/p95/p99 handshake latency, bytes, CPU, memory | Matched hardware and toolchain; compare only after security gates pass |

A model discovery counts only when the issue is independently reproducible against an exact revision, the security notion was specified before scoring, and a fix rejects the witness while positive controls continue to pass. The new static-DH witnesses belong to calibration, not novel cryptanalysis.

## Validation status

Source review and regression-test authoring only. Tests were not compiled or run in this environment. The regression/limitation tests are `legacy_static_x25519_agreement_repeats_key_after_long_term_secret_reuse`, `legacy_static_x25519_derivation_does_not_bind_peer_source_id`, `x25519_low_order_peer_keys_are_rejected_without_storing_keys`, `legacy_pqc_candidate_key_derives_without_verified_classical_handshake_state`, and `legacy_pqc_session_derivation_does_not_bind_peer_id`; the implicit-rejection test is `test_invalid_ciphertext_uses_implicit_rejection_and_needs_key_confirmation`. Additional tests now ensure KEM outputs remain pending rather than active until confirmation exists. CI must provide exact-head build evidence, and independent cryptographic review remains required. No claim is made that the current handshake is deployed or has been exploited.
