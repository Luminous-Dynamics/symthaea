# Mesh Protocol v3: Assurance Plan and State-Machine Boundary

**Status:** research proposal; not a protocol specification or production approval  
**Date:** 2026-10-22  
**Related reviews:** PR #8376 (AEAD nonce lifecycle), PR #8412 (key-agreement claim integrity), PR #8365 (crypto calibration runner)  
**Model:** `docs/security/models/MESH_NONCE_LIFECYCLE.tla`

## Purpose

Convert the nonce and key-agreement findings into a protocol assurance boundary that can be tested independently of Rust CI. The design goal is not a novel cipher or a bespoke hybrid combiner. It is to compose established cryptography with a small, versioned state machine whose security claims have explicit owners, tests, and release gates.

This plan intentionally does **not** claim that the model has been parsed or model-checked. The model is a proposed executable artifact. A TLC run on the exact files and configuration is required before reporting bounded model evidence.

## Decision proposal

1. **Do not enable a production v3 profile until the protocol state machine and wire format are frozen and independently reviewed.**
2. **Prefer a maintained, established secure-channel implementation over a custom handshake.** Where transport and dependencies permit, evaluate TLS 1.3 with the standardized `X25519MLKEM768` group defined for TLS in RFC 10024 and the hybrid framework in RFC 9954. Verify the actually negotiated group in an integration test; source configuration or a requested feature is not evidence of negotiation.
3. **Do not copy TLS's key schedule or hybrid group into a bespoke raw-mesh handshake by analogy.** RFC 10024 specifies TLS 1.3 group behavior, not a generic standalone mesh protocol.
4. **Treat LoRa as a separate transport profile.** Quantify the authenticated handshake, fragment overhead, retries, airtime, energy, maximum frame size and recovery before choosing a suite. If a reviewed interoperable construction does not fit the deployment, defer that security profile rather than claim post-quantum protection from a placeholder backend or a custom combiner.
5. **A real ML-KEM backend is a hard prerequisite for any PQ session claim.** The all-zero compile stub must remain unable to establish or publish a session. Check the current FIPS 203 publication and errata, run official known-answer tests, and compare against an independently maintained implementation before approval.
6. **No established session before mutual confirmation.** Candidate key material stays inaccessible to application traffic until the peer identity, protocol version, selected suite, ephemeral classical exchange, KEM exchange (where used), full transcript and key confirmation have been validated in both directions.


## A standards-first transport decision tree

Do not choose one protocol stack for every physical link. Key establishment and record protection should be separated from the radio framing profile, and the selected profile must be observed at runtime rather than inferred from build flags.

| Deployment profile | First candidate to evaluate | Explicit limitation |
|---|---|---|
| IP transport with mature TLS 1.3 support | A maintained TLS 1.3 provider that can negotiate RFC 10024's `X25519MLKEM768` group, where the provider and policy support it | Provider configuration is not proof of negotiated group; capture and assert the actual negotiated group in integration tests. |
| Constrained CoAP / low-power network | EDHOC (RFC 9528) to establish an authenticated ephemeral-DH context, then OSCORE (RFC 8613; integration details in RFC 9668) for end-to-end object protection | The standardized EDHOC profile is not automatically post-quantum. Do not label it a hybrid/PQ profile unless a specified, reviewed extension and real implementation are selected. |
| Raw mesh / LoRa with the project's 222-byte frame contract | First test whether the existing link can carry a standard authenticated session profile through a gateway or compact transport binding | Large KEM shares/ciphertexts can require many fragments; measure bytes, retransmissions, airtime, energy, and loss recovery before choosing. |
| LoRa profile that cannot support the required security properties within resource/MTU limits | Fail closed / do not enable that profile until a reviewed option exists | Do not invent a bespoke hybrid combiner or falsely claim post-quantum confidentiality to satisfy a product label. |

EDHOC is specifically designed for constrained scenarios and identifies LoRaWAN as a relevant low-power-network use case; it uses established COSE/CBOR constructions and ephemeral authenticated Diffie–Hellman. OSCORE provides end-to-end protection for constrained REST interactions and includes replay-related context semantics. These are relevant design baselines, not drop-in replacements for Symthaea's current raw mesh framing. See [RFC 9528](https://www.rfc-editor.org/rfc/rfc9528.html), [RFC 8613](https://www.rfc-editor.org/rfc/rfc8613.html), and [RFC 9668](https://www.rfc-editor.org/rfc/rfc9668.html).

This split surfaces the actual engineering trade: a compact classical authenticated handshake may be viable on a low-power link while a post-quantum hybrid exchange may incur substantial fragmentation and retransmission cost. The right response is to report the trade-off and require an explicit policy decision—not to silently downgrade the cryptographic property or claim that protocol crypto-agility implies an available PQ suite.

## Restart-epoch collision experiment

The TLA+ model now has an explicit nondeterministic restart operation, RestartWithEpoch(sender, epoch), in addition to the deterministic default-epoch restart. The paired packet_restart_epoch_collision_legacy.cfg and packet_restart_epoch_collision_strict.cfg configurations explore whether an epoch previously used by the sender can be selected again after a counter reset. The legacy configuration must produce a NoNonceReuse counterexample; the strict configuration must preserve that invariant by refusing a repeated complete (effective key, nonce) pair.

This is a possibility proof, not a probability estimate. TLC's nondeterministic choice explores collisions but does not model a cryptographic random generator or show that epochs are independently uniform. The 1/256 collision probability applies only when two epoch draws are independent and uniform over 256 values. Even that idealized probability is far too weak to treat an 8-bit epoch as durable nonce-uniqueness protection.

## Threat model and assumptions

The protected properties are claim-specific and not interchangeable:

- **Confidentiality:** ciphertext is protected under a unique effective key/nonce pair.
- **Integrity:** ciphertext and all security-relevant associated data are authenticated.
- **Peer authentication:** the authenticated key or credential is bound to the claimed full peer identity.
- **Forward secrecy:** later compromise of long-term authentication keys does not reconstruct past session keys, under the protocol's stated assumptions.
- **Hybrid confidentiality:** session derivation obtains independent classical and ML-KEM shared secrets through a reviewed protocol construction; a public challenge or peer ID is not a secret input.
- **Post-quantum status:** the selected protocol and implementation actually use a supported, real ML-KEM implementation; merely compiling a feature or carrying ciphertext bytes is insufficient.
- **Replay protection:** duplicates, old epochs, sequence rollback and unauthorized peer replays cannot cause accepted application effects.
- **Availability/resource bounds:** malformed handshakes and fragments cannot grow per-peer state, allocation or work without explicit limits.
- **Interoperability:** independently built implementations agree on canonical bytes, transcript hash, key schedule, packet encoding and error behavior.

Assumptions must say whether peers share one group secret or have peer-specific keys. With a shared group secret, a member that holds that secret can generally generate valid group-authenticated traffic as another member; a nonce layout alone does not provide individual origin authentication. Key IDs in logs or receipts must be non-secret identifiers for the effective key epoch, not raw secret material.

## Required state machine

The production design should make the following states explicit rather than representing every stage as an optional key in a map:

```text
Disabled
  -> Negotiating
  -> CandidateKeys
  -> PeerAuthenticated
  -> LocallyConfirmed
  -> Established
  -> Closing / Closed
```

All error paths lead to a non-established state and erase transient candidates. The exact ordering may differ by the chosen standard implementation, but these invariants may not:

- `Established` implies an authenticated peer identity, an allowed protocol version/suite, transcript binding, required key confirmation, and a real cryptographic backend.
- A KEM encapsulation/decapsulation result alone does not authenticate the peer or establish a mutually confirmed session.
- A candidate key is never returned by the application-facing session-key API.
- A required PQ/hybrid exchange failure cannot silently become a classical-only session.
- A downgrade is accepted only where an explicit, authenticated policy permits the named lower profile; the chosen profile is bound to the handshake transcript.
- Session teardown, key rotation, timeout and peer revocation invalidate candidate state and prevent its later promotion.
- Classical ephemeral private material and KEM transient secrets are erased according to the selected library's documented guarantees; source-level `drop` alone is not proof of memory zeroization.

## Nonce and record contract

For every AEAD invocation, record the abstract pair `(effective_key_epoch_id, nonce)`. The invariant applies to the **effective key**, not just the bytes of a nonce or the outer packet identity.

- A key epoch identifies one actual derived traffic-key namespace. Epoch IDs are authenticated metadata, not entropy or secret material.
- Use the selected protocol/library's defined key schedule and nonce construction. Do not improvise a truncated sender prefix, small random epoch, wrap-prone thought ID or cross-layer nonce format.
- Counters are unique within the effective key and direction. Rotate to a genuinely new key namespace or stop before wrap.
- If durable counter reservation is used, reserve before encrypting; losing a reserved value is acceptable, reusing it is not.
- Persistent counters alone do not solve snapshot cloning or storage rollback. Either use genuinely fresh session keys after restart, trusted anti-rollback/uniqueness support, or fail closed when the uniqueness precondition cannot be established.
- End-to-end immutable header fields are authenticated as associated data. Mutable relay fields (for example hop count) must be separate and explicitly authenticated at the layer responsible for them.
- Replay acceptance is tracked per authenticated session/key epoch with bounded windows; unauthenticated epochs do not cause unbounded state allocation.

The extra fragment AEAD wrapper must justify its additional 28-byte overhead. If packet-envelope AEAD already gives end-to-end confidentiality and integrity before fragmentation, prefer one encryption layer plus authenticated fragmentation metadata unless a separate hop-by-hop security requirement proves otherwise. If the wrapper remains, the v3 fragment payload capacity, FEC, reassembly limits and transport MTU must all account for it. A 222-byte declared frame cannot safely contain a 250-byte ciphertext merely because a sender API accepts a larger buffer.

## Model and executable evidence

The TLA+ model is deliberately bounded. It isolates three legacy mechanisms:

1. packet nonce collision when distinct source IDs map to the same truncated prefix and the same effective group key;
2. deterministic restart resetting sequence and epoch to their default while retaining the same effective key;
3. fragment nonce collision when the thought-ID projection repeats while the same effective key and fragment index are reused.

The fragment model uses a tiny `FragmentModulus` so a wrap counterexample can be explored in a small state space. It is an abstraction of the real 16-bit field, not a literal model of production byte encoding. The strict configurations model a fail-closed specification oracle: they prevent recording a duplicate pair. They do **not** establish that the Rust implementation obeys the oracle.

### TLC commands

Obtain the official TLA+ tools and use a pinned version of `tla2tools.jar` with Java 11 or newer. From the repository root, run the parser first, then each configuration:

```sh
java -cp /path/to/tla2tools.jar tla2sany.SANY \
  docs/security/models/MESH_NONCE_LIFECYCLE.tla

for cfg in docs/security/models/*.cfg; do
  java -cp /path/to/tla2tools.jar tlc2.TLC \
    -config "$cfg" docs/security/models/MESH_NONCE_LIFECYCLE.tla
done
```

The three `*_legacy.cfg` files are **expected to find a counterexample** for `NoNonceReuse`. The three `*_strict.cfg` files are expected to finish the bounded exploration without violating `NoNonceReuse`. Preserve TLC version, exact model/config digests, command, exit status, state count, invariant result and counterexample trace. Do not call any of those outcomes a universal proof or Rust implementation proof.

## Mapping model evidence to code

Before model results can qualify implementation behavior, add a trace adapter or equivalent deterministic conformance harness that maps each model action to a real boundary:

| Model action/property | Concrete conformance target |
|---|---|
| `Send` | the exact encrypt-and-reserve path, not a detached nonce helper |
| `Restart` | process recreation with the same configured keys and any durable state restored or lost |
| `RotateEpoch` | actual key schedule / key rotation behavior and receiver transition rules |
| `usedPairs` | a test-only, non-secret effective-key identity plus emitted nonce capture |
| `NoNonceReuse` | zero duplicate effective-key/nonce tuples across generated traces |
| strict refusal | no ciphertext, session advancement or application effect when uniqueness cannot be established |

The test-only key identifier must be collision-resistant for the test's bookkeeping purpose or be a unique internal test label. It must never be used as an operational substitute for the actual key bytes or key schedule.

Required trace dimensions include crash before/after reservation and send, restored snapshot, cloned sender state, concurrent peers sharing a key, counter boundary, epoch rollback, rotation grace period, duplicate fragment, dropped/reordered/replayed packets, malformed lengths, peer identity substitution, suite downgrade and candidate handshake timeout.

## Benchmark program

Targets are proposed acceptance criteria, not achieved results:

| Metric | Target / reporting rule |
|---|---|
| Pre-registered nonce/restart/fragment counterexamples | 100% reproduced by legacy fixtures and rejected by the v3 implementation |
| Valid positive/interoperability vectors | 100% accepted by the intended profile |
| Duplicate effective-key/nonce tuples | 0 in every tested trace and captured run |
| Wrong peer identity / transcript mutation / downgrade | 100% rejection of seeded negative controls |
| Established sessions on stub/missing required backend | 0 |
| Established sessions before required key confirmation | 0 |
| Valid-message false rejection | Report separately; target 0 for the frozen corpus |
| State/resource bound violations | 0 beyond explicitly permitted limit behavior |
| Operational profile cost | Report wire bytes, fragments, retries, p50/p95/p99 latency, peak RAM, energy per handshake, airtime, reconnect and recovery time |

For a fair state-of-the-art comparison, freeze a held-out attack corpus and run the same scenarios against (a) conventional unit/property tests, (b) a model-only workflow, (c) one code-reasoning model, (d) a tool-using single agent, and (e) independent generator/evaluator agents with the same time/token/tool budget. Report seeded fault recall, false accepted findings, false alarms, valid-control retention, minimized counterexample rate, human triage time and cost per independently accepted finding. Keep known attacks separate from novel discoveries. A higher raw attack count is not a win if the pipeline generates false findings or breaks valid traffic.

## Release gates

- [ ] Wire format, key model, threat model, profile negotiation, restart/rollback behavior and compatibility policy are frozen in a separate ADR.
- [ ] TLA+ parser and TLC runs are captured with exact artifact/tool identities.
- [ ] Each legacy counterexample maps to a regression in the real implementation.
- [ ] The v3 implementation rejects the old trace while preserving every frozen positive vector.
- [ ] Actual library negotiation/backend identity is observed at runtime in integration tests.
- [ ] Official KATs and independent implementation vectors pass for the real ML-KEM backend.
- [ ] Fuzz/property tests cover parser, fragmentation, replay, bounded allocations and state transitions.
- [ ] Cross-version and transport interoperability is demonstrated on the intended profiles.
- [ ] Independent cryptographic review approves the protocol composition and threat model.
- [ ] Exact-head CI, including format, test, feature-matrix and model checks, is green.
- [ ] Claims stay at the evidence level actually achieved: model check is not implementation proof; test pass is not cryptographic proof; implementation proof is not full protocol proof.

## References

- RFC 8439 — ChaCha20 and Poly1305 for IETF Protocols: https://www.rfc-editor.org/rfc/rfc8439.html
- RFC 5116 — An Interface and Algorithms for Authenticated Encryption: https://www.rfc-editor.org/rfc/rfc5116.html
- RFC 9954 — Hybrid Key Exchange in TLS 1.3: https://www.rfc-editor.org/rfc/rfc9954.html
- RFC 10024 — PQ/T Hybrid Key Agreement Mechanisms for TLS 1.3: https://www.rfc-editor.org/rfc/rfc10024.html
- NIST FIPS 203 — Module-Lattice-Based Key-Encapsulation Mechanism Standard (check live errata/revisions): https://csrc.nist.gov/pubs/fips/203/final
- TLA+ tools and command-line model checking: https://github.com/tlaplus/tlaplus/blob/master/USE.md
