# Threat Model — `mycelix-lawful-identity`

This document is the honest contract. It describes **what the dual-DID architecture mathematically neutralizes** and **what it explicitly does not cover**. Users must understand both before relying on the airlock.

---

## Adversary profile

A well-funded nation-state actor (intelligence agency, tax authority, law-enforcement unit) with:

- Full read-access to the Mycelix Holochain DHT.
- ISP-level network surveillance.
- A known `did:mycelix:legal` for the target (obtained via the target's tax filing, fiat KYC, or court-compelled disclosure).

**Adversary's goal**: link the known `legal` DID to the target's pseudonymous `did:mycelix:primary`, thereby deanonymizing their governance votes, MYCEL reputation, TEND balances, social graph, and published content.

---

## Vector 1 — Cryptographic leakage in the ZKP — **MITIGATED**

**Threat**: a flawed proof construction leaks a deterministic hash, signature fragment, or public-key material of the `legal` DID into proof bytes or public inputs. Observer correlates the leaked material against known `legal` DIDs.

**Mitigation**:
- STARK public inputs are constrained to the minimal set: `{issuer_public_key, claim_predicate, fresh_nonce}`.
- No `legal` DID string, key bytes, or derived hash appears anywhere in the proof payload.
- Merkle commitments use salts unique per proof.

**Test guarantee** (`tests/link_resistance.rs`):
1. Generate 1,000 cross-DID proofs from the same `legal` DID.
2. Assert the `legal` DID string is absent from every proof byte-array.
3. Assert STARK public inputs contain only the three permitted fields.
4. Brute-force correlate public inputs against every record in a 100K-entry synthetic DHT; assert zero matches to the `legal` DID's keypair.

---

## Vector 2 — Replay / correlation — **MITIGATED**

**Threat**: repeated proofs from the same `legal` DID are byte-identical, or timestamps cluster around known `legal` DID actions, allowing an observer to link sessions.

**Mitigation**:
- Every cross-DID proof requires a **fresh nonce provided by the verifier** at request time.
- Verifiers MUST reject any proof whose nonce has been seen before (rolling window of 65,536 most-recent nonces, eviction by LRU).
- Prover includes the nonce in STARK public inputs, making each proof byte-distinct even for identical claims.

**Test guarantee** (`tests/unlinkability.rs`):
1. Same `legal` DID generates two proofs of the same claim to the same verifier with different nonces.
2. Assert proof byte-arrays are distinct.
3. Run a statistical indistinguishability test (chi-squared over byte distribution); assert p > 0.05.
4. Attempt nonce reuse; assert verifier rejects.

---

## Vector 3 — Network metadata (IP correlation) — **NOT MITIGATED**

**Threat**: `primary` and `legal` DIDs run on the same host. They gossip to the DHT from the same IP address. ISP-level observer correlates the two gossip streams trivially.

**Scope statement**: this architecture provides *on-chain cryptographic unlinkability*, not network-layer anonymity.

**User responsibility** (surfaced in CLI first-run onboarding):
- For true unlinkability, run the `legal` DID on a separate physical agent.
- Or route the `legal` DID's gossip through Tor / I2P.
- Or accept that ISP-level adversaries can link the two if they are motivated.

**Mitigation path (future, not in scope)**: integrate tor-rs or libp2p-onion into the lawful-identity conductor config.

---

## Vector 4 — Device compulsion / rubber-hose — **NOT MITIGATED**

**Threat**: law enforcement physically seizes the user's device and compels them to unlock it. Both DID keypairs live in the local keystore. The link is immediately apparent.

**Scope statement**: protection is against mass *passive* on-chain surveillance, not against *targeted endpoint compromise*.

**User responsibility**:
- Recognize the jurisdictional risk of keeping both DIDs on one device.
- Optionally split the keystore across devices.

**Mitigation paths (future, not in scope)**:
- Split keystore across devices (legal DID on a dedicated air-gapped signer).
- Hardware-TEE-bound keys (Android StrongBox, iOS Secure Enclave) that refuse export.
- Duress-wipe: a secondary passphrase that silently destroys the `primary` keystore while appearing to unlock.

---

## What the link-resistance test proves, and what it does not

**Does prove**: Vectors 1 and 2 are mathematically neutralized. No amount of DHT observation or proof collection allows an adversary to link `primary` to `legal`.

**Does not prove**: Vectors 3 and 4. An adversary with network-level surveillance or physical device access can still link the DIDs.

**CLI onboarding (`cli init`) surfaces both honest caveats in plain language** before generating the `legal` keypair. Burying the caveats in this doc is not enough.

---

## Review cadence

This threat model is a living document. Revise when:
- A new adversary capability is demonstrated (e.g., quantum attack on STARK, traffic-analysis attack on Holochain gossip).
- The mitigation set changes (e.g., Tor integration lands, making Vector 3 *partially* mitigated).
- A production incident reveals an assumption was wrong.

Last updated: 2026-04-18 (initial).
