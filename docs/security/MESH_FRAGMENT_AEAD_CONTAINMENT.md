# Mesh Fragment-AEAD Fail-Closed Containment

**Status:** containment patch; exact-head validation pending  
**Scope:** `DualLayerMesh`, `MeshTransport`, and `MeshReceiver` fragmented LoRa path  
**Related:** #8376 (nonce/MTU audit), #8412 (key-agreement claim review), #8471 (nonce lifecycle model), #8491 (handshake admission model)

## Source-grounded issue

The current LoRa sender already encrypts the compressed packet envelope before fragmentation when a packet key is configured. An optional second AEAD wrapper around each LoRa fragment creates another nonce namespace and adds 28 bytes (12-byte nonce + 16-byte tag) to each serialized frame. The declared `LORA_MTU` and fixed receive buffer are 222 bytes; wrapping a full-size 222-byte fragment creates a 250-byte frame.

There is also a source-identity integration hole: `MeshTransport::recv_raw` only returns bytes, but the fragment decryptor expects a source identity that matches the identity embedded in the AEAD nonce. `DualLayerMesh::poll_incoming` previously passed `[0; 8]` as a placeholder. Moreover, sender and receiver configuration could silently fall back to cleartext when fragment encryption was requested without a key.

These are source/API findings. They do not establish that a particular deployed radio truncates frames, that a deployment enables this mode, or that a packet was exploited.

## Containment in this change

- Add `MeshError::SecurityProfileUnavailable` and `MeshError::TransportMtuExceeded` so an unavailable security profile or oversized serialized frame is explicit.
- Make high-level LoRa sending reject the fragment-AEAD mode before any frame is emitted. This guard also rejects the mode if the encryption key is absent; it does not send cleartext.
- Serialize and preflight the complete fragment set against the selected transport's actual `mtu()` before sending the first frame. A small/incorrectly configured transport cannot receive a partial packet before an oversize error.
- Keep the existing packet-envelope AEAD path available when configured, with fragment encryption disabled.
- Add an optional `MeshTransport::recv_raw_with_source` method whose default fails closed. An adapter may implement it only when it can provide a transport-authenticated identity bound to the mesh peer. Copying the nonce's source prefix or trusting an unauthenticated radio address is insufficient.
- When a receiver is configured to require fragment AEAD, absence of its key rejects the frame and increments the decrypt-failure counter instead of parsing plaintext.
- Count an unavailable authenticated-source adapter in receiver failure telemetry.

No wire format or cryptographic primitive is changed. This is intentionally a narrow containment patch, not a production v3 design and not proof that the current overall mesh protocol is secure.

## Why not simply enlarge the buffer?

Increasing the receive buffer cannot make a 250-byte payload fit a physical 222-byte MTU. A proper replacement must be versioned and must jointly define payload capacity, FEC behavior, authenticated headers, reassembly bounds, replay rules, transport-source identity, and interoperability. Packet-envelope AEAD may already provide the required end-to-end payload integrity; any additional per-fragment wrapper needs a distinct, documented security requirement that justifies its cost.

## Release gates for re-enabling fragment AEAD

- [ ] A versioned framing proposal accounts for every byte under the actual transport MTU; sending preflights all frames and never emits a partial packet on an MTU violation.
- [ ] The chosen transport API supplies an authenticated source identity, not a nonce-derived or unauthenticated label.
- [ ] Both source and receiver fail closed on missing key, unsupported identity binding, oversized output, malformed frames, and wrong peer/context.
- [ ] Repeated/reordered/replayed fragments have an explicit state-machine policy and bounded allocations.
- [ ] Positive controls prove valid encrypted packet traffic remains interoperable while the second wrapper is disabled.
- [ ] Resource measurements report bytes, fragments, retries, p50/p95/p99 latency, energy/airtime, peak reassembly memory and recovery.
- [ ] Exact-head Rust tests/feature matrix and an independent cryptographic review are recorded.

The mandatory security objective remains zero duplicate effective-key/nonce pairs and zero accepted unauthorized cleartext fallback. Those correctness gates precede any comparison of speed or radio cost.
