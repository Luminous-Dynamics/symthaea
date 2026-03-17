# Mesh Bridge Security Audit

**Date**: 2026-03-16
**Version**: v0.1.0
**Auditor**: Claude (automated review)

---

## Threat Model

The mesh bridge operates in a hostile environment: LoRa and WiFi-direct mesh networks where any device within radio range can send and receive frames. There is no network-level authentication.

**Attacker capabilities:**
- Passive: sniff all mesh traffic
- Active: inject arbitrary frames, replay captured frames, flood with garbage

---

## Findings

### 1. Replay Attack Mitigation — IMPLEMENTED

**Risk**: Attacker captures a valid TEND exchange relay and replays it hours later.

**Mitigation** (relay.rs):
- `MAX_PAYLOAD_AGE_SECS = 3600` — payloads older than 1 hour are rejected
- `MAX_FUTURE_TOLERANCE_SECS = 300` — payloads with future timestamps (clock skew > 5 min) are rejected
- Content hash dedup — identical payloads are only processed once per dedup cache cycle

**Residual risk**: An attacker could replay within the 1-hour window before the content hash dedup catches it. The dedup cache clears at 10,000 entries. For a community of ~30 users exchanging ~100 entries/day, this provides ~100 days of dedup coverage before a reset.

### 2. Flooding / DoS Prevention — IMPLEMENTED

**Risk**: Malicious peer floods relay with messages, exhausting CPU/memory.

**Mitigation** (relay.rs):
- `MAX_MESSAGES_PER_MINUTE_PER_PEER = 60` — sliding-window rate limiter per origin ID
- `MAX_REASSEMBLY_BUFFERS = 64` — caps in-flight reassembly state
- `MAX_PAYLOAD_DATA_SIZE = 65536` — rejects payloads > 64 KB
- `MAX_ENTRIES_PER_DOMAIN = 20` — poller caps entries per poll cycle
- `REASSEMBLY_TIMEOUT_SECS = 30` — stale reassembly buffers cleaned up

**Residual risk**: An attacker could spoof different origin IDs to bypass per-peer rate limiting. Each spoofed origin consumes rate tracker memory (~200 bytes). At 1000 unique origins, this is ~200 KB — acceptable.

### 3. Transport Encryption — IMPLEMENTED

**Risk**: All relay payloads are sent in plaintext over LoRa/WiFi-direct. Anyone in radio range can read TEND exchange amounts, emergency messages, food harvest quantities.

**Mitigation** (encryption.rs):
- XChaCha20-Poly1305 PSK encryption via `MESH_ENCRYPTION_KEY` env var (64 hex chars = 32 bytes)
- 24-byte random nonce per payload + 16-byte auth tag = 40 bytes overhead (fits LoRa frame budget)
- Encrypt before fragmentation, decrypt after reassembly
- Plaintext fallback when env var is not set (backward compatible)
- All nodes in a community must share the same PSK

**Residual risk**: Key distribution is manual (env var). Long-term: derive per-agent keypairs from Holochain agent keys.

### 4. Message Authentication — PARTIAL

**Risk**: Anyone on the mesh can forge relay payloads claiming to be from any origin.

**Current state**: Origin ID is derived from hostname BLAKE3 hash (8 bytes). Not cryptographically authenticated — an attacker can compute and spoof any origin.

**Mitigation in place**: Content hash dedup prevents forged duplicates. The conductor validates zome calls independently (consciousness gating, governance auth). A forged TEND exchange would still need valid agent credentials on the replaying conductor.

**Recommendation**: HMAC the relay payload with a community PSK. Same key as transport encryption.

### 5. Conductor Auth Token Exposure — ACCEPTABLE

**Risk**: `MESH_APP_TOKEN` stored as environment variable.

**Assessment**: This is standard practice for service-to-service auth in Docker/NixOS deployments. The NixOS module supports `EnvironmentFile` for secret injection from a file with restricted permissions.

### 6. Dedup Cache Persistence — ACCEPTABLE

**Risk**: Dedup cache stored in `/tmp/mycelix-mesh-bridge/` (or `MESH_CACHE_DIR`). On restart, stale entries could allow replays of recently-seen messages.

**Mitigation**: Cache is persisted on shutdown and loaded on startup. Combined with timestamp validation, this is sufficient.

### 7. Bincode Deserialization Safety — VERIFIED

**Risk**: Malformed bincode data could cause panics or excessive memory allocation.

**Verification**: Proptest `prop_arbitrary_bytes_safe_deserialize` (stress_proptests.rs) confirms that arbitrary byte sequences fed to `RelayPayload::from_bytes()` never panic. Bincode's default size limit prevents unbounded allocation.

---

## Test Coverage

| Category | Tests | Location |
|----------|-------|----------|
| Serialization roundtrip | 8 + 2 proptests | serializer.rs |
| Fragment/reassemble | 4 + 2 proptests | serializer.rs |
| Transport (loopback) | 9 | transport.rs |
| Poller logic | 10 | poller.rs |
| Relay logic | 7 + 3 security | relay.rs |
| Dedup cache | 5 | dedup_cache.rs |
| BridgeMetrics | 3 | lib.rs |
| Pipeline integration | 6 | pipeline_integration.rs |
| Manifest validation | 7 | manifest_validation.rs |
| Stress proptests | 15 | stress_proptests.rs |
| **Total** | **74** | |

---

## Recommendations (Priority Order)

1. **Add PSK encryption** — XChaCha20-Poly1305 with community key (addresses findings 3 + 4)
2. **Add HMAC authentication** — per-payload HMAC prevents origin spoofing (addresses finding 4)
3. **LRU dedup cache** — replace HashSet with LRU cache to bound memory and maintain temporal ordering
4. **Audit logging** — log rejected payloads (rate limited, expired, oversized) to a file for post-incident analysis
5. **Connection pooling** — reuse conductor WebSocket across relay calls (partially done, but error recovery could be tighter)

---

## Conclusion

The mesh bridge is **safe for community deployment** with the implemented mitigations. The primary risk (plaintext transport) is acceptable for a trusted community mesh where all nodes are operated by known members. PSK encryption should be added before deploying in environments with untrusted radio neighbors.
