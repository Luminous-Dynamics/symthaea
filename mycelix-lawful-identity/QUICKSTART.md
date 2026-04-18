# QUICKSTART — mycelix-lawful-identity

End-to-end recipe: from a fresh checkout to a live legal DID on the shared Mycelix ecosystem conductor.

For the architectural rationale see [`../MYCELIX_STATE_COEXISTENCE.md`](../MYCELIX_STATE_COEXISTENCE.md). For the threat model see [`docs/THREAT_MODEL.md`](./docs/THREAT_MODEL.md). For claim-key conventions see [`docs/GOV_ID_CLAIM_SHAPES.md`](./docs/GOV_ID_CLAIM_SHAPES.md).

---

## 0. Prerequisites

- A running **Mycelix ecosystem conductor** on `ws://127.0.0.1:33800` (admin) and `ws://127.0.0.1:8888` (app). Verify with `ss -ltn | grep -E '33800|8888'`. If not running, see [`mycelix-workspace/CLAUDE.md`](../mycelix-workspace/CLAUDE.md).
- `hc` CLI available — enter via `nix develop /srv/luminous-dynamics/mycelix-praxis` which provisions `hc`, `holochain`, `lair-keystore`.
- Rust toolchain with `wasm32-unknown-unknown` target.

---

## 1. Build the WASM zomes

```bash
cd /srv/luminous-dynamics/mycelix-lawful-identity
cargo build --workspace --target wasm32-unknown-unknown --release
```

Produces six `.wasm` bundles under `target/wasm32-unknown-unknown/release/`:

- `legal_did_integrity.wasm`, `legal_did.wasm`
- `issuer_trust_tier_integrity.wasm`, `issuer_trust_tier.wasm`
- `cross_did_zkp_integrity.wasm`, `cross_did_zkp.wasm`

Total ~10.8 MB.

---

## 2. Pack DNA + hApp

```bash
nix develop /srv/luminous-dynamics/mycelix-praxis --command bash -c "
  cd /srv/luminous-dynamics/mycelix-lawful-identity
  hc dna pack dna/ -o dna/mycelix_lawful_identity.dna
  hc app pack . -o mycelix-lawful-identity.happ
"
```

Produces:
- `dna/mycelix_lawful_identity.dna` (~1.95 MB)
- `mycelix-lawful-identity.happ` (~1.95 MB)

Both are gitignored by design.

---

## 3. Install + enable on the shared conductor

```bash
nix develop /srv/luminous-dynamics/mycelix-praxis --command bash -c "
  hc sandbox call --running=33800 install-app \
    /srv/luminous-dynamics/mycelix-lawful-identity/mycelix-lawful-identity.happ \
    --app-id mycelix-lawful-identity
  hc sandbox call --running=33800 enable-app mycelix-lawful-identity
"
```

You should see `{"installed_app_id":"mycelix-lawful-identity",...}` followed by `Enabled app: "mycelix-lawful-identity"`.

---

## 4. Build the CLI with live conductor support

The default CLI build is lightweight (call-sheet only). For real zome round-trips, enable the `conductor` feature:

```bash
cd /srv/luminous-dynamics/mycelix-lawful-identity/cli
cargo build --features conductor
# First build: ~5 minutes, pulls ~200 MB transitive deps.
```

Binary lands at `cli/target/debug/lawful-id`.

---

## 5. First-run disclosure

The CLI refuses to do anything until the user has acknowledged the threat model. Run:

```bash
./target/debug/lawful-id init
```

Read the four-vector disclosure (on-chain unlinkability mitigated, network metadata + device compulsion NOT mitigated), then type `ack` to acknowledge.

For automation, bypass the pause with `--no-pause`:

```bash
./target/debug/lawful-id --no-pause init
```

---

## 6. Full user flow

```bash
# Create a legal DID.
./target/debug/lawful-id new-legal-did --live --label "SA passport"
# → Created legal DID: did:mycelix:legal:<64-hex>

# Classify the issuer (Sovereign for a state ID issuer).
./target/debug/lawful-id classify-issuer did:web:home.affairs.gov.za \
    --tier sovereign --rationale "SA Dept of Home Affairs" --live

# Attach a credential to the legal DID. The credential body never
# touches the DHT — only a hash commitment + issuer pointer.
./target/debug/lawful-id import-credential --live \
    --legal-did "did:mycelix:legal:<64-hex>" \
    --credential-hash "blake3:<64-hex>" \
    --issuer-did "did:web:home.affairs.gov.za" \
    --credential-type "SaIdCredential" \
    --issued-at "2022-05-10" \
    --expires-at "2032-05-10"

# Verify.
./target/debug/lawful-id list-credentials "did:mycelix:legal:<64-hex>"
./target/debug/lawful-id lookup-tier did:web:home.affairs.gov.za

# Produce a nonce for a cross-DID proof session.
./target/debug/lawful-id request-nonce did:web:home.affairs.gov.za
```

---

## 7. Known gotchas

### `holochain_client` version must match the conductor

The running ecosystem conductor is Holochain 0.6.x. Older `holochain_client = "0.7"` with `holochain_types = "0.5"` fails on the 0.6-era `Enabled` app-status variant with:

```
Deserialize("unknown variant `enabled`, expected one of `paused`, `disabled`, `running`, `awaiting_memproofs`")
```

The CLI's `Cargo.toml` pins `holochain_client = "0.9.0-dev.20"` and `holochain_types = "0.7.0-dev.19"` which track Holochain 0.6. If you see this error in another mycelix workspace crate, bump the same way.

### `SignZomeCallError("Provenance not found")`

After calling `admin_ws.authorize_signing_credentials(...)`, the returned `SigningCredentials` MUST be registered back into the `ClientAgentSigner` via:

```rust
signer.add_credentials(cell_id, credentials);
```

…BEFORE any `app_ws.call_zome(...)` will succeed. This is easy to miss — `authorize_signing_credentials` only grants the capability on the conductor side; the client-side signer still needs to know how to produce the matching signature. See `cli/src/live.rs::LiveConductor::connect` for the reference pattern.

### URL format

`AdminWebsocket::connect` and `AppWebsocket::connect` take `impl ToSocketAddrs`, NOT URL strings. `"ws://localhost:33800"` hits DNS and can fail with "Name or service not known" even when the port is bound. Use `SocketAddr::from((Ipv4Addr::LOCALHOST, 33800))` instead.

### ActionHash decoding

Zome-call responses containing `ActionHash` fields must deserialize into `holochain_types::prelude::ActionHash` directly (it's MessagePack bytes on the wire), not `serde_json::Value`. Attempting to deserialize into a JSON value yields:

```
Deserialize("invalid type: byte array, expected any valid JSON value")
```

even though the write itself succeeded.

### Two AppWebsocket connections required

The CLI opens the app websocket twice: first with a bootstrap signer to read `app_info()` → resolve the cell id → drop the bootstrap connection; then calls `authorize_signing_credentials`, registers the returned credentials into the signer, and opens the final authorized app websocket. This is the only way the signer knows which cell its credentials apply to.

---

## 8. What to read next

- [`docs/THREAT_MODEL.md`](./docs/THREAT_MODEL.md) — four adversary vectors, what's mitigated vs honest caveats.
- [`docs/GOV_ID_CLAIM_SHAPES.md`](./docs/GOV_ID_CLAIM_SHAPES.md) — claim-key conventions for passport / mDL / SSN-derived credentials.
- [`../MYCELIX_STATE_COEXISTENCE.md`](../MYCELIX_STATE_COEXISTENCE.md) — architectural rationale, dual-DID rule, "NOT built" list.
- [`../mycelix-identity/crates/eidas-zkp/`](../mycelix-identity/crates/eidas-zkp/) — the underlying W3C VC 2.0 + DASTARK + Dilithium5 + Merkle selective-disclosure library that this cluster reuses.
