# mycelix-lawful-identity

The **legal-DID** namespace for Mycelix. Handles government-ID-backed credentials, issuer trust tiers, and cross-DID zero-knowledge bridges — strictly isolated from the primary consciousness-gated identity.

Part of the [state-coexistence extensions](../MYCELIX_STATE_COEXISTENCE.md).

## The dual-DID rule

```
did:mycelix:primary  ──  governance, MYCEL, TEND, social graph  (pseudonymous)
did:mycelix:legal    ──  ONLY state-facing interactions         (this cluster)
```

The two DIDs share **zero on-chain linkability**. Any cross-link is **ZKP-only**: the primary DID can prove "I control a legal DID holding credential X from issuer Y" without revealing *which* legal DID.

## What's here

| Path | What |
|------|------|
| `zomes/legal-did/` | Isolated DID namespace with distinct DHT partition |
| `zomes/issuer-trust-tier/` | Three tiers: Sovereign / RegulatedIntermediary / Peer |
| `zomes/cross-did-zkp/` | Primary ↔ legal cross-proof with fresh-nonce replay protection |
| `cli/` | `lawful-id` CLI for import + proof generation + onboarding disclosures |
| `docs/THREAT_MODEL.md` | Four adversary vectors: what's cryptographically neutralized, what isn't |
| `docs/GOV_ID_CLAIM_SHAPES.md` | Claim-key conventions for mDL / passport / SSN-equivalent |

## What's NOT here

- No KYC enforcement on primary-DID functions.
- No governance tier bonus for verified legal identity.
- No sanctions screening (kills mutual credit; delegates sovereignty).
- No "legal name" field on the primary DID profile.
- No court-order key disclosure protocol (separate `mycelix-lawful-response` cluster if ever built).

## Underlying crypto

Reuses `mycelix-identity/crates/eidas-zkp/` — W3C VC 2.0, DASTARK + Dilithium5, Merkle selective disclosure, range/membership/equality proven claims. A passport credential is just an eIDAS credential with specific claim keys (see `GOV_ID_CLAIM_SHAPES.md`).

Jurisdiction-in-credential range proofs reuse `crates/mycelix-zkp-core/src/circuits/range_proof.rs` via the new `jurisdiction_proof.rs` circuit.

## Build

```bash
nix develop
cargo build --workspace --target wasm32-unknown-unknown --release
hc dna pack dna/ -o dna/mycelix_lawful_identity.dna
hc app pack . -o mycelix-lawful-identity.happ
```

## Port

Frontend reserved at `8133` (see `.claude/rules/PORTS.md`).
Domain: `lawful.mycelix.net`.

## License

AGPL-3.0-or-later (matching the rest of the Mycelix workspace).
