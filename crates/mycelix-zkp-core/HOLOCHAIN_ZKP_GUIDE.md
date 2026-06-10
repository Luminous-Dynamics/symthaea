# Adding ZKP to Any Holochain hApp

## Quick Start

```rust
// 1. Add to your zome's Cargo.toml:
// mycelix-zkp-core = { path = "path/to/crates/mycelix-zkp-core" }

// 2. Generate a proof (client-side, native Rust):
use mycelix_zkp_core::circuits::range_proof::{prove_range, verify_range};

let proof = prove_range(value, min, max, commitment)?;

// 3. Submit to zome via callZome:
// callZome("your_bridge", "submit_proof", { proof_bytes: proof.to_bytes() })

// 4. Verify off-chain:
verify_range(proof, min, max, commitment)?;
```

## Architecture

```
Client (native)              Holochain Zome (WASM)         Off-chain Verifier
┌──────────────┐            ┌─────────────────┐           ┌──────────────┐
│ prove_range() │  callZome  │ validate_struct()│  DHT read │ verify_range()│
│ Dilithium sign│ ─────────→ │ store on DHT    │ ────────→ │ Ed25519 sign  │
│               │            │ link entries     │           │ store attest  │
└──────────────┘            └─────────────────┘           └──────────────┘
```

**Why off-chain verification?** Winterfell CAN compile to WASM (no_std),
but it's untested in Holochain's wasmer runtime. The off-chain pattern
is proven in production (attribution cluster: 31 sweettests).

## Step-by-Step Integration

### 1. Choose your proof type

| Use Case | Circuit | Backend | AND Constraints |
|---|---|---|---|
| Value in range | `range_proof` | Winterfell | ~32 per 16-bit value |
| Consciousness tier | `consciousness` | Winterfell | ~32 (46ms) |
| HDC binding | `bxor` | Binius | 0 (FREE) |
| Majority vote | `band` chain | Binius | 2×N per word |
| Encrypted computation | Triple stack | Binius | Same as plaintext |
| CfC temporal | `band + bxor` | Binius | 1 per neuron per step |

### 2. Add domain tag

```rust
// In mycelix-zkp-core/src/domain.rs, add:
pub fn tag_your_cluster() -> DomainTag {
    DomainTag::new("YourCluster", "YourProofType", 1)
}
```

### 3. Create proof entry type (integrity zome)

```rust
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct YourProofEntry {
    pub proof_bytes: Vec<u8>,        // STARK proof
    pub commitment: Vec<u8>,         // 32-byte SHA-256
    pub domain_tag: String,          // "ZTML:YourCluster:..."
    pub generated_at: Timestamp,
    pub verified: bool,              // Set true after off-chain verify
    pub verifier_pubkey: Option<Vec<u8>>,
    pub verifier_signature: Option<Vec<u8>>,
}
```

### 4. Add submission extern (coordinator zome)

```rust
#[hdk_extern]
pub fn submit_proof(input: SubmitProofInput) -> ExternResult<Record> {
    // Validate structure
    if input.proof_bytes.is_empty() { return Err(...) }
    if input.proof_bytes.len() > 500_000 { return Err(...) }
    if input.commitment.len() != 32 { return Err(...) }

    // Create entry
    let entry = YourProofEntry { ... };
    let hash = create_entry(&EntryTypes::YourProof(entry))?;

    // Link to relevant entity
    create_link(entity_hash, hash.clone(), LinkTypes::EntityToProof, ())?;

    get(hash, GetOptions::default())
}
```

### 5. Off-chain verifier

```rust
// Native Rust binary (not WASM):
use mycelix_zkp_core::circuits::range_proof::verify_range;
use winterfell::Proof;

let proof = Proof::from_bytes(&proof_bytes)?;
verify_range(proof, min, max, commitment)?;

// Sign attestation with Ed25519
let signature = signing_key.sign(&attestation_message);

// Submit attestation back to zome
// callZome("your_bridge", "attest_proof", { hash, pubkey, signature })
```

### 6. Consciousness gating (optional)

```rust
// If your operation requires consciousness tier:
use mycelix_zkp_core::consciousness::{prove_consciousness_tier, ConsciousnessProofRequest, ConsciousnessTier};

let request = ConsciousnessProofRequest {
    phi_score: 0.55,
    required_tier: ConsciousnessTier::Steward,
    agent_did: "did:mycelix:agent001".to_string(),
};
let proof = prove_consciousness_tier(&request)?;
// Submit proof.proof_bytes alongside your operation
```

## Available Circuits

| Circuit | Location | Tests | Measured |
|---|---|---|---|
| Range proof | `src/circuits/range_proof.rs` | 6 | 27ms (32-bit) |
| Winterfell XOR | `src/circuits/winterfell_xor.rs` | 4 | 23.8s (16Kbit) |
| Consciousness | `src/consciousness.rs` | 5 | 46ms |
| PoGQ (FL) | `src/pogq.rs` | 8 | 382ns sim |
| Dilithium5 | `src/dilithium.rs` | 8 | 1.2ms verify |

## Feature Flags

```toml
[dependencies]
mycelix-zkp-core = { path = "...", features = ["backend-winterfell"] }
# Options: backend-winterfell, backend-risc0, backend-dual, dilithium, full
```

- `backend-winterfell`: Winterfell STARK verifier (~200-400KB WASM)
- `dilithium`: Dilithium5 PQ signatures (~2.6KB)
- `full`: Everything (for native testing only, not for zomes)
