# PoGQ Zome - Byzantine Detection with RISC Zero Proofs

**Version**: 0.1.0 (Scaffold)
**Status**: 🚧 Foundational Structure Complete
**Backend**: RISC Zero zkVM (v3.0) primary
**Security**: 127-bit (S128) or 191-bit (S192) conjectured security

---

## Overview

This Holochain zome integrates RISC Zero zero-knowledge proofs for Proof-of-Gradient-Quality (PoGQ) Byzantine detection in decentralized federated learning. It provides tamper-evident proof publishing, cryptographic nonce binding, and sybil-weighted aggregation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PoGQ Zome Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Node generates RISC Zero receipt (vsv-stark host)             │
│  2. Receipt + provenance published to DHT (publish_pogq_proof)    │
│  3. Gradient published with nonce binding (publish_gradient)      │
│  4. Verifiers fetch proofs + verify receipts (verify_pogq_proof)  │
│  5. Aggregator computes sybil weights (compute_sybil_weighted...) │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Entry Types

### PoGQProofEntry
Contains RISC Zero receipt and decision outputs:
- **receipt_bytes**: Serialized RISC Zero Receipt (several MB)
- **prov_hash**: Blake3 hash of build metadata
- **profile_id**: 128 (S128) or 192 (S192)
- **quarantine_out**: 0 (healthy) or 1 (quarantined)
- **nonce**: Cryptographic binding to gradient

### GradientEntry
Links gradient to proof:
- **gradient_commitment**: Hash of gradient tensor
- **pogq_proof_hash**: EntryHash of linked PoGQProofEntry
- **nonce**: MUST match proof nonce (validated)

## Zome Functions

### publish_pogq_proof(entry: PoGQProofEntry) → EntryHash
Publishes RISC Zero receipt to DHT.

**Validations**:
- Receipt non-empty
- Profile ID ∈ {128, 192}
- TODO: Nonce freshness
- TODO: Round advancement

### verify_pogq_proof(proof_hash: EntryHash) → VerificationResult
**Journal-Only Verification (M0 Scope)**

Validates journal integrity:
- Receipt non-empty
- Security profile ∈ {128, 192}
- Quarantine status ∈ {0, 1}
- Round consistency

**Status**: ✅ Implemented (journal validation only)
**Note**: Does NOT perform full ZK proof verification (RISC Zero incompatible with WASM)

### publish_gradient(entry: GradientEntry) → EntryHash
Publishes gradient with nonce binding to proof.

**Critical**: Validates nonce == proof.nonce (prevents replay attacks)

### get_round_gradients(round: u64) → Vec<(GradientEntry, bool)>
Fetches all gradients for a round with quarantine status.

**Returns**: List of (gradient, is_quarantined) tuples

### compute_sybil_weighted_aggregate(round: u64) → SybilWeightedResult
Computes aggregate metadata with sybil weighting.

**Weighting**: 1.0 if healthy, 0.0 if quarantined

## Integration with vsv-stark

### RISC Zero Host Integration
```bash
# Generate proof
cd vsv-stark
export VSV_PROVENANCE_MODE="strict"
export VSV_SECURITY_PROFILE="128"
./target/release/host test_inputs/public.json test_inputs/witness.json output/

# Extract receipt
RECEIPT_BYTES=$(cat output/proof.bin | base64)

# Extract journal
QUARANTINE_OUT=$(jq -r '.quarantine_out' output/decision_journal.json)
PROV_HASH=$(jq -r '.prov_hash' output/decision_journal.json)
```

### Publishing to DHT
```rust
use pogq_zome::{PoGQProofEntry, publish_pogq_proof};

let proof_entry = PoGQProofEntry {
    node_id: agent_info()?.agent_latest_pubkey,
    round: 5,
    nonce: rand::random(),
    receipt_bytes: receipt_bytes,
    prov_hash: prov_hash,
    profile_id: 128,
    air_rev: 1,
    quarantine_out: 0,
    current_round: 5,
    ema_t_fp: scale(0.92),
    consec_viol_t: 0,
    consec_clear_t: 3,
    timestamp: sys_time()?,
};

let proof_hash = publish_pogq_proof(proof_entry)?;
```

## Security Properties

### Tamper-Evident Configuration
- **Property**: Any change to build metadata invalidates proofs
- **Mechanism**: Blake3(rust_ver, git_commit, timestamp, profile_id, air_rev)
- **Enforcement**: Guest fail-fast assertions

### Nonce Binding
- **Property**: Gradients cryptographically bound to proofs
- **Mechanism**: 32-byte random nonce shared between proof and gradient
- **Enforcement**: `publish_gradient` validates nonce match

### Sybil Resistance
- **Property**: Quarantined nodes receive zero weight
- **Mechanism**: PoGQ state machine detects Byzantine behavior
- **Enforcement**: `compute_sybil_weighted_aggregate` filters by quarantine_out

## Development Status

### ✅ Complete (Scaffold)
- Entry types defined
- Zome functions scaffolded
- Nonce binding validation
- DHT path structure
- Sybil weighting logic

### 🚧 TODO (Phase 2 Completion)
- [x] RISC Zero WASM investigation → **BLOCKED** (native deps incompatible)
- [x] Design verification strategy → Journal-only for M0
- [x] Implement journal verification logic
- [ ] Nonce freshness checking (persistent storage)
- [ ] Unit tests for entry validation
- [ ] Integration tests with 2 nodes (1 honest, 1 Byzantine)
- [ ] Nonce replay attack test
- [ ] Provenance hash validation (optional for M0)

### 🔮 Future (M0 Phases 3-5)
- [ ] FL orchestrator integration
- [ ] Byzantine node simulation
- [ ] Metrics collection (proof generation time, verification time)
- [ ] Visualizations (quarantine timeline, quality scores)

## Testing

### Unit Tests
```bash
cd holochain/zomes/pogq_zome
cargo test
```

### Integration Tests (TODO)
```bash
# 2-node test (1 honest, 1 Byzantine)
cd holochain
hc sandbox generate
# Run test scenario
```

### Smoke Test
```bash
# Compile zome to WASM
cargo build --release --target wasm32-unknown-unknown

# Check WASM size
ls -lh target/wasm32-unknown-unknown/release/pogq_zome.wasm
```

## Performance Expectations

### RISC Zero Backend
- **Proof generation**: 30-60 seconds per node per round
- **Proof verification**: <100ms (lightweight, no proving)
- **Receipt size**: ~Several MB (larger than Winterfell's 221KB)
- **DHT storage**: Per-round cleanup recommended (prune old proofs)

### Holochain DHT
- **Throughput**: ~10,000 TPS (Holochain benchmark)
- **Query latency**: <1ms (cached), <10ms (DHT lookup)
- **Scalability**: O(N) proof publishes, O(log N) DHT queries

## References

- **RISC Zero**: https://www.risczero.com/
- **Holochain**: https://holochain.org/
- **M0 Design Doc**: `/srv/luminous-dynamics/Mycelix-Core/0TML/vsv-stark/docs/M0_FIVE_NODE_ZKFL_DEMO_DESIGN.md`
- **Session Status**: `/srv/luminous-dynamics/Mycelix-Core/0TML/vsv-stark/docs/SESSION_STATUS_2025-11-10_PHASE2_START.md`

## License

Same as parent project (vsv-stark)

## Authors

- Luminous Dynamics Team
- RISC Zero provenance integration: November 2025

---

**Next Steps**: Complete TODO items, test with real RISC Zero receipts, integrate with FL orchestrator (M0 Phase 3).
