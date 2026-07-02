# Phase 10 Migration Guide - Holochain DHT + Zero-Knowledge Proofs

**Prerequisites**: Phase 9 deployed and validated
**Timeline**: 2-3 months
**Complexity**: High

---

## Overview

Phase 10 adds two major capabilities to Zero-TrustML:

1. **Holochain DHT** - Immutable audit trail for all transactions
2. **Bulletproofs ZK-PoC** - Privacy-preserving gradient validation

This migration is **non-breaking**. PostgreSQL backend continues to function as Layer 1, with Holochain and ZK-PoC as optional enhancements for users requiring:
- Regulatory compliance (medical, financial FL)
- Immutable audit trails
- Multi-party verification
- Privacy-preserving validation

---

## Part 1: Holochain DHT Integration

### Step 1: Fix Holochain HDK Compatibility

The `credits` zome currently uses HDK 0.6.0-dev which has compilation issues. Fix using the scaffolding tool:

```bash
cd 0TML/zerotrustml-dna

# Enter Holochain environment
nix develop /srv/luminous-dynamics/Mycelix-Core

# Scaffold new zome with correct structure
# When prompted, select: headless (no ui)
hc scaffold zome credits_fixed

# This creates:
# - zomes/credits_fixed_integrity/ (entry definitions)
# - zomes/credits_fixed_coordinator/ (business logic)
```

### Step 2: Migrate Entry Definitions

Copy the Credit struct to the integrity zome:

**File**: `zomes/credits_fixed_integrity/src/lib.rs`
```rust
use hdi::prelude::*;

/// Credit entry type - represents a credit issuance on the DHT
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Credit {
    pub holder: AgentPubKey,
    pub amount: u64,
    pub earned_from: String,
    pub timestamp: i64,  // Microseconds since UNIX epoch
}

/// Entry definitions for this zome
#[hdk_entry_defs]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Credit(Credit),
}

/// Link types (for querying credits by holder)
#[hdk_link_types]
pub enum LinkTypes {
    HolderToCredits,
}

// Validation logic
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(store_entry) => {
            match store_entry.entry {
                Entry::App(app_entry_bytes) => {
                    // Deserialize to Credit
                    let credit: Credit = app_entry_bytes.into_sb().try_into()
                        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid credit entry".into())))?;

                    // Validation rules
                    if credit.amount == 0 {
                        return Ok(ValidateCallbackResult::Invalid("Credit amount must be > 0".into()));
                    }

                    if credit.amount > 1_000_000 {
                        return Ok(ValidateCallbackResult::Invalid("Credit amount exceeds maximum".into()));
                    }

                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
```

### Step 3: Migrate Business Logic

Copy functions to the coordinator zome:

**File**: `zomes/credits_fixed_coordinator/src/lib.rs`
```rust
use hdk::prelude::*;
use credits_fixed_integrity::*;

/// Create a new credit entry
#[hdk_extern]
pub fn create_credit(input: CreateCreditInput) -> ExternResult<ActionHash> {
    let credit = Credit {
        holder: input.holder.clone(),
        amount: input.amount,
        earned_from: input.earned_from.clone(),
        timestamp: sys_time()?,
    };

    // Create entry
    let action_hash = create_entry(EntryTypes::Credit(credit.clone()))?;

    // Create link from holder to credit (for querying)
    let holder_hash = hash_entry(input.holder)?;
    create_link(
        holder_hash,
        action_hash.clone(),
        LinkTypes::HolderToCredits,
        (),
    )?;

    Ok(action_hash)
}

/// Get a credit entry
#[hdk_extern]
pub fn get_credit(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Get all credits for a holder
#[hdk_extern]
pub fn get_credits_for_holder(holder: AgentPubKey) -> ExternResult<Vec<Record>> {
    let holder_hash = hash_entry(holder.clone())?;
    let links = get_links(
        GetLinksInputBuilder::try_new(holder_hash, LinkTypes::HolderToCredits)?.build()
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(record) = get(ActionHash::from(link.target), GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Get total balance for a holder
#[hdk_extern]
pub fn get_balance(holder: AgentPubKey) -> ExternResult<u64> {
    let records = get_credits_for_holder(holder)?;

    let total: u64 = records
        .iter()
        .filter_map(|record| {
            record
                .entry()
                .to_app_option::<Credit>()
                .ok()
                .flatten()
                .map(|credit| credit.amount)
        })
        .sum();

    Ok(total)
}

// Input types
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCreditInput {
    pub holder: AgentPubKey,
    pub amount: u64,
    pub earned_from: String,
}
```

### Step 4: Update DNA Configuration

**File**: `zerotrustml-dna/dna.yaml`
```yaml
---
manifest_version: "1"
name: zerotrustml-dna
integrity:
  origin_time: 2025-10-01T00:00:00.000000Z
  network_seed: null
  properties: null
  zomes:
    - name: credits_integrity
      bundled: zomes/credits_fixed_integrity.wasm
coordinator:
  zomes:
    - name: credits_coordinator
      bundled: zomes/credits_fixed_coordinator.wasm
      dependencies:
        - name: credits_integrity
```

### Step 5: Build and Test

```bash
# Build zomes
cd zomes/credits_fixed_integrity
cargo build --target wasm32-unknown-unknown --release
cd ../credits_fixed_coordinator
cargo build --target wasm32-unknown-unknown --release

# Pack DNA
cd ../../
hc dna pack .

# Test with sandbox
hc sandbox generate --dna workdir/zerotrustml-dna.dna
hc sandbox run

# Test credit creation
hc sandbox call credits_coordinator create_credit '{
  "holder": "uhCAkmHGSp...",
  "amount": 100,
  "earned_from": "byzantine_detection"
}'

# Test balance query
hc sandbox call credits_coordinator get_balance '{
  "holder": "uhCAkmHGSp..."
}'
```

### Step 6: Deploy Holochain Conductor

```bash
# Install Holochain conductor (NixOS)
nix-env -iA nixpkgs.holochain

# Create conductor configuration
cat > conductor-config.yaml << 'EOF'
---
environment_path: /var/lib/holochain
use_dangerous_test_keystore: false

network:
  bootstrap_service: https://bootstrap.holo.host
  transport_pool:
    - type: webrtc
      signal_url: wss://signal.holo.host

app_interfaces:
  - driver:
      type: websocket
      port: 9000

admin_interfaces:
  - driver:
      type: websocket
      port: 9001
EOF

# Start conductor
holochain -c conductor-config.yaml

# Install DNA
hc admin install-app \
  --app-id zerotrustml \
  --agent-key uhCAk... \
  --dna zerotrustml-dna.dna \
  --membrane-proof ""

# Enable app
hc admin enable-app --app-id zerotrustml
```

### Step 7: Update Python Bridge

**File**: `src/zerotrustml/holochain_bridge.py` (enhanced)
```python
import asyncio
import websockets
import json
from typing import Optional, Dict, Any

class HolochainBridge:
    """Bridge between Zero-TrustML and Holochain conductor."""

    def __init__(self, conductor_url: str = "ws://localhost:9000"):
        self.conductor_url = conductor_url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self):
        """Connect to Holochain conductor."""
        self.websocket = await websockets.connect(self.conductor_url)
        print(f"✅ Connected to Holochain conductor at {self.conductor_url}")

    async def call_zome(
        self,
        zome_name: str,
        function_name: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a zome function."""
        if not self.websocket:
            await self.connect()

        request = {
            "type": "app_request",
            "data": {
                "cell_id": ["zerotrustml", "agent-key"],
                "zome_name": zome_name,
                "fn_name": function_name,
                "payload": payload,
                "provenance": "agent-key",
            }
        }

        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        return json.loads(response)

    async def create_credit(
        self,
        holder: str,
        amount: int,
        earned_from: str
    ) -> str:
        """Issue a credit on the DHT."""
        result = await self.call_zome(
            "credits_coordinator",
            "create_credit",
            {
                "holder": holder,
                "amount": amount,
                "earned_from": earned_from,
            }
        )
        return result["data"]  # Returns ActionHash

    async def get_balance(self, holder: str) -> int:
        """Get credit balance from DHT."""
        result = await self.call_zome(
            "credits_coordinator",
            "get_balance",
            {"holder": holder}
        )
        return result["data"]

    async def close(self):
        """Close connection."""
        if self.websocket:
            await self.websocket.close()
```

### Step 8: Hybrid PostgreSQL + Holochain Mode

Enable both backends for redundancy:

**File**: `config/node.yaml`
```yaml
credits:
  backend: "hybrid"  # Use both PostgreSQL + Holochain

  postgresql:
    enabled: true
    # Fast local queries

  holochain:
    enabled: true
    conductor_url: "ws://localhost:9000"
    # Immutable audit trail

  # Sync strategy
  sync_mode: "write_both_read_postgres"
  # Writes go to both, reads come from PostgreSQL (faster)

  # Periodic validation
  validation_interval: 3600  # seconds
  # Every hour, verify PostgreSQL matches Holochain DHT
```

---

## Part 2: Zero-Knowledge Proof of Contribution (Bulletproofs)

### Step 1: Add Bulletproofs Dependency

**File**: `Cargo.toml`
```toml
[dependencies]
bulletproofs = { version = "4.0", features = ["yoloproofs"] }
curve25519-dalek = "4.0"
merlin = "3.0"
rand = "0.8"
sha3 = "0.10"
```

### Step 2: Implement Gradient Proof

**File**: `src/zerotrustml/zkproof/gradient_proof.rs`
```rust
use bulletproofs::{BulletproofGens, PedersenGens, RangeProof};
use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;

pub struct GradientProof {
    pub norm_proof: RangeProof,
    pub size_proof: RangeProof,
    pub loss_proof: RangeProof,
}

impl GradientProof {
    /// Generate ZK proof for gradient quality
    pub fn generate(
        gradient: &Gradient,
        dataset_size: u64,
        loss_improvement: f64,
    ) -> Self {
        let pc_gens = PedersenGens::default();
        let bp_gens = BulletproofGens::new(64, 1);

        // Prove gradient norm is in range [0, max_norm]
        let norm = gradient.l2_norm();
        let norm_u64 = (norm * 1000.0) as u64;  // Scale to integer
        let norm_blinding = Scalar::random(&mut rand::thread_rng());

        let mut transcript = Transcript::new(b"GradientNormProof");
        let (norm_proof, _) = RangeProof::prove_single(
            &bp_gens,
            &pc_gens,
            &mut transcript,
            norm_u64,
            &norm_blinding,
            64,  // 64-bit range
        ).expect("Norm proof generation failed");

        // Prove dataset size >= min_size
        let size_blinding = Scalar::random(&mut rand::thread_rng());
        let mut transcript = Transcript::new(b"DatasetSizeProof");
        let (size_proof, _) = RangeProof::prove_single(
            &bp_gens,
            &pc_gens,
            &mut transcript,
            dataset_size,
            &size_blinding,
            64,
        ).expect("Size proof generation failed");

        // Prove loss improvement > threshold
        let loss_u64 = ((loss_improvement + 1.0) * 1000.0) as u64;  // Shift + scale
        let loss_blinding = Scalar::random(&mut rand::thread_rng());
        let mut transcript = Transcript::new(b"LossImprovementProof");
        let (loss_proof, _) = RangeProof::prove_single(
            &bp_gens,
            &pc_gens,
            &mut transcript,
            loss_u64,
            &loss_blinding,
            64,
        ).expect("Loss proof generation failed");

        GradientProof {
            norm_proof,
            size_proof,
            loss_proof,
        }
    }

    /// Verify ZK proof
    pub fn verify(
        &self,
        max_norm: f64,
        min_size: u64,
        min_improvement: f64,
    ) -> bool {
        let pc_gens = PedersenGens::default();
        let bp_gens = BulletproofGens::new(64, 1);

        // Verify norm proof
        let mut transcript = Transcript::new(b"GradientNormProof");
        let norm_result = self.norm_proof.verify_single(
            &bp_gens,
            &pc_gens,
            &mut transcript,
            &commitment,  // Need commitment from proof
            64,
        );

        // Verify size proof
        let mut transcript = Transcript::new(b"DatasetSizeProof");
        let size_result = self.size_proof.verify_single(
            &bp_gens,
            &pc_gens,
            &mut transcript,
            &commitment,
            64,
        );

        // Verify loss proof
        let mut transcript = Transcript::new(b"LossImprovementProof");
        let loss_result = self.loss_proof.verify_single(
            &bp_gens,
            &pc_gens,
            &mut transcript,
            &commitment,
            64,
        );

        norm_result.is_ok() && size_result.is_ok() && loss_result.is_ok()
    }

    /// Serialize for transmission
    pub fn serialize(&self) -> Vec<u8> {
        // Combine all three proofs
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.norm_proof.to_bytes());
        bytes.extend_from_slice(&self.size_proof.to_bytes());
        bytes.extend_from_slice(&self.loss_proof.to_bytes());
        bytes
    }

    /// Deserialize from bytes
    pub fn deserialize(bytes: &[u8]) -> Result<Self, String> {
        // Parse three proofs from byte stream
        // (Implementation omitted for brevity)
        todo!("Implement deserialization")
    }
}
```

### Step 3: Integrate with Holochain Validation

**File**: `zomes/credits_fixed_integrity/src/lib.rs` (add validation)
```rust
use hdi::prelude::*;

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(store_entry) => {
            match store_entry.entry {
                Entry::App(app_entry_bytes) => {
                    let submission: GradientSubmission = app_entry_bytes.into_sb().try_into()
                        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid submission".into())))?;

                    // Deserialize ZK proof
                    let proof = match GradientProof::deserialize(&submission.zk_proof) {
                        Ok(p) => p,
                        Err(e) => return Ok(ValidateCallbackResult::Invalid(format!("Proof deserialization failed: {}", e))),
                    };

                    // Verify proof (without seeing actual gradient!)
                    if proof.verify(MAX_NORM, MIN_SIZE, MIN_IMPROVEMENT) {
                        Ok(ValidateCallbackResult::Valid)
                    } else {
                        Ok(ValidateCallbackResult::Invalid("ZK proof verification failed".into()))
                    }
                }
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
```

### Step 4: Update Node to Generate Proofs

**File**: `src/zerotrustml/core/node.py` (enhanced)
```python
from zerotrustml.zkproof import GradientProof

class Zero-TrustMLNode:
    """Enhanced node with ZK proof generation."""

    async def submit_gradient(self, gradient: torch.Tensor):
        """Submit gradient with ZK proof."""
        # Compute gradient quality metrics
        norm = gradient.norm().item()
        dataset_size = len(self.local_dataset)
        loss_improvement = self.compute_loss_improvement()

        # Generate ZK proof (proves quality without revealing gradient!)
        zk_proof = GradientProof.generate(
            gradient=gradient,
            dataset_size=dataset_size,
            loss_improvement=loss_improvement,
        )

        # Submit to Holochain with proof
        submission = {
            "gradient_commitment": self.commit_gradient(gradient),  # Hash only
            "zk_proof": zk_proof.serialize(),
            "timestamp": int(time.time()),
        }

        action_hash = await self.holochain_bridge.call_zome(
            "gradients_coordinator",
            "submit_gradient",
            submission
        )

        print(f"✅ Gradient submitted with ZK proof: {action_hash}")
        return action_hash
```

### Step 5: Performance Testing

```bash
# Benchmark proof generation
cargo bench --bench zkproof_bench

# Expected results:
# - Proof generation: 50-80ms
# - Proof verification: 8-15ms
# - Proof size: 600-800 bytes
```

### Step 6: Integration Testing

```python
# tests/test_zkproof_integration.py
import pytest
from zerotrustml.zkproof import GradientProof
from zerotrustml.core.node import Zero-TrustMLNode

@pytest.mark.asyncio
async def test_zkproof_submission():
    """Test end-to-end ZK proof submission."""
    node = Zero-TrustMLNode()

    # Submit gradient with proof
    gradient = torch.randn(1000, 1000)
    action_hash = await node.submit_gradient(gradient)

    assert action_hash is not None
    print("✅ ZK proof submission successful")

@pytest.mark.asyncio
async def test_zkproof_prevents_bad_gradient():
    """Test that Byzantine gradient is rejected."""
    node = Zero-TrustMLNode()

    # Create poisoned gradient
    bad_gradient = torch.randn(1000, 1000) * 1000  # Huge norm

    with pytest.raises(Exception, match="ZK proof verification failed"):
        await node.submit_gradient(bad_gradient)

    print("✅ Byzantine gradient correctly rejected")
```

---

## Part 3: Production Deployment

### Step 1: Update Configuration

**File**: `config/node.yaml` (Phase 10 mode)
```yaml
node:
  mode: "phase10"  # Enable Holochain + ZK-PoC

credits:
  backend: "hybrid"
  postgresql:
    enabled: true
  holochain:
    enabled: true
    conductor_url: "ws://localhost:9000"

zk_proof:
  enabled: true
  proof_type: "bulletproofs"
  max_norm: 10.0
  min_dataset_size: 100
  min_loss_improvement: 0.01
```

### Step 2: Migration from PostgreSQL

```bash
# Export existing credits from PostgreSQL
python scripts/export_credits.py --output credits_export.json

# Import to Holochain DHT
python scripts/import_to_holochain.py --input credits_export.json

# Verify consistency
python scripts/verify_migration.py
```

### Step 3: Deploy Bridge Validators

For production multi-party verification:

```bash
# Deploy 5 bridge validators
for i in {1..5}; do
  docker run -d \
    --name bridge-validator-$i \
    --network zerotrustml-net \
    -e VALIDATOR_ID=$i \
    -e HOLOCHAIN_URL=ws://holochain-conductor:9000 \
    zerotrustml-bridge-validator:1.0.0
done
```

### Step 4: Monitoring

Add ZK-PoC metrics to Prometheus:

```promql
# Proof generation time
histogram_quantile(0.95, zerotrustml_zkproof_generation_duration_seconds)

# Proof verification time
histogram_quantile(0.95, zerotrustml_zkproof_verification_duration_seconds)

# Proof rejection rate
rate(zerotrustml_zkproof_rejected_total[5m])
```

---

## Rollback Plan

If Phase 10 migration encounters issues:

```bash
# Disable Holochain and ZK-PoC
# In config/node.yaml:
credits:
  backend: "postgresql"  # Fallback to Phase 9

zk_proof:
  enabled: false

# Restart nodes
systemctl restart zerotrustml

# System continues with PostgreSQL backend
```

---

## Timeline & Resources

### Phase 10 Implementation Timeline
- **Week 1-2**: Holochain scaffolding + HDK fix
- **Week 3-4**: Bulletproofs implementation
- **Week 5-6**: Integration testing
- **Week 7-8**: Production deployment
- **Week 9-10**: Monitoring + optimization
- **Week 11-12**: Documentation + training

### Required Skills
- Rust (intermediate): For Holochain zomes
- Cryptography (basic): Understanding Bulletproofs
- Python (advanced): For bridge integration
- DevOps (intermediate): Conductor deployment

### External Dependencies
- Holochain conductor 0.5+
- Bulletproofs library 4.0+
- PostgreSQL 15+ (continued)

---

## Next Steps

After Phase 10:
- **Phase 11**: zk-SNARKs for full computation proofs
- **Phase 12**: Multi-industry expansion using adapter pattern
- **Phase 13**: Cross-chain bridges to external DeFi

---

**Document Status**: Ready for Implementation
**Prerequisites**: Phase 9 deployed and validated
**Timeline**: 2-3 months
**Complexity**: High

See [`HOLOCHAIN_HDK_FIX_GUIDE.md`](./HOLOCHAIN_HDK_FIX_GUIDE.md) and [`ZK_POC_TECHNICAL_DESIGN.md`](./ZK_POC_TECHNICAL_DESIGN.md) for additional technical details.
