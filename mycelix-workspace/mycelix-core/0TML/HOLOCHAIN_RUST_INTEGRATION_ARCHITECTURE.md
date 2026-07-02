# 🍄 Holochain + Rust + Python: Distributed FL Architecture

**Date**: November 13, 2025  
**Purpose**: Leverage Holochain for agent-centric coordination, Rust for performance-critical aggregation, Python for ML/research

---

## 🏗️ Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Python Layer (ML Training & Research)           │
│  - PyTorch models                                       │
│  - Local training                                       │
│  - Experiment orchestration                             │
│  - Attack simulations                                   │
└─────────────────┬───────────────────────────────────────┘
                  │ PyO3 Bindings
                  ▼
┌─────────────────────────────────────────────────────────┐
│       Holochain DNA (Distributed Coordination)          │
│  - Proof submission zome                                │
│  - Reputation tracking zome                             │
│  - Model distribution zome                              │
│  - Economic game zome (staking/slashing)                │
│  - DHT for gossip protocol                              │
└─────────────────┬───────────────────────────────────────┘
                  │ HDK Integration
                  ▼
┌─────────────────────────────────────────────────────────┐
│      Rust Core (Performance-Critical Operations)        │
│  - zkSTARK verification (RISC Zero)                     │
│  - Aggregation algorithms (Multi-Krum, etc.)            │
│  - PoGQ oracle computations                             │
│  - Byzantine detection                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🍄 Holochain DNA Architecture

### Zome 1: **Proof Submission** (`gradient_proofs`)

```rust
use hdk::prelude::*;

#[hdk_entry_helper]
pub struct GradientProofEntry {
    pub round_number: u64,
    pub gradient_hash: EntryHash,
    pub proof_bytes: Vec<u8>,          // zkSTARK Receipt
    pub timestamp: Timestamp,
    pub client_pubkey: AgentPubKey,
    pub model_commitment: EntryHash,
}

#[hdk_extern]
pub fn submit_proof(proof: GradientProofEntry) -> ExternResult<ActionHash> {
    // Validation rules (enforced by all nodes):
    // 1. Proof must be for current round
    // 2. Client hasn't submitted for this round yet
    // 3. Client has sufficient stake
    // 4. Proof timestamp is fresh (< 5 minutes old)
    
    validate_proof_freshness(&proof)?;
    validate_no_double_submission(&proof)?;
    validate_stake_requirement(&proof.client_pubkey)?;
    
    // Create entry on agent's source chain
    let action_hash = create_entry(&EntryTypes::GradientProof(proof.clone()))?;
    
    // Emit signal for coordinator to process
    emit_signal(&Signal::NewProofSubmitted {
        submitter: agent_info()?.agent_latest_pubkey,
        proof_hash: action_hash.clone(),
        round: proof.round_number,
    })?;
    
    Ok(action_hash)
}

#[hdk_extern]
pub fn get_proofs_for_round(round: u64) -> ExternResult<Vec<GradientProofEntry>> {
    // Query DHT for all proofs in this round
    let filter = ChainQueryFilter::new()
        .entry_type(EntryTypes::GradientProof.try_into()?)
        .include_entries(true);
    
    let proofs: Vec<_> = query(filter)?
        .into_iter()
        .filter_map(|record| {
            if let Some(entry) = record.entry().as_option() {
                entry.to_app_option::<GradientProofEntry>().ok()
            } else { None }
        })
        .filter(|p| p.round_number == round)
        .collect();
    
    Ok(proofs)
}
```

**Why Holochain for Proofs?**
- ✅ **Agent-centric**: Each client maintains their own source chain of proof submissions
- ✅ **DHT gossip**: Proofs propagate through network without central server
- ✅ **Tamper-proof**: Once committed to source chain, proofs can't be modified
- ✅ **Validation rules**: Network enforces proof freshness, no double-submission
- ✅ **Offline resilience**: Clients can catch up asynchronously

---

### Zome 2: **Reputation Tracking** (`reputation`)

```rust
#[hdk_entry_helper]
pub struct ReputationScore {
    pub agent: AgentPubKey,
    pub total_score: f64,
    pub successful_rounds: u64,
    pub failed_rounds: u64,
    pub last_updated: Timestamp,
    pub stake_amount: u64,
}

#[hdk_extern]
pub fn update_reputation(
    agent: AgentPubKey,
    round_result: RoundResult
) -> ExternResult<ReputationScore> {
    // Only coordinator can update reputations
    verify_coordinator_authority()?;
    
    let mut rep = get_reputation(&agent)?;
    
    match round_result {
        RoundResult::Accepted => {
            rep.total_score += 1.0;
            rep.successful_rounds += 1;
        }
        RoundResult::Rejected { reason } => {
            rep.total_score -= slash_amount(&reason);
            rep.failed_rounds += 1;
        }
    }
    
    rep.last_updated = sys_time()?;
    create_entry(&EntryTypes::Reputation(rep.clone()))?;
    
    Ok(rep)
}

#[hdk_extern]
pub fn get_reputation_weights() -> ExternResult<Vec<(AgentPubKey, f64)>> {
    // Used by PoGQ oracle for trust-weighted aggregation
    let all_reps = query_all_reputations()?;
    
    let weights: Vec<_> = all_reps
        .into_iter()
        .map(|r| {
            let weight = pogq_weight(r.total_score, r.stake_amount);
            (r.agent, weight)
        })
        .collect();
    
    Ok(weights)
}
```

**Why Holochain for Reputation?**
- ✅ **Distributed consensus**: No single point of reputation control
- ✅ **Source chain audit trail**: Full history of reputation changes
- ✅ **Sybil resistance**: Each agent has cryptographic identity
- ✅ **Economic games**: Stake stored on-chain, slashing automatic

---

### Zome 3: **Model Distribution** (`models`)

```rust
#[hdk_entry_helper]
pub struct GlobalModel {
    pub round_number: u64,
    pub model_params_hash: EntryHash,  // Hash of full model
    pub model_chunk: Vec<u8>,           // This chunk
    pub chunk_index: u32,
    pub total_chunks: u32,
    pub coordinator_signature: Signature,
}

#[hdk_extern]
pub fn publish_global_model(model: GlobalModel) -> ExternResult<ActionHash> {
    // Only coordinator can publish global models
    verify_coordinator_signature(&model)?;
    
    // Store chunk on DHT
    let hash = create_entry(&EntryTypes::GlobalModel(model.clone()))?;
    
    // Emit signal for local clients to download
    emit_signal(&Signal::NewGlobalModel {
        round: model.round_number,
        chunk_index: model.chunk_index,
    })?;
    
    Ok(hash)
}

#[hdk_extern]
pub fn get_global_model(round: u64) -> ExternResult<Vec<u8>> {
    // Reconstruct full model from chunks
    let chunks = query_model_chunks(round)?;
    verify_all_chunks_present(&chunks)?;
    
    let full_model = assemble_chunks(chunks)?;
    Ok(full_model)
}
```

**Why Holochain for Models?**
- ✅ **Bandwidth efficiency**: Chunked distribution across DHT
- ✅ **No central bottleneck**: Clients fetch from peers
- ✅ **Resilience**: Model replicated across network
- ✅ **Gossip protocol**: Natural fit for model propagation

---

### Zome 4: **Economic Game** (`stake_slashing`)

```rust
#[hdk_entry_helper]
pub struct StakePosition {
    pub agent: AgentPubKey,
    pub amount: u64,
    pub locked_until: Timestamp,
    pub slashing_events: Vec<SlashingEvent>,
}

#[hdk_extern]
pub fn stake(amount: u64, rounds: u64) -> ExternResult<StakePosition> {
    let agent = agent_info()?.agent_latest_pubkey;
    
    // Lock stake for N rounds
    let locked_until = sys_time()? + Duration::from_rounds(rounds);
    
    let stake = StakePosition {
        agent: agent.clone(),
        amount,
        locked_until,
        slashing_events: vec![],
    };
    
    create_entry(&EntryTypes::Stake(stake.clone()))?;
    
    emit_signal(&Signal::StakeDeposited {
        agent,
        amount,
    })?;
    
    Ok(stake)
}

#[hdk_extern]
pub fn slash(
    agent: AgentPubKey,
    reason: SlashReason,
    amount: u64
) -> ExternResult<StakePosition> {
    // Only coordinator can slash
    verify_coordinator_authority()?;
    
    let mut stake = get_stake(&agent)?;
    
    stake.amount -= amount;
    stake.slashing_events.push(SlashingEvent {
        reason: reason.clone(),
        amount,
        timestamp: sys_time()?,
    });
    
    create_entry(&EntryTypes::Stake(stake.clone()))?;
    
    emit_signal(&Signal::AgentSlashed {
        agent,
        reason,
        amount,
    })?;
    
    Ok(stake)
}
```

**Why Holochain for Economics?**
- ✅ **Transparent slashing**: All agents can audit slashing events
- ✅ **Economic security**: Stake locked in cryptographic commitments
- ✅ **Game-theoretic incentives**: Enforced by validation rules
- ✅ **No trusted third party**: Network validates economic rules

---

## 🦀 Rust Core (Centralized Coordinator)

**Located**: `src/coordinator/` (new Rust workspace)

```rust
// src/coordinator/aggregator.rs
use risc0_zkvm::Receipt;

pub struct TrustWeightedAggregator {
    reputation_weights: HashMap<AgentPubKey, f64>,
}

impl TrustWeightedAggregator {
    pub async fn aggregate(
        &self,
        proofs: Vec<(AgentPubKey, Receipt, Vec<f32>)>,
    ) -> Result<Vec<f32>, AggregationError> {
        // 1. Verify all zkSTARK proofs (parallel)
        let verified_gradients = self.verify_all_proofs(proofs).await?;
        
        // 2. Compute PoGQ trust weights
        let weights = self.compute_pogq_weights(&verified_gradients)?;
        
        // 3. Multi-Krum selection
        let selected = multi_krum_select(&verified_gradients, weights, self.f)?;
        
        // 4. Trust-weighted average
        let aggregated = trust_weighted_mean(&selected, &self.reputation_weights)?;
        
        Ok(aggregated)
    }
    
    async fn verify_all_proofs(
        &self,
        proofs: Vec<(AgentPubKey, Receipt, Vec<f32>)>,
    ) -> Result<Vec<VerifiedGradient>, VerificationError> {
        // Parallel proof verification (CPU-bound)
        let tasks: Vec<_> = proofs
            .into_iter()
            .map(|(agent, receipt, gradient)| {
                tokio::task::spawn_blocking(move || {
                    gen7_zkstark::verify_receipt(&receipt)?;
                    Ok(VerifiedGradient { agent, gradient })
                })
            })
            .collect();
        
        // Await all verifications
        let results = futures::future::try_join_all(tasks).await?;
        Ok(results)
    }
}
```

**Why Rust Coordinator?**
- ✅ **Performance**: Aggregation is O(n²), needs to be fast
- ✅ **Correctness**: Type safety prevents Byzantine bugs
- ✅ **zkSTARK integration**: RISC Zero is Rust-native
- ✅ **Concurrency**: Tokio for async proof verification

---

## 🐍 Python Layer (ML & Research)

**Remains**: `src/zerotrustml/` (existing Python code)

```python
# Client-side integration with Holochain
from holochain_client import HolochainClient, AgentPubKey
import gen7_zkstark

class HolochainFederatedClient:
    def __init__(self, agent_pubkey: AgentPubKey):
        self.hc = HolochainClient("ws://localhost:8888")
        self.agent = agent_pubkey
        
    async def participate_in_round(self, round_number: int):
        # 1. Get global model from Holochain DHT
        global_model = await self.hc.call_zome(
            "models", "get_global_model", {"round": round_number}
        )
        
        # 2. Train locally (Python/PyTorch)
        gradient = self.train_local_model(global_model)
        
        # 3. Generate zkSTARK proof (Rust via PyO3)
        proof_bytes = gen7_zkstark.prove_gradient_zkstark(
            model_params=global_model.tolist(),
            gradient=gradient.tolist(),
            local_data=self.data.tolist(),
            local_labels=self.labels.tolist(),
            num_samples=len(self.data),
            input_dim=self.data.shape[1],
            num_classes=10,
            epochs=5,
            learning_rate=0.05,
        )
        
        # 4. Submit proof to Holochain
        proof_entry = {
            "round_number": round_number,
            "gradient_hash": hashlib.sha256(gradient.tobytes()).hexdigest(),
            "proof_bytes": proof_bytes,
            "timestamp": int(time.time()),
            "client_pubkey": str(self.agent),
            "model_commitment": global_model_hash,
        }
        
        await self.hc.call_zome(
            "gradient_proofs", "submit_proof", proof_entry
        )
```

**Why Keep Python for ML?**
- ✅ **Ecosystem**: PyTorch, NumPy, scikit-learn
- ✅ **Research velocity**: Rapid experimentation
- ✅ **Visualization**: Matplotlib, Plotly for analysis
- ✅ **Data pipelines**: torchvision, datasets

---

## 🔄 Integration Flow

### Round N Execution:

```
1. Coordinator (Rust)
   ├─> Publishes global_model_N to Holochain DHT
   └─> Emits "round_N_started" signal

2. Clients (Python) listen for signal
   ├─> Fetch global_model_N from DHT
   ├─> Train locally (PyTorch)
   ├─> Generate zkSTARK proof (Rust via PyO3)
   └─> Submit proof to Holochain (gradient_proofs zome)

3. Holochain DNA validates submissions
   ├─> Check proof freshness
   ├─> Check no double-submission
   ├─> Check stake requirement
   └─> Commit to submitter's source chain + DHT

4. Coordinator (Rust) collects proofs
   ├─> Query Holochain for all round_N proofs
   ├─> Verify zkSTARK proofs (parallel, Rust)
   ├─> Fetch reputation weights from Holochain
   ├─> Aggregate with PoGQ (trust-weighted)
   └─> Publish global_model_N+1 to Holochain

5. Holochain updates reputations
   ├─> Coordinator calls reputation zome
   ├─> Increment scores for accepted gradients
   └─> Slash stakes for rejected/malicious gradients
```

---

## 📁 Directory Structure

```
Mycelix-Core/0TML/
├── holochain-dna/              # NEW: Holochain workspace
│   ├── zomes/
│   │   ├── gradient_proofs/    # Proof submission
│   │   ├── reputation/         # Reputation tracking
│   │   ├── models/             # Model distribution
│   │   └── stake_slashing/     # Economic game
│   ├── dna.yaml
│   └── Cargo.toml
│
├── rust-coordinator/           # NEW: Rust aggregator
│   ├── src/
│   │   ├── aggregator.rs       # Trust-weighted aggregation
│   │   ├── pogq.rs             # PoGQ oracle
│   │   └── byzantine.rs        # Attack detection
│   └── Cargo.toml
│
├── gen7-zkstark/               # EXISTING: zkSTARK proofs
│   ├── guest/                  # zkVM guest code
│   ├── host/                   # Proof generation
│   └── bindings/               # Python bindings
│
└── src/zerotrustml/            # EXISTING: Python ML code
    ├── client.py               # Holochain integration
    ├── models.py               # PyTorch models
    └── experiments/            # Research code
```

---

## 🎯 Migration Path

### Phase 1: Holochain DNA (Week 1-2)
- [ ] Create `holochain-dna` workspace
- [ ] Implement `gradient_proofs` zome
- [ ] Implement `reputation` zome
- [ ] Local hApp testing

### Phase 2: Rust Coordinator (Week 3-4)
- [ ] Create `rust-coordinator` workspace
- [ ] Port aggregation algorithms to Rust
- [ ] Integrate with Holochain DNA
- [ ] Benchmark vs Python (target 10-100x speedup)

### Phase 3: Python Integration (Week 5)
- [ ] `holochain-client` Python library
- [ ] Update `client.py` to use Holochain
- [ ] Preserve existing PyTorch training code
- [ ] Integration tests

### Phase 4: Economic Layer (Week 6-7)
- [ ] Implement `stake_slashing` zome
- [ ] Game-theoretic attack simulations
- [ ] Economic security proofs

---

## 🎁 Benefits of This Architecture

| Feature | Centralized (Python) | Holochain + Rust |
|---------|---------------------|------------------|
| **Single Point of Failure** | ❌ Coordinator crash = system down | ✅ DHT continues gossip |
| **Bandwidth** | ❌ Coordinator bottleneck | ✅ P2P distribution |
| **Aggregation Speed** | ❌ O(n²) in Python = slow | ✅ O(n²) in Rust = fast |
| **Reputation Tampering** | ❌ Coordinator can manipulate | ✅ Cryptographically secured |
| **Proof Verification** | ❌ Sequential in Python | ✅ Parallel in Rust + RISC Zero |
| **Economic Security** | ❌ Requires separate blockchain | ✅ Built into Holochain DNA |
| **Sybil Resistance** | ❌ Manual identity management | ✅ AgentPubKey cryptographic IDs |
| **ML Ecosystem** | ✅ PyTorch/NumPy integration | ✅ Preserved in Python layer |

---

## 🚀 Next Steps

1. **Finish Gen-7 zkSTARK integration** (current - 90% done!)
2. **Create Holochain DNA prototype** (gradient_proofs zome)
3. **Benchmark Rust aggregator** vs Python
4. **Red-team the economic game** (stake/slash mechanics)
5. **Integrate all three layers**

---

**Status**: Architecture designed, ready to implement after Gen-7 Phase 2 completes ✨

**Build Progress**: Python wheel compiling in background (monitor `/tmp/maturin-pyo3-compat-build.log`)
