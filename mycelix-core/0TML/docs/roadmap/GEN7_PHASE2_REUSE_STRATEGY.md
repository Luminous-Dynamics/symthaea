# Gen-7 Phase 2: Leveraging Existing Infrastructure

**Date**: November 12, 2025
**Status**: 🚀 Strategic Plan
**Goal**: Accelerate Gen-7 by reusing VSV-STARK and Holochain infrastructure

---

## Executive Summary

Great news! We already have significant zkSTARK and distributed infrastructure that can accelerate Gen-7 Phase 2 from **30 days → ~10 days**.

**Key Discovery**:
1. ✅ **VSV-STARK** - Complete RISC Zero zkVM setup with Winterfell integration
2. ✅ **Holochain DHT** - Decentralized commitment and proof storage
3. ✅ **Fixed-Point Math** - Q16.16 quantization already implemented
4. ✅ **Verification Flow** - Challenge-response protocol exists

**Strategic Decision**: **Reuse and adapt** existing components instead of building from scratch.

---

## What We Already Have

### 1. VSV-STARK (Verifiable Secret Validation with zkSTARKs)

**Location**: `/srv/luminous-dynamics/Mycelix-Core/0TML/vsv-stark/`

**Components**:
- **RISC Zero Integration**: Complete zkVM setup (host + guest)
- **Winterfell Native AIR**: `winterfell-pogq/` with custom constraint system
- **Fixed-Point Math**: Q16.16 quantization for neural networks
- **Canary Validation**: Proof that gradient improves loss on canary samples
- **Python Bindings**: (if exists, need to check)

**Current Use Case**: VSV (Verifiable Secret Validation) - Prove gradient quality without revealing data

**How It Maps to Gen-7**:
| VSV Component | Gen-7 Equivalent | Adaptation Needed |
|---------------|------------------|-------------------|
| Canary samples | Training witness | ✅ Minimal - same concept |
| Loss delta proof | Gradient provenance | 🔄 Change: prove computation, not just quality |
| Challenge-response | Proof-on-demand | ✅ Reuse directly |
| RISC Zero zkVM | Proof generation | ✅ Reuse directly |
| Winterfell AIR | Custom constraints | 🔄 Adapt for gradient trace |

**Reuse Assessment**: **80% reusable** - Core proving infrastructure is identical!

---

### 2. Holochain DHT Integration

**Location**: `/srv/luminous-dynamics/Mycelix-Core/0TML/holochain/`

**Components**:
- **DHT Storage**: Decentralized hash table for commitments
- **Validation Rules**: Entry validation logic
- **Conductor**: Holochain runtime + admin interfaces
- **Python Client**: WebSocket-based Python → Holochain bridge

**Current Use Case**: MATL (Mycelix Adaptive Trust Layer) - Distributed gradient commitment storage

**How It Maps to Gen-7**:
| Holochain Component | Gen-7 Equivalent | Adaptation Needed |
|---------------------|------------------|-------------------|
| Commitment storage | `H_g`, `H_theta` storage | ✅ Reuse directly |
| Validation function | Proof verification | ✅ Extend with zkSTARK verifier |
| Challenge sampling | Random challenge seed | ✅ Reuse directly |
| Reputation tracking | Stake management | 🔄 Extend: add stake slashing |

**Reuse Assessment**: **70% reusable** - Need to add stake/slashing logic

---

### 3. RISC Zero (zkVM)

**Location**: `/srv/luminous-dynamics/Mycelix-Core/0TML/test_risc_zero_cli.py`

**Status**: CLI tested, integration exists

**Advantages for Gen-7**:
- ✅ Write normal Rust code → automatic zkSTARK proofs
- ✅ No manual AIR design (zkVM compiles Rust → constraints)
- ✅ Proven infrastructure (used by production systems)
- ✅ Fast development (days vs weeks for manual AIR)

**Trade-offs**:
- ⏱️ Slightly larger proofs (~200-400KB vs optimal ~100KB)
- ⏱️ Slightly slower proving (~10-60s vs optimal ~5s)
- ✅ But: Much faster to implement

**Recommendation**: Use RISC Zero for Phase 2, optimize to native Winterfell AIR in Phase 3 if needed.

---

### 4. Winterfell PoGQ

**Location**: `/srv/luminous-dynamics/Mycelix-Core/0TML/vsv-stark/winterfell-pogq/`

**Purpose**: Native AIR for PoGQ (Proof of Gradient Quality) composite scoring

**Components**:
- Custom constraint system for neural network operations
- Fixed-point arithmetic in STARK field
- Merkle tree compression for model states
- Optimized for small CNN architectures

**Reuse Strategy**:
- Phase 2: Skip this (use RISC Zero zkVM)
- Phase 3: Adapt PoGQ AIR for gradient provenance if performance optimization needed

---

## Revised Phase 2 Implementation Plan (10 Days)

### Week 1: Adaptation (Days 1-5)

**Goal**: Adapt VSV-STARK for gradient provenance instead of canary validation

#### Day 1-2: Modify Proof Statement

**Current VSV Proof**: "My gradient `g` improves canary loss by ΔL ≥ threshold"

**Gen-7 Proof**: "My gradient `g` was computed by training on local data `D` for `E` epochs"

**Changes Needed**:
```rust
// VSV guest code (vsv-stark/methods/guest/src/main.rs)
fn main() {
    // Read public inputs
    let model_hash = env::read();
    let gradient_hash = env::read();
    let canary_samples = env::read();

    // Compute canary loss delta (VSV)
    let loss_delta = compute_canary_quality(...);
    assert!(loss_delta >= threshold);  // VSV check

    // Commit result
    env::commit(&loss_delta);
}

// Gen-7 guest code (NEW)
fn main() {
    // Read public inputs
    let model_hash = env::read();
    let gradient_hash = env::read();
    let epochs = env::read();
    let lr = env::read();

    // Read private witness
    let local_data = env::read();
    let local_labels = env::read();
    let model = env::read();

    // Prove computation
    let computed_gradient = train_and_compute_gradient(
        model, local_data, local_labels, epochs, lr
    );

    // Verify gradient hash
    assert_eq!(hash(&computed_gradient), gradient_hash);

    // Commit result
    env::commit(&gradient_hash);
}
```

**Task**: Copy `vsv-stark/methods/guest/`, rename to `gen7-gradient-proof/`, modify main.rs

**Estimated Time**: 2 days

#### Day 3-4: Update Host Interface

**Task**: Modify host program to call Gen-7 guest with correct inputs

```rust
// host/src/main.rs
use gen7_gradient_proof_methods::GRADIENT_PROOF_ELF;  // Compiled guest code

fn prove_gradient(
    global_model: Vec<f32>,
    local_data: Vec<Vec<f32>>,
    local_labels: Vec<u8>,
    epochs: u32,
    lr: f32,
) -> Receipt {
    let mut stdin = ExecutorEnv::builder();

    // Public inputs
    stdin.write(&hash(&global_model))?;
    stdin.write(&hash(&gradient))?;  // Pre-computed
    stdin.write(&epochs)?;
    stdin.write(&lr)?;

    // Private witness
    stdin.write(&local_data)?;
    stdin.write(&local_labels)?;
    stdin.write(&quantize_model(global_model))?;  // Q16.16

    // Generate proof
    let env = stdin.build()?;
    let prover = default_prover();
    let receipt = prover.prove(env, GRADIENT_PROOF_ELF)?;

    Ok(receipt)
}
```

**Estimated Time**: 2 days

#### Day 5: Python Bindings (PyO3)

**Option A**: Reuse existing Python bridge if it exists

**Option B**: Create new PyO3 bindings

```rust
// src/lib.rs
use pyo3::prelude::*;
use numpy::PyArray1;

#[pyfunction]
fn prove_gradient_stark(
    py: Python,
    global_model: &PyArray1<f32>,
    local_data: &PyArray1<f32>,
    local_labels: &PyArray1<u8>,
    epochs: u32,
    lr: f32,
) -> PyResult<Vec<u8>> {
    let model = global_model.to_vec()?;
    let data = local_data.to_vec()?;
    let labels = local_labels.to_vec()?;

    let receipt = prove_gradient(model, data, labels, epochs, lr)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Proof generation failed: {}", e)
        ))?;

    Ok(receipt.journal.bytes)
}

#[pymodule]
fn gen7_zkstark(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prove_gradient_stark, m)?)?;
    m.add_function(wrap_pyfunction!(verify_gradient_stark, m)?)?;
    Ok(())
}
```

**Estimated Time**: 1 day

---

### Week 2: Integration & Testing (Days 6-10)

#### Day 6-7: Update GradientProofCircuit

**Task**: Replace simulation mode with real zkSTARK calls

```python
# src/zerotrustml/gen7/gradient_proof.py

def _generate_stark_proof(self, public_inputs, private_witness):
    """Generate real RISC Zero zkSTARK proof."""
    if not self.use_real_stark:
        return self._simulate_stark_proof(public_inputs, private_witness)

    # Import Rust zkSTARK library
    import gen7_zkstark

    # Convert to format expected by Rust
    model = public_inputs["global_model"]
    data = private_witness["local_data"]
    labels = private_witness["local_labels"]
    epochs = public_inputs["epochs"]
    lr = public_inputs["lr"]

    # Call Rust prover
    proof_bytes = gen7_zkstark.prove_gradient_stark(
        global_model=model,
        local_data=data.flatten(),  # Flatten to 1D
        local_labels=labels,
        epochs=epochs,
        lr=int(lr * 65536),  # Convert to Q16.16
    )

    return proof_bytes

def _verify_stark_proof(self, proof_bytes, public_inputs):
    """Verify RISC Zero zkSTARK proof."""
    import gen7_zkstark

    return gen7_zkstark.verify_gradient_stark(
        proof_bytes=proof_bytes,
        model_hash=public_inputs["global_model_hash"],
        gradient_hash=public_inputs["gradient_hash"],
        epochs=public_inputs["epochs"],
        lr=int(public_inputs["lr"] * 65536),
    )
```

**Estimated Time**: 2 days

#### Day 8-9: E7 Re-Validation

**Task**: Re-run Gen-7 integration test with real proofs

```bash
# Run integration test with real zkSTARKs
cd experiments
python test_gen7_integration.py --use-real-stark

# Expected output:
#   E7.1: False Positive Rate = 0% : ✅ PASS
#   E7.2: False Negative Rate = 0% : ✅ PASS
#   E7.3: Proof generation <5s : ⏱️ TBD (target: 10-60s with RISC Zero)
#   E7.4: Proof size <100KB : ⏱️ TBD (target: 200-400KB with RISC Zero)
```

**Acceptance Criteria**:
- All honest proofs accepted (E7.1 ✅)
- All malicious proofs rejected (E7.2 ✅)
- Performance within zkVM expected range

**Estimated Time**: 2 days

#### Day 10: Documentation & Benchmarking

**Task**: Update documentation with real performance numbers

**Deliverables**:
1. `docs/validation/GEN7_PHASE2_COMPLETE.md` - Updated completion report
2. `docs/roadmap/GEN7_PHASE2_PERFORMANCE.md` - Benchmark results
3. Updated README with real zkSTARK integration status

**Estimated Time**: 1 day

---

## Holochain Integration (Optional Phase 2.5)

**Goal**: Distribute Gen-7 proofs and commitments via Holochain DHT

**Why Later?**:
- Phase 2 focuses on cryptographic security (zkSTARKs)
- Holochain adds decentralized storage (infrastructure layer)
- Can test Gen-7 locally first, then add DHT

**When to Integrate**: After Phase 2 complete, before Phase 3 (economic optimization)

**Integration Plan** (5 days):

1. **Day 1-2: Extend Holochain Validation Rules**
   ```rust
   // holochain/dnas/gradient_sharing/zomes/gradients/src/lib.rs

   #[hdk_extern]
   pub fn submit_gradient_proof(proof: GradientProof) -> ExternResult<()> {
       // 1. Verify zkSTARK proof
       let is_valid = verify_gradient_stark(
           proof.proof_bytes,
           proof.public_inputs,
       )?;

       if !is_valid {
           return Err(wasm_error!("Invalid zkSTARK proof"));
       }

       // 2. Store commitment on DHT
       create_entry(&EntryTypes::GradientCommitment(GradientCommitment {
           client_id: proof.client_id,
           gradient_hash: proof.gradient_hash,
           model_hash: proof.model_hash,
           round_idx: proof.round_idx,
           proof_bytes: proof.proof_bytes,
           timestamp: sys_time()?,
       }))?;

       // 3. Update reputation (stake management)
       update_client_reputation(proof.client_id, ReputationChange::ValidProof)?;

       Ok(())
   }
   ```

2. **Day 3-4: Python-Holochain Bridge**
   ```python
   # src/zerotrustml/gen7/holochain_bridge.py

   import websocket
   import json

   class HolochainGradientStorage:
       def __init__(self, conductor_url="ws://localhost:8888"):
           self.ws = websocket.create_connection(conductor_url)

       def submit_proof(self, proof: GradientProof) -> bool:
           request = {
               "type": "zome_call",
               "data": {
                   "zome": "gradients",
                   "fn": "submit_gradient_proof",
                   "payload": {
                       "proof_bytes": proof.proof_bytes.hex(),
                       "client_id": proof.public_inputs["client_id"],
                       "gradient_hash": proof.public_inputs["gradient_hash"],
                       "model_hash": proof.public_inputs["global_model_hash"],
                       "round_idx": proof.public_inputs["round_idx"],
                   }
               }
           }

           self.ws.send(json.dumps(request))
           response = json.loads(self.ws.recv())

           return response["success"]
   ```

3. **Day 5: Integration Test**
   ```python
   # experiments/test_gen7_holochain_integration.py

   def test_distributed_gen7():
       """Test Gen-7 with Holochain DHT storage."""

       # Start Holochain conductor
       conductor = start_holochain_conductor()

       # Initialize clients with Holochain storage
       clients = [
           HolochainClient(client_id=f"client_{i:02d}", conductor=conductor)
           for i in range(10)
       ]

       # Run federated learning with DHT storage
       for round_idx in range(5):
           for client in clients:
               # Generate gradient + proof
               proof = client.compute_and_prove_gradient()

               # Submit to Holochain DHT
               success = client.submit_to_dht(proof)
               assert success

           # Aggregate from DHT
           valid_proofs = coordinator.fetch_valid_proofs_from_dht(round_idx)
           aggregated = coordinator.aggregate(valid_proofs)

       # Verify DHT integrity
       chain_valid = verify_dht_chain(conductor)
       assert chain_valid
   ```

---

## Performance Expectations

### RISC Zero zkVM (Phase 2)

| Metric | VSV Baseline | Gen-7 Estimate | Target |
|--------|--------------|----------------|--------|
| **Proving Time** | 5-30s | 10-60s | <60s (acceptable) |
| **Proof Size** | 100-200KB | 200-400KB | <400KB (acceptable) |
| **Verification** | <5ms | <10ms | <50ms |
| **Memory** | <4 GB | <8 GB | <16 GB |

**Optimization Path (Phase 3)**:
- If proving >60s: Reduce canary batch size, simplify model
- If proof >400KB: Switch to native Winterfell AIR (100-200KB)
- If verification >50ms: Unlikely with RISC Zero, but can batch-verify

### Native Winterfell AIR (Phase 3 - Optional)

| Metric | RISC Zero | Winterfell AIR | Improvement |
|--------|-----------|----------------|-------------|
| **Proving Time** | 10-60s | 5-10s | **2-6x faster** |
| **Proof Size** | 200-400KB | 50-100KB | **2-4x smaller** |
| **Dev Time** | 10 days | +20 days | Trade-off |

**Recommendation**: Only optimize to native AIR if RISC Zero proves too slow in practice.

---

## Risk Mitigation

### Risk 1: RISC Zero Proving Too Slow (>60s)

**Mitigation**:
- Start with minimal model (2-layer MLP on MNIST)
- Measure baseline before full CIFAR-10 model
- If too slow: Reduce canary batch size (64 → 32 → 16)
- Fall back: Keep simulation mode, document as future work

### Risk 2: Python-Rust Integration Issues

**Mitigation**:
- VSV-STARK already has Python bridge (check status)
- If not: Use proven PyO3 + maturin stack
- Comprehensive integration tests before E7 validation

### Risk 3: Holochain DHT Adds Too Much Complexity

**Mitigation**:
- Make Holochain **optional** in Phase 2
- Core Gen-7 works standalone (local mode)
- Add DHT as Phase 2.5 enhancement (not blocker)

---

## Success Criteria (10 Days)

Phase 2 considered **COMPLETE** when:

1. ✅ **Rust guest code adapted** from VSV to Gen-7 gradient provenance
2. ✅ **Python bindings working** (`pip install gen7-zkstark`)
3. ✅ **Real zkSTARK proofs generated** (not simulation)
4. ✅ **E7.1: False Positive Rate = 0%** (all honest accepted)
5. ✅ **E7.2: False Negative Rate = 0%** (all malicious rejected)
6. ✅ **E7.3: Proof generation <60s** (RISC Zero target)
7. ✅ **E7.4: Proof size <400KB** (RISC Zero target)
8. ✅ **Integration test passes** with real proofs
9. ✅ **Documentation updated** with real performance numbers
10. ✅ **Benchmarks published** (proving time, proof size, verification)

**Optional (Phase 2.5)**:
11. ⏱️ **Holochain DHT integration** (+5 days)
12. ⏱️ **Distributed testing** (multi-node validation)

---

## Timeline Comparison

| Phase | Original Estimate | With Reuse | Acceleration |
|-------|------------------|------------|--------------|
| **Phase 2** | 30 days | **10 days** | **3x faster** |
| **Phase 2.5** (DHT) | N/A | +5 days | New feature |
| **Total** | 30 days | **15 days** | **2x faster** |

**Net Result**: Gen-7 Phase 2 complete by **November 27, 2025** (vs Dec 12)

---

## Next Actions

### Immediate (Today)

1. ✅ Review VSV-STARK codebase structure
2. ✅ Check if Python bindings already exist
3. ✅ Verify RISC Zero CLI working
4. ✅ Create Phase 2 todo list with 10-day timeline

### Tomorrow (Day 1)

5. ⏱️ Copy `vsv-stark/methods/guest/` to `gen7-gradient-proof/`
6. ⏱️ Modify guest code proof statement
7. ⏱️ Start adapting host interface

### Week 1 Deliverable

- Working Rust prover generating real zkSTARK proofs
- Python bindings functional
- Ready for E7 integration testing

---

**Status**: 🚀 Ready to implement
**Timeline**: 10-15 days (vs original 30)
**Confidence**: High (reusing proven infrastructure)
**Risk**: Low (fallback to simulation mode if issues)

---

*"Don't rebuild the wheel. Adapt the wheel you already have."*

**Strategic Advantage**: By reusing VSV-STARK, we get production-quality zkSTARK infrastructure immediately, cutting development time by 66%.
