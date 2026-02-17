# VSV with ZK-STARKs: Practical Implementation Guide

**Status**: Implementation Planning
**Timeline**: 1-2 weeks for proof-of-concept
**Goal**: Sub-30s proving, <200KB proofs, <5ms verification

---

## Why STARKs for VSV?

**Alignment with Zero-Trust Philosophy**:
- ✅ **Transparent**: No trusted setup ceremony (perfect for trustless FL)
- ✅ **Post-quantum**: Hash-based security (future-proof)
- ✅ **Fast verification**: 1-10ms for DHT validators
- ✅ **Simple UX**: Validators just verify receipts, no complex ceremony
- ✅ **Rotation-friendly**: Model changes don't require new setup

**Trade-offs**:
- ❌ Larger proofs than Groth16 (100-400KB vs 200 bytes)
- ❌ Slower proving than Groth16 (5-60s vs <1s)
- ✅ But: Can optimize with fixed-point quantization + small canary model
- ✅ And: No setup = less operational complexity

---

## 1. Minimal Viable ZK-STARK Receipt for VSV

### What We Prove (Don't Over-Prove!)

**Claim**: "My committed gradient g improves canary loss for the committed model θ_t"

**Specifically**:
```
1. Start with model θ_t and canary samples S_c
2. Compute loss before: L_before = Loss(θ_t, S_c)
3. Apply SGD step: θ' = θ_t - η * g
4. Compute loss after: L_after = Loss(θ', S_c)
5. Prove: ΔL = L_before - L_after ≥ Δ_min (threshold)
```

**Why This Works**:
- Byzantine nodes committed to g BEFORE challenge
- Can't fake ΔL without actually computing correct gradient
- Verifier checks commitments match + ΔL threshold

### Public Inputs (On-Chain/DHT)

```rust
pub struct PublicInputs {
    H_g: Hash,           // Commitment to gradient g
    H_theta: Hash,       // Commitment to model θ_t
    H_Sc: Hash,          // Commitment to canary samples S_c
    eta: FixedPoint,     // Learning rate η
    delta_min: FixedPoint, // Minimum required loss improvement
    challenge_seed: u64, // Deterministic canary selection
}
```

**Size**: ~256 bytes (8 × 32-byte hashes/values)

### Witness (Private, Inside Proof)

```rust
struct Witness {
    g: Vec<FixedPoint>,        // Gradient (committed via H_g)
    theta_t: Vec<FixedPoint>,  // Model weights (committed via H_theta)
    S_c: Vec<Sample>,          // Canary samples (committed via H_Sc)
}
```

**Size**: ~1-10 MB depending on model/canary size (stays inside proof)

### Proof Statement (What Circuit Checks)

```python
def verify_gradient_quality(public: PublicInputs, witness: Witness) -> bool:
    # 1. Verify commitments
    assert hash(witness.g) == public.H_g
    assert hash(witness.theta_t) == public.H_theta
    assert hash(witness.S_c) == public.H_Sc

    # 2. Forward pass BEFORE gradient
    L_before = fixed_point_forward(witness.theta_t, witness.S_c)

    # 3. Apply SGD step
    theta_prime = [t - public.eta * g for t, g in zip(witness.theta_t, witness.g)]

    # 4. Forward pass AFTER gradient
    L_after = fixed_point_forward(theta_prime, witness.S_c)

    # 5. Check loss improvement
    delta_L = L_before - L_after
    assert delta_L >= public.delta_min

    return True
```

**Circuit Complexity**: 2 forward passes + 1 vector subtraction

---

## 2. Circuit Optimization (Make It Fast Enough)

### 2.1 Quantization to Fixed-Point

**Why**: STARKs work best with integer arithmetic (no floating-point!)

**Approach**:
```rust
// Q16.16 fixed-point (16 integer bits, 16 fractional bits)
type FixedPoint = i32;

fn to_fixed(f: f32) -> FixedPoint {
    (f * 65536.0) as i32
}

fn from_fixed(fp: FixedPoint) -> f32 {
    (fp as f32) / 65536.0
}

fn mul_fixed(a: FixedPoint, b: FixedPoint) -> FixedPoint {
    ((a as i64 * b as i64) >> 16) as i32
}
```

**Impact**:
- 10-100x faster proving vs float32
- Acceptable accuracy for loss delta measurement
- Naturally fits STARK field arithmetic

### 2.2 Canary Model Architecture

**Frozen, Tiny CNN for CIFAR-10**:
```python
class CanaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # No batch norm!
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Use ReLU (piecewise linear)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # No softmax! Use MSE on logits
```

**Key Design Choices**:
- No batch normalization (complicates circuit)
- ReLU activation (piecewise linear = cheap)
- Small feature dimensions (16, 32, 64)
- MSE loss on logits (no exp/softmax)
- Total params: ~100K (vs 11M for ResNet-18)

### 2.3 Loss Function (Circuit-Friendly)

**Mean Squared Error on Logits**:
```python
def mse_loss_fixed(logits: List[FixedPoint], labels: List[int]) -> FixedPoint:
    """
    MSE between logits and one-hot labels (no softmax/exp!)
    """
    total_error = 0
    for i, logit_vec in enumerate(logits):
        true_class = labels[i]
        # One-hot target: [0, 0, ..., 1, ..., 0]
        for j, logit in enumerate(logit_vec):
            target = 1 if j == true_class else 0
            error = logit - to_fixed(target)
            total_error += mul_fixed(error, error)  # Square error

    return total_error / len(logits)  # Mean
```

**Canonical Decision**: MSE-on-logits remains the mandated loss for all Phase 1 zkVM/STARK implementations. Revisit only after a quantified cost analysis of any cross-entropy approximation.

**Why MSE Instead of Cross-Entropy**:
- No exp() computation (saves 100x+ constraints)
- No log() computation (another 100x+ constraints)
- Still measures gradient quality effectively
- Slightly less accurate but vastly cheaper

**Alternative (if needed)**: Piecewise-linear approximation of cross-entropy:
```python
def approx_cross_entropy(logits, labels):
    # Use lookup table for small value ranges
    # Or piece-wise linear segments
    pass
```

### 2.4 Batching Strategy

**Fixed Batch Size**: 32 or 64 canary samples per proof
- Amortizes circuit overhead
- Deterministic circuit (same AIR for all proofs)
- ~2-4 KB of canary data per proof

**Sample Selection**:
```python
def select_canary_samples(challenge_seed: int, dataset: List, k: int = 32) -> List:
    """
    Deterministically select k samples using challenge seed.
    Both prover and verifier can reproduce this.
    """
    np.random.seed(challenge_seed)
    indices = np.random.choice(len(dataset), size=k, replace=False)
    return [dataset[i] for i in indices]
```

---

## 3. Expected Performance Envelope

### 3.1 Realistic Targets (Rule of Thumb)

**Small CNN on CIFAR-10-scale canaries (32-64 images, Q16.16)**:

| Metric | Target | Acceptable | Worst Case |
|--------|--------|------------|------------|
| **Proving Time** | <30s | 5-60s | <120s |
| **Proof Size** | <200 KB | 100-400 KB | <1 MB |
| **Verification Time** | <5ms | 1-10ms | <50ms |
| **Memory Usage** | <4 GB | <8 GB | <16 GB |

**Challenge Window**: 30-120 seconds per round without killing UX

### 3.2 Optimization Knobs

**If proving is too slow**:
1. Reduce canary batch size (64 → 32 → 16)
2. Simplify model (32 filters → 16 → 8)
3. Use Q8.8 instead of Q16.16 (less precision)
4. GPU proving (if available)
5. Parallel proving (split canary batches)

**If proof is too large**:
1. More aggressive quantization (Q16.16 → Q8.8)
2. Smaller model (reduce layer sizes)
3. Recursive composition (prove in stages, fold)

**If verification is too slow**:
1. This shouldn't happen with STARKs (inherently fast)
2. But if it does: batch-verify multiple proofs together

---

## 4. Validator Flow (Drop-in for DHT Validators)

### 4.1 Challenge Phase

```rust
// Validator samples node j for challenge
fn initiate_challenge(node_j: NodeID, dht: &DHT) -> Challenge {
    // 1. Fetch commitments from DHT
    let commitment = dht.get_commitment(node_j)?;
    assert!(commitment.H_g.is_valid());
    assert!(commitment.H_theta.is_valid());

    // 2. Generate deterministic canary seed
    let seed = hash(commitment.H_g || current_round || validator_id);

    // 3. Send challenge
    Challenge {
        node: node_j,
        seed: seed,
        deadline: now() + 120s,
        delta_min: 0.001,  // Minimum required improvement
    }
}
```

### 4.2 Response Phase

```rust
// Node j generates STARK proof
fn respond_to_challenge(challenge: Challenge, node_state: &NodeState) -> Response {
    // 1. Derive canary samples from seed
    let canary_samples = select_canary_samples(challenge.seed, &CANARY_DATASET, 32);

    // 2. Compute loss delta
    let theta_t = node_state.current_model;
    let g = node_state.committed_gradient;
    let L_before = fixed_forward(&theta_t, &canary_samples);
    let theta_prime = apply_sgd(&theta_t, &g, LEARNING_RATE);
    let L_after = fixed_forward(&theta_prime, &canary_samples);
    let delta_L = L_before - L_after;

    // 3. Generate STARK proof
    let public_inputs = PublicInputs {
        H_g: node_state.H_g,
        H_theta: node_state.H_theta,
        H_Sc: hash(&canary_samples),
        eta: LEARNING_RATE,
        delta_min: challenge.delta_min,
        challenge_seed: challenge.seed,
    };

    let witness = Witness {
        g: g,
        theta_t: theta_t,
        S_c: canary_samples,
    };

    let proof = stark_prove(public_inputs, witness)?;  // 5-60s

    Response {
        node: challenge.node,
        proof: proof,  // ~200 KB
        delta_L: delta_L,  // Optional: for transparency
    }
}
```

### 4.3 Verification Phase

```rust
// Validator verifies STARK proof
fn verify_response(challenge: Challenge, response: Response) -> VerificationResult {
    // 1. Check proof validity
    let public_inputs = reconstruct_public_inputs(&challenge, &response);
    let is_valid = stark_verify(&public_inputs, &response.proof)?;  // <5ms

    if !is_valid {
        return VerificationResult::Invalid(
            "STARK proof failed verification"
        );
    }

    // 2. Check commitments match DHT records
    if public_inputs.H_g != dht.get_commitment(response.node).H_g {
        return VerificationResult::Invalid(
            "Gradient commitment mismatch"
        );
    }

    // 3. Record result
    return VerificationResult::Valid {
        node: response.node,
        passed: true,
        delta_L: response.delta_L,
    };
}
```

### 4.4 Reputation Update

```rust
// Update node reputation based on challenge results
fn update_reputation(node: NodeID, result: VerificationResult) {
    match result {
        VerificationResult::Valid { passed: true, .. } => {
            reputation[node] += 1;  // Passed challenge
        }
        VerificationResult::Valid { passed: false, .. } => {
            reputation[node] -= 10;  // Failed challenge (Byzantine detected)
            if reputation[node] < THRESHOLD {
                slash(node);  // Remove from network
            }
        }
        VerificationResult::Invalid(_) => {
            reputation[node] -= 100;  // Critical failure
            slash(node);  // Immediate removal
        }
    }
}
```

---

## 5. Implementation Roadmap (1-2 Weeks)

### Week 1: Foundation

**Day 1-2: Fix the Statement**
- [ ] Finalize proof claim (loss delta ≥ threshold)
- [ ] Define public inputs struct
- [ ] Define witness struct
- [ ] Write formal security argument (informal proof)

**Day 3-4: Freeze Canary Model Spec**
- [ ] Design CanaryCNN architecture (layers, activations)
- [ ] Implement in PyTorch (float32 baseline)
- [ ] Convert to fixed-point Q16.16
- [ ] Validate accuracy loss is acceptable (<5%)

**Day 5-7: Pick STARK Path**

**Option A: zkVM Route (Easiest)**
- [ ] Write canary loss-delta checker in Rust
- [ ] Compile to STARK-proven guest (RISC Zero or similar)
- [ ] Get receipt from normal Rust code
- [ ] Pro: Fast prototyping, reuse normal code
- [ ] Con: Less optimized, larger proofs

**Option B: Native AIR Route (Faster Long-Term)**
- [ ] Hand-craft AIR for matmul + ReLU + MSE
- [ ] Implement custom constraint system
- [ ] Pro: 10-100x faster proofs
- [ ] Con: Weeks of development, harder to debug

**Recommendation**: Start with zkVM (Option A), optimize to native AIR later if needed

### Week 2: Prototype

**Day 8-10: MNIST Proof-of-Concept**
- [ ] Generate 16-sample canaries from MNIST
- [ ] Train tiny model (2-layer MLP)
- [ ] Generate STARK proof for one gradient
- [ ] Measure: proving time, proof size, verification time

**Target Budget**:
- ≤ 30s proving time
- ≤ 200 KB proof size
- ≤ 5ms verification time

**Day 11-12: VSV Integration**
- [ ] Wire STARK verifier into validator flow
- [ ] Test challenge-response protocol
- [ ] Simulate Byzantine node (fails challenges)
- [ ] Simulate honest node (passes challenges)

**Day 13-14: Performance Tuning**
- [ ] Optimize model (reduce layers, filters)
- [ ] Optimize quantization (Q16.16 vs Q8.8)
- [ ] Optimize batch size (16 vs 32 vs 64)
- [ ] Document final performance numbers

---

## 6. Optional Refinements (Don't Do First!)

### 6.1 Batched Multi-Challenge Receipts

**Idea**: One proof for k canary sets (amortize prover cost)

```rust
struct BatchedProof {
    challenges: Vec<Challenge>,  // k challenges
    proofs: Vec<STARKProof>,     // k proofs
    // Or: single aggregated proof (if using recursive SNARKs)
}
```

**Benefit**: k challenges in ~2k proving time (vs k × prover time)
**Complexity**: Requires proof aggregation or recursive composition

### 6.2 Recursive Folding for Rounds Aggregation

**Idea**: Prove "I passed challenges in rounds 1-10" with one proof

```rust
// STARK-inside-STARK (IVC-style)
fn aggregate_proofs(proofs: Vec<STARKProof>) -> STARKProof {
    // Prove: "I verified k STARK proofs"
    // Output: Single aggregated proof
    // Size: ~constant regardless of k
}
```

**Benefit**: Constant-size audit trail over many rounds
**Complexity**: Requires Nova/SuperNova style folding or STARK recursion

### 6.3 Directional Derivative Variant

**Alternative Approach**: Instead of 2 forward passes, prove gradient direction

```rust
// Prove: g · ∇L_canary ≥ 0 (gradient aligns with loss reduction)
fn prove_gradient_direction(g: Gradient, canary_grad: Gradient) -> bool {
    let dot_product = dot(g, canary_grad);
    assert!(dot_product >= 0);  // Same direction
}
```

**Benefit**: Smaller circuit (no second forward pass)
**Cost**: Must prove backprop correct (might be more expensive overall)

---

## 7. Why This Beats Mode 1 & Mode 2

### Mode 1 (Ground Truth) Limitations

| Limitation | VSV Solution |
|------------|--------------|
| Server needs validation set | Public canary samples (no trust) |
| Single point of trust | Decentralized validators |
| Server can cheat | Cryptographic proofs (can't fake) |

### Mode 2 (TEE) Limitations

| Limitation | VSV Solution |
|------------|--------------|
| Expensive hardware | Software-only STARKs |
| Supply chain attacks | No hardware dependency |
| Limited scalability | Scales to millions of nodes |

### VSV Unique Advantages

1. **Zero Trust**: No validation set, no TEE, no trusted party
2. **Decentralized**: DHT validators, no central aggregation
3. **Cost-Effective**: $0 hardware, ~$0.01/proof compute
4. **Scalable**: O(log N) verification via sampling
5. **Future-Proof**: Post-quantum secure (hash-based)

---

## 8. Security Argument (Informal)

### Claim: Byzantine nodes cannot fake high-quality gradients

**Proof Sketch**:

1. **Commitment prevents selective response**:
   - Node commits H(g || nonce) BEFORE seeing challenge
   - Cannot change g after seeing canary seed
   - Binding property of hash prevents equivocation

2. **STARK proof prevents fabrication**:
   - Prover must actually compute loss delta
   - Cannot fake ΔL without knowing g, θ_t, S_c
   - Soundness of STARK guarantees honest computation

3. **Challenge unpredictability prevents pre-computation**:
   - Validator chooses seed = H(H_g || round || validator_id)
   - Node cannot predict which canary samples before commit
   - Must commit to one gradient (can't have multiple)

4. **Reputation prevents Sybil attacks**:
   - Creating identity costs stake (PoS) or computation (PoW)
   - Failed challenges slash reputation
   - Need sustained honesty to accumulate reputation

**Attack Scenarios & Mitigations**:

| Attack | Why It Fails |
|--------|--------------|
| Pre-compute canary responses | Can't predict seed before commit |
| Submit good canary grad, bad aggregation grad | Commitment is to ONE gradient |
| Fake STARK proof | Soundness property prevents |
| Sybil to avoid challenges | Stake/PoW + random sampling |
| Collude with validators | Need >50% validator control |

---

## 9. Comparison: STARKs vs Groth16 vs PLONK

| Property | Groth16 | PLONK | STARK (Recommended) |
|----------|---------|-------|---------------------|
| **Setup** | ⚠️ Trusted (per circuit) | ⚠️ Universal (one-time) | ✅ Transparent (none) |
| **Proving** | ✅ ~1s | ⏱️ ~5s | ⏱️ 5-60s |
| **Proof Size** | ✅ ~200 bytes | ✅ ~400 bytes | ⏱️ 100-400 KB |
| **Verification** | ✅ ~1ms | ✅ ~5ms | ✅ 1-10ms |
| **Post-Quantum** | ❌ No (pairing-based) | ❌ No (pairing-based) | ✅ Yes (hash-based) |
| **Rotation** | ❌ New setup per model | ✅ Same setup | ✅ No setup |
| **DHT-Friendly** | ⏱️ OK | ⏱️ OK | ✅ Excellent |

**Verdict**: STARKs win for VSV due to transparency + post-quantum + rotation-friendliness

**Fallback**: Keep Groth16 interface for special deployments (trusted setup acceptable, need <1s proving)

---

## 10. Next Steps (This Week)

### Immediate Actions (Day 1-3)

1. **Set up STARK development environment**:
   ```bash
   # Option A: RISC Zero (zkVM route)
   cargo install risc0

   # Option B: Winterfell (native AIR route)
   git clone https://github.com/facebook/winterfell
   ```

2. **Implement fixed-point CanaryCNN**:
   - Convert PyTorch model to Q16.16
   - Test accuracy loss vs float32 baseline
   - Target: <5% accuracy degradation

3. **Write proof-of-concept prover**:
   - Simple Rust program: takes (g, θ_t, S_c) → ΔL
   - No STARK yet, just validate logic
   - Measure baseline compute time (without proving)

### Stretch Goals (Day 4-7)

4. **Generate first STARK proof**:
   - Wrap PoC in zkVM or native AIR
   - Produce receipt for one canary challenge
   - Measure: proving time, proof size

5. **Implement validator side**:
   - STARK verifier in Rust
   - DHT mock (just in-memory for now)
   - Challenge-response flow

6. **MNIST end-to-end test**:
   - 20 nodes (13 honest, 7 Byzantine)
   - 5 rounds of challenges
   - Measure detection accuracy

---

## 11. Resources & Tools

### STARK Libraries

**Recommended (zkVM Route)**:
- **RISC Zero**: https://github.com/risc0/risc0
  - Easiest to get started
  - Write normal Rust, get STARK proofs
  - ~100-400 KB proofs
  - ~5-60s proving time

**Alternative (Native AIR)**:
- **Winterfell**: https://github.com/facebook/winterfell (Facebook's STARK library)
- **Plonky2**: https://github.com/mir-protocol/plonky2 (faster recursion)
- **Starkware Cairo**: https://www.cairo-lang.org/ (mature ecosystem)

### Fixed-Point Libraries

- **fixed crate**: https://crates.io/crates/fixed (Rust)
- **numpy.int16**: https://numpy.org/doc/stable/user/basics.types.html (Python prototyping)

### Learning Resources

- **STARK101**: https://starkware.co/stark-101/ (interactive tutorial)
- **ZK MOOC**: https://zk-learning.org/ (Stanford course)
- **A16Z Crypto Canon**: https://a16zcrypto.com/zero-knowledge-canon/ (curated papers)

---

## 12. Success Criteria

### Minimum Viable Demo (2 weeks)

- ✅ STARK proof for one canary challenge (<30s proving)
- ✅ Verification works (<5ms)
- ✅ Proof size <200 KB
- ✅ MNIST detection: >90% accuracy at 35% BFT

### Research Paper Quality (8-12 weeks)

- ✅ Formal security proof (soundness, completeness)
- ✅ Sybil mitigation implemented
- ✅ CIFAR-10 validation (>95% detection, <10% FPR)
- ✅ Performance analysis (overhead vs Mode 1)
- ✅ Open-source implementation

### Production Ready (16-20 weeks)

- ✅ Optimized native AIR (5-10s proving)
- ✅ Holochain DHT integration
- ✅ 50+ node testnet
- ✅ Documentation & deployment guide

---

**Status**: Ready to implement ✅
**Next Action**: Set up RISC Zero and implement CanaryCNN in fixed-point
**Timeline**: First STARK proof by end of Week 2
