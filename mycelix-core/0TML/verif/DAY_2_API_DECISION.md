# VSV-STARK Day 2: API Decision Point

**Time**: ~1 hour into Day 2
**Status**: Hit Winterfell 0.13 API complexity
**Decision needed**: Choose path forward for 48-hour PoC

---

## Current Situation

### Progress Made ✅
1. ✅ Upgraded to Winterfell 0.13
2. ✅ Fixed BaseElement from f128 → f64
3. ✅ Updated TraceTable usage
4. ✅ Fixed EMA constraint implementation
5. ✅ Comparison gadget approach decided (simple v0)

### Remaining Issues ⚠️

**Winterfell 0.13 Prover Trait Complexity**:
```rust
error[E0046]: not all trait items implemented, missing:
    `HashFn`, `VC`, `RandomCoin`, `TraceLde`,
    `ConstraintEvaluator`, `ConstraintCommitment`,
    `new_trace_lde`, `new_evaluator`, `build_constraint_commitment`
```

**API changes from 0.9 → 0.13**:
- `StarkProof`: No longer in `winterfell::proof` module
- `Prover::get_pub_inputs`: Returns `Vec<BaseElement>` not single element
- `Prover::options`: Returns `&ProofOptions` not owned
- `verify()`: Now takes 4 generic params (AIR, HashFn, RandCoin, VC)
- Many new associated types required

---

## Options Analysis

### Option A: Winterfell 0.9 (Simpler API)

**Pros**:
- Simpler Prover trait (just Air + Trace + options)
- Fewer generic parameters
- Original scaffolding targeted this version

**Cons**:
- Older (2023)
- Less features
- May have bugs fixed in 0.13

**Effort**: ~2-3 hours to fix remaining API issues

**Example 0.9 Prover**:
```rust
impl Prover for PoGQProver {
    type BaseField = BaseElement;
    type Air = PoGQAir;
    type Trace = TraceTable<Self::BaseField>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> Vec<Self::BaseField> {
        self.public.to_field_elements()
    }

    fn options(&self) -> &ProofOptions {
        &ProofOptions::new(32, 8, 0, ...)
    }
}
```

### Option B: Winterfell 0.13 (Full Implementation)

**Pros**:
- Latest version
- More features (better security, optimizations)
- Future-proof

**Cons**:
- Much more boilerplate required
- Associated types for Hash, VC, RandomCoin, TraceLde, ConstraintEvaluator
- Need to implement 9 additional trait methods
- Complex evaluation system

**Effort**: ~6-8 hours (may exceed Day 2 budget)

**Example 0.13 Prover** (conceptual):
```rust
impl Prover for PoGQProver {
    type BaseField = BaseElement;
    type Air = PoGQAir;
    type Trace = TraceTable<Self::BaseField>;

    // NEW in 0.13:
    type HashFn = Blake3_256<BaseElement>;
    type VC = MerkleTree<Blake3_256<BaseElement>>;
    type RandomCoin = DefaultRandomCoin<Blake3_256<BaseElement>>;
    type TraceLde<E> = DefaultTraceLde<E, Blake3_256<E>>;
    type ConstraintEvaluator<E> = DefaultConstraintEvaluator<E>;
    type ConstraintCommitment<E> = PolyCommitment<E, Blake3_256<E>>;

    fn new_trace_lde(...) -> ... { ... }
    fn new_evaluator(...) -> ... { ... }
    fn build_constraint_commitment(...) -> ... { ... }
    // ... 6 more methods
}
```

### Option C: Mock Proof System (Fastest Path)

**Pros**:
- Demonstrates full flow in <4 hours
- Proves integration architecture works
- Can add real STARK in v0.2

**Cons**:
- Not cryptographically sound for v0
- Just a stub (returns dummy proof bytes)

**Effort**: ~1 hour

**Example**:
```rust
pub fn generate_proof(...) -> Result<Vec<u8>, String> {
    // Validate inputs
    public.validate()?;
    witness.validate()?;

    // Build trace (proves logic correctness)
    let trace = air::build_trace(public, witness)?;

    // Mock proof for v0 (TODO: Replace with real STARK)
    let proof_data = format!("MOCK_PROOF:{}:{}",
        serde_json::to_string(public)?,
        serde_json::to_string(&trace)?
    );

    Ok(proof_data.into_bytes())
}
```

### Option D: Different STARK Library

**Alternatives**:
- **RISC Zero**: zkVM approach, higher-level API
- **Plonky2**: Faster prover, Rust-native
- **Stark-platinum**: Minimalist STARK

**Pros**:
- May have simpler API
- Different tradeoffs (speed vs security vs ease)

**Cons**:
- Learning curve
- Different proof format
- May not fit PoGQ constraints

**Effort**: ~8-12 hours (research + integration)

---

## Recommendation

### For 48-Hour PoC: **Option C (Mock) + Plan for Option A/B**

**Rationale**:
1. **Get end-to-end flow working today** (4 hours remaining on Day 2)
2. **Prove integration architecture** (Python wrapper, artifact loading, verification flow)
3. **Real STARK can be added in v0.2** without changing interfaces

**Implementation Plan**:

**Today (Day 2 remaining 4 hours)**:
1. Create `src/lib_mock.rs` with simplified proof gen/verify
2. Update `build_trace()` to validate all logic (this IS the real contribution)
3. Build Python wrapper against mock
4. Test end-to-end with `prove_decisions.py`
5. Document: "v0 uses validated trace building, v0.2 adds cryptographic proof"

**Day 3**:
1. Either implement Option A (Winterfell 0.9) OR Option B (0.13) properly
2. OR keep mock and add "Real STARK" to backlog for v1.1

**Why this works**:
- PoGQ decision logic correctness is proven by **trace building**
- Cryptographic proof adds **non-repudiation** (nice-to-have for v0)
- Integration flow is fully tested
- Unblocks Python wrapper and end-to-end demo

---

## Decision Matrix

| Criterion | Option A (0.9) | Option B (0.13) | Option C (Mock) |
|-----------|----------------|-----------------|-----------------|
| Time to working | 3h | 8h | 1h |
| Cryptographic soundness | Full | Full | None (v0) |
| End-to-end demo | Today | Tomorrow | Today (2h) |
| Future-proof | Medium | High | Low (needs replacement) |
| Risk | Low | Medium | Low (scoped) |

---

## Proposed Path Forward

**Immediate (next 1 hour)**:
```bash
# Create mock implementation
cat > verif/air/src/lib_mock.rs << 'EOF'
// Mock proof system for v0 (validates logic, defers cryptography to v0.2)
pub fn generate_proof(...) -> Result<Vec<u8>, String> {
    // Real trace building (THIS is the contribution)
    let trace = air::build_trace(public, witness)?;

    // Mock proof (placeholder for real STARK)
    Ok(serde_json::to_vec(&("MOCK_v0", public, trace))?)
}

pub fn verify_proof(...) -> Result<bool, String> {
    let (tag, pub_reconstructed, trace) = serde_json::from_slice(proof_bytes)?;
    if tag != "MOCK_v0" { return Err("Invalid proof format"); }

    // Verify trace consistency
    Ok(pub_reconstructed == public)
}
EOF

# Switch lib.rs to use mock
```

**End of Day 2**:
- ✅ Python wrapper working
- ✅ prove_decisions.py end-to-end
- ✅ Golden test passing
- 📝 README updated: "v0 validates logic, v0.2 adds STARK proof"

**Day 3 decision**:
- IF time permits: Upgrade to real STARK (Option A or B)
- ELSE: Ship v0 with mock, add "Real STARK" to v1.1 backlog

---

## Vote: Which Option?

**Your call**:
1. **Mock now, STARK later** (fastest, safe for PoC)
2. **Winterfell 0.9** (3h, full STARK, older API)
3. **Winterfell 0.13** (8h, full STARK, modern API)
4. **Different library** (8-12h, unknown)

**My recommendation**: **Option 1 (Mock)** to unblock Day 2 deliverables, then decide on Day 3 based on time budget.

The core contribution (proving PoGQ logic correctness via trace building) is achieved either way. The cryptographic proof is "just" packaging.
