# Winterfell PoGQ Range Check Implementation Design

**Purpose:** Add rigorous soundness guarantees via bit decomposition for counter and remainder values

**Status:** Design document - implementation in progress

## Current Vulnerability

**Problem:** Counters and remainder are unconstrained field elements, allowing overflow attacks:

```rust
// Malicious prover can set:
viol_t = Field::MODULUS - 1  // Huge value
viol_next = viol_t + 1 = 0    // Overflows to 0, avoiding quarantine!

// Or:
rem = SCALE + 1000            // Invalid remainder breaks EMA arithmetic
```

## Solution: Bit Decomposition

**Technique:** For value `x` with `n` bits, add columns `b_0, b_1, ..., b_{n-1}` where:
1. Each `b_i ∈ {0, 1}` (boolean constraint)
2. `x = Σ(b_i * 2^i)` (reconstruction constraint)

**Effect:** Prover must provide valid bit representation → `x < 2^n` guaranteed

## New Trace Architecture

### Current: TRACE_WIDTH = 12
```
[ 0] ema_t       - EMA score
[ 1] viol_t      - Violation counter (UNCONSTRAINED)
[ 2] clear_t     - Clear counter (UNCONSTRAINED)
[ 3] quar_t      - Quarantine flag (implicit boolean)
[ 4] x_t         - Hybrid score (witness)
[ 5] threshold   - Conformal threshold (constant)
[ 6] round       - Round number
[ 7] is_violation - Boolean selector
[ 8] is_warmup    - Boolean selector
[ 9] is_release   - Boolean selector
[10] is_entering  - Boolean selector
[11] rem_t       - EMA remainder (UNCONSTRAINED)
```

### New: TRACE_WIDTH = 38 (+26 bit columns)

```
[12-16] viol_bits[0..4]  - 5-bit decomposition of viol_t (supports k ≤ 31)
[17-21] clear_bits[0..4] - 5-bit decomposition of clear_t (supports m ≤ 31)
[22-37] rem_bits[0..15]  - 16-bit decomposition of rem_t (0 ≤ rem < 65536)
```

**Why these bit widths?**
- **viol_t, clear_t:** 5 bits = max value 31, covers realistic k,m ≤ 16
- **rem_t:** 16 bits = max 65535, matches SCALE = 65536 (Q16.16)

## New AIR Constraints

### C10-C14: viol_t Bit Decomposition (5 bits)
```rust
// Boolean constraints (degree 2)
for i in 0..5:
    b_i * (b_i - 1) = 0  // Enforces b_i ∈ {0, 1}

// Reconstruction constraint (degree 1)
viol_t - (b_0 + 2*b_1 + 4*b_2 + 8*b_3 + 16*b_4) = 0
```

**Result:** `0 ≤ viol_t ≤ 31` proven

### C15-C19: clear_t Bit Decomposition (5 bits)
```rust
// Boolean constraints
for i in 0..5:
    b_i * (b_i - 1) = 0

// Reconstruction
clear_t - (b_0 + 2*b_1 + ... + 16*b_4) = 0
```

**Result:** `0 ≤ clear_t ≤ 31` proven

### C20-C35: rem_t Bit Decomposition (16 bits)
```rust
// Boolean constraints (16 of them)
for i in 0..16:
    b_i * (b_i - 1) = 0

// Reconstruction
rem_t - (b_0 + 2*b_1 + 4*b_2 + ... + 32768*b_15) = 0
```

**Result:** `0 ≤ rem_t ≤ 65535 < SCALE` proven

### C36: quar_t Boolean (Explicit)
```rust
quar_t * (quar_t - 1) = 0
```

**Result:** `quar_t ∈ {0, 1}` enforced (was implicit before)

## Total New Constraints

- **5 boolean constraints** for viol_t bits
- **1 reconstruction constraint** for viol_t
- **5 boolean constraints** for clear_t bits
- **1 reconstruction constraint** for clear_t
- **16 boolean constraints** for rem_t bits
- **1 reconstruction constraint** for rem_t
- **1 boolean constraint** for quar_t

**Total:** 30 new constraints (all degree 1-2, efficient!)

## Implementation Plan

### Phase 1: Update AIR (air.rs)
1. Change `TRACE_WIDTH` from 12 to 38
2. Update column layout documentation
3. Add 30 new constraint evaluations in `evaluate_transition()`
4. Ensure all constraints are degree ≤ 2

### Phase 2: Update Trace Builder (trace.rs)
1. Allocate 38-column trace matrix (was 12)
2. Implement bit decomposition helper:
   ```rust
   fn decompose_u64(value: u64, bits: usize) -> Vec<u64> {
       (0..bits).map(|i| (value >> i) & 1).collect()
   }
   ```
3. Fill bit columns for viol_t, clear_t, rem_t at each step
4. Verify reconstruction: `viol_t == reconstruct(viol_bits)`

### Phase 3: Update Prover (prover.rs)
1. Verify `TRACE_WIDTH` matches AIR expectation
2. No other changes needed (Winterfell handles constraint evaluation)

### Phase 4: Update Tests (tests.rs)
1. All existing tests should pass (logic unchanged)
2. Add new test: `test_range_check_enforcement`
   - Try to build trace with viol_t = 100 (> 31)
   - Should fail during trace building (assertion)
3. Add test: `test_invalid_remainder`
   - Try rem_t = 70000 (> SCALE)
   - Should fail

### Phase 5: Adversarial Testing
1. **Overflow attack test:**
   - Maliciously set viol_t = Field::MAX in trace
   - Proof generation should fail (constraint violation)
2. **Fake proof test:**
   - Try to prove quar_out=0 with viol_t=k (should be quar_out=1)
   - Verification should fail

## Performance Impact

### Expected Slowdown: ~2-3×
- **TRACE_WIDTH:** 12 → 38 (3.2× wider)
- **Constraints:** 9 → 39 (4.3× more)
- **Prove time:** ~0.2ms → ~0.5-1ms (still 50,000× faster than zkVM!)

### Proof Size Impact: ~1.5-2×
- More trace columns → larger Merkle tree commitment
- More constraints → larger quotient polynomial
- Expected: ~10KB → ~20KB (still 11× smaller than zkVM's 221KB)

## Benefits

✅ **Soundness:** Overflow attacks impossible
✅ **Completeness:** Honest provers unaffected (bit decomp is trivial)
✅ **Auditability:** Clear, verifiable constraints
✅ **Performance:** Still massively faster than zkVM

## Alternative Approaches Considered

### Option A: Native Range Check Gadget (Rejected)
- Some STARK frameworks have built-in range checks
- Winterfell v0.13 doesn't expose this
- Would require custom extension to Air trait

### Option B: Lookup Tables (Rejected for now)
- More efficient for many range checks
- Requires Winterfell v0.14+ (LogUp/multiset checks)
- Bit decomposition is simpler and sufficient

### Option C: Field-Native Bounds (Rejected)
- E.g., prove `viol_t < p/4` where p = field modulus
- Less intuitive, doesn't match PoGQ semantics (k,m are small)
- Bit decomposition is more direct

## Testing Strategy

### Unit Tests
- [x] Bit decomposition helper function
- [ ] Trace building with valid values
- [ ] Trace building rejects invalid values (viol > 31)
- [ ] AIR constraints satisfied for honest traces

### Integration Tests
- [ ] All 3 scenarios (normal, enter, release) pass with range checks
- [ ] Scaled traces (T=128, 256, 512) work correctly
- [ ] Edge cases: viol=k-1, clear=m-1, rem=SCALE-1

### Adversarial Tests
- [ ] Overflow attack (viol=2^32) - proof fails
- [ ] Invalid remainder (rem=SCALE+1) - proof fails
- [ ] Fake quarantine decision - verification fails
- [ ] Dual-backend agreement: 100 random cases match

## Migration Path

**Backwards Compatibility:** Breaking change (TRACE_WIDTH modified)

**Rollout:**
1. Implement in new branch `feature/range-checks`
2. Full test suite passes (including adversarial)
3. Re-benchmark and update Table VII-bis
4. Merge to main
5. Tag as v0.2.0 (production-grade soundness)

## Open Questions

1. **Bit width sufficiency:** 5 bits supports k,m ≤ 31. Is this enough?
   - **Answer:** Yes, realistic PoGQ uses k,m ≤ 10

2. **Selector constraints:** Do selectors need range checks too?
   - **Answer:** No, selectors are boolean by construction (is_* flags)
   - But add explicit `s*(s-1)=0` for defense-in-depth

3. **Performance vs security:** Can we reduce bit widths?
   - **Answer:** 5 bits is minimum for k=16 scenarios
   - 16 bits for remainder is required (SCALE = 2^16)

## Next Steps

1. ✅ Write this design document
2. [ ] Implement bit decomposition helper in trace.rs
3. [ ] Update TRACE_WIDTH and column layout
4. [ ] Add 30 new AIR constraints
5. [ ] Update trace builder to fill bit columns
6. [ ] Run full test suite
7. [ ] Add adversarial tests
8. [ ] Benchmark and update paper

**Timeline:** 2-3 days implementation + 1 day testing + 1 day benchmarking = ~5 days total

---

*This design ensures mathematically rigorous soundness while maintaining 1000×+ speedup over zkVMs.*
