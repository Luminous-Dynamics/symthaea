# Witness Pattern Implementation Status

**Date**: 2025-11-10
**Status**: Partial Success - Need Guidance on Final Pattern

## ✅ Completed

1. **Updated dithering helpers** with user's LFSR-based implementation:
   - `lfsr_step()` - Integer LFSR for deterministic bit toggling
   - `dithered_band()` - Uses LFSR with configurable span_lsb
   - `band_clear()` - Accepts threshold and span_lsb parameters
   - `band_violate()` - Accepts threshold and span_lsb parameters

2. **Applied first test patch**: `test_normal_operation_no_violation`
   - Lowered threshold to q(0.1) = 6553
   - Attempted multiple witness generation strategies

## ⚠️ Challenge: Achieving Full 16-Bit Variation

### Attempts Made

#### Attempt 1: LFSR with base=0.101, span=65536
```rust
let witness = band_clear(0.1, 65536);  // User's recommended pattern
```
**Result**: ❌ LFSR generates steps that don't reach high values in 8 iterations
**Bit variation**: Bits 0-14 vary, but bit 15 constant at 0 (all values < 32768)
**Max witness value**: ~28747

#### Attempt 2: LFSR with base=0.001, span=65536
```rust
let witness = band_clear(0.0, 65536);
```
**Result**: ✅ Constraint degree validation **PASSED!**
**Issue**: ❌ Semantic failure - low values trigger quarantine (quar_out=1 vs expected 0)
**Max witness value**: ~22193

#### Attempt 3: Manual pattern with q(0.15), q(0.25), ..., q(0.95)
**Result**: ❌ Partial bit variation, affine constraints (degree 1 instead of 7)
**Issue**: Some bit combinations missing from pattern

#### Attempt 4: Evenly-spaced values [6563, 14987, 23411, ..., 65535]
```rust
let min_val = q(0.1) + 10;
let step = (65535 - min_val) / 7;
let witness = vec![min_val, min_val+step, min_val+2*step, ..., 65535];
```
**Bit variation**: ✅ Bits 13-15 vary perfectly (0b000 through 0b111 all represented)
**Result**: ❌ Lower bits (0, 1, 2, 11, 12) **constant** due to arithmetic pattern
**Constraint degrees**:
```
expected: [0,0,0,0,0,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,0,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,0,7]
actual:   [0,0,0,0,0,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,0,0,0,7,1,5,7,7,7,7,7,7,0,0,1,5,7,0,0]
```
**Mismatches**: Indices 22 (x_bit[0]), 23 (x_bit[1]), 25, 26, 33, 34, 35, 36, 38, 39

### Root Cause Analysis

**Mathematical Pattern Issue**:
- When step size has certain bits as 0, adding multiples of step preserves those bit patterns
- Example: step=8424=0x20E8 has bit 0=0, bit 1=0, bit 2=0
- Starting from 6563=0x19A3 with bit 0=1, bit 1=1, bit 2=0
- Result: All values have bit 0=1, bit 1=1, bit 2=0 (constant!)

**LFSR Distribution Issue**:
- LFSR generates pseudo-random steps, but in just 8 iterations doesn't cover full u32 range
- Early LFSR output tends toward lower values
- With span_lsb=65536, need LFSR to generate steps close to 65535 to reach high values
- Current LFSR seed (0xACE1) doesn't reach bit 15=1 region in 8 steps

## 🤔 Questions for User

### Option A: Accept Pragmatic Partial Variation
Following user's guidance: *"Partial is acceptable—not strictly necessary for all 16 bits to vary in every test"*

**Proposal**: Use Attempt 4 pattern (evenly-spaced), accept that bits 0, 1, 2, 11, 12 are constant in this specific test

**Pros**: Bits 3-10 and 13-15 vary well (13/16 bits)
**Cons**: Still have constraint degree mismatches at indices 22-23, 33-34

### Option B: Manual Bit-Pattern Construction
Explicitly construct 8 values where each bit position varies:

```rust
let witness = vec![
    0b0001100110100011,  // 6563  - Start pattern
    0b0010101010101010,  // 10922 - Alternating bits
    0b0101010101010101,  // 21845 - Inverse alternating
    0b1000110011001100,  // 35020 - Pattern with bit 15=1
    0b1010101010101010,  // 43690 - High alternating
    0b1100110011001100,  // 52428 - High pattern
    0b1111000011110000,  // 61680 - Near max
    0b1111111111111111,  // 65535 - Maximum
];
```

**Pros**: Explicit control over bit patterns
**Cons**: Manual, not using helper functions, may need per-test tuning

### Option C: Enhanced LFSR Seed Strategy
Use multiple LFSR sequences or adjusted seeds to ensure high-value coverage:

```rust
fn dithered_band_enhanced(base: f32, span_lsb: u64) -> Vec<u64> {
    let base_q = q(base);
    let seeds = [0xACE1, 0xFACE, 0xBEEF, 0xDEAD, 0xCAFE, 0xBABE, 0xF00D, 0xFEED];
    seeds.iter().map(|&seed| {
        let mut state = seed;
        state = lfsr_step(state);
        let step = (state as u64) & (span_lsb - 1);
        (base_q + step).min(65535)
    }).collect()
}
```

**Pros**: Uses LFSR as intended, better distribution
**Cons**: Not the original user-provided pattern

## 📊 Current Status

**Files Ready**:
- `src/tests.rs`: Helpers updated, first test modified (currently Attempt 4 pattern)
- `src/air.rs`: Option B implementation complete ✅
- `src/prover.rs`: Safety switch complete ✅
- `src/provenance.rs`: Blake3 + limb helpers complete ✅

**Test Status**:
- Helper functions: ✅ Implemented and compile
- First test: ⏸️ Blocked by witness pattern decision
- Remaining 31 tests: ⏸️ Waiting for pattern strategy

**Next Steps** (awaiting user guidance):
1. Choose witness pattern strategy (A, B, or C)
2. Apply pattern to first test and verify it passes
3. Apply remaining test patches (test_enter_quarantine, test_warmup_override, etc.)
4. Run full test suite: `cargo test --lib`
5. Add 4 new provenance-specific tests
6. Deploy to M0 Phase 2

## 💡 Recommendation

**Pragmatic Path**: Option B (Manual Bit-Pattern Construction)

**Rationale**:
- User prioritized semantic correctness (#1) over forced variation
- Tests should validate PoGQ logic, not demonstrate maximal bit patterns
- Manual patterns give explicit control and are easy to verify
- Can document why each pattern was chosen for each test

**Implementation**:
1. Create helper function `witness_full_bits()` that returns manually-crafted pattern
2. Use this for tests requiring full variation (like test_normal_operation_no_violation)
3. Use `band_clear()` for tests where partial variation acceptable
4. Document the tradeoff in comments

**Alternative**: If user prefers LFSR approach, provide specific seeds that achieve desired distribution (Option C)

---

**ETA to Unblock**: 30 minutes once pattern strategy confirmed
