# Provenance Integration Blocker - Constraint Degree Mismatch

## Status: BLOCKED - Winterfell AIR Validation Failure

**Date**: 2025-11-10
**Last Updated**: 2025-11-10 (Option B attempted)
**Impact**: All 20 integration/unit tests failing (21 declared constraints vs runtime evaluation mismatch)
**Root Cause**: PoGQAir struct extension breaks Winterfell's constraint degree validation

## Option B Attempt Results ❌

Attempted converting provenance fields to BaseElement in PoGQAir:
- Changed `prov_hash: [u64; 4]` to `prov_hash: [BaseElement; 4]`
- Changed `profile_id: u32` to `profile_id: BaseElement`
- Changed `air_rev: u32` to `air_rev: BaseElement`
- Added safety switch VSV_PROVENANCE_MODE ("strict"/"off")
- Added limb conversion helpers to provenance.rs

**Result**: SAME constraint degree mismatch error persists
**Confirmed**: Error occurs even with VSV_PROVENANCE_MODE=off
**Conclusion**: Issue is NOT with field types or verification logic, but with PoGQAir struct having additional fields AT ALL

---

## Problem Summary

After extending `PoGQAir` struct to include provenance fields (`prov_hash`, `profile_id`, `air_rev`), Winterfell's proof generation fails with:

```
assertion `left == right` failed: transition constraint degrees didn't match
expected: [  0,   0,   0,   0,   0,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   0,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   0,   7]
actual:   [  0,   0,   0,   0,   0,   7,   7,   7,   7,   7,   7,   7,   7,   7,   0,   7,   7,   7,   7,   7,   7,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
```

### Mismatched Constraints
- **Index 14**: Expected degree 7 (rem_bit[2] boolean), actual 0
- **Indices 22-37**: Expected degree 7 (x_t bit booleans), actual 0
- **Index 39**: Expected degree 7 (quar_t boolean), actual 0

---

## What Was Changed

### 1. PublicInputs Extension (WORKING)
```rust
pub struct PublicInputs {
    // ... 11 original fields ...
    pub trace_length: usize,
    // NEW: Provenance fields
    pub prov_hash: [u64; 4],    // Blake3 hash commitment
    pub profile_id: u32,         // S128=128, S192=192
    pub air_rev: u32,            // AIR schema version
}
```

### 2. PoGQAir Struct Extension (CAUSING FAILURE)
```rust
pub struct PoGQAir {
    context: AirContext<BaseElement>,
    // ... 11 existing BaseElement fields ...
    quar_out: BaseElement,
    // NEW: Provenance fields (NOT BaseElement!)
    prov_hash: [u64; 4],
    profile_id: u32,
    air_rev: u32,
}
```

### 3. get_public_inputs() Extension (WORKING)
```rust
pub fn get_public_inputs(&self) -> Vec<BaseElement> {
    vec![
        // ... 11 existing fields ...
        self.quar_out,
        // NEW: Convert provenance to BaseElement
        BaseElement::from(self.prov_hash[0]),
        BaseElement::from(self.prov_hash[1]),
        BaseElement::from(self.prov_hash[2]),
        BaseElement::from(self.prov_hash[3]),
        BaseElement::from(self.profile_id as u64),
        BaseElement::from(self.air_rev as u64),
    ]
}
```

---

## Why This Breaks

**Hypothesis**: Winterfell's internal AIR validation depends on:
1. **Struct field layout**: Adding non-BaseElement fields after BaseElement fields may break memory alignment expectations
2. **Public input size**: Changing from 11 to 17 elements affects how Winter fell calculates constraint degrees
3. **Context initialization**: `AirContext::new()` may be sensitive to struct size or field types

The fact that **diagnostic test passes** (TraceBuilder works fine) but **proof generation fails** (constraint validation fails) suggests the issue is in how Winterfell interprets the AIR definition, not in the trace itself.

---

## What Works

✅ **TraceBuilder**: Correctly builds 44×8 trace with all values
✅ **Diagnostic test**: EMA constraints validate correctly on trace
✅ **PublicInputs extension**: 17 field struct compiles and serializes
✅ **Provenance computation**: Blake3 hash computed correctly
✅ **Test fixtures**: All tests updated with provenance fields

---

## What Doesn't Work

❌ **Proof generation**: Winter fell rejects AIR during prove()
❌ **All 20 tests**: Integration tests fail at proof generation
❌ **S192 tests**: Cross-profile tests can't run until base tests pass
❌ **Adversarial tests**: Can't validate tamper resistance until proving works

---

## Investigation Attempts

1. **Borrow checker fix**: Fixed tampered_proof.len() borrow issue ✓
2. **Test fixture updates**: Updated all PublicInputs constructors ✓
3. **ProofResult extension**: Return provenance-populated inputs ✓
4. **Git history check**: No prior working version found (new codebase)
5. **Diagnostic verification**: Trace building works correctly ✓

---

## Possible Solutions

### Option A: Don't Store Provenance in PoGQAir
**Idea**: Keep provenance only in PublicInputs, access via get_public_inputs()
**Pros**: Minimal struct changes, Winter fell sees same AIR structure
**Cons**: Need to refactor how provenance is accessed during verification

### Option B: Store as BaseElement from the Start
**Idea**: Convert provenance to BaseElement before storing in PoGQAir
**Pros**: Maintains all-BaseElement field layout
**Cons**: Awkward conversions, may still affect struct size

### Option C: Use Auxiliary Columns
**Idea**: Add provenance as auxiliary trace columns instead of AIR fields
**Pros**: Winterfell designed for this pattern
**Cons**: Significant refactor, more complex verification

### Option D: Debug Winterfell Internals
**Idea**: Add extensive debug logging to understand why degrees mismatch
**Pros**: Root cause understanding
**Cons**: Time-intensive, may hit Winterfell limitations

---

## Recommended Next Steps

1. **Try Option A first** (simplest, least invasive):
   - Remove provenance fields from PoGQAir struct
   - Store in prover/verifier objects instead
   - Access via get_public_inputs() when needed

2. **If Option A fails, try Option B**:
   - Convert provenance to BaseElement array: [BaseElement; 6]
   - Maintain consistent field types throughout struct

3. **If both fail, escalate**:
   - Post issue to Winterfell GitHub
   - Consult with zkSTARK domain experts
   - Consider alternative proving backend

---

## Impact on Timeline

**Current blocking**: Tasks #15-16 (adversarial tests, options mismatch tests)
**Downstream blocking**: M0 Phase 2 (Holochain integration requires working proofs)
**Critical path**: Week 3 Day 5 deliverables depend on passing tests

**ETA to unblock**: 2-4 hours if Option A works, 1-2 days if deeper investigation needed

---

## Test Command to Verify Fix

```bash
# Single test (fastest feedback)
cargo test --lib test_normal_operation_no_violation -- --nocapture

# Full suite
cargo test --lib

# With adversarial tests (once fixed)
cargo test --lib | grep -E "(test result|passed|failed)"
```

## Success Criteria

✅ All 21 tests passing
✅ S128 and S192 profiles both work
✅ Provenance hash mismatch correctly rejects proofs
✅ Adversarial tamper tests can be added

---

**Contact**: Ready for user guidance on preferred solution approach.
