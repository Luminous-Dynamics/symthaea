# Provenance Integration Status Update - Need Guidance

**Date**: 2025-11-10
**Status**: Partial Success - Provenance working, test witness design challenge

## ✅ What's Working

### Provenance Infrastructure (100% Complete)
1. **Blake3 hash module**: All 4 unit tests passing
2. **PublicInputs extended**: `prov_hash: [u64; 4]`, `profile_id: u32`, `air_rev: u32` ✅
3. **PoGQAir with Option B**: Provenance stored as `BaseElement` fields ✅
4. **Limb conversion helpers**: `u64x4_le_to_bytes`, `base_elements_to_u64x4` ✅
5. **Safety switch**: `VSV_PROVENANCE_MODE` ("strict"/"off") implemented ✅
6. **All 32 test fixtures updated**: Provenance fields populated ✅
7. **8 adversarial tests written**: Syntax-checked and ready ✅

## ⚠️ Current Challenge: Constraint Degree Validation

### Problem Summary
Winterfell's degree validation fails with various witness patterns:

```
assertion `left == right` failed: transition constraint degrees didn't match
expected: [0,0,0,0,0,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,0,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,0,7]
actual:   [0,7,7,0,0,5,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,0,7,7,7,0,7,7,7,7,7,7,7,7,7,0,0,0,0,0]
```

### Root Cause Analysis

**User's Guidance**: "With constant witness_scores, x_t columns are constant → bit columns constant → boolean constraints evaluate to degree 0."

**Tested Solutions**:

| Witness Pattern | Result | Issues |
|-----------------|---------|--------|
| Constant (0.92×8) | ❌ Degree mismatch | Many bits constant (indices 14, 22-37, 39) |
| Linear varied (0.92-0.99) | ❌ Degree mismatch | High bits (13-15) constant, affine constraints OK |
| Wide varied (0.85-0.99) | ❌ Test assertion | Crosses threshold → quar_out=1 (expected 0) |
| Single violation (0.88 + clears) | ❌ Degree mismatch | Affine C1/C2 become non-constant |
| User's example (0.89-0.96) | ❌ Degree mismatch | C1/C2 vary, bits 13-15 constant, C5 degree 5 |

### Key Findings

1. **Crossing threshold affects affine constraints**:
   - C1 (viol counter) and C2 (clear counter) depend on `is_violation`
   - Varying `is_violation` makes these constraints non-constant (degree 7 instead of 0)

2. **Bit range limitations**:
   - Witnesses in range [0.90, 1.0] → bits 13-15 always 1 (constant)
   - Witnesses in range [0.0, 0.5] → bits 13-15 vary
   - But threshold is 0.90, so can't go too low without creating violations

3. **Test semantics conflict**:
   - `test_normal_operation_no_violation` expects quar_out=0
   - Varied witnesses that ensure all bits vary often create violations
   - Violations → quarantine → quar_out=1 → test fails

## 🤔 Questions for User

### 1. Test Design Philosophy
Should tests be redesigned to accommodate varied witnesses, or should witnesses be carefully crafted per test scenario?

**Option A**: Accept that some constraints are constant for specific test scenarios (e.g., quar_t=0 for "no violation")
**Option B**: Design test scenarios where ALL constraints vary (e.g., rename test, allow violations)

### 2. Witness Range Strategy
How should we handle the bit variation vs. threshold constraint tradeoff?

**Your example** (0.89-0.96) crosses threshold, making affine constraints vary.
Should we:
- Use this pattern and update test expectations (quar_out might be 1)?
- Use different witness patterns for different test types?
- Accept partial bit variation (e.g., bits 13-15 constant)?

### 3. Constraint Degree Declaration
Are the current constraint degree declarations correct?

Current: All boolean constraints declared as degree 2 → adjusted to 7
Reality: Some booleans evaluate to constant 0 → actual degree 0

Should we:
- Keep declarations as-is and ensure test witnesses make all constraints vary?
- Investigate if declarations should be conditional/dynamic?

## 📊 Detailed Test Results

### User's Example Values (0.89-0.96)
```rust
let witness = vec![
    q(0.93), q(0.91), q(0.92), q(0.94),
    q(0.90), q(0.95), q(0.89), q(0.96),
];
```

**Constraint Degree Mismatches**:
- Index 1 (C1): Expected 0, got 7 - viol counter varies
- Index 2 (C2): Expected 0, got 7 - clear counter varies
- Index 5 (C5): Expected 7, got 5 - rem_bit[0] partial variation
- Index 25 (C25): Expected 7, got 0 - x_bit[3] constant
- Index 35-37 (C35-37): Expected 7, got 0 - x_bits[13-15] constant
- Index 39 (C39): Expected 7, got 0 - quar_t constant

### Successful Partial Result (0.55-0.98 range)
With `[q(0.55), q(0.62), q(0.70), q(0.78), q(0.85), q(0.91), q(0.95), q(0.98)]`:

✅ **Constraint degree validation PASSED!**
❌ **Test assertion failed**: quar_out expected 0, got 1

This suggests:
1. Wide bit variation resolves degree issues
2. But creates violations → quarantine → wrong test expectation

## 💡 Proposed Solutions

### Solution 1: Redesign This Test
Rename and refactor `test_normal_operation_no_violation` to allow varied behavior:

```rust
#[test]
fn test_varied_witness_with_single_violation() {
    let witness = vec![
        q(0.55), q(0.62), q(0.70), q(0.78),
        q(0.85), q(0.91), q(0.95), q(0.98),
    ];
    // ... update quar_out expectation based on actual logic ...
}
```

### Solution 2: Multiple Test Patterns
Create different witness patterns for different test types:

- **Narrow range tests** (0.91-0.98): For scenarios where high bits constant is acceptable
- **Wide range tests** (0.55-0.98): For scenarios testing full bit variation
- **Threshold crossing tests**: Explicitly test violation logic with appropriate expectations

### Solution 3: Investigate Original Tests
Check if the ORIGINAL tests (before provenance integration) had constant witnesses and how they handled this issue.

## 🎯 Immediate Next Steps (Awaiting Guidance)

1. **Clarify test design philosophy** - Should we accept partial constraint variation per test?
2. **Provide witness pattern recommendations** - What values ensure all constraints vary while matching test semantics?
3. **Confirm quar_out behavior** - With your example witnesses, what should quar_out be?

## 📂 Files Ready for Review

- `src/air.rs`: PoGQAir with Option B provenance fields ✅
- `src/prover.rs`: Safety switch implementation ✅
- `src/provenance.rs`: Blake3 hash + limb helpers (4/4 tests passing) ✅
- `src/tests.rs`: Updated first test with varied witnesses (blocked) ⏸️
- `src/tests.rs`: 8 new adversarial tests (syntax-checked, waiting) ⏸️

## 🔮 Once Unblocked

1. Update remaining 31 integration tests with appropriate witness patterns
2. Run full test suite: `cargo test --lib`
3. Verify all 21+ tests pass
4. Add 4 new provenance-specific tests (limb roundtrip, tamper_limb, tamper_profile, mode_off)
5. Deploy to M0 Phase 2 (Holochain integration)

---

**Contact**: Awaiting user guidance on test design philosophy and witness pattern recommendations.

**ETA to unblock**: 1-2 hours once guidance received, assuming recommended patterns work.
