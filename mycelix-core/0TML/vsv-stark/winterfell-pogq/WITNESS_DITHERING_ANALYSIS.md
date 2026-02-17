# Witness Dithering Analysis - Technical Challenge

**Date**: 2025-11-10
**Status**: Fundamental tradeoff between bit variation and semantic constraints

## 🎯 Core Problem

Winterfell's constraint degree validation requires **dynamic bit columns** (all 16 bits of x_t varying), but test semantic constraints (e.g., "no violation" = all scores >= threshold) **limit the achievable bit variation range**.

## 📊 Experimental Results

### Attempt 1: base=0.93, span_lsb=32 (FAILED)
**Witness**: `[60948, 60978, 60976...]` (binary: `0b1110111000...`)
**Bit variation**: Only lowest 5 bits vary
**Result**: ❌ Constraint degree mismatch (indices 23-24, 28-39)
**Issue**: Bits 8-15 constant

### Attempt 2: base=0.93, span_lsb=8192 (FAILED)
**Witness**: `[60948, 61106, 61008...]` (binary: `0b1110111...`)
**Bit variation**: Bits 0-12 vary
**Result**: ❌ Constraint degree mismatch (indices 23-24, 27-39)
**Issue**: Bits 13-15 constant (all `111`)

### Attempt 3: base=0.60, span_lsb=32768 (PASSED CONSTRAINTS, FAILED SEMANTICS)
**Witness**: `[39321, 39479...]` (binary: `0b1001100...`)
**Bit variation**: All 16 bits vary
**Result**: ✅ Constraint validation PASSED | ❌ quar_out=1 (expected 0)
**Issue**: Many scores < 0.90 threshold → violations → quarantine

### Attempt 4: base=0.91, span_lsb=16384 (FAILED)
**Witness**: Not printed, but similar pattern
**Bit variation**: ~14 bits vary
**Result**: ❌ Constraint degree mismatch
**Issue**: Still insufficient variation for highest bits

## 🔬 Root Cause Analysis

### Q16.16 Fixed-Point Constraints
- Value range: [0, 65535] represents [0.0, 0.9999]
- Threshold: q(0.90) = 58982 = `0b1110011010000110`
- Binary structure:
  - Bits 15-14: Must be `11` for values >= 0.75
  - Bit 13: Must be `1` for values >= 0.875
  - Bits 12-0: Variable based on specific value

### The Fundamental Tradeoff

| Base Value | Span Needed | Semantic | Bit Variation | Result |
|------------|-------------|----------|---------------|---------|
| 0.93 (safe above threshold) | 8192 | ✅ No violations | ❌ Bits 13-15 constant | Degree mismatch |
| 0.91 (safe above threshold) | 16384 | ✅ No violations | ❌ Bit 15 constant | Degree mismatch |
| 0.60 (crosses threshold) | 32768 | ❌ Causes violations | ✅ All 16 bits vary | Wrong quar_out |

**Mathematical Reality**: To vary all 16 bits while staying >= threshold (0.90):
- Min value: q(0.90) = 58982
- Max value: q(0.9999) = 65535
- Available range: 6553 values
- Span needed: 6553

But span_lsb must be power of 2 for masking, so closest is 8192, which gives range [base, base+8192], exceeding the available 6553-value window!

## 🤔 Possible Solutions

### Option 1: Accept Partial Bit Variation
**Approach**: Use base=0.93, span_lsb=8192, accept bits 13-15 constant
**Pros**: Semantics correct, reasonable variation (13/16 bits)
**Cons**: Constraint degrees for x_bits[13-15] remain 0
**Status**: ❌ Failed validation

### Option 2: Redesign Test Expectations
**Approach**: Use base=0.60, span_lsb=32768, update test to expect quar_out=1
**Pros**: Full bit variation, constraint validation passes
**Cons**: Test no longer represents "no violation" scenario
**Status**: ✅ Constraints pass, but semantic mismatch

### Option 3: Use Non-Uniform Distribution
**Approach**: Manually specify 8 diverse Q16.16 values that:
- Cover full 16-bit range
- Most stay >= threshold (to minimize violations)
- Are carefully chosen to satisfy EMA/quar logic

**Example**:
```rust
let witness = vec![
    q(0.91), // 59637 = 0b1110100100000101
    q(0.94), // 61604 = 0b1111000010100100
    q(0.88), // 57672 = 0b1110000101011000 (one violation)
    q(0.96), // 62914 = 0b1111010110000010
    q(0.92), // 60293 = 0b1110101110010101
    q(0.99), // 64880 = 0b1111110101110000
    q(0.95), // 62259 = 0b1111001101010011
    q(0.93), // 60948 = 0b1110111000010100
];
```
**Pros**: Could balance variation + semantics
**Cons**: Manual, brittle, needs verification
**Status**: NOT TRIED

### Option 4: Modify AIR Degree Declarations
**Approach**: Investigate if constraint degrees should be conditional/dynamic
**Pros**: Addresses root cause in AIR design
**Cons**: May not be supported by Winterfell, significant refactor
**Status**: NOT TRIED

### Option 5: Use Different Test Strategy Per Scenario
**Approach**: Different witness patterns for different test types:
- "No violation" tests: Accept partial bit variation (bits 0-12)
- "With violation" tests: Use full range for bit variation
- "Boundary" tests: Mix exact values with varied values

**Pros**: Pragmatic, matches test semantics
**Cons**: Some tests still have degree mismatches
**Status**: PARTIALLY VIABLE

## 📋 User Guidance Needed

### Question 1: Bit Variation Requirements
Is it **strictly necessary** for ALL 16 bits to vary in EVERY test, or is it acceptable for some bits to be constant in specific scenarios?

**If acceptable**: We can proceed with Option 5 (different patterns per test)
**If not acceptable**: We need Option 3 (manual non-uniform) or Option 4 (AIR redesign)

### Question 2: Test Semantic Priority
Should tests prioritize:
1. **Semantic correctness** (e.g., "no violation" actually has no violations)
2. **Constraint validation** (all degrees match, even if semantics change)

**If (1)**: Option 3 or 5 with careful value selection
**If (2)**: Option 2 (redefine test expectations to match witness behavior)

### Question 3: Specific Witness Pattern Recommendation
Could you provide a **concrete witness array** (8 Q16.16 values) that:
- Ensures all 16 x_t bits vary
- Keeps most/all values >= threshold (0.90)
- Produces expected quar_out=0 for "no violation" test

**Or** confirm that such an array is mathematically impossible given the constraints.

## 🎯 Immediate Next Steps (Awaiting Guidance)

1. **Clarify requirements**: Full 16-bit variation required, or partial acceptable?
2. **Choose strategy**: Manual values (Option 3), test redesign (Option 2), or pragmatic mix (Option 5)?
3. **Provide example**: Concrete witness values that satisfy all constraints

---

**Current Implementation**: Dithered band helpers working correctly, just need guidance on parameter tuning or strategy pivot.

**Files**: `src/tests.rs` (helpers implemented), ready for final witness pattern once decided.
