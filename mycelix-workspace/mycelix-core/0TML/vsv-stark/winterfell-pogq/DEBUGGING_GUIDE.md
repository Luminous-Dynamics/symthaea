# Winterfell PoGQ - Debugging Guide

**Status**: 22 of 32 tests failing
**Goal**: Fix test failures and achieve production-ready Winterfell backend
**Estimated Fix Time**: 2-4 days

---

## Test Failure Summary

### Compilation Status: ✅ SUCCESS
```bash
$ cargo check
Finished `dev` profile [optimized + debuginfo] target(s) in 0.49s
```
Only 2 minor warnings (unused functions in provenance.rs - non-critical)

### Test Status: ❌ FAILURES
```bash
$ cargo test --lib
test result: FAILED. 10 passed; 22 failed; 0 ignored
```

---

## Failed Tests Breakdown

### Category 1: Core PoGQ Logic (7 tests) - HIGH PRIORITY

1. **test_normal_operation_no_violation**
   - **Expected**: Node stays healthy (quar=0) when quality scores high
   - **Likely Issue**: EMA constraint wrong or selector logic broken
   - **Debug**: Check trace builder fills EMA column correctly

2. **test_enter_quarantine**
   - **Expected**: Node enters quarantine after k violations
   - **Likely Issue**: Violation counter constraint incorrect
   - **Debug**: Verify is_violation selector and viol_t increment logic

3. **test_release_from_quarantine**
   - **Expected**: Node released after m clear rounds
   - **Likely Issue**: Clear counter or is_release selector wrong
   - **Debug**: Check clear_t increment and release condition

4. **test_warmup_override**
   - **Expected**: Quarantine disabled during warm-up (round ≤ w)
   - **Likely Issue**: is_warmup selector or quarantine override constraint
   - **Debug**: Verify warm-up period logic in constraint 3

5. **test_release_exactly_m_clears**
   - **Expected**: Release happens at exactly m clears, not before
   - **Likely Issue**: Off-by-one error in clear counter or release condition
   - **Debug**: Check m comparison in is_release selector

6. **test_hysteresis_k_minus_one**
   - **Expected**: Node stays healthy with k-1 violations (hysteresis)
   - **Likely Issue**: Violation counter reset logic
   - **Debug**: Verify reset condition in constraint 1

7. **test_boundary_threshold_exact_match**
   - **Expected**: x_t == threshold counts as "not violation"
   - **Likely Issue**: is_violation selector uses strict < instead of ≤
   - **Debug**: Check comparison logic in is_violation computation

### Category 2: Adversarial Inputs (6 tests) - MEDIUM PRIORITY

8. **test_adversarial_zero_values**
   - **Expected**: Handle x_t = 0 gracefully
   - **Likely Issue**: Division by zero or field arithmetic issue
   - **Debug**: Check EMA computation with zero inputs

9. **test_adversarial_max_witness_value**
   - **Expected**: Handle x_t = 2^32 - 1 (max Q16.16)
   - **Likely Issue**: Overflow in EMA multiplication
   - **Debug**: Verify field element size and overflow handling

10. **test_adversarial_max_remainder**
    - **Expected**: Handle rem_t = SCALE - 1 (edge case)
    - **Likely Issue**: Range check for remainder failing
    - **Debug**: Check 16-bit decomposition constraints

11. **test_adversarial_alternating_extremes**
    - **Expected**: Handle x_t alternating between 0 and max
    - **Likely Issue**: EMA accumulation overflow or underflow
    - **Debug**: Multi-round trace analysis

### Category 3: Security/Tamper Detection (5 tests) - HIGH PRIORITY

12. **test_tamper_proof_bytes**
    - **Expected**: Verification fails if proof modified
    - **Likely Issue**: Verification not checking proof integrity
    - **Debug**: Ensure verify() actually validates proof bytes

13. **test_tamper_provenance_hash**
    - **Expected**: Verification fails if prov_hash modified
    - **Likely Issue**: Provenance not checked in verification
    - **Debug**: Add provenance validation in verify()

14. **test_tamper_air_rev**
    - **Expected**: Verification fails if air_rev mismatches
    - **Likely Issue**: AIR revision not validated
    - **Debug**: Check air_rev in PublicInputs validation

15. **test_tamper_profile_id**
    - **Expected**: Verification fails if profile_id mismatches
    - **Likely Issue**: Security profile not enforced
    - **Debug**: Validate profile_id against proof options

16. **test_tamper_expected_output**
    - **Expected**: Verification fails if quar_out doesn't match actual
    - **Likely Issue**: Output assertion not working
    - **Debug**: Check boundary constraint for quar_out

### Category 4: Options Validation (4 tests) - MEDIUM PRIORITY

17. **test_options_match_s192**
    - **Expected**: S192 profile proof verifies with S192 options
    - **Likely Issue**: Proof options mismatch detection
    - **Debug**: Ensure prove() and verify() use same options

18. **test_options_mismatch_s128_vs_s192**
    - **Expected**: S128 proof fails verification with S192 options
    - **Likely Issue**: Options not checked in verification
    - **Debug**: Add options validation in verify()

19. **test_options_mismatch_s192_vs_s128**
    - **Expected**: S192 proof fails verification with S128 options
    - **Likely Issue**: Same as above
    - **Debug**: Same fix as #18

20. **test_provenance_mode_off**
    - **Expected**: Provenance disabled when mode=off
    - **Likely Issue**: Provenance hash still computed/validated
    - **Debug**: Check provenance mode flag handling

21. **test_provenance_mode_invalid**
    - **Expected**: Invalid provenance mode rejected
    - **Likely Issue**: Mode validation missing
    - **Debug**: Add mode validation in PublicInputs construction

22. **test_prove_and_verify** (in prover::tests)
    - **Expected**: Basic prove→verify roundtrip works
    - **Likely Issue**: Core prover/verifier integration broken
    - **Debug**: This is THE fundamental test - fix FIRST

---

## Debugging Strategy

### Phase 1: Fix Core Prover/Verifier (Day 1 Morning)

**Priority**: ⚠️ **CRITICAL - FIX FIRST**

```bash
# Run the fundamental test
cargo test prover::tests::test_prove_and_verify -- --nocapture

# Expected error will reveal root cause
# Common issues:
# 1. Trace width mismatch (expected 44, got X)
# 2. Constraint evaluation panic
# 3. Proof serialization error
# 4. Verification assertion failure
```

**Fix Approach**:
1. Check trace width matches AIR definition (44 columns)
2. Verify all constraints evaluate without panic
3. Ensure boundary assertions are correct
4. Test with minimal trace (T=4)

### Phase 2: Fix AIR Constraints (Day 1 Afternoon - Day 2)

**Step 1: Verify Selectors**
```rust
// src/air.rs - Check selector computations

// is_violation: Should be 1 if x_t < threshold, else 0
// Current implementation may be wrong - verify field arithmetic

let is_violation = if x_t < threshold { E::ONE } else { E::ZERO };

// Fix: Use field comparison carefully (field elements wrap!)
```

**Step 2: Fix EMA Constraint**
```rust
// Constraint 0: EMA update
// ema[t+1] = (beta * ema[t] + (SCALE - beta) * x[t]) / SCALE

// Check:
// 1. Division by SCALE implemented correctly?
// 2. Remainder handling correct?
// 3. Field arithmetic doesn't overflow?

// Debug: Add assertion in trace builder
assert_eq!(
    trace.get(0, t+1),  // ema[t+1] from trace
    expected_ema,        // computed from vsv_core
    "EMA mismatch at row {}", t
);
```

**Step 3: Fix Counter Constraints**
```rust
// Constraint 1: Violation counter
// viol[t+1] = is_violation ? viol[t] + 1 : 0

// Issue: Conditional logic in field arithmetic is tricky
// Fix: Use selector multiplication

// Correct form:
// viol[t+1] = is_violation * (viol[t] + 1)
```

**Step 4: Fix Quarantine Logic**
```rust
// Constraint 3: Quarantine state
// Most complex - has 4 cases:
// 1. Warm-up: quar[t+1] = 0
// 2. Release: quar[t+1] = 0
// 3. Enter: quar[t+1] = 1
// 4. Maintain: quar[t+1] = quar[t]

// Use cascading selectors:
// quar[t+1] = is_warmup * 0
//           + (!is_warmup && is_release) * 0
//           + (!is_warmup && !is_release && is_entering) * 1
//           + (!is_warmup && !is_release && !is_entering) * quar[t]

// Simplify to avoid nested conditionals:
// quar[t+1] = (1 - is_warmup) * (
//               (1 - is_release) * (
//                 is_entering + (1 - is_entering) * quar[t]
//               )
//             )
```

### Phase 3: Fix Range Checks (Day 2 Afternoon)

**Remainder Range Check**
```rust
// Constraint: 0 <= rem_t < SCALE
// Implementation: 16-bit decomposition

// rem_t = rem_bits[0] + rem_bits[1]*2 + rem_bits[2]*4 + ... + rem_bits[15]*32768

// Each rem_bits[i] must be 0 or 1
// Constraint: rem_bits[i] * (rem_bits[i] - 1) = 0

// Verify: All 16 bits constrained
for i in 12..28 {  // rem_bits columns
    // C(8+i): rem_bits[i] is boolean
    frame.current()[i] * (frame.current()[i] - E::ONE)
}
```

**Witness Range Check**
```rust
// Similar for x_t (columns 28-43)
for i in 28..44 {  // x_bits columns
    // C(24+i): x_bits[i] is boolean
    frame.current()[i] * (frame.current()[i] - E::ONE)
}
```

### Phase 4: Fix Security Tests (Day 3)

**Tamper Detection**
```rust
// In src/prover.rs - verify() method

pub fn verify(&self, proof: &Proof, public: PublicInputs) -> Result<bool, ...> {
    // Add validation BEFORE calling Winterfell verify

    // 1. Check provenance hash
    let computed_prov = ProvenanceHash::compute(...);
    if public.prov_hash != computed_prov.to_u64x4_le() {
        return Err(ProofError::ProvenanceHashMismatch);
    }

    // 2. Check AIR revision
    if public.air_rev != AIR_SCHEMA_REV {
        return Err(ProofError::AirRevisionMismatch);
    }

    // 3. Check security profile
    if public.profile_id != self.profile.id() {
        return Err(ProofError::SecurityProfileMismatch);
    }

    // 4. Verify output assertion
    // (This should be done by Winterfell boundary constraint)

    // 5. Call Winterfell verify
    winterfell::verify(proof, public_inputs, &self.options)
}
```

---

## Systematic Debugging Commands

### Run Single Test
```bash
# Test with full backtrace
RUST_BACKTRACE=full cargo test test_prove_and_verify -- --nocapture

# Test with specific security profile
VSV_SECURITY_PROFILE=S192 cargo test test_prove_and_verify
```

### Debug Trace Construction
```rust
// Add to src/trace.rs

impl TraceBuilder {
    pub fn build_debug(&self, inputs: PublicInputs, witness: Vec<u64>) -> TraceTable {
        let trace = self.build(inputs.clone(), witness.clone());

        // Print trace for inspection
        eprintln!("=== TRACE DEBUG ===");
        eprintln!("Trace length: {}", trace.length());
        eprintln!("Trace width: {}", trace.width());

        for t in 0..trace.length() {
            eprintln!("Row {}: ema={}, viol={}, clear={}, quar={}, x={}, threshold={}",
                t,
                trace.get(0, t),  // ema_t
                trace.get(1, t),  // viol_t
                trace.get(2, t),  // clear_t
                trace.get(3, t),  // quar_t
                trace.get(4, t),  // x_t
                trace.get(5, t),  // threshold
            );
        }

        trace
    }
}
```

### Compare with Reference Implementation
```rust
// Cross-validate against vsv-core

use vsv_core::pogq::compute_decision;

let reference_result = compute_decision(inputs.clone(), witness.clone());
let winterfell_result = prove_and_extract_decision(inputs.clone(), witness.clone());

assert_eq!(reference_result.quar_out, winterfell_result.quar_out,
    "Winterfell disagrees with reference implementation!");
```

---

## Expected Timeline

### Day 1 (8 hours)
- **Morning**: Fix test_prove_and_verify (core prover/verifier) - 4 hours
- **Afternoon**: Fix EMA constraint - 4 hours
- **Expected**: 1-2 tests passing

### Day 2 (8 hours)
- **Morning**: Fix counter constraints (violation, clear) - 4 hours
- **Afternoon**: Fix range checks (remainder, witness) - 4 hours
- **Expected**: 5-10 tests passing

### Day 3 (8 hours)
- **Morning**: Fix quarantine logic constraint - 4 hours
- **Afternoon**: Fix security tests (tamper detection) - 4 hours
- **Expected**: 15-20 tests passing

### Day 4 (4 hours)
- **Morning**: Fix remaining edge cases and options validation - 2 hours
- **Afternoon**: Final validation and benchmarking - 2 hours
- **Expected**: 32/32 tests passing ✅

---

## Known Issues & Fixes

### Issue 1: Field Element Comparisons
**Problem**: `x_t < threshold` doesn't work in finite field arithmetic
**Fix**: Use integer comparison, then convert to field element
```rust
// WRONG
let is_violation = x_t.lt(&threshold);  // Doesn't exist!

// CORRECT
let is_violation = if x_t_u64 < threshold_u64 {
    E::ONE
} else {
    E::ZERO
};
```

### Issue 2: Division in Constraints
**Problem**: `ema[t+1] = ... / SCALE` - division not allowed in constraints
**Fix**: Add remainder column and use multiplication constraint
```rust
// Instead of: ema[t+1] = numerator / SCALE
// Use: ema[t+1] * SCALE + rem[t+1] = numerator
//      AND: 0 <= rem[t+1] < SCALE (range check)
```

### Issue 3: Conditional Logic in Constraints
**Problem**: Can't use if-else in polynomial constraints
**Fix**: Use selector multiplication
```rust
// Instead of: if condition { A } else { B }
// Use: selector * A + (1 - selector) * B

// Where selector = 1 if condition, 0 otherwise
```

### Issue 4: Trace Length Must Be Power of 2
**Problem**: PoGQ might run for non-power-of-2 rounds
**Fix**: Pad trace to next power of 2
```rust
let trace_length = witness.len();
let padded_length = trace_length.next_power_of_two();

// Extend witness with last value
let mut padded_witness = witness.clone();
padded_witness.resize(padded_length, witness.last().unwrap());
```

---

## Validation Checklist

Before marking Winterfell as "production-ready":

### Functionality
- [ ] All 32 tests passing
- [ ] Cross-validation with vsv-core (100% agreement)
- [ ] Boundary cases handled (zero, max, exact threshold)
- [ ] Adversarial inputs don't crash

### Security
- [ ] Tamper detection working (all 5 tamper tests pass)
- [ ] Provenance validation enforced
- [ ] AIR revision checked
- [ ] Security profile validated
- [ ] Output assertions correct

### Performance
- [ ] Proving time 5-15s (3-10× faster than RISC Zero)
- [ ] Verification time <50ms
- [ ] Proof size ~200KB
- [ ] Memory usage <1GB during proving

### Code Quality
- [ ] No warnings (or only acceptable ones)
- [ ] Code comments explain constraints
- [ ] Test coverage >90%
- [ ] Fuzzing passes (proptest)

---

## Success Metrics

### Immediate (After Day 4)
- ✅ 32/32 tests passing
- ✅ Proving faster than RISC Zero (46.6s → 5-15s)
- ✅ Cross-validation 100% match with vsv-core

### Production Deployment (v2.0)
- ✅ 1000+ proofs generated without error
- ✅ Byzantine detection accuracy maintained
- ✅ Performance stable under load
- ✅ Zero security incidents

---

## Emergency Contacts

**Winterfell Expert**: [Name/Email]
**STARK Cryptographer**: [Name/Email]
**PoGQ Algorithm Lead**: [Name/Email]

---

**Document Version**: 1.0
**Last Updated**: November 11, 2025
**Next Review**: After Day 1 debugging session
