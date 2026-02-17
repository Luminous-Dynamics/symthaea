# Winterfell PoGQ Security Audit Checklist

**Purpose:** Verify soundness of hand-rolled STARK before production deployment

## Critical Soundness Issues (Must Fix)

### 1. Missing Range Checks 🚨 HIGH PRIORITY
**Problem:** Counters and remainder are unconstrained, allowing overflow attacks

**Current State:**
- `viol_t`, `clear_t`: No bounds checking (could be Field::MAX)
- `rem_t`: No proof that 0 ≤ rem < SCALE
- `quar_t`: Implicitly boolean but not enforced

**Attack Vector:**
```rust
// Malicious prover sets:
viol_t = Field::MAX
// After increment: viol_next = 0 (overflow)
// Prover avoids entering quarantine despite k violations!
```

**Mitigation:**
- [ ] Add bit decomposition for `viol_t` (5 bits for k=16 max)
- [ ] Add bit decomposition for `clear_t` (5 bits for m=16 max)
- [ ] Add bit decomposition for `rem_t` (16 bits for Q16.16)
- [ ] Enforce `quar_t * (quar_t - 1) = 0` (boolean constraint)

**Estimated Impact:** +26 trace columns (38 total), ~2× prove time, still 1000×+ faster than zkVM

### 2. Selector Independence 🚨 HIGH PRIORITY
**Problem:** Are selectors uniquely determined by public/witness, or can prover manipulate?

**Current Constraints:**
```rust
// C7: is_violation = (x_t < threshold) ? 1 : 0
// But is this ENFORCED or just computed?
```

**Verification Needed:**
- [ ] Prove selectors are uniquely determined by (x_t, threshold, round, etc.)
- [ ] Check: Can prover set `is_violation=0` when `x_t < threshold`?
- [ ] Review transition constraint C3: Does it force correct selector computation?

**Test:** Try generating proof with manually flipped selector in trace

### 3. Counter Update Logic 🟡 MEDIUM PRIORITY
**Problem:** Implicit reset via multiplication - is this sound?

**Current Logic:**
```rust
// viol_next = is_violation * (viol_t + 1)
// When is_violation=0, this resets to 0 (implicit)
// Is this vulnerable to manipulation?
```

**Verification:**
- [ ] Formal proof that implicit reset is equivalent to full mux
- [ ] Test: Can prover maintain viol_t when is_violation=0?
- [ ] Check: Does constraint degree allow alternative interpretations?

### 4. Field Arithmetic Overflow 🟡 MEDIUM PRIORITY
**Problem:** Q16.16 arithmetic could overflow field boundary

**Scenario:**
```rust
ema_t = Field::MODULUS - 1
x_t = Field::MODULUS - 1
sum = beta * ema_t + (1-beta) * x_t
// Does this wrap? Break EMA calculation?
```

**Mitigation:**
- [ ] Verify all intermediate products < Field::MODULUS
- [ ] Add constraints: ema_t < 2^48 (reasonable bound for Q16.16 probabilities)
- [ ] Test with maximum field values

## Moderate Soundness Issues

### 5. Public Input Integrity 🟡 MEDIUM PRIORITY
**Problem:** Prover and Verifier must use IDENTICAL PublicInputs

**Current Risk:**
- Prover could use different (beta, k, m, w) than claimed
- Verifier would accept proof with wrong parameters

**Verification:**
- [ ] Confirm PublicInputs are hashed into Fiat-Shamir transcript
- [ ] Test: Generate proof with beta=0.85, verify with beta=0.90 (should fail)
- [ ] Review Winterfell's public input commitment mechanism

### 6. Constraint Degree Audit 🟢 LOW PRIORITY
**Problem:** High-degree constraints increase proving cost

**Current Claim:** All constraints ≤ degree 2

**Verification:**
- [ ] Audit `evaluate_transition()`: Count multiplications per constraint
- [ ] Confirm: No constraint uses >2 witness columns multiplicatively
- [ ] Check EMA constraint: `ema_next * SCALE + rem = beta * ema + (SCALE-beta) * x` is degree 2 ✓

### 7. Trace Padding 🟢 LOW PRIORITY
**Problem:** Traces padded to 2^k - what values are used?

**Current:** Likely repeats last row (standard practice)

**Verification:**
- [ ] Confirm padding doesn't create fake transitions
- [ ] Test: T=9 (requires padding to 16) - verify last 7 rows don't trigger state changes

## Dual-Backend Cross-Verification Tests

### 8. Output Agreement Testing 🟡 MEDIUM PRIORITY
**Purpose:** Prove both backends compute identical results

**Test Suite:**
- [ ] Same inputs → both output `quar_out = 0`
- [ ] Same inputs → both output `quar_out = 1`
- [ ] Edge cases: k-1 violations, m-1 clears
- [ ] Random fuzzing: 1000 random (beta, k, m, w, witness) pairs
- [ ] Adversarial: Inputs designed to trigger overflow/boundary conditions

**Script:** `dual_backend_smoke.sh` (run in CI)

### 9. Proof Soundness Testing 🚨 HIGH PRIORITY
**Purpose:** Adversarially try to generate fake proofs

**Attack Scenarios:**
```bash
# Test 1: Proof with wrong output
# Generate proof claiming quar_out=0 when actually should be 1
prover.prove(quar_init=0, witness=[low_scores], quar_out=0)  # WRONG!

# Test 2: Proof with manipulated counters
# Set viol=k manually without k violations
prover.prove(viol_init=2, witness=[high_scores], quar_out=1)  # WRONG!

# Test 3: Invalid EMA update
# Force ema_next = arbitrary value
prover.prove(ema_init=0.9, witness=[0.5], ema_final=0.99)  # WRONG!
```

**Expected:** All should FAIL verification

## Security Level Analysis

### 10. FRI Parameters Review 🟢 LOW PRIORITY
**Current:** 42 queries, 8× blowup, 0 grinding → ~96-bit conjectured security

**Verification:**
- [ ] Confirm 96-bit is sufficient for PoGQ threat model (yes, for FL sybil detection)
- [ ] Compare to RISC Zero's 100-bit claim
- [ ] Document in paper: "Security-performance trade-off"

### 11. Cryptographic Assumptions 🟢 LOW PRIORITY
**Winterfell uses:**
- Blake3 for hashing (Fiat-Shamir)
- FRI for polynomial commitment
- Field: Probably Goldilocks (2^64 - 2^32 + 1)

**Verification:**
- [ ] Document exact field used
- [ ] Confirm no known attacks on FRI with these parameters
- [ ] Reference Winterfell security audit (if exists)

## Edge Case Testing

### 12. Fix Failing Tests 🟡 MEDIUM PRIORITY
**Current:** 2/9 tests failing
- `test_hysteresis_k_minus_one`: k-1 violations boundary
- `test_warmup_override`: Warmup period edge case

**Action:**
- [ ] Debug: Add trace logging to see exact state at failure
- [ ] Fix: Adjust trace_length or initial conditions
- [ ] Verify: Not actual soundness bugs, just test setup issues

### 13. Comprehensive Edge Case Suite 🟡 MEDIUM PRIORITY
**Add tests for:**
- [ ] Exact threshold match (x_t = threshold)
- [ ] Maximum field values
- [ ] Zero values (ema=0, threshold=0)
- [ ] Single-step traces (T=1, minimal valid)
- [ ] Longest trace (T=2^16 if feasible)
- [ ] All-zeros witness
- [ ] Alternating violations/clears (no convergence)

## Recommended Priority Order

**Week 1 (Before paper submission):**
1. ✅ Dual-backend smoke test (output agreement on 100 random cases)
2. ✅ Soundness testing (try to generate fake proofs - confirm failure)
3. ✅ Fix 2 failing tests
4. ✅ Public input integrity test

**Week 2-3 (Before production):**
5. Add range checks for counters/remainder (critical for soundness)
6. Selector independence audit
7. Comprehensive edge case suite

**Week 4+ (Hardening):**
8. Field overflow testing
9. Formal constraint audit
10. External security review

## Honest Assessment

**Current Status:** Production-ready for **trusted prover** scenarios (our own FL nodes)

**NOT production-ready for:** Adversarial prover scenarios without range checks

**Why the speedup is legitimate:**
- zkVM overhead is REAL (10-100× for simple circuits)
- Our circuit is genuinely simple (counters + multiplication)
- STARK scaling favors wide traces with simple constraints
- **But** we're trading off security completeness for performance

**Recommendation:**
1. Ship dual-backend with Winterfell as "fast path" + RISC Zero as "verification path"
2. Add range checks before using Winterfell in adversarial settings
3. Document limitations in paper
4. Get external audit before claiming "production-grade"
