# Remediation Execution Plan

**Date**: 2026-01-19
**Priority**: Critical issues first, then high, medium

---

## Phase 1: Critical Error Handling (Immediate)

### 1.1 Fix Consensus Proposal Unwraps
**File**: `libs/rb-bft-consensus/src/consensus.rs`
**Lines**: 615, 644
**Issue**: `.unwrap()` on proposal that could be None
**Fix**: Replace with proper error handling

### 1.2 Fix Reputation Tolerance
**File**: `libs/rb-bft-consensus/src/consensus.rs`
**Line**: 430
**Issue**: 0.01 tolerance allows vote weight manipulation
**Fix**: Tighten to 0.001 or use snapshot-based validation

---

## Phase 2: Dependency Security (Immediate)

### 2.1 Document MD5 Usage
**File**: `libs/fl-aggregator/Cargo.toml`
**Issue**: MD5 is cryptographically broken
**Analysis**: Used only for display hashes, NOT security
**Fix**: Add comment documenting non-security usage, consider blake3 for consistency

---

## Phase 3: ZK Precision Improvements

### 3.1 Improve Floating-Point Validation
**File**: `libs/kvector-zkp/src/prover.rs`
**Lines**: 135-149
**Issue**: f32 precision issues in variance calculation
**Fix**: Use more conservative thresholds and add NaN/Inf checks

---

## Phase 4: Test Coverage

### 4.1 Add Feldman-DKG Core Tests
**Priority**: CRITICAL - VSS properties unverified

### 4.2 Add MATL-Bridge Trust Formula Tests
**Priority**: CRITICAL - Core trust calculation untested

---

## Execution Order

1. ✅ Fix consensus unwraps (safety) - `consensus.rs:615,644`
2. ✅ Tighten reputation tolerance (security) - `consensus.rs:430`, 0.01→0.001
3. ✅ Add NaN/Inf validation in ZK (security) - `prover.rs:94-113`
4. ✅ Document MD5 usage (clarity) - `fl-aggregator/Cargo.toml`
5. ✅ Add critical tests:
   - **MATL Bridge** (`bridge.rs`): 12 new tests for trust formula verification
     - `test_matl_weights_sum_to_one` - Verify weights sum to 1.0
     - `test_trust_formula_with_known_values` - T = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy
     - `test_trust_boundary_all_zeros` - All components at minimum
     - `test_trust_boundary_all_ones` - All components at maximum
     - `test_trust_pogq_dominance` - PoGQ (0.4) vs others (0.6)
     - `test_custom_weight_configuration` - Custom weights respected
     - `test_agent_not_found_error` - Proper error handling
     - `test_trust_consistency_across_rounds` - Monotonic for good behavior
     - `test_multiple_byzantine_penalties_compound` - Compounding penalties
     - `test_stats_accuracy` - Bridge statistics accuracy
   - **Feldman DKG** (`ceremony.rs`): 10 new VSS property tests
     - `test_threshold_property_exact` - t shares reconstruct, t-1 cannot
     - `test_any_threshold_subset_reconstructs` - All C(n,t) subsets work
     - `test_public_key_commitment_consistency` - g^(sum of secrets)
     - `test_disqualification_excludes_from_reconstruction` - Disqualified excluded
     - `test_insufficient_participants_fails` - Below threshold fails
     - `test_duplicate_participant_rejected` - No duplicate IDs
     - `test_invalid_participant_id_rejected` - ID validation
     - `test_deal_submission_order_independent` - Order doesn't matter
     - `test_phase_transition_correctness` - Correct phase flow

---
