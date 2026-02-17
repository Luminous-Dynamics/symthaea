# Provenance Integration Status

**Date**: 2025-01-XX (Continuation Session)
**Phase**: Breaking Changes Complete, Integration Pending
**Est. Completion Time**: 1-2 hours remaining

## Summary

Implementing tamper-evident provenance system with Blake3 commitment to all proof generation parameters. This is a **breaking change** to PublicInputs that requires updating all test fixtures.

---

## ✅ Completed Work

### 1. PublicInputs Extension (Breaking Change)

**Added Fields**:
```rust
// 32-byte Blake3 hash split into 4× u64 (little-endian)
pub prov_hash: [u64; 4],

// Security profile ID (128 or 192 for S128/S192)
pub profile_id: u32,

// AIR schema revision (monotone version number)
pub air_rev: u32,
```

**Updated Methods**:
- `to_elements()` - Now includes 6 additional field elements (4 hash u64s + profile_id + air_rev)
- Total public inputs: 11 (original) + 6 (provenance) = 17 field elements

### 2. AIR Schema Versioning

**Added Constant**:
```rust
/// Current AIR schema revision (v1 = PoGQ v4.1 LEAN)
pub const AIR_SCHEMA_REV: u32 = 1;
```

**Exported** from `lib.rs` for public access

**Version History**:
- v1: PoGQ v4.1 LEAN (44 cols, 40 constraints, 32-bit range checks)

### 3. SecurityProfile Enhancement

**Added Method**:
```rust
pub fn profile_id(&self) -> u32 {
    match self {
        SecurityProfile::S128 => 128,
        SecurityProfile::S192 => 192,
    }
}
```

**Returns numeric ID** for provenance commitment

### 4. Provenance Hash Infrastructure

**Module**: `src/provenance.rs` ✅ Already complete

**Features**:
- Blake3-based canonical hashing
- Domain separation: "VSV-STARK-PROVENANCE-v1"
- Commits to: Winterfell version, Rust version, Git commit, build mode, AIR config, ProofOptions, SecurityProfile
- 4/4 tests passing

---

## ⏳ Remaining Work

### Step 1: Update Prover to Compute Provenance (Est. 30-45 min)

**In `prover.rs`**:

```rust
pub fn prove_exec(
    &self,
    public: PublicInputs,  // NOTE: Now expects prov_hash to be populated
    witness_scores: Vec<u64>,
) -> Result<ProofResult, String> {
    // ... existing code ...
}
```

**Options**:

**Option A: Prover computes hash** (recommended)
- Extract PublicInputs params into helper struct
- Pass to `ProvenanceHash::compute()`
- Populate `public.prov_hash` before proof generation
- **Pros**: Automatic, no user error
- **Cons**: User can't pre-compute hash

**Option B: User provides hash**
- User calls `ProvenanceHash::compute()` manually
- Passes populated PublicInputs to `prove_exec()`
- **Pros**: Explicit, user control
- **Cons**: Easy to forget, error-prone

**Recommendation**: Option A with validation

### Step 2: Update All Test Fixtures (Est. 30-45 min)

**Files to Update**:
- `src/prover.rs` - `test_prove_and_verify()`
- `src/bin/lean_benchmarks.rs` - All 3 scenarios
- `src/bin/s192_spot_bench.rs` - S192 benchmark
- Any other test files using PublicInputs

**Required Changes**:
```rust
// OLD
let public = PublicInputs {
    beta: q(0.85),
    // ... 11 fields ...
    trace_length: 32,
};

// NEW
let public = PublicInputs {
    beta: q(0.85),
    // ... 11 original fields ...
    trace_length: 32,
    prov_hash: [0, 0, 0, 0],  // Placeholder (prover will compute)
    profile_id: 128,           // S128
    air_rev: AIR_SCHEMA_REV,  // Current version
};
```

### Step 3: Add Verifier Check (Est. 15-30 min)

**In `prover.rs::verify_proof()`**:

```rust
pub fn verify_proof(&self, proof_bytes: &[u8], public: PublicInputs) -> Result<bool, String> {
    // BEFORE FRI verification:

    // 1. Compute expected provenance hash
    let expected_hash = ProvenanceHash::compute(
        self.profile,
        &self.options,
        "PoGQ-v4.1",
        44,  // TRACE_WIDTH
        40,  // constraint count
    );

    // 2. Convert to u64×4 LE
    let expected_u64s = hash_to_u64x4_le(expected_hash.as_bytes());

    // 3. Compare with proof's provenance
    if public.prov_hash != expected_u64s {
        return Err(format!(
            "Provenance mismatch: expected {:?}, got {:?}",
            expected_u64s,
            public.prov_hash
        ));
    }

    // 4. Verify profile_id matches
    if public.profile_id != self.profile.profile_id() {
        return Err(format!(
            "Profile mismatch: expected {}, got {}",
            self.profile.profile_id(),
            public.profile_id
        ));
    }

    // 5. Verify AIR revision
    if public.air_rev != AIR_SCHEMA_REV {
        return Err(format!(
            "AIR revision mismatch: expected {}, got {}",
            AIR_SCHEMA_REV,
            public.air_rev
        ));
    }

    // THEN proceed with existing FRI verification...
    let proof = Proof::from_bytes(proof_bytes)?;
    // ...
}
```

**Utility Function Needed**:
```rust
fn hash_to_u64x4_le(hash: &[u8; 32]) -> [u64; 4] {
    let mut result = [0u64; 4];
    for i in 0..4 {
        result[i] = u64::from_le_bytes([
            hash[i*8], hash[i*8+1], hash[i*8+2], hash[i*8+3],
            hash[i*8+4], hash[i*8+5], hash[i*8+6], hash[i*8+7],
        ]);
    }
    result
}
```

### Step 4: Update ProvenanceHash::compute() Signature (Est. 15 min)

**Current**:
```rust
pub fn compute(
    profile: SecurityProfile,
    options: &ProofOptions,
    air_version: &str,
    trace_width: usize,
    constraint_count: usize,
) -> Self
```

**Needs**: Add PoGQ runtime params (beta, k, m, threshold, trace_length)

**Reasoning**: These are part of the commitment and should be hashed

---

## 🧪 Testing Plan

### Unit Tests
- ✅ `provenance::tests` - All passing
- ⏳ `prover::tests::test_prove_and_verify` - Needs PublicInputs update
- ⏳ AIR construction tests - Needs PublicInputs update

### Integration Tests
- ⏳ S128 prove + verify with correct provenance
- ⏳ S192 prove + verify with correct provenance
- ⏳ Profile mismatch detection (S128 proof, S192 verifier)
- ⏳ AIR revision mismatch detection
- ⏳ Hash tamper detection

### Benchmarks
- ⏳ `lean-benchmarks` - Needs PublicInputs update
- ⏳ `s192_spot_bench` - Needs PublicInputs update

---

## 🚧 Build Status

**Current**: Will not compile due to breaking changes

**Expected Errors**:
- Missing fields in PublicInputs construction (all test fixtures)
- Possible mismatched public input counts in AIR assertions

**Next Command**:
```bash
cargo build --release 2>&1 | grep "error"
```

---

## 📊 Impact Assessment

### Breaking Changes
- **PublicInputs struct**: +3 fields
- **to_elements()**: +6 field elements
- **All test fixtures**: Require updates
- **Proof format**: Unchanged (provenance in public inputs, not proof body)

### Security Impact
- **Replay attack resistance**: ✅ Profile/options bound to proof
- **Mixture attack resistance**: ✅ Can't mix S128/S192 proofs
- **Configuration tampering**: ✅ Detected before FRI verification
- **Performance**: Negligible (~1 Blake3 hash + 3 field comparisons)

### Compatibility
- **Forward compat**: Old proofs will fail verification (expected)
- **Backward compat**: None (breaking change by design)
- **Migration**: Re-generate all proofs with new format

---

## ✅ Acceptance Criteria

Per user requirements:

- [x] PublicInputs extended with provenance fields
- [ ] Prover computes and embeds provenance hash
- [ ] Verifier checks hash equality before FRI
- [ ] Profile mismatch detected and rejected
- [ ] AIR revision mismatch detected and rejected
- [ ] All tests updated and passing
- [ ] Benchmarks updated and running

---

## 🎯 Next Steps (Priority Order)

1. **Add hash_to_u64x4_le() utility** (~5 min)
2. **Update prover to compute provenance** (~30 min)
3. **Update test fixtures** (~45 min)
4. **Add verifier check** (~30 min)
5. **Build and fix errors** (~30 min)
6. **Run tests** (~15 min)
7. **Run benchmarks** (~10 min)

**Total Remaining**: ~2.5 hours (conservative estimate)

---

**Status**: Foundation complete, integration in progress 🏗️
**Risk Level**: Low (well-planned breaking change)
**Recommendation**: Complete integration before proceeding to negative tests
