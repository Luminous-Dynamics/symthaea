# RISC Zero Provenance Integration - Recommended Path Forward

**Date**: 2025-11-10
**Status**: PIVOT TO RISC ZERO - Avoids Winterfell Degree Validation Blocker

## 🎯 Key Discovery

After investigating the existing RISC Zero implementation, **RISC Zero avoids the entire Winterfell constraint degree validation blocker**. RISC Zero doesn't validate AIR constraint degrees - it just proves Rust execution correctness in the zkVM.

## 📐 Current Architecture

### 1. vsv-core (Shared Business Logic) ✅
**Location**: `vsv-core/src/pogq.rs`
**Purpose**: Core PoGQ decision logic used by BOTH backends

```rust
pub struct PublicInputs {
    // PoGQ parameters (fixed-point)
    pub beta_fp: Fixed,
    pub w: u64,
    pub k: u64,
    pub m: u64,
    pub threshold_fp: Fixed,

    // Previous state
    pub ema_prev_fp: Fixed,
    pub consec_viol_prev: u64,
    pub consec_clear_prev: u64,
    pub quarantined_prev: u64,

    // Current round
    pub current_round: u64,

    // Expected output
    pub quarantine_out: u64,
}
```

**Status**: ❌ No provenance fields

### 2. winterfell-pogq (Winterfell Backend) 🚧
**Location**: `winterfell-pogq/src/air.rs`
**Purpose**: STARK prover using Facebook's Winterfell v0.13.1

```rust
pub struct PublicInputs {
    // PoGQ parameters (u64 for Q16.16)
    pub beta: u64,
    pub w: u64,
    pub k: u64,
    pub m: u64,
    pub threshold: u64,

    // Initial state
    pub ema_init: u64,
    pub viol_init: u64,
    pub clear_init: u64,
    pub quar_init: u64,
    pub round_init: u64,

    // Expected output
    pub quar_out: u64,

    // Trace length
    pub trace_length: usize,

    // ✅ PROVENANCE (added)
    pub prov_hash: [u64; 4],    // Blake3 hash
    pub profile_id: u32,        // S128/S192
    pub air_rev: u32,           // Schema version
}
```

**Status**:
- ✅ Provenance integration 95% complete (Option B: BaseElement fields)
- ❌ Blocked by constraint degree validation (static vs dynamic degrees)
- ❌ 24-27 tests fail with degree mismatches

### 3. RISC Zero (host + guest) ✅
**Location**: `host/src/main.rs`, `methods/guest/src/main.rs`
**Purpose**: Alternative STARK backend using RISC Zero zkVM

**Host** (proof generator):
```rust
use vsv_core::pogq::{PrivateWitness, PublicInputs};

let public: PublicInputs = serde_json::from_str(&public_json)?;
let witness: PrivateWitness = serde_json::from_str(&witness_json)?;

let env = ExecutorEnv::builder()
    .write(&public)?
    .write(&witness)?
    .build()?;

let prove_info = prover.prove(env, METHOD_ELF)?;
```

**Guest** (zkVM program):
```rust
use vsv_core::pogq::{compute_decision, PublicInputs, PrivateWitness};

fn main() {
    let public: PublicInputs = env::read();
    let witness: PrivateWitness = env::read();
    let output = compute_decision(&public, &witness).expect("valid decision");

    assert_eq!(output.quarantine_out, public.quarantine_out);
    env::commit_slice(&journal_bytes);
}
```

**Status**:
- ✅ Full RISC Zero zkVM setup working
- ✅ No AIR constraint degree validation
- ✅ Just proves Rust execution correctness
- ❌ No provenance fields yet (uses vsv-core's PublicInputs directly)

## 🔍 Architectural Analysis

### Current Type Hierarchy

```
vsv-core::pogq::PublicInputs (lightweight, no provenance)
    ↑                                    ↑
    |                                    |
    |                                    |
winterfell-pogq::PublicInputs      RISC Zero (uses vsv-core directly)
(extended with provenance)         (no provenance yet)
```

### The Key Difference

**Winterfell**:
- Declares AIR constraints with static degrees
- Formula: `(base_degree - 1) * (trace_length - 1)`
- Boolean constraints (base degree 2) → adjusted degree 7 (trace_length=8)
- **Problem**: Test semantics force certain columns constant → degree 0 ≠ 7

**RISC Zero**:
- No AIR constraints
- Just runs Rust code in zkVM
- Proves execution correctness
- **No degree validation blocker** ✅

## 🚀 Recommended Path: Add Provenance to vsv-core

### Why This Approach

1. **Uniform API**: Both backends use same PublicInputs type
2. **Single Source of Truth**: Provenance defined once in vsv-core
3. **Simpler Integration**: RISC Zero automatically gets provenance
4. **Clean Architecture**: Business logic (vsv-core) includes provenance metadata

### Implementation Plan

#### Step 1: Add Provenance Fields to vsv-core (30 minutes)

**File**: `vsv-core/src/pogq.rs`

```rust
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PublicInputs {
    // PoGQ parameters (fixed-point)
    pub beta_fp: Fixed,
    pub w: u64,
    pub k: u64,
    pub m: u64,
    pub threshold_fp: Fixed,

    // Previous state
    pub ema_prev_fp: Fixed,
    pub consec_viol_prev: u64,
    pub consec_clear_prev: u64,
    pub quarantined_prev: u64,

    // Current round
    pub current_round: u64,

    // Expected output
    pub quarantine_out: u64,

    // 🆕 PROVENANCE (tamper-evident configuration commitment)
    // Blake3 hash of (Rust version, git commit, build timestamp, security profile)
    // Split into 4× u64 for field element compatibility
    #[cfg_attr(feature = "serde", serde(default = "default_prov_hash"))]
    pub prov_hash: [u64; 4],

    // Security profile ID (128 for S128, 192 for S192)
    #[cfg_attr(feature = "serde", serde(default = "default_profile_id"))]
    pub profile_id: u32,

    // AIR schema revision (increment on constraint/width changes)
    #[cfg_attr(feature = "serde", serde(default = "default_air_rev"))]
    pub air_rev: u32,
}

// Default values for backward compatibility (mode=off)
fn default_prov_hash() -> [u64; 4] {
    [0, 0, 0, 0]
}

fn default_profile_id() -> u32 {
    0
}

fn default_air_rev() -> u32 {
    1
}
```

**Rationale**:
- `#[serde(default)]` ensures backward compatibility with existing JSON test files
- Zero values mean "provenance disabled" (mode=off)
- Non-zero values mean "provenance enabled" (mode=strict)

#### Step 2: Add Blake3 Provenance Module to vsv-core (30 minutes)

**File**: `vsv-core/src/provenance.rs`

```rust
//! Provenance hash computation for VSV-STARK proofs
//!
//! Computes Blake3 hash of (Rust version, git commit, build timestamp, profile)
//! to create tamper-evident configuration commitment.

use blake3::Hasher;

/// Security profile identifiers
pub const S128_PROFILE_ID: u32 = 128;
pub const S192_PROFILE_ID: u32 = 192;

/// AIR schema revision (increment on breaking changes)
pub const AIR_SCHEMA_REV: u32 = 1;

/// Compute provenance hash from build metadata
///
/// Hash input: "rust={rust_ver}|git={git_rev}|build={timestamp}|profile={profile_id}|air={air_rev}"
pub fn compute_provenance_hash(
    rust_version: &str,
    git_commit: &str,
    build_timestamp: &str,
    profile_id: u32,
    air_rev: u32,
) -> [u64; 4] {
    let input = format!(
        "rust={}|git={}|build={}|profile={}|air={}",
        rust_version, git_commit, build_timestamp, profile_id, air_rev
    );

    let hash = blake3::hash(input.as_bytes());
    let bytes = hash.as_bytes(); // 32 bytes

    // Split into 4× u64 (little-endian)
    [
        u64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]),
        u64::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]]),
        u64::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23]]),
        u64::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31]]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_provenance_hash() {
        let hash = compute_provenance_hash("1.85.0", "abc123", "2025-11-10", S128_PROFILE_ID, AIR_SCHEMA_REV);

        // Hash should be deterministic
        let hash2 = compute_provenance_hash("1.85.0", "abc123", "2025-11-10", S128_PROFILE_ID, AIR_SCHEMA_REV);
        assert_eq!(hash, hash2);

        // Different input should produce different hash
        let hash3 = compute_provenance_hash("1.85.0", "xyz789", "2025-11-10", S128_PROFILE_ID, AIR_SCHEMA_REV);
        assert_ne!(hash, hash3);
    }
}
```

**Update vsv-core/src/lib.rs**:
```rust
pub mod provenance;

pub use provenance::{compute_provenance_hash, S128_PROFILE_ID, S192_PROFILE_ID, AIR_SCHEMA_REV};
```

#### Step 3: Update RISC Zero Host to Compute Provenance (30 minutes)

**File**: `host/src/main.rs`

```rust
use vsv_core::{
    pogq::{PrivateWitness, PublicInputs},
    provenance::{compute_provenance_hash, S128_PROFILE_ID, AIR_SCHEMA_REV},
};
use std::env;

fn main() -> Result<()> {
    // ... existing argument parsing ...

    // Read and parse inputs
    let mut public: PublicInputs = serde_json::from_str(&public_json)?;
    let witness: PrivateWitness = serde_json::from_str(&witness_json)?;

    // 🆕 Compute provenance hash (if enabled)
    let provenance_mode = env::var("VSV_PROVENANCE_MODE").unwrap_or_else(|_| "off".to_string());

    if provenance_mode == "strict" {
        let rust_version = env!("CARGO_PKG_RUST_VERSION");
        let git_commit = option_env!("GIT_COMMIT_HASH").unwrap_or("dev");
        let build_timestamp = env!("BUILD_TIMESTAMP");

        public.prov_hash = compute_provenance_hash(
            rust_version,
            git_commit,
            build_timestamp,
            S128_PROFILE_ID, // or read from CLI arg
            AIR_SCHEMA_REV,
        );
        public.profile_id = S128_PROFILE_ID;
        public.air_rev = AIR_SCHEMA_REV;

        println!("🔒 Provenance enabled:");
        println!("   Hash: {:016x}{:016x}{:016x}{:016x}",
                 public.prov_hash[0], public.prov_hash[1],
                 public.prov_hash[2], public.prov_hash[3]);
        println!("   Profile: S{}", public.profile_id);
        println!("   AIR Rev: {}", public.air_rev);
    } else {
        println!("⚠️  Provenance disabled (mode={})", provenance_mode);
    }

    // ... rest of existing code ...
}
```

#### Step 4: Update RISC Zero Guest to Verify Provenance (30 minutes)

**File**: `methods/guest/src/main.rs`

```rust
use risc0_zkvm::guest::env;
use vsv_core::{
    pogq::{compute_decision, PublicInputs, PrivateWitness},
    provenance::{compute_provenance_hash, AIR_SCHEMA_REV},
};

fn main() {
    let public: PublicInputs = env::read();
    let witness: PrivateWitness = env::read();

    // 🆕 Verify provenance if enabled (non-zero hash)
    if public.prov_hash != [0, 0, 0, 0] {
        // In production, guest would recompute hash from embedded build metadata
        // For now, just verify AIR schema revision matches
        assert_eq!(
            public.air_rev, AIR_SCHEMA_REV,
            "AIR schema revision mismatch: expected {}, got {}",
            AIR_SCHEMA_REV, public.air_rev
        );
    }

    // Compute decision
    let output = compute_decision(&public, &witness).expect("valid decision");

    // Verify output matches expected
    assert_eq!(
        output.quarantine_out, public.quarantine_out,
        "Computed decision doesn't match expected"
    );

    // Commit journal
    let journal = DecisionJournal {
        mode: if public.prov_hash != [0, 0, 0, 0] { "strict" } else { "off" }.to_string(),
        current_round: public.current_round,
        ema_t_fp: output.ema_t_fp.to_raw(),
        consec_viol_t: output.consec_viol_t,
        consec_clear_t: output.consec_clear_t,
        quarantine_out: output.quarantine_out,
        // 🆕 Include provenance in journal
        prov_hash: public.prov_hash,
        profile_id: public.profile_id,
        air_rev: public.air_rev,
    };

    env::commit_slice(&rmp_serde::to_vec(&journal).expect("journal serialization"));
}

#[derive(Serialize, Deserialize)]
struct DecisionJournal {
    mode: String,
    current_round: u64,
    ema_t_fp: i32,
    consec_viol_t: u64,
    consec_clear_t: u64,
    quarantine_out: u64,
    // 🆕 Provenance
    prov_hash: [u64; 4],
    profile_id: u32,
    air_rev: u32,
}
```

#### Step 5: Update winterfell-pogq to Use vsv-core Provenance (1 hour)

**Challenge**: Winterfell's PublicInputs uses u64, vsv-core uses Fixed (i32)

**Solution**: Add conversion helper in winterfell-pogq:

```rust
impl PublicInputs {
    /// Convert from vsv-core's PublicInputs to Winterfell's
    pub fn from_core(core: &vsv_core::pogq::PublicInputs, trace_length: usize) -> Self {
        Self {
            beta: core.beta_fp.to_raw() as u64,
            w: core.w,
            k: core.k,
            m: core.m,
            threshold: core.threshold_fp.to_raw() as u64,
            ema_init: core.ema_prev_fp.to_raw() as u64,
            viol_init: core.consec_viol_prev,
            clear_init: core.consec_clear_prev,
            quar_init: core.quarantined_prev,
            round_init: core.current_round,
            quar_out: core.quarantine_out,
            trace_length,
            // 🆕 Copy provenance directly
            prov_hash: core.prov_hash,
            profile_id: core.profile_id,
            air_rev: core.air_rev,
        }
    }
}
```

#### Step 6: Update Test Fixtures (1 hour)

All JSON test files need provenance fields (with defaults):

```json
{
  "beta_fp": 55705,
  "w": 3,
  "k": 2,
  "m": 3,
  "threshold_fp": 58982,
  "ema_prev_fp": 60000,
  "consec_viol_prev": 0,
  "consec_clear_prev": 2,
  "quarantined_prev": 0,
  "current_round": 5,
  "quarantine_out": 0,
  "prov_hash": [0, 0, 0, 0],
  "profile_id": 0,
  "air_rev": 1
}
```

**Automation**: Create script to update all test files:

```bash
# Update all test JSON files with provenance defaults
find test_inputs -name "public*.json" -exec \
  jq '. + {prov_hash: [0,0,0,0], profile_id: 0, air_rev: 1}' {} -c \
  > {}.new && mv {}.new {} \;
```

## ✅ Benefits of This Approach

1. **Single Source of Truth**: Provenance defined in vsv-core
2. **RISC Zero Ready**: Automatically gets provenance (no degree validation blocker)
3. **Winterfell Compatible**: Can still use Winterfell if degree issues resolved
4. **Backward Compatible**: `#[serde(default)]` handles old JSON files
5. **Clean Architecture**: Business logic layer includes metadata
6. **Safety Switch**: `VSV_PROVENANCE_MODE=off` disables provenance
7. **Dual Backend**: Can ship with RISC Zero primary, Winterfell optional

## 📊 Estimated Timeline

| Task | Time | Complexity |
|------|------|------------|
| Step 1: Add fields to vsv-core | 30 min | Low |
| Step 2: Blake3 module in vsv-core | 30 min | Low |
| Step 3: Update RISC Zero host | 30 min | Low |
| Step 4: Update RISC Zero guest | 30 min | Medium |
| Step 5: Update winterfell-pogq | 1 hour | Medium |
| Step 6: Update test fixtures | 1 hour | Low |
| **Total** | **4 hours** | **Low-Medium** |

## 🎯 Comparison with Winterfell-Only Approach

| Aspect | Winterfell-Only | vsv-core Provenance (RISC Zero Primary) |
|--------|-----------------|----------------------------------------|
| Degree Validation Blocker | ❌ Unsolved | ✅ RISC Zero avoids it |
| Test Suite Passing | ❌ 24-27 tests fail | ✅ All tests should pass |
| Implementation Time | 2.5 hours (Path 1) | 4 hours |
| Production Ready | ⚠️ Partial (ignored tests) | ✅ Fully tested |
| Backend Flexibility | ❌ Winterfell only | ✅ RISC Zero + Winterfell |
| Architecture Cleanliness | ⚠️ Backend-specific | ✅ Unified API |

## 🚀 Recommended Next Step

**Proceed with vsv-core provenance integration** using the 6-step plan above. This:

1. ✅ Avoids Winterfell's degree validation blocker entirely
2. ✅ Makes RISC Zero the primary production backend
3. ✅ Preserves option to use Winterfell later (if degree issues resolved)
4. ✅ Creates clean, unified provenance API
5. ✅ Ships fully tested provenance integration

**ETA**: 4 hours to complete all 6 steps

---

**Status**: Awaiting approval to proceed with vsv-core provenance integration (RISC Zero primary path)
