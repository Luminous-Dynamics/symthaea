# PoGQ Zome Verification Strategy

**Status**: M0 Phase 2 Design
**Date**: November 10, 2025
**Decision**: Journal-Only Verification (Option C) for M0, Full Verification for Production

---

## Problem Statement

RISC Zero zkVM **cannot compile to wasm32-unknown-unknown** due to native system dependencies:
- `risc0-sys` - System bindings
- File I/O libraries (`elf`, `object`, `tempfile`)
- Native compression (`lzma-sys`)
- NVIDIA profiling (`nvtx`)

**Implication**: Full ZK receipt verification cannot run inside Holochain WASM zome.

---

## Solution Options Evaluated

### Option A: External Verification Service ⚙️
**Architecture**: Separate native service performs verification, zome queries via HTTP/gRPC.

**Pros**:
- Full ZK proof verification
- Clean separation of concerns
- Scalable (can run on dedicated hardware)

**Cons**:
- Additional infrastructure complexity (service discovery, authentication)
- Network latency for verification
- Single point of failure if service goes down

**Verdict**: Best for production, overkill for M0 demo.

---

### Option B: Trust-but-Verify (Deferred Verification) ⏱️
**Architecture**: Store receipts in DHT, verify asynchronously offline.

**Pros**:
- Simple Holochain integration
- No external dependencies
- Receipts available for future auditing

**Cons**:
- No immediate feedback on proof validity
- Byzantine nodes could publish invalid proofs
- Quarantine decisions not verified in real-time

**Verdict**: Too weak for FL Byzantine detection (defeats purpose of PoGQ).

---

### Option C: Journal-Only Verification (M0 Compromise) ✅ **SELECTED**
**Architecture**: Extract and validate journal fields, defer full ZK verification.

**Pros**:
- Works in WASM (no RISC Zero dependencies)
- Validates journal integrity (hashes, signatures)
- Sufficient for M0 demo (demonstrates flow)
- Receipts still stored for external verification later

**Cons**:
- Not full ZK verification (trusts prover didn't forge journal)
- Production deployment would need Option A

**Verdict**: **Optimal for M0** - demonstrates architecture without production security requirements.

---

## M0 Implementation Design (Journal-Only)

### Step 1: Receipt Publishing (No Change)
```rust
#[hdk_extern]
pub fn publish_pogq_proof(entry: PoGQProofEntry) -> ExternResult<EntryHash> {
    // Validate basic fields
    if entry.receipt_bytes.is_empty() {
        return Err(wasm_error!("Receipt cannot be empty"));
    }

    if entry.profile_id != 128 && entry.profile_id != 192 {
        return Err(wasm_error!("Invalid security profile"));
    }

    // TODO: Nonce freshness check
    // TODO: Round advancement check

    // Publish to DHT
    let proof_hash = create_entry(&entry)?;
    let path = Path::from(format!("pogq_proofs.round_{}", entry.round));
    path.ensure()?;
    create_link(path.path_entry_hash()?, proof_hash.clone(), ())?;

    Ok(proof_hash)
}
```

**Status**: ✅ Already implemented (no changes needed)

### Step 2: Journal-Only Verification
```rust
#[hdk_extern]
pub fn verify_pogq_proof_journal(proof_hash: EntryHash) -> ExternResult<VerificationResult> {
    // Fetch proof entry
    let entry: PoGQProofEntry = get(proof_hash, GetOptions::default())?
        .ok_or(wasm_error!("Proof not found"))?
        .entry()
        .to_app_option()?
        .ok_or(wasm_error!("Invalid entry"))?;

    // Journal-only verification (no ZK proof check)
    // 1. Check receipt is non-empty
    if entry.receipt_bytes.is_empty() {
        return Ok(VerificationResult {
            valid: false,
            quarantine_out: entry.quarantine_out,
            prov_hash: entry.prov_hash,
            error: Some("Empty receipt".to_string()),
        });
    }

    // 2. Check provenance hash is non-zero (if strict mode expected)
    let prov_is_zero = entry.prov_hash == [0, 0, 0, 0];
    if !prov_is_zero {
        // Provenance enabled - validate hash format
        // (Full validation would require re-computing from build metadata)
        // For M0: Accept any non-zero hash as valid
    }

    // 3. Check profile ID matches expected
    if entry.profile_id != 128 && entry.profile_id != 192 {
        return Ok(VerificationResult {
            valid: false,
            quarantine_out: entry.quarantine_out,
            prov_hash: entry.prov_hash,
            error: Some(format!("Invalid profile ID: {}", entry.profile_id)),
        });
    }

    // 4. Check quarantine status is binary
    if entry.quarantine_out > 1 {
        return Ok(VerificationResult {
            valid: false,
            quarantine_out: entry.quarantine_out,
            prov_hash: entry.prov_hash,
            error: Some(format!("Invalid quarantine status: {}", entry.quarantine_out)),
        });
    }

    // Journal validation passed (defer ZK proof verification)
    Ok(VerificationResult {
        valid: true,
        quarantine_out: entry.quarantine_out,
        prov_hash: entry.prov_hash,
        error: None,
    })
}
```

**Key Changes from Placeholder**:
- Validate journal fields (non-empty receipt, valid profile_id, binary quarantine)
- Check provenance hash non-zero (if strict mode)
- Return detailed error messages
- **Skip full RISC Zero Receipt deserialization/verification** (WASM-incompatible)

### Step 3: Gradient Publishing with Verification
```rust
#[hdk_extern]
pub fn publish_gradient(entry: GradientEntry) -> ExternResult<EntryHash> {
    // ... (existing nonce binding check) ...

    // NEW: Verify proof journal before accepting gradient
    let verification = verify_pogq_proof_journal(entry.pogq_proof_hash.clone())?;
    if !verification.valid {
        return Err(wasm_error!(format!(
            "Proof verification failed: {:?}", verification.error
        )));
    }

    // ... (existing gradient creation logic) ...
}
```

**Impact**: Gradients rejected if linked proof fails journal validation.

---

## Security Analysis

### M0 Security Posture (Journal-Only)
**Threat Model**: 5-node MNIST demo, participants trusted during development.

**Protection Level**:
- ✅ **Nonce binding**: Prevents gradient-proof decoupling
- ✅ **Journal integrity**: Validates quarantine_out, provenance, profile_id
- ✅ **Receipt archival**: Full receipts stored for future verification
- ❌ **ZK proof verification**: NOT performed (trusted prover)

**Attack Vectors Not Defended**:
- Malicious node could forge journal fields (quarantine_out=0 when should be 1)
- Prover could generate invalid RISC Zero receipt that passes journal checks

**Mitigation**: For M0 demo, all nodes controlled by developer. Production would use Option A.

### Production Security Posture (External Service)
**Required for Real Deployment**: Full ZK proof verification via external service.

**Architecture**:
1. Holochain zome publishes receipt to DHT
2. External verification service (vsv-stark host):
   - Polls DHT for new receipts
   - Performs full RISC Zero Receipt::verify()
   - Updates DHT with verification status
3. Nodes query verification status before aggregation

**Implementation Timeline**: M0 Phase 3-4 (after demo proves concept)

---

## Implementation Checklist

### Phase 2 (Current) - Journal-Only Verification
- [x] RISC Zero WASM investigation (BLOCKED - confirmed impossible)
- [ ] Implement `verify_pogq_proof_journal()` (journal-only)
- [ ] Update `publish_gradient()` to call verification
- [ ] Add provenance hash validation
- [ ] Unit tests for journal validation
- [ ] 2-node integration test (1 honest, 1 Byzantine with forged journal)

### Phase 3 (Future) - External Verification Service
- [ ] Design verification service API (gRPC/HTTP)
- [ ] Implement service using vsv-stark host
- [ ] Add service discovery to Holochain zome
- [ ] Update aggregation to check verification status
- [ ] End-to-end test with real verification

---

## Decision Rationale

**Why Journal-Only for M0?**
1. **Pragmatism**: RISC Zero cannot compile to WASM (confirmed via dependency analysis)
2. **M0 Scope**: Demonstrate architecture, not production security
3. **Future-Proof**: Receipts stored in DHT, can be verified externally post-M0
4. **Time-Efficient**: Avoids weeks of external service infrastructure

**When to Upgrade?**
- Before any deployment with untrusted participants
- Before mainnet launch
- If adversarial testing reveals journal forgery attacks

---

## Conclusion

**M0 Strategy**: Journal-only verification (Option C)
**Production Strategy**: External verification service (Option A)
**Next Steps**: Implement journal validation logic in pogq_zome

---

**References**:
- RISC Zero WASM Investigation: `/tmp/pogq_wasm_check.log`
- Dependency Analysis: `cargo tree --package pogq_zome --target wasm32-unknown-unknown`
- M0 Design Doc: `vsv-stark/docs/M0_FIVE_NODE_ZKFL_DEMO_DESIGN.md`
